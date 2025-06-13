import numpy as np
import cv2
import os
import subprocess
import sys
import numba
from tqdm import tqdm
from numba import njit, prange

def apply_ripple_effect(image_path, output_folder, duration=5, fps=30, max_amplitude=7.0, num_waves=20):
    # Charger l'image avec OpenCV (préserve les couleurs sans conversion implicite)
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image_np = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width, _ = image_np.shape

    total_frames = duration * fps
    hold_frames = int(0.2 * fps)  # 0.5 seconde d'image stable

    # Pré-calculs
    y_indices, x_indices = np.indices((height, width))
    center_x, center_y = width // 2, height // 2
    dx = x_indices - center_x
    dy = y_indices - center_y
    distance = np.sqrt(dx**2 + dy**2)
    distance_safe = np.where(distance == 0, 1, distance)

    os.makedirs(output_folder, exist_ok=True)

    from tqdm import tqdm

    for frame_num in tqdm(range(total_frames), desc="Génération des frames", unit="frame"):
        if frame_num < hold_frames or frame_num >= total_frames - hold_frames:
            # Image stable sans effet au début et à la fin
            distorted = image_np.copy()
        else:
            time = frame_num / fps
            progress = (frame_num - hold_frames) / (total_frames - 2 * hold_frames)

            # Fréquence et portée de l'onde diminuent progressivement
            dynamic_ripple_scale = 60.0 + 100.0 * progress
            dynamic_damping = 300 + 700 * progress

            angle_base = distance / dynamic_ripple_scale
            wave_decay = (1 - progress)
            dynamic_num_waves = num_waves * (1 - 0.5 * progress)

            angle = angle_base - time * (dynamic_num_waves * np.pi / duration)
            intensity = max_amplitude * 3.0 * np.sin(angle) * wave_decay * np.exp(-distance / dynamic_damping)

            offset_dx = (dx / distance_safe) * intensity
            offset_dy = (dy / distance_safe) * intensity

            src_x = np.clip((x_indices + offset_dx).astype(np.int32), 0, width - 1)
            src_y = np.clip((y_indices + offset_dy).astype(np.int32), 0, height - 1)

            # Appliquer la déformation accélérée avec Numba
            distorted = apply_displacement(image_np, src_y, src_x)

        frame_filename = os.path.join(output_folder, f"frame_{frame_num:04d}.png")
        cv2.imwrite(frame_filename, cv2.cvtColor(distorted, cv2.COLOR_RGB2BGR))

    print(f"✅ Images PNG générées dans : {output_folder}")

def assemble_video_from_frames(frame_folder, output_video, fps=30):
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate", str(fps),
        "-i", os.path.join(frame_folder, "frame_%04d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        output_video
    ]

    try:
        from tqdm import tqdm
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in tqdm(process.stdout, desc="Assemblage vidéo", unit="ligne"):
            pass
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd)
        print(f"✅ Vidéo générée dans : {output_video}")
    except subprocess.CalledProcessError as e:
        print("❌ Erreur lors de l'assemblage vidéo avec ffmpeg :", e)

@njit(parallel=True)
def apply_displacement(image, src_y, src_x):
    height, width = src_y.shape
    output = np.empty((height, width, 3), dtype=np.uint8)
    for y in prange(height):
        for x in prange(width):
            output[y, x, 0] = image[src_y[y, x], src_x[y, x], 0]
            output[y, x, 1] = image[src_y[y, x], src_x[y, x], 1]
            output[y, x, 2] = image[src_y[y, x], src_x[y, x], 2]
    return output

# Exemple d'utilisation modifié : le script prend maintenant en argument un
# dossier contenant des images (PNG ou JPG). Pour chaque image trouvée, l'effet
# goutte est appliqué et une vidéo MP4 est générée dans le dossier "result".
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage : python script.py dossier_images")
        sys.exit(1)

    images_dir = sys.argv[1]
    if not os.path.isdir(images_dir):
        print("Le chemin spécifié n'est pas un dossier")
        sys.exit(1)

    result_dir = "result"
    os.makedirs(result_dir, exist_ok=True)

    images = sorted(
        [f for f in os.listdir(images_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    )

    if not images:
        print("Aucune image PNG ou JPG trouvée dans le dossier")
        sys.exit(1)

    for image_name in images:
        input_image = os.path.join(images_dir, image_name)
        base_name = os.path.splitext(image_name)[0]

        frame_folder = os.path.join(result_dir, f"{base_name}_frames")
        output_video = os.path.join(result_dir, base_name + ".mp4")

        apply_ripple_effect(
            input_image,
            frame_folder,
            duration=5,
            fps=30,
            max_amplitude=7.0,
            num_waves=20,
        )
        assemble_video_from_frames(frame_folder, output_video)
