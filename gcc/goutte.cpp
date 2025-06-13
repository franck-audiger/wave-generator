#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>
#include <iomanip>
#include <cmath>

namespace fs = std::filesystem;

void apply_ripple_effect(const std::string& image_path,
                         const std::string& frame_folder,
                         int duration = 5,
                         int fps = 30,
                         double max_amplitude = 7.0,
                         int num_waves = 20)
{
    cv::Mat image_bgr = cv::imread(image_path, cv::IMREAD_COLOR);
    if (image_bgr.empty()) {
        std::cerr << "Cannot open image: " << image_path << std::endl;
        return;
    }
    cv::Mat image;
    cv::cvtColor(image_bgr, image, cv::COLOR_BGR2RGB);

    int height = image.rows;
    int width = image.cols;

    int total_frames = duration * fps;
    int hold_frames = static_cast<int>(0.2 * fps);

    cv::Mat dx(height, width, CV_32F);
    cv::Mat dy(height, width, CV_32F);
    cv::Mat distance(height, width, CV_32F);
    cv::Mat distance_safe(height, width, CV_32F);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float fx = static_cast<float>(x - width / 2);
            float fy = static_cast<float>(y - height / 2);
            dx.at<float>(y, x) = fx;
            dy.at<float>(y, x) = fy;
            float dist = std::sqrt(fx * fx + fy * fy);
            distance.at<float>(y, x) = dist;
            distance_safe.at<float>(y, x) = dist == 0.0f ? 1.0f : dist;
        }
    }

    fs::create_directories(frame_folder);

    for (int frame_num = 0; frame_num < total_frames; ++frame_num) {
        cv::Mat distorted(image.rows, image.cols, image.type());

        if (frame_num < hold_frames || frame_num >= total_frames - hold_frames) {
            distorted = image.clone();
        } else {
            double time = static_cast<double>(frame_num) / fps;
            double progress = static_cast<double>(frame_num - hold_frames) /
                              (total_frames - 2 * hold_frames);
            double dynamic_ripple_scale = 60.0 + 100.0 * progress;
            double dynamic_damping = 300.0 + 700.0 * progress;
            double dynamic_num_waves = num_waves * (1 - 0.5 * progress);
            double wave_decay = 1 - progress;

            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    double angle_base = distance.at<float>(y, x) / dynamic_ripple_scale;
                    double angle = angle_base -
                                   time * (dynamic_num_waves * M_PI / duration);
                    double intensity = max_amplitude * 3.0 * std::sin(angle) *
                                       wave_decay *
                                       std::exp(-distance.at<float>(y, x) /
                                                dynamic_damping);
                    double offset_dx = dx.at<float>(y, x) / distance_safe.at<float>(y, x) * intensity;
                    double offset_dy = dy.at<float>(y, x) / distance_safe.at<float>(y, x) * intensity;
                    int src_x = std::clamp(static_cast<int>(x + offset_dx), 0, width - 1);
                    int src_y = std::clamp(static_cast<int>(y + offset_dy), 0, height - 1);
                    distorted.at<cv::Vec3b>(y, x) = image.at<cv::Vec3b>(src_y, src_x);
                }
            }
        }

        std::ostringstream frame_name;
        frame_name << frame_folder << "/frame_" << std::setw(4) << std::setfill('0')
                   << frame_num << ".png";
        cv::imwrite(frame_name.str(), distorted);
    }
}

void assemble_video(const std::string& frame_folder,
                    const std::string& output_video,
                    int fps = 30)
{
    std::ostringstream cmd;
    cmd << "ffmpeg -y -framerate " << fps
        << " -i " << frame_folder << "/frame_%04d.png"
        << " -c:v libx264 -pix_fmt yuv420p "
        << output_video << " >/dev/null 2>&1";
    std::system(cmd.str().c_str());
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " image_directory" << std::endl;
        return 1;
    }
    fs::path images_dir = argv[1];
    if (!fs::is_directory(images_dir)) {
        std::cerr << "Le chemin specifie n'est pas un dossier" << std::endl;
        return 1;
    }

    fs::path result_dir = "result";
    fs::create_directories(result_dir);

    std::vector<fs::path> images;
    for (const auto& entry : fs::directory_iterator(images_dir)) {
        if (!entry.is_regular_file()) continue;
        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".png" || ext == ".jpg" || ext == ".jpeg")
            images.push_back(entry.path());
    }
    std::sort(images.begin(), images.end());

    if (images.empty()) {
        std::cerr << "Aucune image PNG ou JPG trouvee dans le dossier" << std::endl;
        return 1;
    }

    for (const auto& image_path : images) {
        std::string base_name = image_path.stem().string();
        fs::path frame_folder = result_dir / (base_name + "_frames");
        fs::path output_video = result_dir / (base_name + ".mp4");

        apply_ripple_effect(image_path.string(), frame_folder.string());
        assemble_video(frame_folder.string(), output_video.string());
        std::cout << "Nom du fichier genere : " << output_video.filename().string() << std::endl;
    }

    return 0;
}

