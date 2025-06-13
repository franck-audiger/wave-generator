# Native C Version

This folder provides a native implementation of the ripple effect.
The program uses OpenCV to read the images, generate frames, and calls
`ffmpeg` to assemble the resulting MP4 videos.

## Build

Make sure the OpenCV development libraries are installed. On Debian/Ubuntu:

```bash
sudo apt-get install libopencv-dev
```

Compile with g++:

```bash
g++ -std=c++17 goutte.cpp `pkg-config --cflags --libs opencv4` -o goutte
```

## Usage

```bash
./goutte chemin/vers/dossier_images
```

Videos are written to the `result` folder with the same base name as the
input image but with an `.mp4` extension.
