#!/bin/zsh
ffmpeg -f image2 -framerate 2 -i plots/step_%d.png -vf scale=928x1152,setsar=1 -c:v libx264 -pix_fmt yuv420p -profile:v main -coder 0 -preset veryslow -crf 22 -threads 0 output_video.mp4
# ffmpeg -framerate 1 -i plots/step_%d.png video.mp4