#!/bin/bash

# Usage: -r run_name -i iteration

while getopts ":r:i:" opt; do
  case $opt in
    r) run_name="$OPTARG"
    ;;
    i) iteration="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

printf "Argument run_name is %s\n" "$run_name"
printf "Argument iteration is %s\n" "$iteration"

input_path=plots/run_"$run_name"/iteration"$iteration"/step_%d.png
output_path=videos/"$run_name"

mkdir -p "$output_path"
ffmpeg -f image2 -framerate 2 -i "$input_path" -c:v libx264 -pix_fmt yuv420p -profile:v main -coder 0 -preset veryslow -crf 22 -threads 0 "$output_path"/iteration_"$iteration".mp4