#!/bin/bash

# Usage: ./viswall_gifs.sh

# Create movie for Video Wall
# see: https://wiki.flatironinstitute.org/SCC/Hardware/Viswall

module purge
module load ffmpeg/7.1.1-nix  # or another ffmpeg/*-nix version, depending on your modules set
module load mpv

convert ../examples/cosmological_box/movie.gif -coalesce input1_frame_%04d.png
convert ../examples/dynamical_friction/movie.gif  -coalesce input2_frame_%04d.png
convert ../examples/heating_gas/movie.gif  -coalesce input3_frame_%04d.png
convert ../examples/heating_stars/movie.gif  -coalesce input4_frame_%04d.png
convert ../examples/kinetic_condensation/movie.gif -coalesce input5_frame_%04d.png
convert ../examples/logo/movie.gif  -coalesce input6_frame_%04d.png
convert ../examples/soliton_binary_merger/movie.gif  -coalesce input7_frame_%04d.png
convert ../examples/soliton_merger/movie.gif  -coalesce input8_frame_%04d.png
convert ../examples/tidal_stripping/movie.gif  -coalesce input9_frame_%04d.png

for frame in $(seq -f "%04g" 0 100); do
  montage input1_frame_${frame}.png input2_frame_${frame}.png input3_frame_${frame}.png \
                 input4_frame_${frame}.png input5_frame_${frame}.png input6_frame_${frame}.png \
                 input7_frame_${frame}.png input8_frame_${frame}.png input9_frame_${frame}.png \
                 -tile 3x3 -geometry +0+0 output_frame_${frame}.png
done


ffmpeg -framerate 12 -pattern_type glob -i 'output*.png' -c:v libvpx-vp9 -b:v 0 -crf 24 -threads 16 -row-mt 1 -pass 1 -f null /dev/null && \
ffmpeg -framerate 12 -pattern_type glob -i 'output*.png' -c:v libvpx-vp9 -b:v 0 -crf 24 -threads 16 -row-mt 1 -pass 2 jaxion.webm

rm input*.png
rm output*.png

mpv --sub-scale=0.4 --sub-file=jaxion-subtitles.srt --fs jaxion.webm

