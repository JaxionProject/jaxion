#!/bin/bash

# Usage: ./combine_gifs.sh

magick ../examples/cosmological_box/movie.gif -coalesce input1_frame_%04d.png
magick ../examples/dynamical_friction/movie.gif  -coalesce input2_frame_%04d.png
magick ../examples/heating_gas/movie.gif  -coalesce input3_frame_%04d.png
magick ../examples/heating_stars/movie.gif  -coalesce input4_frame_%04d.png
magick ../examples/kinetic_condensation/movie.gif -coalesce input5_frame_%04d.png
magick ../examples/logo_inverse_problem/movie.gif  -coalesce input6_frame_%04d.png
magick ../examples/soliton_binary_merger/movie.gif  -coalesce input7_frame_%04d.png
magick ../examples/soliton_merger/movie.gif  -coalesce input8_frame_%04d.png
magick ../examples/tidal_stripping/movie.gif  -coalesce input9_frame_%04d.png

magick -delay 5 -loop 0 input*.png -resize 512x512 -background black -gravity center -extent 512x512 ../jaxion.mp4

for frame in $(seq -f "%04g" 0 100); do
  magick montage input1_frame_${frame}.png input2_frame_${frame}.png input3_frame_${frame}.png \
                 input4_frame_${frame}.png input5_frame_${frame}.png input6_frame_${frame}.png \
                 input7_frame_${frame}.png input8_frame_${frame}.png input9_frame_${frame}.png \
                 -tile 3x3 -geometry +0+0 output_frame_${frame}.png
done

magick -delay 10 -loop 0 output_frame_*.png jaxion.gif

gifsicle -O3 --lossy=40 --resize 384x384 jaxion.gif -o ../jaxion.gif --colors 256

rm input*.png
rm output*.png
rm jaxion.gif
