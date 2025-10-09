#!/bin/bash

# Usage: ./make_gif.sh /path/to/folder output.gif

folder="$1"
output="${2:-output.gif}"

if [ -z "$folder" ]; then
  echo "Usage: $0 /path/to/folder [output.gif]"
  exit 1
fi

magick "$folder"/dm*.png "$output"
echo "GIF created: $output"