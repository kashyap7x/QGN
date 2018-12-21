#!/bin/bash
for file in [0-9]*.png; do
  # strip the postfix (".png") off the file name
  number=${file%.png}
  # subtract 1 from the resulting number
  # i=$((number-1))
  # copy to a new name in a new folder
  mv ${file} $(printf %08d.png $number)
done
