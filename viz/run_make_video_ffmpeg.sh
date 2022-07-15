#!/bin/bash

METHOD=rootsift
FPS=2

# for METHOD in dbow rootsift superglue
for METHOD in superglue
do
  echo $METHOD

# gate
ffmpeg -framerate $FPS -i tmp/gate_1vs2/$METHOD/%06d.png -c:v libx264 -preset fast -crf 22 -pix_fmt yuv420p tmp/${METHOD}_gate_1vs2.mp4
ffmpeg -framerate $FPS -i tmp/gate_1vs3/$METHOD/%06d.png -c:v libx264 -preset fast -crf 22 -pix_fmt yuv420p tmp/${METHOD}_gate_1vs3.mp4
ffmpeg -framerate $FPS -i tmp/gate_1vs4/$METHOD/%06d.png -c:v libx264 -preset fast -crf 22 -pix_fmt yuv420p tmp/${METHOD}_gate_1vs4.mp4
ffmpeg -framerate $FPS -i tmp/gate_2vs3/$METHOD/%06d.png -c:v libx264 -preset fast -crf 22 -pix_fmt yuv420p tmp/${METHOD}_gate_2vs3.mp4
ffmpeg -framerate $FPS -i tmp/gate_2vs4/$METHOD/%06d.png -c:v libx264 -preset fast -crf 22 -pix_fmt yuv420p tmp/${METHOD}_gate_2vs4.mp4
ffmpeg -framerate $FPS -i tmp/gate_3vs4/$METHOD/%06d.png -c:v libx264 -preset fast -crf 22 -pix_fmt yuv420p tmp/${METHOD}_gate_3vs4.mp4

# # office
ffmpeg -framerate $FPS -i tmp/gate_1vs2/$METHOD/%06d.png -c:v libx264 -preset fast -crf 22 -pix_fmt yuv420p tmp/${METHOD}_gate_1vs2.mp4
ffmpeg -framerate $FPS -i tmp/gate_1vs3/$METHOD/%06d.png -c:v libx264 -preset fast -crf 22 -pix_fmt yuv420p tmp/${METHOD}_gate_1vs3.mp4
ffmpeg -framerate $FPS -i tmp/gate_2vs3/$METHOD/%06d.png -c:v libx264 -preset fast -crf 22 -pix_fmt yuv420p tmp/${METHOD}_gate_2vs3.mp4

# # backyard
ffmpeg -framerate $FPS -i tmp/gate_1vs4/$METHOD/%06d.png -c:v libx264 -preset fast -crf 22 -pix_fmt yuv420p tmp/${METHOD}_gate_1vs4.mp4
ffmpeg -framerate $FPS -i tmp/gate_2vs3/$METHOD/%06d.png -c:v libx264 -preset fast -crf 22 -pix_fmt yuv420p tmp/${METHOD}_gate_2vs3.mp4

# # courtyard
ffmpeg -framerate $FPS -i tmp/gate_1vs2/$METHOD/%06d.png -c:v libx264 -preset fast -crf 22 -pix_fmt yuv420p tmp/${METHOD}_gate_1vs2.mp4
ffmpeg -framerate $FPS -i tmp/gate_1vs3/$METHOD/%06d.png -c:v libx264 -preset fast -crf 22 -pix_fmt yuv420p tmp/${METHOD}_gate_1vs3.mp4
ffmpeg -framerate $FPS -i tmp/gate_1vs4/$METHOD/%06d.png -c:v libx264 -preset fast -crf 22 -pix_fmt yuv420p tmp/${METHOD}_gate_1vs4.mp4
ffmpeg -framerate $FPS -i tmp/gate_2vs3/$METHOD/%06d.png -c:v libx264 -preset fast -crf 22 -pix_fmt yuv420p tmp/${METHOD}_gate_2vs3.mp4
ffmpeg -framerate $FPS -i tmp/gate_2vs4/$METHOD/%06d.png -c:v libx264 -preset fast -crf 22 -pix_fmt yuv420p tmp/${METHOD}_gate_2vs4.mp4
ffmpeg -framerate $FPS -i tmp/gate_3vs4/$METHOD/%06d.png -c:v libx264 -preset fast -crf 22 -pix_fmt yuv420p tmp/${METHOD}_gate_3vs4.mp4

done