#!/bin/bash
echo -e "The pretrained models will stored in the 'pretrained_models' folder\n"
mkdir -p pretrained_models
python -m gdown.cli "https://drive.google.com/uc?id=1CNXnqxGcVHDYzjrfZOlBge7A54OfCt2d"

echo -e "Please check that the md5sum is: e96adb318075a278e9b7d3c9a982fee9"
echo -e "+ md5sum pretrained_models.tgz"
md5sum pretrained_models.tgz

echo -e "If it is not, please rerun this script"

sleep 5
tar xfzv pretrained_models.tgz

echo -e "Cleaning\n"
rm pretrained_models.tgz

echo -e "Downloading done!"
