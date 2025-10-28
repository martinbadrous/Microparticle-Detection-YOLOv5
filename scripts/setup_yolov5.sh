#!/usr/bin/env bash
set -e
if [ ! -d "./yolov5" ]; then
  git submodule add https://github.com/ultralytics/yolov5.git yolov5 || true
fi
python -m pip install -U pip
pip install -r yolov5/requirements.txt
pip install -r requirements.txt
echo "Setup complete."
