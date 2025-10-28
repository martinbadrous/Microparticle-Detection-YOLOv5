import subprocess, os, sys

DATA="data/data.yaml"
EPOCHS=50
IMG=640
BATCH=16
WEIGHTS=os.environ.get("YOLOV5_WEIGHTS", "yolov5s.pt")  # set to yolov5m.pt to try a larger model
HYP="yolov5_hyp_no_rotate.yaml"

cmd = [
  sys.executable, "yolov5/train.py",
  "--img", str(IMG),
  "--batch", str(BATCH),
  "--epochs", str(EPOCHS),
  "--data", DATA,
  "--weights", WEIGHTS,
  "--project", "runs/train",
  "--name", "mp_yolov5",
  "--hyp", HYP,
]
print(" ".join(cmd))
subprocess.run(cmd, check=True)
