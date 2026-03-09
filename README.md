# Kids Robotics Gesture Control (macOS ARM)

Python CLI app that maps hand gestures from a USB camera to LewanSoul miniArm motion commands.

## Quickstart

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Commands

```bash
python -m app.cli calibrate-camera --camera-index 0
python -m app.cli calibrate-gestures --camera-index 0
python -m app.cli test-robot --port /dev/tty.usb* --dry-run
python -m app.cli run --camera-index 0 --port /dev/tty.usb* --dry-run
python -m app.cli demo --duration-sec 20 --no-preview
python -m app.cli demo-robot --cycles 3 --step-deg 3 --pause-ms 250
python -m app.cli detect-hardware --config-file config/hardware_paths.json --camera-max-index 8
```

`detect-hardware` writes detected values to `config/hardware_paths.json`:
- `arm_path`: serial path for the robot arm (for example `/dev/tty.usbmodem*`)
- `stereo_camera_path`: detected camera source path (for example `/dev/video0` or `index:0`)
- `stereo_camera_index`: OpenCV camera index used by the detector
