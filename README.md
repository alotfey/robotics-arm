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
```
