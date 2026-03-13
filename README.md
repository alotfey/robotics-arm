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
python -m app.cli test-robot --live --port /dev/cu.usbmodem101 --baud-rate 9600 --protocol default-program --startup-delay-sec 2.0 --move-time-ms 700 --base-home-angle 45 --shoulder-home-angle 90 --elbow-home-angle 90 --wrist-home-angle 90 --gripper-home-angle 130 --default-base-channel B --default-shoulder-channel C --default-elbow-channel D --default-wrist-channel E --default-gripper-channel A
python -m app.cli run --camera-index 0 --port /dev/tty.usb* --dry-run
python -m app.cli demo --duration-sec 20 --no-preview
python -m app.cli demo-robot --cycles 3 --step-deg 3 --pause-ms 250 --moves-per-direction 3 --dry-run
python -m app.cli demo-robot --live --protocol default-program --port /dev/cu.usbmodem101 --baud-rate 9600 --startup-delay-sec 2.0 --move-time-ms 700 --cycles 2 --step-deg 8 --moves-per-direction 2 --pause-ms 200 --servo-hold-ms 1200 --base-home-angle 45 --shoulder-home-angle 90 --elbow-home-angle 90 --wrist-home-angle 90 --gripper-home-angle 130 --default-base-channel B --default-shoulder-channel C --default-elbow-channel D --default-wrist-channel E --default-gripper-channel A
python -m app.cli demo-robot --live --protocol arduino-pwm --port /dev/cu.usbmodem1101 --baud-rate 115200 --startup-delay-sec 2.0 --move-time-ms 700 --cycles 2 --step-deg 12 --moves-per-direction 2 --pause-ms 200 --servo-hold-ms 1500 --base-pwm-pin 3 --shoulder-pwm-pin 5 --elbow-pwm-pin 6 --wrist-pwm-pin 9 --gripper-pwm-pin 10 --gripper-open-angle 130 --gripper-closed-angle 40
python -m app.cli demo-robot --cycles 2 --step-deg 8 --pause-ms 500 --moves-per-direction 5 --live --port /dev/cu.usbmodem1101 --baud-rate 115200 --protocol lx16a --startup-delay-sec 2.0 --move-time-ms 500 --base-servo-id 1 --shoulder-servo-id 2 --elbow-servo-id 3 --wrist-servo-id 4 --gripper-servo-id 5
python -m app.cli detect-hardware --config-file config/hardware_paths.json --camera-max-index 8
python -m app.cli detect-hardware --config-file config/hardware_paths.json --camera-max-index 8 --prefer-bluetooth
python -m app.cli run --camera-index 0 --protocol default-program --port auto --prefer-bluetooth --baud-rate 9600
```

Use `--protocol arduino-pwm` only when `firmware/arduino_pwm_servo_bridge/arduino_pwm_servo_bridge.ino` is flashed on the Arduino.
Use `--protocol default-program` only when `Default_Program/MiniArm/MiniArm.ino` is flashed, with `--baud-rate 9600`.
For Default_Program after USB reconnect/reset, hold `K1+K2` for around 3 seconds until `Start...` appears.
If your gripper direction is reversed, add `--invert-gripper`.

`detect-hardware` writes detected values to `config/hardware_paths.json`:
- `arm_path`: serial path for the robot arm (for example `/dev/tty.usbmodem*`)
- `stereo_camera_path`: detected camera source path (for example `/dev/video0` or `index:0`)
- `stereo_camera_index`: OpenCV camera index used by the detector

## Bluetooth serial module support

The app supports Bluetooth serial modules the same way as USB serial:
- Pair the module in macOS first.
- Use either `--port auto --prefer-bluetooth` or a direct port such as `/dev/cu.HC-05-DevB`.
- `detect-hardware --prefer-bluetooth` will prioritize BT serial ports when writing `arm_path`.

## Arduino LX16A Bridge Probe

If your arm is connected through an Arduino USB device (for example `/dev/cu.usbmodem1101`), flash the bridge sketch and scan servo IDs.

1. Flash sketch:
   - Open `firmware/arduino_lx16a_bridge/arduino_lx16a_bridge.ino` in Arduino IDE
   - Board: Arduino Uno
   - Port: `/dev/cu.usbmodem1101` (or your current Arduino port)
   - Upload

2. Probe IDs from Python:

```bash
python scripts/probe_arduino_lx16a_bridge.py --port /dev/cu.usbmodem1101 --start-id 1 --end-id 20
```

Expected output includes lines like `FOUND <id> <position>`. If all IDs are `MISS`, the servo bus wiring/bridge wiring is still not connected correctly.
