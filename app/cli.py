from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
from rich.console import Console

from app.camera.uvc_camera import UvcCamera
from app.models import CameraConfig, RobotConfig, RuntimeConfig, SafetyConfig
from app.robot.lewansoul_miniarm import LewanSoulConfig, LewanSoulMiniArmDriver
from app.runtime.control_loop import ControlLoop
from app.runtime.demo_mode import DemoCamera, DemoGestureClassifier
from app.runtime.hardware_discovery import (
    DEFAULT_CONFIG_PATH,
    detect_and_save_hardware_paths,
    detect_arm_path,
)
from app.vision.gestures import GestureClassifier

console = Console()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Kids robotics gesture control CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    cam = sub.add_parser("calibrate-camera", help="Preview and validate camera stream")
    _add_camera_flags(cam)

    gest = sub.add_parser("calibrate-gestures", help="Visualize gesture detection and stability")
    _add_camera_flags(gest)
    gest.add_argument("--debounce-frames", type=int, default=5)

    run = sub.add_parser("run", help="Run live gesture -> robot control loop")
    _add_camera_flags(run)
    _add_robot_flags(run)
    _add_runtime_safety_flags(run)
    run.add_argument("--no-preview", action="store_true")
    run.add_argument("--no-home", action="store_true")

    demo = sub.add_parser("demo", help="Run simulated gesture control with no hardware")
    _add_demo_camera_flags(demo)
    _add_runtime_safety_flags(demo)
    demo.add_argument("--duration-sec", type=float, default=20.0)
    demo.add_argument("--no-preview", action="store_true")
    demo.add_argument("--no-home", action="store_true")

    demo_robot = sub.add_parser(
        "demo-robot",
        help="Run 5-servo robot motion demo sequence (supports dry-run or --live hardware mode)",
    )
    _add_robot_flags(demo_robot)
    demo_robot.add_argument("--live", action="store_true")
    demo_robot.add_argument("--cycles", type=int, default=3)
    demo_robot.add_argument("--step-deg", type=float, default=3.0)
    demo_robot.add_argument("--pause-ms", type=int, default=250)
    demo_robot.add_argument("--moves-per-direction", type=int, default=3)
    demo_robot.add_argument("--servo-hold-ms", type=int, default=1200)

    test = sub.add_parser("test-robot", help="Send safe test commands to robot")
    _add_robot_flags(test)
    test.add_argument("--live", action="store_true")

    detect_hw = sub.add_parser(
        "detect-hardware",
        help="Detect connected arm and stereo camera, then write their paths to config",
    )
    detect_hw.add_argument("--config-file", type=str, default=str(DEFAULT_CONFIG_PATH))
    detect_hw.add_argument("--camera-max-index", type=int, default=8)
    detect_hw.add_argument(
        "--prefer-bluetooth",
        action="store_true",
        help="Prefer Bluetooth serial adapters over USB when selecting arm_path",
    )

    return parser


def _add_camera_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)


def _add_demo_camera_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)


def _add_robot_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--port", type=str, default="/dev/tty.usbmodem")
    parser.add_argument(
        "--prefer-bluetooth",
        action="store_true",
        help="When --port auto, prefer a Bluetooth serial adapter over USB",
    )
    parser.add_argument("--baud-rate", type=int, default=9600)
    parser.add_argument(
        "--protocol",
        type=str,
        choices=(
            "lx16a",
            "text",
            "hiwonder",
            "dual",
            "arduino-pwm",
            "pwm",
            "default-program",
            "miniarm-default",
            "miniarm",
        ),
        default="default-program",
        help="Robot serial protocol to use",
    )
    parser.add_argument(
        "--startup-delay-sec",
        type=float,
        default=2.0,
        help="Delay after opening serial port (Arduino boards often reset on open)",
    )
    parser.add_argument(
        "--move-time-ms",
        type=int,
        default=500,
        help="Servo travel time per command in milliseconds",
    )
    parser.add_argument("--base-servo-id", type=int, default=1)
    parser.add_argument("--shoulder-servo-id", type=int, default=2)
    parser.add_argument("--elbow-servo-id", type=int, default=3)
    parser.add_argument("--wrist-servo-id", type=int, default=4)
    parser.add_argument("--gripper-servo-id", type=int, default=5)
    parser.add_argument("--gripper-open-position", type=int, default=650)
    parser.add_argument("--gripper-closed-position", type=int, default=380)
    parser.add_argument("--gripper-open-angle", type=int, default=130)
    parser.add_argument("--gripper-closed-angle", type=int, default=40)
    parser.add_argument("--invert-gripper", action="store_true")
    parser.add_argument("--base-pwm-pin", type=int, default=3)
    parser.add_argument("--shoulder-pwm-pin", type=int, default=5)
    parser.add_argument("--elbow-pwm-pin", type=int, default=6)
    parser.add_argument("--wrist-pwm-pin", type=int, default=9)
    parser.add_argument("--gripper-pwm-pin", type=int, default=10)
    parser.add_argument("--base-home-angle", type=int, default=90)
    parser.add_argument("--shoulder-home-angle", type=int, default=90)
    parser.add_argument("--elbow-home-angle", type=int, default=90)
    parser.add_argument("--wrist-home-angle", type=int, default=90)
    parser.add_argument("--gripper-home-angle", type=int, default=130)
    parser.add_argument("--default-base-channel", type=str, default="B")
    parser.add_argument("--default-shoulder-channel", type=str, default="C")
    parser.add_argument("--default-elbow-channel", type=str, default="D")
    parser.add_argument("--default-wrist-channel", type=str, default="E")
    parser.add_argument("--default-gripper-channel", type=str, default="A")
    parser.add_argument(
        "--no-verify-positions",
        action="store_true",
        help="Skip LX16A position readback checks during connect/move",
    )
    parser.add_argument("--dry-run", action="store_true")


def _add_runtime_safety_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--debounce-frames", type=int, default=5)
    parser.add_argument("--no-hand-timeout-ms", type=int, default=800)
    parser.add_argument("--max-command-hz", type=float, default=10.0)
    parser.add_argument("--max-joint-step-deg", type=float, default=5.0)


def _warn_default_program_baud(protocol: str, baud_rate: int) -> None:
    if protocol not in {"default-program", "miniarm-default", "miniarm"}:
        return
    if int(baud_rate) == 9600:
        return
    console.log(
        "[yellow]WARN[/yellow] Default_Program firmware expects 9600 baud. "
        f"Current value is {baud_rate}."
    )


def _effective_baud_rate(protocol: str, baud_rate: int) -> int:
    if protocol in {"default-program", "miniarm-default", "miniarm"} and int(baud_rate) == 115200:
        console.log(
            "[yellow]INFO[/yellow] auto-switching baud rate to 9600 for Default_Program protocol."
        )
        return 9600
    return int(baud_rate)


def _resolve_robot_port(port: str, prefer_bluetooth: bool = False) -> str:
    normalized = str(port).strip()
    if normalized.lower() not in {"auto", "detect", "auto-detect"}:
        return normalized

    detected = detect_arm_path(prefer_bluetooth=prefer_bluetooth)
    if detected is None:
        if prefer_bluetooth:
            raise RuntimeError(
                "No robot serial port detected (Bluetooth preferred). "
                "Pair the module first, or pass --port /dev/cu.<your-module> explicitly."
            )
        raise RuntimeError(
            "No robot serial port detected. "
            "Pass --port /dev/cu.<device> explicitly, or retry with --port auto --prefer-bluetooth."
        )
    console.log(f"Auto-detected robot port: {detected}")
    return detected


def run_calibrate_camera(args: argparse.Namespace) -> int:
    camera = UvcCamera(args.camera_index, args.width, args.height, args.fps)
    camera.start()
    started = time.time()
    frames = 0

    console.log("Camera calibration started. Press 'q' to exit.")
    try:
        while True:
            frame = camera.read()
            frames += 1
            cv2.imshow("Camera Calibration", frame.frame_bgr)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        camera.stop()
        cv2.destroyAllWindows()

    elapsed = max(0.001, time.time() - started)
    measured_fps = frames / elapsed
    console.log(f"Frames: {frames}, elapsed: {elapsed:.2f}s, measured FPS: {measured_fps:.1f}")
    return 0


def run_calibrate_gestures(args: argparse.Namespace) -> int:
    camera = UvcCamera(args.camera_index, args.width, args.height, args.fps)
    classifier = GestureClassifier(stable_frames_required=args.debounce_frames)
    camera.start()
    console.log("Gesture calibration started. Press 'q' to exit.")

    try:
        while True:
            frame = camera.read()
            detection = classifier.detect(frame.frame_bgr, frame.timestamp_ms)
            if detection.event is not None:
                console.log(
                    f"Gesture={detection.event.name.value} confidence={detection.event.confidence:.2f} "
                    f"stable_frames={detection.event.stable_frames}"
                )
            cv2.imshow("Gesture Calibration", detection.annotated_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        classifier.close()
        camera.stop()
        cv2.destroyAllWindows()

    return 0


def run_test_robot(args: argparse.Namespace) -> int:
    if args.live and args.dry_run:
        raise ValueError("Choose either --live or --dry-run, not both")
    dry_run = not args.live
    if args.dry_run:
        dry_run = True
    port = _resolve_robot_port(
        args.port,
        prefer_bluetooth=getattr(args, "prefer_bluetooth", False),
    )
    protocol = getattr(args, "protocol", "default-program")
    baud_rate = _effective_baud_rate(protocol, getattr(args, "baud_rate", 9600))
    _warn_default_program_baud(protocol=protocol, baud_rate=baud_rate)

    driver = LewanSoulMiniArmDriver(
        LewanSoulConfig(
            port=port,
            baud_rate=baud_rate,
            dry_run=dry_run,
            protocol=protocol,
            startup_delay_sec=getattr(args, "startup_delay_sec", 2.0),
            move_time_ms=getattr(args, "move_time_ms", 500),
            base_servo_id=getattr(args, "base_servo_id", 1),
            shoulder_servo_id=getattr(args, "shoulder_servo_id", 2),
            elbow_servo_id=getattr(args, "elbow_servo_id", 3),
            wrist_servo_id=getattr(args, "wrist_servo_id", 4),
            gripper_servo_id=getattr(args, "gripper_servo_id", 5),
            gripper_open_position=getattr(args, "gripper_open_position", 650),
            gripper_closed_position=getattr(args, "gripper_closed_position", 380),
            gripper_open_angle=getattr(args, "gripper_open_angle", 130),
            gripper_closed_angle=getattr(args, "gripper_closed_angle", 40),
            verify_positions=not getattr(args, "no_verify_positions", False),
            invert_gripper=getattr(args, "invert_gripper", False),
            base_pwm_pin=getattr(args, "base_pwm_pin", 3),
            shoulder_pwm_pin=getattr(args, "shoulder_pwm_pin", 5),
            elbow_pwm_pin=getattr(args, "elbow_pwm_pin", 6),
            wrist_pwm_pin=getattr(args, "wrist_pwm_pin", 9),
            gripper_pwm_pin=getattr(args, "gripper_pwm_pin", 10),
            base_home_angle=getattr(args, "base_home_angle", 90),
            shoulder_home_angle=getattr(args, "shoulder_home_angle", 90),
            elbow_home_angle=getattr(args, "elbow_home_angle", 90),
            wrist_home_angle=getattr(args, "wrist_home_angle", 90),
            gripper_home_angle=getattr(args, "gripper_home_angle", 130),
            default_base_channel=getattr(args, "default_base_channel", "B"),
            default_shoulder_channel=getattr(args, "default_shoulder_channel", "C"),
            default_elbow_channel=getattr(args, "default_elbow_channel", "D"),
            default_wrist_channel=getattr(args, "default_wrist_channel", "E"),
            default_gripper_channel=getattr(args, "default_gripper_channel", "A"),
        ),
        console=console,
    )
    driver.connect()
    try:
        driver.home()
        driver.move_axis("base", 3.0)
        driver.move_axis("base", -3.0)
        driver.move_axis("shoulder", 3.0)
        driver.move_axis("shoulder", -3.0)
        driver.set_gripper(open=False)
        driver.set_gripper(open=True)
        driver.stop_all()
    finally:
        driver.disconnect()

    return 0


def run_control(args: argparse.Namespace) -> int:
    protocol = getattr(args, "protocol", "default-program")
    baud_rate = _effective_baud_rate(protocol, getattr(args, "baud_rate", 9600))
    port = _resolve_robot_port(
        args.port,
        prefer_bluetooth=getattr(args, "prefer_bluetooth", False),
    )
    camera_cfg = CameraConfig(
        camera_index=args.camera_index,
        width=args.width,
        height=args.height,
        fps=args.fps,
    )
    robot_cfg = RobotConfig(
        port=port,
        baud_rate=baud_rate,
        dry_run=args.dry_run,
        home_on_startup=not args.no_home,
    )
    safety_cfg = SafetyConfig(
        debounce_frames=args.debounce_frames,
        no_hand_timeout_ms=args.no_hand_timeout_ms,
        max_command_hz=args.max_command_hz,
        max_joint_step_deg=args.max_joint_step_deg,
    )
    runtime_cfg = RuntimeConfig(
        camera=camera_cfg,
        robot=robot_cfg,
        safety=safety_cfg,
        preview=not args.no_preview,
    )
    _warn_default_program_baud(protocol=protocol, baud_rate=runtime_cfg.robot.baud_rate)

    camera = UvcCamera(
        runtime_cfg.camera.camera_index,
        runtime_cfg.camera.width,
        runtime_cfg.camera.height,
        runtime_cfg.camera.fps,
    )
    classifier = GestureClassifier(stable_frames_required=runtime_cfg.safety.debounce_frames)
    driver = LewanSoulMiniArmDriver(
        LewanSoulConfig(
            port=runtime_cfg.robot.port,
            baud_rate=runtime_cfg.robot.baud_rate,
            dry_run=runtime_cfg.robot.dry_run,
            protocol=protocol,
            startup_delay_sec=getattr(args, "startup_delay_sec", 2.0),
            move_time_ms=getattr(args, "move_time_ms", 500),
            base_servo_id=getattr(args, "base_servo_id", 1),
            shoulder_servo_id=getattr(args, "shoulder_servo_id", 2),
            elbow_servo_id=getattr(args, "elbow_servo_id", 3),
            wrist_servo_id=getattr(args, "wrist_servo_id", 4),
            gripper_servo_id=getattr(args, "gripper_servo_id", 5),
            gripper_open_position=getattr(args, "gripper_open_position", 650),
            gripper_closed_position=getattr(args, "gripper_closed_position", 380),
            gripper_open_angle=getattr(args, "gripper_open_angle", 130),
            gripper_closed_angle=getattr(args, "gripper_closed_angle", 40),
            verify_positions=not getattr(args, "no_verify_positions", False),
            invert_gripper=getattr(args, "invert_gripper", False),
            base_pwm_pin=getattr(args, "base_pwm_pin", 3),
            shoulder_pwm_pin=getattr(args, "shoulder_pwm_pin", 5),
            elbow_pwm_pin=getattr(args, "elbow_pwm_pin", 6),
            wrist_pwm_pin=getattr(args, "wrist_pwm_pin", 9),
            gripper_pwm_pin=getattr(args, "gripper_pwm_pin", 10),
            base_home_angle=getattr(args, "base_home_angle", 90),
            shoulder_home_angle=getattr(args, "shoulder_home_angle", 90),
            elbow_home_angle=getattr(args, "elbow_home_angle", 90),
            wrist_home_angle=getattr(args, "wrist_home_angle", 90),
            gripper_home_angle=getattr(args, "gripper_home_angle", 130),
            default_base_channel=getattr(args, "default_base_channel", "B"),
            default_shoulder_channel=getattr(args, "default_shoulder_channel", "C"),
            default_elbow_channel=getattr(args, "default_elbow_channel", "D"),
            default_wrist_channel=getattr(args, "default_wrist_channel", "E"),
            default_gripper_channel=getattr(args, "default_gripper_channel", "A"),
        ),
        console=console,
    )
    loop = ControlLoop(camera, classifier, driver, runtime_cfg, console=console)
    loop.run()
    return 0


def run_demo(args: argparse.Namespace) -> int:
    if args.duration_sec <= 0:
        raise ValueError("--duration-sec must be greater than 0")

    camera_cfg = CameraConfig(camera_index=0, width=args.width, height=args.height, fps=args.fps)
    robot_cfg = RobotConfig(
        port="demo://virtual-arm",
        baud_rate=9600,
        dry_run=True,
        home_on_startup=not args.no_home,
    )
    safety_cfg = SafetyConfig(
        debounce_frames=args.debounce_frames,
        no_hand_timeout_ms=args.no_hand_timeout_ms,
        max_command_hz=args.max_command_hz,
        max_joint_step_deg=args.max_joint_step_deg,
    )
    runtime_cfg = RuntimeConfig(
        camera=camera_cfg,
        robot=robot_cfg,
        safety=safety_cfg,
        preview=not args.no_preview,
    )

    camera = DemoCamera(
        width=runtime_cfg.camera.width,
        height=runtime_cfg.camera.height,
        fps=runtime_cfg.camera.fps,
        duration_sec=args.duration_sec,
    )
    classifier = DemoGestureClassifier(stable_frames_required=runtime_cfg.safety.debounce_frames)
    driver = LewanSoulMiniArmDriver(
        LewanSoulConfig(
            port=runtime_cfg.robot.port,
            baud_rate=runtime_cfg.robot.baud_rate,
            dry_run=True,
            protocol="default-program",
            startup_delay_sec=0.0,
            move_time_ms=300,
            base_servo_id=1,
            shoulder_servo_id=2,
            elbow_servo_id=3,
            wrist_servo_id=4,
            gripper_servo_id=5,
            gripper_open_position=650,
            gripper_closed_position=380,
            verify_positions=False,
            default_base_channel="B",
            default_shoulder_channel="C",
            default_elbow_channel="D",
            default_wrist_channel="E",
            default_gripper_channel="A",
        ),
        console=console,
    )
    loop = ControlLoop(camera, classifier, driver, runtime_cfg, console=console)
    try:
        loop.run()
    except KeyboardInterrupt:
        console.log("Demo complete")
    return 0


def run_demo_robot(args: argparse.Namespace) -> int:
    if args.live and args.dry_run:
        raise ValueError("Choose either --live or --dry-run, not both")
    if args.cycles <= 0:
        raise ValueError("--cycles must be greater than 0")
    if args.step_deg <= 0:
        raise ValueError("--step-deg must be greater than 0")
    if args.pause_ms < 0:
        raise ValueError("--pause-ms must be >= 0")
    if args.moves_per_direction <= 0:
        raise ValueError("--moves-per-direction must be greater than 0")
    if args.servo_hold_ms < 0:
        raise ValueError("--servo-hold-ms must be >= 0")

    dry_run = not args.live
    if args.dry_run:
        dry_run = True
    port = _resolve_robot_port(
        args.port,
        prefer_bluetooth=getattr(args, "prefer_bluetooth", False),
    )
    protocol = getattr(args, "protocol", "default-program")
    baud_rate = _effective_baud_rate(protocol, getattr(args, "baud_rate", 9600))
    _warn_default_program_baud(protocol=protocol, baud_rate=baud_rate)

    driver = LewanSoulMiniArmDriver(
        LewanSoulConfig(
            port=port,
            baud_rate=baud_rate,
            dry_run=dry_run,
            protocol=protocol,
            startup_delay_sec=getattr(args, "startup_delay_sec", 2.0),
            move_time_ms=getattr(args, "move_time_ms", 500),
            base_servo_id=getattr(args, "base_servo_id", 1),
            shoulder_servo_id=getattr(args, "shoulder_servo_id", 2),
            elbow_servo_id=getattr(args, "elbow_servo_id", 3),
            wrist_servo_id=getattr(args, "wrist_servo_id", 4),
            gripper_servo_id=getattr(args, "gripper_servo_id", 5),
            gripper_open_position=getattr(args, "gripper_open_position", 650),
            gripper_closed_position=getattr(args, "gripper_closed_position", 380),
            gripper_open_angle=getattr(args, "gripper_open_angle", 130),
            gripper_closed_angle=getattr(args, "gripper_closed_angle", 40),
            verify_positions=not getattr(args, "no_verify_positions", False),
            invert_gripper=getattr(args, "invert_gripper", False),
            base_pwm_pin=getattr(args, "base_pwm_pin", 3),
            shoulder_pwm_pin=getattr(args, "shoulder_pwm_pin", 5),
            elbow_pwm_pin=getattr(args, "elbow_pwm_pin", 6),
            wrist_pwm_pin=getattr(args, "wrist_pwm_pin", 9),
            gripper_pwm_pin=getattr(args, "gripper_pwm_pin", 10),
            base_home_angle=getattr(args, "base_home_angle", 90),
            shoulder_home_angle=getattr(args, "shoulder_home_angle", 90),
            elbow_home_angle=getattr(args, "elbow_home_angle", 90),
            wrist_home_angle=getattr(args, "wrist_home_angle", 90),
            gripper_home_angle=getattr(args, "gripper_home_angle", 130),
            default_base_channel=getattr(args, "default_base_channel", "B"),
            default_shoulder_channel=getattr(args, "default_shoulder_channel", "C"),
            default_elbow_channel=getattr(args, "default_elbow_channel", "D"),
            default_wrist_channel=getattr(args, "default_wrist_channel", "E"),
            default_gripper_channel=getattr(args, "default_gripper_channel", "A"),
        ),
        console=console,
    )

    pause_s = float(args.pause_ms) / 1000.0
    hold_s = float(args.servo_hold_ms) / 1000.0
    reset_wait_s = max(pause_s, float(getattr(args, "move_time_ms", 500)) / 1000.0, hold_s)
    sweep_deg = float(args.step_deg) * float(args.moves_per_direction)
    driver.connect()
    try:
        console.log("Reading current positions for all 5 servos")
        if hasattr(driver, "read_positions"):
            positions = driver.read_positions()
            console.log(f"Current positions: {positions}")

        console.log("Moving all 5 servos to middle position")
        if hasattr(driver, "center_all"):
            driver.center_all()
        else:
            driver.home()
        time.sleep(reset_wait_s)
        console.log(
            f"Robot 5-servo demo started in {'dry-run' if dry_run else 'live'} mode "
            f"(cycles={args.cycles}, step={args.step_deg:.2f}deg, "
            f"moves-per-direction={args.moves_per_direction}, sweep={sweep_deg:.2f}deg, "
            f"hold={args.servo_hold_ms}ms)"
        )
        for cycle in range(args.cycles):
            console.log(f"Cycle {cycle + 1}/{args.cycles}")
            for axis in ("base", "shoulder", "elbow", "wrist"):
                console.log(f"Axis {axis}: +{sweep_deg:.2f}deg")
                driver.move_axis(axis, sweep_deg)
                time.sleep(reset_wait_s)
                console.log(f"Axis {axis}: -{sweep_deg:.2f}deg")
                driver.move_axis(axis, -sweep_deg)
                time.sleep(reset_wait_s)
            for _ in range(args.moves_per_direction):
                console.log("Gripper: close")
                driver.set_gripper(open=False)
                time.sleep(reset_wait_s)
                console.log("Gripper: open")
                driver.set_gripper(open=True)
                time.sleep(reset_wait_s)

        console.log("Moving all 5 servos to middle position")
        if hasattr(driver, "center_all"):
            driver.center_all()
        else:
            driver.home()
        time.sleep(reset_wait_s)
        driver.stop_all()
        console.log("Robot 5-servo demo complete")
    finally:
        driver.disconnect()

    return 0


def run_detect_hardware(args: argparse.Namespace) -> int:
    if args.camera_max_index < 0:
        raise ValueError("--camera-max-index must be >= 0")

    result = detect_and_save_hardware_paths(
        config_path=Path(args.config_file),
        camera_max_index=args.camera_max_index,
        prefer_bluetooth=getattr(args, "prefer_bluetooth", False),
    )
    console.log(f"Hardware config updated: {args.config_file}")
    console.log(
        f"arm_path={result.get('arm_path')} stereo_camera_path={result.get('stereo_camera_path')} "
        f"stereo_camera_index={result.get('stereo_camera_index')}"
    )
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "calibrate-camera":
        return run_calibrate_camera(args)
    if args.command == "calibrate-gestures":
        return run_calibrate_gestures(args)
    if args.command == "test-robot":
        return run_test_robot(args)
    if args.command == "run":
        return run_control(args)
    if args.command == "demo":
        return run_demo(args)
    if args.command == "demo-robot":
        return run_demo_robot(args)
    if args.command == "detect-hardware":
        return run_detect_hardware(args)

    raise RuntimeError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
