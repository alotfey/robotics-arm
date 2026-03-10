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
from app.runtime.hardware_discovery import DEFAULT_CONFIG_PATH, detect_and_save_hardware_paths
from app.vision.gestures import GestureClassifier

console = Console()


def build_parser() -> argparse.ArgumentParser:
    """Build and return the top-level CLI parser.

    Returns:
        argparse.ArgumentParser: Configured parser with all subcommands.
    """
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
        help="Run robot command demo in dry-run mode without connected hardware",
    )
    _add_robot_flags(demo_robot)
    demo_robot.add_argument("--cycles", type=int, default=3)
    demo_robot.add_argument("--step-deg", type=float, default=3.0)
    demo_robot.add_argument("--pause-ms", type=int, default=250)

    test = sub.add_parser("test-robot", help="Send safe test commands to robot")
    _add_robot_flags(test)
    test.add_argument("--live", action="store_true")

    detect_hw = sub.add_parser(
        "detect-hardware",
        help="Detect connected arm and stereo camera, then write their paths to config",
    )
    detect_hw.add_argument("--config-file", type=str, default=str(DEFAULT_CONFIG_PATH))
    detect_hw.add_argument("--camera-max-index", type=int, default=8)

    return parser


def _add_camera_flags(parser: argparse.ArgumentParser) -> None:
    """Attach live camera options to a subcommand parser.

    Args:
        parser: Parser to mutate with camera arguments.
    """
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)


def _add_demo_camera_flags(parser: argparse.ArgumentParser) -> None:
    """Attach demo camera options to a subcommand parser.

    Args:
        parser: Parser to mutate with demo camera arguments.
    """
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)


def _add_robot_flags(parser: argparse.ArgumentParser) -> None:
    """Attach robot connection options to a subcommand parser.

    Args:
        parser: Parser to mutate with robot arguments.
    """
    parser.add_argument("--port", type=str, default="/dev/tty.usbmodem")
    parser.add_argument("--baud-rate", type=int, default=115200)
    parser.add_argument("--dry-run", action="store_true")


def _add_runtime_safety_flags(parser: argparse.ArgumentParser) -> None:
    """Attach runtime safety controls to a subcommand parser.

    Args:
        parser: Parser to mutate with safety arguments.
    """
    parser.add_argument("--debounce-frames", type=int, default=5)
    parser.add_argument("--no-hand-timeout-ms", type=int, default=800)
    parser.add_argument("--max-command-hz", type=float, default=10.0)
    parser.add_argument("--max-joint-step-deg", type=float, default=5.0)


def run_calibrate_camera(args: argparse.Namespace) -> int:
    """Run camera preview calibration until the user exits.

    Args:
        args: Parsed CLI arguments for camera settings.

    Returns:
        int: Process-style status code.
    """
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
    """Run gesture classification preview for calibration.

    Args:
        args: Parsed CLI arguments for camera and debounce settings.

    Returns:
        int: Process-style status code.
    """
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
    """Send a short safe command sequence to the robot driver.

    Args:
        args: Parsed CLI arguments for robot connectivity and dry-run mode.

    Returns:
        int: Process-style status code.

    Raises:
        ValueError: If both live and dry-run flags are requested together.
    """
    if args.live and args.dry_run:
        raise ValueError("Choose either --live or --dry-run, not both")
    dry_run = not args.live
    if args.dry_run:
        dry_run = True

    driver = LewanSoulMiniArmDriver(
        LewanSoulConfig(port=args.port, baud_rate=args.baud_rate, dry_run=dry_run),
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
    """Start the live camera -> gesture -> robot control loop.

    Args:
        args: Parsed CLI arguments for camera, robot, and safety settings.

    Returns:
        int: Process-style status code.
    """
    camera_cfg = CameraConfig(
        camera_index=args.camera_index,
        width=args.width,
        height=args.height,
        fps=args.fps,
    )
    robot_cfg = RobotConfig(
        port=args.port,
        baud_rate=args.baud_rate,
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
        ),
        console=console,
    )
    loop = ControlLoop(camera, classifier, driver, runtime_cfg, console=console)
    loop.run()
    return 0


def run_demo(args: argparse.Namespace) -> int:
    """Run the full control loop with synthetic camera and gestures.

    Args:
        args: Parsed CLI arguments for demo mode settings.

    Returns:
        int: Process-style status code.

    Raises:
        ValueError: If the requested demo duration is not positive.
    """
    if args.duration_sec <= 0:
        raise ValueError("--duration-sec must be greater than 0")

    camera_cfg = CameraConfig(camera_index=0, width=args.width, height=args.height, fps=args.fps)
    robot_cfg = RobotConfig(
        port="demo://virtual-arm",
        baud_rate=115200,
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
    """Run a dry-run robot movement demo without camera input.

    Args:
        args: Parsed CLI arguments for dry-run robot demo behavior.

    Returns:
        int: Process-style status code.

    Raises:
        ValueError: If numeric options are outside valid ranges.
    """
    if args.cycles <= 0:
        raise ValueError("--cycles must be greater than 0")
    if args.step_deg <= 0:
        raise ValueError("--step-deg must be greater than 0")
    if args.pause_ms < 0:
        raise ValueError("--pause-ms must be >= 0")

    driver = LewanSoulMiniArmDriver(
        LewanSoulConfig(port=args.port, baud_rate=args.baud_rate, dry_run=True),
        console=console,
    )

    pause_s = float(args.pause_ms) / 1000.0
    driver.connect()
    try:
        driver.home()
        console.log(
            f"Robot demo started in dry-run mode (cycles={args.cycles}, step={args.step_deg:.2f}deg)"
        )
        for _ in range(args.cycles):
            driver.move_axis("base", args.step_deg)
            time.sleep(pause_s)
            driver.move_axis("base", -args.step_deg)
            time.sleep(pause_s)
            driver.move_axis("shoulder", args.step_deg)
            time.sleep(pause_s)
            driver.move_axis("shoulder", -args.step_deg)
            time.sleep(pause_s)
            driver.set_gripper(open=False)
            time.sleep(pause_s)
            driver.set_gripper(open=True)
            time.sleep(pause_s)

        driver.stop_all()
        console.log("Robot demo complete")
    finally:
        driver.disconnect()

    return 0


def run_detect_hardware(args: argparse.Namespace) -> int:
    """Probe arm and camera hardware and persist discovered paths.

    Args:
        args: Parsed CLI arguments for output file and camera scan range.

    Returns:
        int: Process-style status code.

    Raises:
        ValueError: If the maximum camera index is negative.
    """
    if args.camera_max_index < 0:
        raise ValueError("--camera-max-index must be >= 0")

    result = detect_and_save_hardware_paths(
        config_path=Path(args.config_file),
        camera_max_index=args.camera_max_index,
    )
    console.log(f"Hardware config updated: {args.config_file}")
    console.log(
        f"arm_path={result.get('arm_path')} stereo_camera_path={result.get('stereo_camera_path')} "
        f"stereo_camera_index={result.get('stereo_camera_index')}"
    )
    return 0


def main() -> int:
    """Parse CLI arguments and dispatch to the selected command handler.

    Returns:
        int: Process-style status code.

    Raises:
        RuntimeError: If an unknown command is encountered.
    """
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
