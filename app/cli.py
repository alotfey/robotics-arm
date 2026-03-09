from __future__ import annotations

import argparse
import time

import cv2
from rich.console import Console

from app.camera.uvc_camera import UvcCamera
from app.models import CameraConfig, RobotConfig, RuntimeConfig, SafetyConfig
from app.robot.lewansoul_miniarm import LewanSoulConfig, LewanSoulMiniArmDriver
from app.runtime.control_loop import ControlLoop
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

    test = sub.add_parser("test-robot", help="Send safe test commands to robot")
    _add_robot_flags(test)
    test.add_argument("--live", action="store_true")

    return parser


def _add_camera_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)


def _add_robot_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--port", type=str, default="/dev/tty.usbmodem")
    parser.add_argument("--baud-rate", type=int, default=115200)
    parser.add_argument("--dry-run", action="store_true")


def _add_runtime_safety_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--debounce-frames", type=int, default=5)
    parser.add_argument("--no-hand-timeout-ms", type=int, default=800)
    parser.add_argument("--max-command-hz", type=float, default=10.0)
    parser.add_argument("--max-joint-step-deg", type=float, default=5.0)


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

    raise RuntimeError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
