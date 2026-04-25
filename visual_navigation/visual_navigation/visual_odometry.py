#!/usr/bin/env python3
"""
=============================================================================
 Visual Odometry Estimation Node  — Student 1
 Project: Distributed Visual Navigation Hint System using ROS2
=============================================================================

 Responsibility:
   - Subscribe to /motion_data  (from Motion Tracking Node)
   - Estimate camera movement direction: LEFT / RIGHT / FORWARD / BACKWARD
   - Detect unreliable estimation (blur, low features, dynamic scenes)
   - Publish /camera_motion with status and direction
   - Provide /estimate_motion  service for on-demand estimation

 ROS2 Communication:
   Subscriptions : /motion_data       (std_msgs/String  — JSON payload)
   Publications  : /camera_motion     (std_msgs/String  — JSON payload)
   Services      : /estimate_motion   (std_srvs/Trigger)

 Parameters:
   focal_length      (float, default 600.0) — conceptual, in pixels
   min_features      (int,   default 10)    — below this → unreliable
   motion_threshold  (float, default 2.0)   — minimum pixel shift
   blur_threshold    (float, default 80.0)  — Laplacian variance cutoff
   history_size      (int,   default 30)    — frames kept in history

=============================================================================
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger

import json
import math
import time


# ─────────────────────────────────────────────────────────────────────────────
#  Helper: direction reasoning from optical-flow vectors
# ─────────────────────────────────────────────────────────────────────────────

def estimate_direction(dx: float, dy: float, magnitude: float,
                       threshold: float = 2.0) -> str:
    """
    Determine camera movement direction from 2-D motion vectors.

    Convention (image coordinates):
      dx > 0  →  scene moves RIGHT  →  camera moves RIGHT
      dx < 0  →  scene moves LEFT   →  camera moves LEFT
      dy > 0  →  scene moves DOWN   →  camera moves BACKWARD
                  (objects appear lower → you backed away)
      dy < 0  →  scene moves UP     →  camera moves FORWARD
                  (objects appear higher → you moved toward them)

    Args:
        dx        : mean horizontal pixel shift
        dy        : mean vertical   pixel shift
        magnitude : overall motion magnitude
        threshold : minimum movement (px) to leave STATIONARY state

    Returns:
        Direction string consumed by Navigation Decision Node.
    """
    if magnitude < threshold:
        return "STATIONARY"

    abs_dx = abs(dx)
    abs_dy = abs(dy)

    # Primary axis wins if difference > 30 %
    if abs_dx > abs_dy * 1.3:
        return "MOVING_RIGHT" if dx > 0 else "MOVING_LEFT"

    if abs_dy > abs_dx * 1.3:
        return "MOVING_FORWARD" if dy < 0 else "MOVING_BACKWARD"

    # Diagonal movement
    angle = math.degrees(math.atan2(dy, dx))
    return f"MOVING_DIAGONAL({angle:.1f}deg)"


# ─────────────────────────────────────────────────────────────────────────────
#  Main Node class
# ─────────────────────────────────────────────────────────────────────────────

class VisualOdometryNode(Node):
    """
    ROS2 node: Visual Odometry Estimation.

    Integrates motion vectors received from the Motion Tracking Node and
    produces camera motion estimates consumed by the Navigation Decision Node.
    """

    # ── construction ──────────────────────────────────────────────────────────
    def __init__(self):
        super().__init__('visual_odometry_node')

        # ── declare & read parameters ────────────────────────────────────────
        self.declare_parameter('focal_length',     600.0)
        self.declare_parameter('min_features',     10)
        self.declare_parameter('motion_threshold', 2.0)
        self.declare_parameter('blur_threshold',   80.0)
        self.declare_parameter('history_size',     30)

        self.focal_length     = self.get_parameter('focal_length').value
        self.min_features     = self.get_parameter('min_features').value
        self.motion_threshold = self.get_parameter('motion_threshold').value
        self.blur_threshold   = self.get_parameter('blur_threshold').value
        self.history_size     = self.get_parameter('history_size').value

        # ── internal state ────────────────────────────────────────────────────
        self.last_motion_data: dict | None = None   # most recent raw payload
        self.motion_history: list[dict]    = []     # rolling window
        self.estimation_reliable: bool     = True

        # Conceptual dead-reckoning pose (x, y, z in arbitrary units)
        self.camera_pose = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        self.total_frames_processed = 0
        self.unreliable_count       = 0

        # ── ROS2 subscriber ───────────────────────────────────────────────────
        self.motion_sub = self.create_subscription(
            String,
            '/motion_data',
            self._motion_callback,
            10
        )

        # ── ROS2 publisher ────────────────────────────────────────────────────
        self.camera_motion_pub = self.create_publisher(
            String,
            '/camera_motion',
            10
        )

        # ── ROS2 service ─────────────────────────────────────────────────────
        self.estimate_srv = self.create_service(
            Trigger,
            '/estimate_motion',
            self._estimate_motion_callback
        )

        # ── periodic publish timer (10 Hz) ────────────────────────────────────
        self.publish_timer = self.create_timer(0.1, self._publish_camera_motion)

        # ── startup log ───────────────────────────────────────────────────────
        self.get_logger().info('=' * 60)
        self.get_logger().info(' Visual Odometry Estimation Node  [ROS2]')
        self.get_logger().info(' Student 1 — Visual Navigation Hint System')
        self.get_logger().info('=' * 60)
        self.get_logger().info(f'  focal_length     = {self.focal_length}')
        self.get_logger().info(f'  min_features     = {self.min_features}')
        self.get_logger().info(f'  motion_threshold = {self.motion_threshold}')
        self.get_logger().info(f'  blur_threshold   = {self.blur_threshold}')
        self.get_logger().info('  Subscribed : /motion_data')
        self.get_logger().info('  Publishing : /camera_motion')
        self.get_logger().info('  Service    : /estimate_motion')
        self.get_logger().info('=' * 60)

    # ── callbacks ─────────────────────────────────────────────────────────────

    def _motion_callback(self, msg: String) -> None:
        """
        Receive JSON motion data from the Motion Tracking Node.

        Expected JSON keys:
          dx            (float) — mean horizontal optical-flow shift (pixels)
          dy            (float) — mean vertical   optical-flow shift (pixels)
          magnitude     (float) — overall motion magnitude
          feature_count (int)   — number of tracked feature points
          is_blurry     (bool)  — True when Laplacian variance is low
          variance      (float) — pixel-level scene variance (optional)
        """
        try:
            data: dict = json.loads(msg.data)
        except (json.JSONDecodeError, ValueError) as exc:
            self.get_logger().error(f'[VO] JSON parse error: {exc}')
            return

        self.last_motion_data = data
        self.total_frames_processed += 1
        self._process_frame(data)

    def _publish_camera_motion(self) -> None:
        """
        Timer callback — publish /camera_motion at 10 Hz.
        Outputs STOP payload when estimation is unreliable.
        """
        if self.last_motion_data is None:
            return  # no data yet

        if not self.estimation_reliable:
            payload = self._build_stop_payload()
        else:
            payload = self._build_motion_payload()

        out = String()
        out.data = json.dumps(payload)
        self.camera_motion_pub.publish(out)

    def _estimate_motion_callback(self, request, response: Trigger.Response):
        """
        Service handler for /estimate_motion.
        Returns the latest motion estimate on demand.
        """
        if self.last_motion_data is None:
            response.success = False
            response.message = json.dumps({
                'status':  'NO_DATA',
                'command': 'STOP',
                'reason':  'No motion data has been received yet'
            })
            self.get_logger().warn('[VO] /estimate_motion called but no data yet')
            return response

        if not self.estimation_reliable:
            response.success = False
            response.message = json.dumps(self._build_stop_payload())
            self.get_logger().warn('[VO] /estimate_motion → UNRELIABLE')
            return response

        response.success = True
        response.message = json.dumps(self._build_motion_payload())
        self.get_logger().info('[VO] /estimate_motion → OK')
        return response

    # ── core processing ───────────────────────────────────────────────────────

    def _process_frame(self, data: dict) -> None:
        """
        Run the visual odometry pipeline for a single motion frame.

        Steps:
          1. Extract motion vectors and metadata
          2. Reliability check  (blur, low features, excessive motion)
          3. Direction estimation
          4. Dead-reckoning pose update
          5. Append to history
        """
        dx            = float(data.get('dx',            0.0))
        dy            = float(data.get('dy',            0.0))
        magnitude     = float(data.get('magnitude',     0.0))
        feature_count = int(data.get('feature_count',   0))
        is_blurry     = bool(data.get('is_blurry',      False))

        # ── 1. reliability check ─────────────────────────────────────────────
        reliable, reasons = self._check_reliability(
            feature_count, is_blurry, magnitude
        )
        self.estimation_reliable = reliable

        if not reliable:
            self.unreliable_count += 1
            self.get_logger().warn(
                f'[VO] Frame {self.total_frames_processed} UNRELIABLE '
                f'→ {", ".join(reasons)}'
            )
            return

        # ── 2. direction estimation ──────────────────────────────────────────
        direction = estimate_direction(
            dx, dy, magnitude, threshold=self.motion_threshold
        )

        # ── 3. pose update (conceptual dead-reckoning) ───────────────────────
        self._update_pose(direction, dx, dy)

        # ── 4. history ───────────────────────────────────────────────────────
        entry = {
            'frame':     self.total_frames_processed,
            'direction': direction,
            'dx':        round(dx, 3),
            'dy':        round(dy, 3),
            'magnitude': round(magnitude, 3),
            'features':  feature_count,
            'timestamp': time.time(),
        }
        self.motion_history.append(entry)
        if len(self.motion_history) > self.history_size:
            self.motion_history.pop(0)

        self.get_logger().info(
            f'[VO] Frame {self.total_frames_processed:04d} | '
            f'dir={direction:<22} | '
            f'dx={dx:+.1f} dy={dy:+.1f} mag={magnitude:.1f} '
            f'feat={feature_count}'
        )

    # ── reliability ───────────────────────────────────────────────────────────

    def _check_reliability(
        self,
        feature_count: int,
        is_blurry: bool,
        magnitude: float
    ) -> tuple[bool, list[str]]:
        """
        Decide whether the current motion estimation is trustworthy.

        Failure conditions:
          - Image is blurry             (Laplacian variance too low)
          - Too few tracked features    (sparse scene)
          - Excessive motion magnitude  (dynamic scene / camera shake)

        Returns:
            (is_reliable, list_of_failure_reasons)
        """
        reasons: list[str] = []

        if is_blurry:
            reasons.append('IMAGE_BLURRY')

        if feature_count < self.min_features:
            reasons.append(f'LOW_FEATURES({feature_count}<{self.min_features})')

        # Very large optical-flow → camera shake or chaotic scene
        if magnitude > 150.0:
            reasons.append(f'EXCESSIVE_MOTION({magnitude:.1f}px)')

        return (len(reasons) == 0), reasons

    # ── pose update ───────────────────────────────────────────────────────────

    def _update_pose(self, direction: str, dx: float, dy: float) -> None:
        """
        Conceptual dead-reckoning using focal-length normalisation.

        The scale factor maps pixel shifts to arbitrary distance units:
            scale = 1 / focal_length
        This is a simplified monocular VO approach — no metric scale.
        """
        scale = 1.0 / max(self.focal_length, 1.0)

        if direction == 'MOVING_FORWARD':
            self.camera_pose['z'] += abs(dy) * scale
        elif direction == 'MOVING_BACKWARD':
            self.camera_pose['z'] -= abs(dy) * scale
        elif direction == 'MOVING_RIGHT':
            self.camera_pose['x'] += abs(dx) * scale
        elif direction == 'MOVING_LEFT':
            self.camera_pose['x'] -= abs(dx) * scale
        # STATIONARY and DIAGONAL → no pose update

    # ── payload builders ──────────────────────────────────────────────────────

    def _build_motion_payload(self) -> dict:
        """Build a well-formed /camera_motion payload for reliable frames."""
        latest = self.motion_history[-1] if self.motion_history else {}
        return {
            'status':    'OK',
            'reliable':  True,
            'direction': latest.get('direction', 'STATIONARY'),
            'dx':        latest.get('dx',         0.0),
            'dy':        latest.get('dy',         0.0),
            'magnitude': latest.get('magnitude',  0.0),
            'features':  latest.get('features',   0),
            'pose':      {k: round(v, 4) for k, v in self.camera_pose.items()},
            'frames_processed': self.total_frames_processed,
            'unreliable_count': self.unreliable_count,
            'timestamp': time.time(),
        }

    def _build_stop_payload(self) -> dict:
        """Build a STOP payload used when estimation is unreliable."""
        return {
            'status':    'UNRELIABLE',
            'reliable':  False,
            'command':   'STOP',            # consumed by Navigation Decision Node
            'direction': 'UNKNOWN',
            'dx':        0.0,
            'dy':        0.0,
            'magnitude': 0.0,
            'pose':      {k: round(v, 4) for k, v in self.camera_pose.items()},
            'frames_processed': self.total_frames_processed,
            'unreliable_count': self.unreliable_count,
            'timestamp': time.time(),
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = VisualOdometryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('[VO] KeyboardInterrupt — shutting down.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
