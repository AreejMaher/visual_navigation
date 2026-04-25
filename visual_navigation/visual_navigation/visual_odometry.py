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

from visual_navigation_interfaces.msg import MotionList, CameraMotion
from visual_navigation_interfaces.srv import EstimateMotion # Use your custom srv
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
            MotionList,
            '/motion_data',
            self._motion_callback,
            10
        )

        # ── ROS2 publisher ────────────────────────────────────────────────────
        self.camera_motion_pub = self.create_publisher(
            CameraMotion,
            '/camera_motion',
            10
        )

        # ── ROS2 service ─────────────────────────────────────────────────────
        self.estimate_srv = self.create_service(
            EstimateMotion,
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

    def _motion_callback(self, msg: MotionList) -> None:
        reliable_boxes = [m for m in msg.motion if m.is_reliable]
        
        if not reliable_boxes:
            self.estimation_reliable = False
            self._publish_camera_motion()
            return

        # Calculate the average movement across the grid
        avg_dx = sum(m.dx for m in reliable_boxes) / len(reliable_boxes)
        avg_dy = sum(m.dy for m in reliable_boxes) / len(reliable_boxes)
        avg_mag = sum(m.magnitude for m in reliable_boxes) / len(reliable_boxes)

        # Build a data dictionary to keep the rest of the logic working
        data = {
            'dx': avg_dx,
            'dy': avg_dy,
            'magnitude': avg_mag,
            'reliable_count': len(reliable_boxes),
            'total_boxes': len(msg.motion)
        }

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
            msg = self._build_stop_payload()
        else:
            msg = self._build_motion_payload()

        self.camera_motion_pub.publish(msg)

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
    ##
    def _check_reliability(self, data: dict) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        
        reliable_count = data.get('reliable_count', 0)
        if reliable_count < 5:  # If more than half the grid is unreliable
            reasons.append(f'LOW_RELIABLE_CELLS({reliable_count}/9)')

        if data.get('magnitude', 0.0) > 150.0:
            reasons.append('EXCESSIVE_MOTION')

        return (len(reasons) == 0), reasons
    ##

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

    def _build_motion_payload(self) -> CameraMotion:
        """Build a well-formed CameraMotion message for reliable frames."""
        msg = CameraMotion()
        latest = self.motion_history[-1] if self.motion_history else {}
        
        msg.linear_x = float(latest.get('dy', 0.0) * -0.01) # Forward/Backward
        msg.angular_z = float(latest.get('dx', 0.0) * -0.01) # Rotation
        msg.direction = str(latest.get('direction', 'STATIONARY'))
        msg.is_reliable = True
        
        msg.reliability_score = float(self.last_motion_data.get('reliable_count', 0) / 9.0)
        
        return msg

    def _build_stop_payload(self) -> CameraMotion:
        """Build a STOP payload used when estimation is unreliable."""
        msg = CameraMotion()
        msg.is_reliable = False
        msg.direction = "STOP"
        msg.reliability_score = 0.0
        msg.linear_x = 0.0
        msg.angular_z = 0.0
        
        return msg


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
