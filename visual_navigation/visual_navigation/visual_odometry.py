#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger
import math
import time

# Custom interfaces 
from visual_navigation_interfaces.msg import MotionList, CameraMotion
from visual_navigation_interfaces.srv import EstimateMotion 

def estimate_direction(dx: float, dy: float, magnitude: float, threshold: float = 2.0) -> str:
    """Determine camera movement direction from pixel shifts."""
    if magnitude < threshold:
        return "STATIONARY"

    abs_dx = abs(dx)
    abs_dy = abs(dy)

    if abs_dx > abs_dy * 1.3:
        return "MOVING_RIGHT" if dx > 0 else "MOVING_LEFT"

    if abs_dy > abs_dx * 1.3:
        return "MOVING_FORWARD" if dy < 0 else "MOVING_BACKWARD"

    angle = math.degrees(math.atan2(dy, dx))
    return f"MOVING_DIAGONAL({angle:.1f}deg)"

class VisualOdometryNode(Node):
    def __init__(self):
        super().__init__('visual_odometry_node')

        # Declare parameters
        self.declare_parameter('focal_length', 600.0)
        self.declare_parameter('min_features', 5)
        self.declare_parameter('motion_threshold', 2.0)
        self.declare_parameter('max_magnitude', 150.0)

        self.focal_length = self.get_parameter('focal_length').value
        self.min_features = self.get_parameter('min_features').value
        self.motion_threshold = self.get_parameter('motion_threshold').value
        self.max_magnitude = self.get_parameter('max_magnitude').value

        # Internal State (Still using a dict internally is fine for performance)
        self.last_motion_data = None
        self.estimation_reliable = True
        self.current_direction = "STATIONARY"

        # Communication
        self.motion_sub = self.create_subscription(MotionList, '/motion_data', self._motion_callback, 10)
        self.camera_motion_pub = self.create_publisher(CameraMotion, '/camera_motion', 10)
        self.estimate_srv = self.create_service(EstimateMotion, '/estimate_motion', self._estimate_motion_callback)
        
        # 10Hz Timer for steady output
        self.publish_timer = self.create_timer(0.1, self._publish_camera_motion)

        self.get_logger().info('Visual Odometry Node started without JSON dependencies.')

    def _motion_callback(self, msg: MotionList) -> None:
        """Processes motion vectors from Student 4's node[cite: 95, 186]."""
        reliable_boxes = [m for m in msg.motion if m.is_reliable]
        
        if not reliable_boxes:
            self.estimation_reliable = False
            return

        avg_dx = sum(m.dx for m in reliable_boxes) / len(reliable_boxes)
        avg_dy = sum(m.dy for m in reliable_boxes) / len(reliable_boxes)
        avg_mag = sum(m.magnitude for m in reliable_boxes) / len(reliable_boxes)

        # Update reliability check logic [cite: 106, 182]
        self.estimation_reliable = (len(reliable_boxes) >= self.min_features and avg_mag < self.max_magnitude)
        
        if self.estimation_reliable:
            self.current_direction = estimate_direction(avg_dx, avg_dy, avg_mag, self.motion_threshold)
            self.last_motion_data = {
                'dx': avg_dx, 
                'dy': avg_dy, 
                'mag': avg_mag, 
                'score': len(reliable_boxes) / 9.0
            }

    def _publish_camera_motion(self) -> None:
        """Publishes the CameraMotion message used by Student 2[cite: 112, 185]."""
        msg = CameraMotion()
        
        if not self.estimation_reliable:
            msg.is_reliable = False
            msg.direction = "STOP"  # System Rule: STOP on unreliable motion [cite: 148]
            msg.reliability_score = 0.0
        elif self.last_motion_data:
            msg.is_reliable = True
            msg.direction = self.current_direction
            msg.reliability_score = self.last_motion_data['score']
            # Direct field mapping (No strings/JSON)
            msg.linear_x = self.last_motion_data['dy'] * -0.01 
            msg.angular_z = self.last_motion_data['dx'] * -0.01

        self.camera_motion_pub.publish(msg)

    def _estimate_motion_callback(self, request, response: EstimateMotion.Response):
        """Service to provide motion on demand without JSON[cite: 165, 166]."""
        if self.last_motion_data is None or not self.estimation_reliable:
            response.success = False
            return response

        # Populate response fields directly
        response.success = True
        response.direction = self.current_direction
        response.magnitude = self.last_motion_data['mag']
        return response