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

        self.last_motion_data = None
        self.estimation_reliable = True
        self.current_direction = "STATIONARY"

        # Communication
        self.motion_sub = self.create_subscription(MotionList, '/motion_data', self._motion_callback, 10)
        self.camera_motion_pub = self.create_publisher(CameraMotion, '/camera_motion', 10)
        self.estimate_srv = self.create_service(EstimateMotion, '/estimate_motion', self._estimate_motion_callback)
        
        # 10Hz Timer for steady output
        self.publish_timer = self.create_timer(0.1, self._publish_camera_motion)

        self.get_logger().info('Visual Odometry Node started.')

    def _motion_callback(self, msg: MotionList) -> None:
        """Processes motion vectors from Student 4's node[cite: 95, 186]."""
        reliable_boxes = [m for m in msg.motion if m.is_reliable]
        count = len(reliable_boxes)

        if not reliable_boxes:
            self.get_logger().warn('NO RELIABLE BOXES: Forcing STOP state.')
            self.estimation_reliable = False
            return

        avg_dx = sum(m.dx for m in reliable_boxes) / count
        avg_dy = sum(m.dy for m in reliable_boxes) / count
        avg_mag = sum(m.magnitude for m in reliable_boxes) / count

        # Update reliability check logic 
        is_reliable = (count >= self.min_features and avg_mag < self.max_magnitude)
        
        if not is_reliable:
            reason = "LOW_FEATURES" if count < self.min_features else "EXCESSIVE_MOTION"
            self.get_logger().warn(f'UNRELIABLE ({reason}): Boxes={count}/9, Mag={avg_mag:.2f}') #
        
        self.estimation_reliable = is_reliable
        
        if self.estimation_reliable:
            new_dir = estimate_direction(avg_dx, avg_dy, avg_mag, self.motion_threshold)

            if new_dir != self.current_direction:
                self.get_logger().info(f'Direction Changed: {new_dir}') 
            
            self.current_direction = new_dir
            self.last_motion_data = {
                'dx': avg_dx, 
                'dy': avg_dy, 
                'mag': avg_mag, 
                'score': count / 9.0
            }

    def _publish_camera_motion(self) -> None:
        msg = CameraMotion()
        
        if not self.estimation_reliable:
            msg.is_reliable = False
            msg.direction = "STOP"  # System Rule: STOP on unreliable motion [cite: 148]
            msg.reliability_score = 0.0
        elif self.last_motion_data:
            msg.is_reliable = True
            msg.direction = self.current_direction
            msg.reliability_score = self.last_motion_data['score']
            # Direct field mapping 
            msg.linear_x = self.last_motion_data['dy'] * -0.01 
            msg.angular_z = self.last_motion_data['dx'] * -0.01

        self.camera_motion_pub.publish(msg)

    def _estimate_motion_callback(self, request, response: EstimateMotion.Response):
        self.get_logger().info('Service /estimate_motion called.')
        if self.last_motion_data is None or not self.estimation_reliable:
            response.success = False
            return response

        # Populate response fields directly
        response.success = True
        response.direction = self.current_direction
        response.magnitude = self.last_motion_data['mag']
        return response

def main(args=None):
    rclpy.init(args=args)
    node = VisualOdometryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
