import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from visual_navigation_interfaces.msg import DetectionList, CameraMotion
import numpy as np
from cv_bridge import CvBridge

class VehicleNavigationNode(Node):
    def __init__(self):
        super().__init__('navigation_node')
        self.bridge = CvBridge()
        
        # Subscriptions
        self.create_subscription(DetectionList, '/object_data', self.object_callback, 10)
        self.create_subscription(Image, '/depth_data', self.depth_callback, 10)
        self.create_subscription(CameraMotion, '/camera_motion', self.motion_callback, 10)

        # Publisher
        self.publisher = self.create_publisher(String, '/navigation_command', 10)
        
        # Parameters
        self.declare_parameter('safety_distance', 20.0)
        self.declare_parameter('motion_reliability_threshold', 0.5)
        
        self.current_obstacles = []
        self.latest_depth_map = None

        self.motion_is_reliable = True
        self.motion_reliability_val = 1.0
    
    def object_callback(self, msg):
        self.current_obstacles = msg.detections


    def motion_callback(self, msg):
        self.motion_is_reliable = msg.is_reliable
        self.motion_reliability_val = msg.reliability_score

    def depth_callback(self, msg):
        self.latest_depth_map = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
        self.make_decision()

    def make_decision(self):
        if self.latest_depth_map is None:
            return

        cmd = String()
        safety_thresh = self.get_parameter('safety_distance').value
        reliability_thresh = self.get_parameter('motion_reliability_threshold').value
        
        h, w = self.latest_depth_map.shape
        # Define the center corridor (middle third of the frame)
        center_corridor = self.latest_depth_map[:, w//3:2*w//3]
        center_depth = np.mean(center_corridor)

        ##
        if not self.motion_is_reliable or self.motion_reliability_val < reliability_thresh:
            cmd.data = "STOP"
            self.get_logger().warn(f"Decision: {cmd.data} | Reason: Motion Unreliable ({self.motion_reliability_val:.2f})")
        ##
        
        elif center_depth < safety_thresh:
            left_depth = np.mean(self.latest_depth_map[:, :w//3])
            right_depth = np.mean(self.latest_depth_map[:, 2*w//3:])
            
            # Count detected obstacles on either side
            left_obs = sum(1 for o in self.current_obstacles if (o.x1 + o.x2)/2 < w/2)
            right_obs = sum(1 for o in self.current_obstacles if (o.x1 + o.x2)/2 >= w/2)

            if left_depth > right_depth and left_obs <= right_obs:
                cmd.data = "MOVE LEFT" # 
            elif right_depth > left_depth and right_obs <= left_obs:
                cmd.data = "MOVE RIGHT" # 
            else:
                cmd.data = "STOP" # 
            
            self.get_logger().info(f"Decision: {cmd.data} | Reason: Center Blocked (Depth: {center_depth:.2f})")
        
        # 3. Path is Clear
        else:
            cmd.data = "FORWARD"
            self.get_logger().info(f"Decision: {cmd.data} | Reason: Center Clear")

        self.publisher.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = VehicleNavigationNode()
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