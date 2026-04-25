import rclpy
from rclpy.node import Node 
from visual_navigation_interfaces.msg import RoiFeature, RoiFeatureList, MotionData, MotionList
import cv2
import numpy as np


class MotionTracking(Node):
    def __init__(self):
        super().__init__('motion_tracking')

        self.previous_features = None 

        self.create_subscription(RoiFeatureList, '/roi_features', self.motion_callback, 10)
        self.motion_pub = self.create_publisher(MotionList, '/motion_data', 10)
        self.declare_parameter('motion_threshold', 1.5)

    def motion_callback(self, msg):
        if self.previous_features is None:
            self.previous_features = msg.features
            return
        
        motion_List = MotionList()

        for prev_box, current_box in zip(self.previous_features, msg.features):
            motion_data = MotionData()

            motion_data.grid_i = current_box.grid_i
            motion_data.grid_j = current_box.grid_j
            if prev_box.is_reliable and current_box.is_reliable:
                # ---- pixel shift ----
                dx = current_box.centroid_x - prev_box.centroid_x 
                dy = current_box.centroid_y - prev_box.centroid_y

                magnitude = np.hypot(dx, dy)
                angle = np.arctan2(dy,dx)

                motion_data.magnitude = float(magnitude)
                motion_data.angle = float(angle)
                motion_data.dx = float(dx)
                motion_data.dy = float(dy)
                motion_data.is_reliable = True

                threshold = self.get_parameter('motion_threshold').value

                if magnitude < threshold :
                    motion_data.direction = "STOP"
                else:
                    if abs(dx) > abs(dy): 
                        if dx < 0 :
                            motion_data.direction = "LEFT"
                        elif dx > 0 :
                            motion_data.direction = "RIGHT"
                    else:
                        if dy < 0 :
                            motion_data.direction = "UP"
                        elif dy > 0 :
                            motion_data.direction = "DOWN"
            else :
                motion_data.is_reliable = False
                motion_data.direction = "UNRELIABLE"

            motion_List.motion.append(motion_data)

        self.previous_features = msg.features
        self.motion_pub.publish(motion_List)


def main(args=None):
    try:
        rclpy.init(args=args)
        node = MotionTracking()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()

if __name__ == '__main__':
    main()