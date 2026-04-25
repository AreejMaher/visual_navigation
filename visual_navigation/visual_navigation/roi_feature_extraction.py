import rclpy
from rclpy.node import Node 
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import cv2


class Feature_Extraction(Node):
    def __init__(self):
        super().__init__('roi_feature_extraction')

        self.create_subscription(Image, '/camera_frames', self.feacture_callback(), 100)
        # self.create_publisher(Image, '/roi_features', , 100)
        self.declare_parameter('roi_size', 0)

    def feacture_callback(delf, img):
        pass

def main(args=None):
    try:
        rclpy.init(args=args)
        node = Feature_Extraction()
    except KeyboardInterrupt:
        pass
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()