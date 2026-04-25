import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import sys
import os
import torch

base_path = os.path.expanduser('~/ros2_ws/src/exam_proctoring/exam_proctoring/')
src_path = os.path.join(base_path,'Depth-Anything-V2/metric_depth/')
sys.path.append(src_path)

from depth_anything_v2.dpt import DepthAnythingV2

class Depth_Estimator(Node):
    def __init__(self):
        super().__init__("depth_estimator")

        self.bridge = CvBridge()
        self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

        self.model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
         }

        self.encoder = 'vits' # or 'vits', 'vitb', 'vitg'
        self.depth_model = DepthAnythingV2(**self.model_configs[self.encoder ])
        checkpoint_path = os.path.join(base_path, f'models/depth_anything_v2_{self.encoder}.pth')
        self.depth_model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        self.depth_model = self.depth_model.to(self.device).eval()
        

        self.create_subscription(Image, "/camera_frames", self.camera_callback, 10)
        self.depth_pub = self.create_publisher(Image, "/depth_data", 10)
        self.declare_parameter('depth_threshold', 10)

    def camera_callback(self, img):
        frame = self.bridge.imgmsg_to_cv2(img, desired_encoding='bgr8')
        depth_map_raw = self.depth_model.infer_image(frame) # HxW raw depth map in numpy

        # --- DISTANCE ESTIMATION ---
        # Extract a 40x40 pixel box in the center and calculate the mean depth
        h, w = depth_map_raw.shape
        # center_target = depth_map_raw[h//2-20:h//2+20, w//2-20:w//2+20]
        # proximity_score = center_target.mean()
        center_zone = depth_map_raw[h//3:2*h//3, w//3:2*w//3]
        avg_distance = float(center_zone.mean())

        threshold = self.get_parameter('depth_threshold').value
    
        if avg_distance > threshold:
            self.get_logger().warn(f"OBSTACLE DETECTED! Distance: {avg_distance:.2f}")
        else:
            self.get_logger().info(f"Path Clear | Distance: {avg_distance:.2f}")

        depth_map = self.bridge.cv2_to_imgmsg(depth_map_raw, encoding='32FC1', header=img.header)
    
        self.depth_pub.publish(depth_map)

        # # Visualization 
        # depth_normalized = cv2.normalize(depth_map_raw, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        # depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
        # cv2.imshow('Depth Map', depth_colormap)
        # cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = Depth_Estimator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
