import rclpy
from rclpy.node import Node 
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from visual_navigation_interfaces.msg import RoiFeature, RoiFeatureList
import cv2
import numpy as np
import time


class ROIFeatureNode(Node):
    def __init__(self):
        super().__init__('roi_feature_extraction')

        self.bridge = CvBridge()
        self.orb = cv2.ORB_create()

        self.create_subscription(Image, '/camera_frames', self.feacture_callback, 10)
        self.roi_pub = self.create_publisher(RoiFeatureList, '/roi_features', 100)
        self.declare_parameter('roi_size', 3)

        self.features = []

        self.get_logger().info('ROI Feature Extraction Node started.')

    def feacture_callback(self, img):
        try:
            frame = self.bridge.imgmsg_to_cv2(img, desired_encoding= "bgr8")
            # self.get_logger().info(f"frame (height, width, channel) :  {frame.shape}") // (h, w, ch) (480, 640, 3)

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # rgb_frame =cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

            grid_size = self.get_parameter('roi_size').value
            h , w = gray_frame.shape
            h_roi = h // grid_size
            w_roi = w // grid_size

            roi_list = RoiFeatureList()
            roi_list.header.frame_id = img.header.frame_id
            roi_list.header.stamp = self.get_clock().now().to_msg()

            for i in range(grid_size):
                for j in range(grid_size):
                    y_start = i*h_roi
                    x_start = j*w_roi
                    y_end = y_start + h_roi
                    x_end = x_start + w_roi
                    
                    # --- Region of interest slicing ---
                    roi_gray = gray_frame[y_start:y_end, x_start:x_end]
                    roi_color = frame[y_start:y_end, x_start:x_end]

                    # --- Pixel Statistics ---
                    mean_intensity = float(np.mean(roi_gray))
                    std_dev = float(np.std(roi_gray))

                    # --- Intensity-Weighted Centroid ---
                    # xs: matrix of column indices, ys: matrix of row indices
                    # each pixel's position is weighted by its brightness
                    # centroid = sum(position * brightness) / sum(brightness)
                    ys, xs = np.indices(roi_gray.shape)
                    total_intensity = float(np.sum(roi_gray))

                    if total_intensity > 0:
                        centroid_x = float(np.sum(xs * roi_gray) / total_intensity) + x_start
                        centroid_y = float(np.sum(ys * roi_gray) / total_intensity) + y_start
                    else:
                        centroid_x = float(x_start + w_roi // 2)
                        centroid_y = float(y_start + h_roi // 2)


                    # --- Feature Extraction ---
                    keypoints_1, descriptors_1 = self.orb.detectAndCompute(roi_gray, None)
                    num_keypoints = len(keypoints_1)

                    # img2 = cv2.drawKeypoints(frame,keypoints,None,color=(0,255,0), flags=0)

                    roi_features = RoiFeature()
                    roi_features.is_reliable = std_dev > 10.0 and num_keypoints > 3
                    roi_features.grid_i = i
                    roi_features.grid_j = j
                    roi_features.centroid_x = centroid_x
                    roi_features.centroid_y = centroid_y
                    roi_features.mean_intensity = mean_intensity
                    roi_features.std_dev = std_dev

                    roi_list.features.append(roi_features)

                     # --- Visualize ---
                    color = (0, 255, 0) if roi_features.is_reliable else (0, 0, 255)
                    cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), color, 1)
                    cv2.putText(frame,
                                f"std:{std_dev:.1f} kp:{num_keypoints}",
                                (x_start + 4, y_start + 16),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                                (0, 255, 255), 1)
                    cv2.circle(frame,
                               (int(centroid_x), int(centroid_y)),
                               3, (0, 0, 255), -1)
            # --- Publish ---
            self.roi_pub.publish(roi_list)
            reliable_count = sum(1 for f in roi_list.features if f.is_reliable)
            self.get_logger().info(
                f'Published {len(roi_list.features)} ROI cells | '
                f'{reliable_count} reliable'
            )

            cv2.imshow('ROI Feature Extraction', frame)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f'Callback failed: {e}')


def main(args=None):
    try:
        rclpy.init(args=args)
        node = ROIFeatureNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()