import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
from exam_interfaces.msg import DetectionList, BoundingBox

class ObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('object_node')

        self.bridge = CvBridge()

        self.subscription = self.create_subscription(
            Image,
            '/camera_frames',
            self.image_callback,
            10)

        self.publisher = self.create_publisher(
            DetectionList,
            '/object_data',
            10)

        self.declare_parameter('confidence_threshold', 0.5) 

        self.model = YOLO("yolov8n.pt")  # lightweight model

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        results = self.model(frame)

        detections = DetectionList()
        detections.header = msg.header

        conf_thresh = self.get_parameter('confidence_threshold').value

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                label = self.model.names[cls]
                confidence = float(box.conf[0])

                if label in ["cell phone", "book"] and confidence >= conf_thresh:
                    b = box.xyxy[0]

                    bbox = BoundingBox()

                    bbox.x1 = int(b[0])
                    bbox.y1 = int(b[1])
                    bbox.x2 = int(b[2])
                    bbox.y2 = int(b[3])

                    bbox.confidence = float(box.conf[0])
                    bbox.class_name = label
    
                    detections.detections.append(bbox)

        self.publisher.publish(detections)
        plotted_frame = results[0].plot()
        cv2.imshow('YOLO Object Detection', plotted_frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()
