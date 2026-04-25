import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
from std_msgs.msg import String
from visual_navigation_interfaces.msg import DetectionList, BoundingBox
import cv2  
import json

class ObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('object_detection')
        
        
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('model_path', 'yolov8n.pt')
        
        conf_val = self.get_parameter('confidence_threshold').value
        model_p = self.get_parameter('model_path').value
        
       
        self.model = YOLO(model_p)
        self.bridge = CvBridge()
        
        self.subscription = self.create_subscription(
            Image,
            '/camera_frames',
            self.camera_callback,
            10)
            
        self.publisher_ = self.create_publisher(DetectionList, '/object_data', 10)
       
        self.get_logger().info(f'Node started with model: {model_p} and confidence: {conf_val}')
    
    def camera_callback(self, msg):
        self.get_logger().info("Frame received") 
        
        # Convert image and run YOLO f
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        results = self.model(cv_img, conf=self.get_parameter('confidence_threshold').value)
        
        # Create the DetectionList message
        msg_out = DetectionList()
        msg_out.header = msg.header # Keep the timestamp synced!

        # Extract bounding boxes and fill the message
        for r in results:
            for box in r.boxes:
                b = BoundingBox()
                coords = box.xyxy[0].tolist()
                
                # Fill your custom message fields
                b.x1 = int(coords[0])
                b.y1 = int(coords[1])
                b.x2 = int(coords[2])
                b.y2 = int(coords[3])
                b.confidence = float(box.conf[0].item())
                
                # YOLO classes are numbers (0=person), convert to string
                # b.class_name = str(int(box.cls[0].item())) 
                b.class_name = self.model.names[int(box.cls[0])]
                
                msg_out.detections.append(b)
        
        # Publish 
        self.publisher_.publish(msg_out)

        # visualization
        annotated_frame = results[0].plot()
        cv2.imshow("Camera View - YOLO Detections", annotated_frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
