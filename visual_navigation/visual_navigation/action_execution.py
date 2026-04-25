import rclpy  # ros2 Python client library (ros1: rospy)
from rclpy.node import Node  # base class for ros2 nodes (ros1: no node class)

from rclpy.action import ActionServer  # creates action server in ros2 (ros1: actionlib)
from std_msgs.msg import String  # standard string message type (same in ros1)

from my_robot_interfaces.action import NavigateAction
# custom action interface
# ros1: imported from package msg folder

import time  # used to simulate action execution delay


class ActionExecution(Node):  # define ros2 node class
    def __init__(self):
        super().__init__("action_node")
        # initialize node with name "action_node"
        # ros1: rospy.init_node("action_node")

        # Parameter
        self.declare_parameter("action_duration", 3.0)
        # declare action execution duration
        # ros1: rospy.get_param("~action_duration", 3.0)

        self.action_duration = self.get_parameter("action_duration").value
        # get parameter value

        # Publisher
        self.publisher = self.create_publisher(String, "action_status", 10)
        # publish execution status
        # ros1: rospy.Publisher("action_status", String, queue_size=10)

        # Subscriber
        self.create_subscription(String,"navigation_command",self.command_callback,10)
        # subscribe to navigation commands
        # ros1: rospy.Subscriber("navigation_command", String, callback)

        # Action Server
        self._action_server = ActionServer(self,NavigateAction,"navigate_action",self.execute_callback)
        # create action server
        # ros1: actionlib.SimpleActionServer()

        self.last_command = ""
        # stores last navigation command

        self.get_logger().info("Action Execution Node Started")
        # ros1: rospy.loginfo(...)

    def command_callback(self, msg):
        # called when command is received from navigation node

        self.last_command = msg.data
        # save received command

        self.get_logger().info(f"Command received: {msg.data}")
        # ros1: rospy.loginfo(...)

    def execute_callback(self, goal_handle):
        # called when action goal is sent

        command = goal_handle.request.command
        # get command from action goal
        # ros1: goal.command

        self.get_logger().info(f"Executing: {command}")

        feedback_msg = NavigateAction.Feedback()
        # create feedback object

        for i in range(int(self.action_duration)):
            feedback_msg.feedback = f"Executing {command} step {i+1}"
            # update feedback message

            goal_handle.publish_feedback(feedback_msg)
            # send feedback to client
            # ros1: publish_feedback()

            time.sleep(1)
            # simulate action time

        goal_handle.succeed()
        # mark goal as completed
        # ros1: set_succeeded()

        status_msg = String()
        status_msg.data = f"{command} completed"
        # create status message

        self.publisher.publish(status_msg)
        # publish execution result
        # ros1: same publish()

        result = NavigateAction.Result()
        # create result object

        result.result = "Navigation completed"
        # final action result

        return result
        # return result to client


def main(args=None):
    rclpy.init(args=args)
    # initialize ros2
    # ros1: rospy.init_node()

    node = ActionExecution()
    # create node object

    try:
        rclpy.spin(node)
        # keep node running
        # ros1: rospy.spin()

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        # destroy node
        # ros2 only

        rclpy.shutdown()
        # shutdown ros2
        # ros1 handles automatically


if __name__ == "__main__":
    main()


# command to run the node
# ros2 run my_py_pkg action_node

# publish test command
# ros2 topic pub --once /navigation_command std_msgs/msg/String "{data: 'MOVE LEFT'}"

# send action goal
# ros2 action send_goal /navigate_action my_robot_interfaces/action/NavigateAction "{command: 'MOVE LEFT'}"
