import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode
from geometry_msgs.msg import PoseStamped
import sys
import math

class SwarmController(Node):
    def __init__(self, drone_ns):
        super().__init__(f'swarm_controller_{drone_ns}')
        self.namespace = drone_ns
        self.setpoint_counter = 0
        
        # BOTH Subscriber and Publisher MUST use Sensor Data QoS for MAVROS
        self.state = State()
        self.state_sub = self.create_subscription(State, f'/{drone_ns}/state', self.state_cb, qos_profile_sensor_data)
        self.pose_pub = self.create_publisher(PoseStamped, f'/{drone_ns}/setpoint_position/local', qos_profile_sensor_data)
        
        self.arming_client = self.create_client(CommandBool, f'/{drone_ns}/cmd/arming')
        self.mode_client = self.create_client(SetMode, f'/{drone_ns}/set_mode')
        
        # 20Hz Loop
        self.timer = self.create_timer(0.05, self.control_loop)
        self.last_req = self.get_clock().now().nanoseconds / 1e9
        
        self.get_logger().info(f'[{self.namespace}] System Ready. Initiating foolproof sequence...')

    def state_cb(self, msg):
        self.state = msg

    def control_loop(self):
        now = self.get_clock().now().nanoseconds / 1e9
        
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = "map" # CRITICAL: MAVROS will drop the message without this!
        
        if self.namespace == 'drone0':
            t = now
            pose.pose.position.z = 5.0 # Evader flies at 5m
        else:
            t = now - 2.5
            pose.pose.position.z = 4.0 # Pursuer chases at 4m

        radius = 3.0
        speed = 0.5
        pose.pose.position.x = radius * math.cos(speed * t)
        pose.pose.position.y = radius * math.sin(speed * t)
        pose.pose.orientation.w = 1.0
        
        self.pose_pub.publish(pose)
        self.setpoint_counter += 1

        # Warmup: Stream 40 setpoints (2 seconds) before asking PX4 for anything
        if self.setpoint_counter < 40:
            if self.setpoint_counter % 10 == 0:
                self.get_logger().info(f'[{self.namespace}] Streaming setpoints: {self.setpoint_counter}/40...')
            return

        # State Machine: OFFBOARD first, then ARM (Standard PX4 Protocol)
        if (now - self.last_req) > 2.0:
            if self.state.mode != "OFFBOARD":
                self.get_logger().info(f'[{self.namespace}] Requesting OFFBOARD...')
                mode_req = SetMode.Request(custom_mode='OFFBOARD')
                self.mode_client.call_async(mode_req)
            elif not self.state.armed:
                self.get_logger().info(f'[{self.namespace}] OFFBOARD confirmed! Requesting ARM...')
                arm_req = CommandBool.Request(value=True)
                self.arming_client.call_async(arm_req)
            else:
                self.get_logger().info(f'[{self.namespace}] Flying! Executing chase loop.')
                
            self.last_req = now

def main(args=None):
    rclpy.init(args=args)
    drone_ns = sys.argv[1] if len(sys.argv) > 1 else 'drone0'
    controller = SwarmController(drone_ns)
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
        
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
