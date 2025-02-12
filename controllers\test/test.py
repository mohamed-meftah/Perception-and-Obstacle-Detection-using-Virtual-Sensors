import numpy as np
from controller import Robot

class ObstacleAvoider:
    def __init__(self, robot):
        self.robot = robot
        self.TIME_STEP = 32
        self.max_speed = 10.0
        
        # Wheel configuration [front_left, front_right, rear_left, rear_right]
        self.wheels = [
            robot.getDevice('wheel1'),
            robot.getDevice('wheel2'),
            robot.getDevice('wheel3'),
            robot.getDevice('wheel4')
        ]
        for wheel in self.wheels:
            wheel.setPosition(float('inf'))
            wheel.setVelocity(0.0)
            
        # LiDAR configuration
        self.lidar = robot.getDevice('lidar')
        self.lidar.enable(self.TIME_STEP)
        self.lidar.enablePointCloud()
        
        # Get actual LiDAR parameters
        self.lidar_fov = self.lidar.getFov()
        self.lidar_res = self.lidar.getHorizontalResolution()
        
        # State machine
        self.STATE_MOVING = 0
        self.STATE_TURNING = 1
        self.state = self.STATE_MOVING
        
        # Obstacle detection parameters
        self.obstacle_distance = 0.4  # Distance to start avoiding
        self.side_obstacle_distance = 0.2 # Distance for side obstacles
        self.min_safe_distance = 0.01  # Emergency stop distance
        
        # Turning parameters
        self.angular_speed = 2.0  # Angular speed for turning
        self.turn_direction = 0  # -1: left, 1: right
        self.turn_reason = None  # 'front', 'left', 'right'
        
        # Debugging counters
        self.loop_count = 0

    def set_speeds(self, left, right):
        """Set speeds with front/rear wheel synchronization"""
        for i, wheel in enumerate(self.wheels):
            if i % 2 == 0:  # Left wheels (0 and 2)
                wheel.setVelocity(left)
            else:           # Right wheels (1 and 3)
                wheel.setVelocity(right)

    def get_front_distance(self):
        """Get filtered front distance with emergency detection"""
        ranges = self.lidar.getRangeImage()
        
        # Check 30° cone in front of robot
        center = len(ranges) // 2
        view_span = int(len(ranges) * 30 / 360)  # ±15 degrees
        frontal_ranges = ranges[center-view_span:center+view_span]
        
        # Filter out infinite values and find minimum
        valid_ranges = [r for r in frontal_ranges if r < 10]
        return min(valid_ranges) if valid_ranges else float('inf')

    def get_left_distance(self):
        """Get minimum distance in the left sector"""
        ranges = self.lidar.getRangeImage()
        mid = len(ranges) // 2
        left_scan = ranges[:mid//2]  # First quarter (left side)
        valid_left = [r for r in left_scan if r < 10]
        return min(valid_left) if valid_left else float('inf')

    def get_right_distance(self):
        """Get minimum distance in the right sector"""
        ranges = self.lidar.getRangeImage()
        mid = len(ranges) // 2
        right_scan = ranges[3*mid//2:]  # Last quarter (right side)
        valid_right = [r for r in right_scan if r < 10]
        return min(valid_right) if valid_right else float('inf')

    def find_best_turn_direction(self):
        """Determine safest turn direction using full LiDAR data"""
        ranges = self.lidar.getRangeImage()
        mid = len(ranges) // 2
        
        # Analyze left and right quarters of the scan
        left_scan = ranges[:mid//2]
        right_scan = ranges[3*mid//2:]
        
        # Filter valid ranges and handle empty cases
        valid_left = [r for r in left_scan if r < 10]
        valid_right = [r for r in right_scan if r < 10]
        
        # Assign default high value if no valid readings
        left_avg = np.mean(valid_left) if len(valid_left) > 0 else 10.0 
        right_avg = np.mean(valid_right) if len(valid_right) > 0 else 10.0
        
        # Prefer direction with higher average distance
        return 1 if right_avg > left_avg else -1

    def emergency_stop(self):
        self.set_speeds(0, 0)
        print("EMERGENCY STOP ACTIVATED!")

    def run(self):
        while self.robot.step(self.TIME_STEP) != -1:
            self.loop_count += 1
            front_dist = self.get_front_distance()
            left_dist = self.get_left_distance()
            right_dist = self.get_right_distance()
            
            # Debug print every 10 steps
            if self.loop_count % 10 == 0:
                print(f"Front: {front_dist:.2f}m | Left: {left_dist:.2f}m | Right: {right_dist:.2f}m | State: {self.state}")

            # Emergency stop check
            if front_dist < self.min_safe_distance:
                self.emergency_stop()
                continue

            if self.state == self.STATE_MOVING:
                if front_dist < self.obstacle_distance:
                    # Initiate turn for front obstacle
                    self.turn_reason = 'front'
                    self.turn_direction = self.find_best_turn_direction()
                    self.state = self.STATE_TURNING
                    turn_speed = self.angular_speed
                    
                    # Set turning speeds
                    if self.turn_direction == 1:  # Right turn
                        self.set_speeds(turn_speed, -turn_speed)
                    else:  # Left turn
                        self.set_speeds(-turn_speed, turn_speed)
                        
                    print(f"Front obstacle detected. Turning {'RIGHT' if self.turn_direction == 1 else 'LEFT'}") 
                elif left_dist < self.side_obstacle_distance:
                    # Obstacle on left, turn right
                    self.turn_reason = 'left'
                    self.state = self.STATE_TURNING
                    self.set_speeds(self.angular_speed, -self.angular_speed)
                    print("Left obstacle detected. Turning RIGHT")
                elif right_dist < self.side_obstacle_distance:
                    # Obstacle on right, turn left
                    self.turn_reason = 'right'
                    self.state = self.STATE_TURNING
                    self.set_speeds(-self.angular_speed, self.angular_speed)
                    print("Right obstacle detected. Turning LEFT")
                else:
                    # Move forward
                    self.set_speeds(self.max_speed, self.max_speed)

            elif self.state == self.STATE_TURNING:
                # Check exit condition based on turn reason
                current_front = self.get_front_distance()
                current_left = self.get_left_distance()
                current_right = self.get_right_distance()

                clear = False
                if self.turn_reason == 'front':
                    clear = current_front > self.obstacle_distance * 1.5
                elif self.turn_reason == 'left':
                    clear = current_left >= self.side_obstacle_distance
                elif self.turn_reason == 'right':
                    clear = current_right >= self.side_obstacle_distance

                if clear:
                    # Path clear, resume moving
                    self.state = self.STATE_MOVING
                    self.set_speeds(self.max_speed, self.max_speed)
                    print(f"Clear path detected, resuming movement")

if __name__ == '__main__':
    robot = Robot()
    avoider = ObstacleAvoider(robot)
    avoider.run()
