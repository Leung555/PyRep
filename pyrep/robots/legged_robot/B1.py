from pyrep.robots.legged_robot.legged_robot_component import LeggedRobotComponent

class B1(LeggedRobotComponent):

    def __init__(self, count: int = 0):
        super().__init__(count, 'B1')