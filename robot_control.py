from enum import Enum

class Action(Enum):
    STOP = 0
    LEFT = 1
    RIGHT = 2
    FORWARD = 3
    BACKWARD = 4

# todo: IMPLEMENT execute_action such it actually communicate with the robot properly
def execute_action(x, y, direction, action: Action):
    x_new, y_new, direction_new = x, y, direction
    if action == Action.STOP:
        # todo: complete robot control
        pass
    elif action == Action.LEFT:
        # todo: complete robot control
        direction_new = (direction-15)%360
        pass
    elif action == Action.RIGHT:
        # todo: complete robot control
        direction_new = (direction + 15) % 360
        pass
    elif action == Action.FORWARD:
        pass
    elif action == Action.BACKWARD:
        pass

    return x_new, y_new, direction_new