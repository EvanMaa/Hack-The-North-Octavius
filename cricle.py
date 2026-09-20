import time

# Each tuple is:
# (Motor 1, Motor 2, Motor 3)

CIRCLE_SEQUENCE = [
    (-2,  1,  1),
    (-2,  0,  2),
    (-1, -1,  2),
    ( 0, -2,  2),
    ( 1, -2,  1),
    ( 2, -2,  0),
    ( 2, -1, -1),
    ( 2,  0, -2),
    ( 1,  1, -2),
    ( 0,  2, -2),
    (-1,  2, -1),
    (-2,  2,  0),
]

CIRCLE_STEP_TIME = 0.30


def circle_test():
    global motor_commands

    print("Starting circle...")

    try:
        while True:

            for command in CIRCLE_SEQUENCE:

                with state_lock:
                    motor_commands = list(command)

                print("Circle:", command)

                time.sleep(CIRCLE_STEP_TIME)

    except KeyboardInterrupt:
        pass

    finally:
        with state_lock:
            motor_commands = [0, 0, 0]

        print("Circle stopped")