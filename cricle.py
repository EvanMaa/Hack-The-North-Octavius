import rpi_gpio as GPIO
import time

# ============================================================
# CONFIG
# ============================================================

MOTORS = [
    [17, 18, 27, 22],  # Motor 1
    [23, 24, 25, 5],   # Motor 2
    [6, 13, 19, 26],   # Motor 3
]

# Motor 1 is physically reversed
MOTOR_DIRECTION = [-1, 1, 1]

SEQUENCE = [
    [1, 0, 0, 1],
    [1, 1, 0, 0],
    [0, 1, 1, 0],
    [0, 0, 1, 1],
]

# Three motor speeds
SPEED_DELAYS = {
    1: 0.020,
    2: 0.010,
    3: 0.005,
}

# How quickly we move around the circle
CIRCLE_STEP_TIME = 0.30

# Commands used to approximate a circular trajectory
CIRCLE_SEQUENCE = [
    (-3,  1,  1),
    (-2, -2,  1),

    ( 1, -3,  1),
    ( 1, -2, -2),

    ( 1,  1, -3),
    (-2,  1, -2),
]

CIRCLE_STEP_TIME = 0.5


# ============================================================
# GPIO SETUP
# ============================================================

GPIO.setmode(GPIO.BCM)

for motor in MOTORS:
    for pin in motor:
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, GPIO.LOW)


def set_motor(pins, state):
    for pin, value in zip(pins, state):
        GPIO.output(
            pin,
            GPIO.HIGH if value else GPIO.LOW
        )


def stop_motor(pins):
    for pin in pins:
        GPIO.output(pin, GPIO.LOW)


def stop_all():
    for motor in MOTORS:
        stop_motor(motor)


# ============================================================
# MOTOR STATE
# ============================================================

phases = [0, 0, 0]

last_steps = [
    time.monotonic(),
    time.monotonic(),
    time.monotonic(),
]


# ============================================================
# MOTOR UPDATE
# ============================================================

def update_motors(commands):
    """
    commands example:
        (-2, 1, 1)

    magnitude = speed
    sign      = direction
    """

    now = time.monotonic()

    for i in range(3):

        command = commands[i]

        # Stop
        if command == 0:
            stop_motor(MOTORS[i])
            continue

        # Speed
        speed = abs(command)
        step_delay = SPEED_DELAYS[speed]

        # Direction
        direction = 1 if command > 0 else -1

        # Motor 1 is mounted backwards
        direction *= MOTOR_DIRECTION[i]

        # Time for another step?
        if now - last_steps[i] >= step_delay:

            phases[i] = (
                phases[i] + direction
            ) % len(SEQUENCE)

            set_motor(
                MOTORS[i],
                SEQUENCE[phases[i]]
            )

            last_steps[i] = now


# ============================================================
# CIRCLE
# ============================================================

def run_circle():

    print("Starting circular motion")
    print("Ctrl+C to stop")

    circle_index = 0

    current_command = CIRCLE_SEQUENCE[0]

    next_circle_update = (
        time.monotonic() + CIRCLE_STEP_TIME
    )

    while True:

        now = time.monotonic()

        # Move to next part of circle
        if now >= next_circle_update:

            circle_index = (
                circle_index + 1
            ) % len(CIRCLE_SEQUENCE)

            current_command = (
                CIRCLE_SEQUENCE[circle_index]
            )

            print("Command:", current_command)

            next_circle_update = (
                now + CIRCLE_STEP_TIME
            )

        # Continuously step motors
        update_motors(current_command)

        # Small CPU break
        time.sleep(0.0005)


# ============================================================
# MAIN
# ============================================================

try:

    run_circle()

except KeyboardInterrupt:

    print("\nStopping circle...")

finally:

    stop_all()
    GPIO.cleanup()

    print("Done.")