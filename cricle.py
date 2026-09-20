import rpi_gpio as GPIO
import time

# ============================================================
# CONFIG
# ============================================================

# GPIO pins for each motor
MOTORS = [
    [17, 18, 27, 22],  # Motor 1
    [23, 24, 25, 5],   # Motor 2
    [6, 13, 19, 26],   # Motor 3
]

# Motor 1 is physically reversed
MOTOR_DIRECTION = [-1, 1, 1]

# Two-coil full-step sequence
SEQUENCE = [
    [1, 0, 0, 1],
    [1, 1, 0, 0],
    [0, 1, 1, 0],
    [0, 0, 1, 1],
]

# ------------------------------------------------------------
# MOTOR SPEEDS
#
# Smaller delay = faster
# ------------------------------------------------------------

SPEED_DELAYS = {
    1: 0.020,  # slow
    2: 0.010,  # medium
    3: 0.005,  # fast
}

# ============================================================
# CIRCLE SETTINGS
# ============================================================

# Faster transitions
CIRCLE_STEP_TIME = 0.30

# Tightness-biased circular trajectory
CIRCLE_SEQUENCE = [
    (-2,  2,  2),
    (-2,  1,  3),
    (-2,  0,  3),

    (-1, -1,  3),
    ( 0, -2,  3),
    ( 1, -2,  3),

    ( 2, -2,  2),
    ( 3, -2,  1),
    ( 3, -2,  0),

    ( 3, -1, -1),
    ( 3,  0, -2),
    ( 3,  1, -2),

    ( 2,  2, -2),
    ( 1,  3, -2),
    ( 0,  3, -2),

    (-1,  3, -1),
    (-2,  3,  0),
    (-2,  3,  1),
]

# ============================================================
# GPIO SETUP
# ============================================================

GPIO.setmode(GPIO.BCM)

for motor in MOTORS:
    for pin in motor:
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, GPIO.LOW)


# ============================================================
# GPIO FUNCTIONS
# ============================================================

def set_motor(pins, state):
    """Apply one electrical phase to a motor."""

    for pin, value in zip(pins, state):

        GPIO.output(
            pin,
            GPIO.HIGH if value else GPIO.LOW
        )


def stop_motor(pins):
    """Turn all coils off for one motor."""

    for pin in pins:
        GPIO.output(pin, GPIO.LOW)


def stop_all():
    """Turn all three motors off."""

    for motor in MOTORS:
        stop_motor(motor)


# ============================================================
# MOTOR STATE
# ============================================================

# Each motor needs its own phase
phases = [0, 0, 0]

# Each motor also needs its own timer
last_steps = [
    time.monotonic(),
    time.monotonic(),
    time.monotonic(),
]

# Track whether each motor is energized
motor_active = [False, False, False]


# ============================================================
# MOTOR UPDATE
# ============================================================

def update_motors(commands):
    """
    commands is a tuple such as:

        (-2, 1, 1)

    Sign:
        + = forward
        - = backward
        0 = stop

    Magnitude:
        1 = slow
        2 = medium
        3 = fast
    """

    now = time.monotonic()

    for i in range(3):

        command = commands[i]
        pins = MOTORS[i]

        # ----------------------------------------------------
        # STOP
        # ----------------------------------------------------

        if command == 0:

            if motor_active[i]:
                stop_motor(pins)
                motor_active[i] = False

            continue

        # ----------------------------------------------------
        # SPEED
        # ----------------------------------------------------

        speed_level = abs(command)

        step_delay = SPEED_DELAYS[speed_level]

        # ----------------------------------------------------
        # DIRECTION
        # ----------------------------------------------------

        if command > 0:
            direction = 1
        else:
            direction = -1

        # Motor 1 is physically reversed
        direction *= MOTOR_DIRECTION[i]

        # ----------------------------------------------------
        # STEP
        # ----------------------------------------------------

        if now - last_steps[i] >= step_delay:

            phases[i] = (
                phases[i] + direction
            ) % len(SEQUENCE)

            set_motor(
                pins,
                SEQUENCE[phases[i]]
            )

            last_steps[i] = now
            motor_active[i] = True


# ============================================================
# CIRCLE TEST
# ============================================================

def run_circle():

    print()
    print("========================================")
    print("       SLOW CIRCLE TEST")
    print("========================================")
    print()
    print(
        f"Number of circle states: "
        f"{len(CIRCLE_SEQUENCE)}"
    )

    print(
        f"Time per state: "
        f"{CIRCLE_STEP_TIME} seconds"
    )

    print(
        f"Approx circle time: "
        f"{len(CIRCLE_SEQUENCE) * CIRCLE_STEP_TIME} seconds"
    )

    print()
    print("Ctrl+C to stop")
    print()

    # Start at first command
    circle_index = 0

    current_command = (
        CIRCLE_SEQUENCE[circle_index]
    )

    print(
        f"Step {circle_index + 1}/"
        f"{len(CIRCLE_SEQUENCE)}:",
        current_command
    )

    next_circle_update = (
        time.monotonic()
        + CIRCLE_STEP_TIME
    )

    while True:

        now = time.monotonic()

        # ----------------------------------------------------
        # MOVE TO NEXT PART OF CIRCLE
        # ----------------------------------------------------

        if now >= next_circle_update:

            circle_index = (
                circle_index + 1
            ) % len(CIRCLE_SEQUENCE)

            current_command = (
                CIRCLE_SEQUENCE[circle_index]
            )

            print(
                f"Step {circle_index + 1}/"
                f"{len(CIRCLE_SEQUENCE)}:",
                current_command
            )

            next_circle_update = (
                now + CIRCLE_STEP_TIME
            )

        # ----------------------------------------------------
        # CONTINUOUSLY DRIVE MOTORS
        # ----------------------------------------------------

        update_motors(current_command)

        # Small CPU break
        time.sleep(0.0005)


# ============================================================
# MAIN
# ============================================================

try:

    run_circle()

except KeyboardInterrupt:

    print()
    print("Stopping circle...")

finally:

    stop_all()

    GPIO.cleanup()

    print("All motors stopped.")
    print("Done.")