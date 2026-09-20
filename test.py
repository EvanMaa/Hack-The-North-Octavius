import rpi_gpio as GPIO
import time
# Direction of the Spool tightening
DIRECTION = {
    "1": -1,
    "2": 1,
    "3": 1,
}

# ============================================================
# CONFIG
# ============================================================

# GPIO numbers (BCM numbering)
MOTORS = {
    "1": [17, 18, 27, 22],
    "2": [23, 24, 25, 5],
    "3": [6, 13, 19, 26],
}

# Single-coil / wave-drive sequence
# Only ONE coil is powered at a time
SEQUENCE = [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
]

# Start conservative.
# Try 0.003, 0.0025, 0.002 later.
STEP_DELAY = 0.004

# How long each command spins the motor
SPIN_TIME = 0.5


# ============================================================
# GPIO SETUP
# ============================================================

GPIO.setmode(GPIO.BCM)

for pins in MOTORS.values():
    for pin in pins:
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, GPIO.LOW)


# ============================================================
# MOTOR FUNCTIONS
# ============================================================

def set_motor(pins, state):
    """Apply one phase to a motor."""
    for pin, value in zip(pins, state):
        GPIO.output(
            pin,
            GPIO.HIGH if value else GPIO.LOW
        )


def stop_motor(pins):
    """Turn off all coils."""
    for pin in pins:
        GPIO.output(pin, GPIO.LOW)


def spin_motor(motor_number, direction):
    """
    direction:
        +1 = forward
        -1 = backward
    """

    pins = MOTORS[motor_number]
    direction =* DIRECTION[motor_number]

    if direction > 0:
        direction_name = "FORWARD"
    else:
        direction_name = "BACKWARD"

    print(f"Motor {motor_number}: {direction_name}")

    # Start at known phase
    phase = 0

    set_motor(pins, SEQUENCE[phase])

    # Give rotor a moment to align
    time.sleep(0.05)

    start = time.monotonic()

    while time.monotonic() - start < SPIN_TIME:

        # Move forward or backward through phases
        phase = (phase + direction) % len(SEQUENCE)

        set_motor(
            pins,
            SEQUENCE[phase]
        )

        time.sleep(STEP_DELAY)

    stop_motor(pins)

    print(f"Motor {motor_number}: STOPPED")


# ============================================================
# COMMAND LOOP
# ============================================================

try:

    print("""
================================
       3 MOTOR TESTER
================================

 1  = Motor 1 forward
-1  = Motor 1 backward

 2  = Motor 2 forward
-2  = Motor 2 backward

 3  = Motor 3 forward
-3  = Motor 3 backward

 q  = Quit

Wave drive:
1000 -> 0100 -> 0010 -> 0001
================================
""")

    while True:

        command = input("> ").strip()

        # Quit
        if command.lower() == "q":
            break

        # Forward
        elif command in ("1", "2", "3"):
            spin_motor(
                motor_number=command,
                direction=1
            )

        # Backward
        elif command in ("-1", "-2", "-3"):
            spin_motor(
                motor_number=command[1:],
                direction=-1
            )

        else:
            print(
                "Enter 1, 2, 3, "
                "-1, -2, -3, or q"
            )


# ============================================================
# CLEANUP
# ============================================================

finally:

    print("Shutting down motors...")

    for pins in MOTORS.values():
        stop_motor(pins)

    GPIO.cleanup()

print("Done.")