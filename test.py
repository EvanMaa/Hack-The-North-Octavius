import rpi_gpio as GPIO
import time

MOTORS = {
    "1": [17, 18, 27, 22],
    "2": [23, 24, 25, 5],
    "3": [6, 13, 19, 26],
}

SEQUENCE = [
    [1, 0, 0, 1],
    [1, 1, 0, 0],
    [0, 1, 1, 0],
    [0, 0, 1, 1],
]

STEP_DELAY = 0.003
SPIN_TIME = 1.0

GPIO.setmode(GPIO.BCM)

# Initialize GPIO
for motor in MOTORS.values():
    for pin in motor:
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, GPIO.LOW)


def set_motor(pins, state):
    for pin, value in zip(pins, state):
        GPIO.output(pin, GPIO.HIGH if value else GPIO.LOW)


def stop_motor(pins):
    for pin in pins:
        GPIO.output(pin, GPIO.LOW)


def spin_motor(motor_number, direction):
    pins = MOTORS[motor_number]

    direction_name = "forward" if direction == 1 else "backward"
    print(f"Spinning motor {motor_number} {direction_name}...")

    start = time.monotonic()

    # Start at opposite ends depending on direction
    phase = 0

    while time.monotonic() - start < SPIN_TIME:
        set_motor(pins, SEQUENCE[phase])

        # Move forward or backward through sequence
        phase = (phase + direction) % len(SEQUENCE)

        time.sleep(STEP_DELAY)

    stop_motor(pins)

    print(f"Motor {motor_number} stopped.")


try:
    print("Motor test ready")
    print()
    print(" 1 = Motor 1 forward")
    print(" 2 = Motor 2 forward")
    print(" 3 = Motor 3 forward")
    print("-1 = Motor 1 backward")
    print("-2 = Motor 2 backward")
    print("-3 = Motor 3 backward")
    print(" q = Quit")

    while True:
        command = input("> ").strip()

        if command.lower() == "q":
            break

        # Forward
        elif command in ("1", "2", "3"):
            spin_motor(command, 1)

        # Backward
        elif command in ("-1", "-2", "-3"):
            motor_number = command[1:]
            spin_motor(motor_number, -1)

        else:
            print("Enter 1, 2, 3, -1, -2, -3, or q")

finally:
    # Turn everything off
    for motor in MOTORS.values():
        stop_motor(motor)

    GPIO.cleanup()

print("Done")