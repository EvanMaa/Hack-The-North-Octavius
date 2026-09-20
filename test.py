import rpi_gpio as GPIO
import time

# GPIO numbers
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

# Initialize all 12 GPIO pins
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


def spin_motor(motor_number):
    pins = MOTORS[motor_number]

    print(f"Spinning motor {motor_number}...")

    start = time.monotonic()
    phase = 0

    while time.monotonic() - start < SPIN_TIME:
        set_motor(pins, SEQUENCE[phase])

        phase = (phase + 1) % len(SEQUENCE)

        time.sleep(STEP_DELAY)

    stop_motor(pins)

    print(f"Motor {motor_number} stopped.")


try:
    print("Motor test ready")
    print("1 = Motor 1")
    print("2 = Motor 2")
    print("3 = Motor 3")
    print("q = Quit")

    while True:
        command = input("> ").strip()

        if command in MOTORS:
            spin_motor(command)

        elif command.lower() == "q":
            break

        else:
            print("Enter 1, 2, 3, or q")

finally:
    # Make absolutely sure everything is off
    for motor in MOTORS.values():
        stop_motor(motor)

    GPIO.cleanup()

print("Done")