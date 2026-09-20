import rpi_gpio as GPIO
import time

# GPIO numbers, NOT physical pin numbers
MOTORS = [
    [17, 18, 27, 22],  # Motor 1
    [23, 24, 25, 5],   # Motor 2
    [6, 13, 19, 26],   # Motor 3
]

# Full-step sequence
SEQUENCE = [
    [1, 0, 0, 1],
    [1, 1, 0, 0],
    [0, 1, 1, 0],
    [0, 0, 1, 1],
]

GPIO.setmode(GPIO.BCM)

# Configure all 12 pins
for motor in MOTORS:
    for pin in motor:
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, GPIO.LOW)


def set_motor(motor_pins, state):
    for pin, value in zip(motor_pins, state):
        GPIO.output(pin, GPIO.HIGH if value else GPIO.LOW)


try:
    print("All 3 motors starting...")

    # ~10 seconds depending on motor/OS timing
    for _ in range(800):
        for state in SEQUENCE:

            # Apply same phase to all three motors
            for motor in MOTORS:
                set_motor(motor, state)

            time.sleep(0.003)

    print("Finished!")

finally:
    # Turn off every motor coil
    for motor in MOTORS:
        for pin in motor:
            GPIO.output(pin, GPIO.LOW)

    GPIO.cleanup()