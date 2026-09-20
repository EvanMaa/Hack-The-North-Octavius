import rpi_gpio as GPIO
import time

PINS = [17, 18, 27, 22]

SEQUENCE = [
    [1, 0, 0, 1],
    [1, 1, 0, 0],
    [0, 1, 1, 0],
    [0, 0, 1, 1],
]

GPIO.setmode(GPIO.BCM)

for pin in PINS:
    GPIO.setup(pin, GPIO.OUT)


def set_motor(state):
    for pin, value in zip(PINS, state):
        GPIO.output(
            pin,
            GPIO.HIGH if value else GPIO.LOW
        )


try:
    print("GO!")

    for _ in range(500):
        for state in SEQUENCE:
            set_motor(state)

            # Start here
            time.sleep(0.003)

finally:
    # Turn all coils off
    for pin in PINS:
        GPIO.output(pin, GPIO.LOW)

    GPIO.cleanup()

print("Done")