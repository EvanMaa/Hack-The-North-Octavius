import subprocess
import time

PINS = [17, 18, 27, 22]

SEQUENCE = [
    [1, 0, 0, 0],
    [1, 1, 0, 0],
    [0, 1, 0, 0],
    [0, 1, 1, 0],
    [0, 0, 1, 0],
    [0, 0, 1, 1],
    [0, 0, 0, 1],
    [1, 0, 0, 1],
]

def set_pin(pin, value):
    subprocess.run([
        "gpio-rp1",
        "set",
        str(pin),
        "dh" if value else "dl"
    ], check=True)


def set_motor(state):
    for pin, value in zip(PINS, state):
        set_pin(pin, value)


# Configure GPIO
for pin in PINS:
    subprocess.run([
        "gpio-rp1", "set", str(pin), "op", "pn", "dl"
    ], check=True)


print("Motor moving...")

# Move a small amount
for _ in range(20):
    for state in SEQUENCE:
        set_motor(state)
        time.sleep(0.01)

# Release motor
set_motor([0, 0, 0, 0])

print("Done")