import subprocess
import time

PINS = [17, 18, 27, 22]

# Half-step sequence for 28BYJ-48
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


def gpio(pin, value):
    level = "dh" if value else "dl"
    subprocess.run(
        ["gpio-rp1", "set", str(pin), level],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )


def setup():
    for pin in PINS:
        subprocess.run(
            ["gpio-rp1", "set", str(pin), "op", "pn", "dl"]
        )


def set_motor(state):
    for pin, value in zip(PINS, state):
        gpio(pin, value)


def stop():
    set_motor([0, 0, 0, 0])


setup()

print("Forward")

# Just move it a little
for _ in range(30):
    for state in SEQUENCE:
        set_motor(state)
        time.sleep(0.005)

stop()

time.sleep(1)

print("Backward")

for _ in range(30):
    for state in reversed(SEQUENCE):
        set_motor(state)
        time.sleep(0.005)

stop()

print("Done")