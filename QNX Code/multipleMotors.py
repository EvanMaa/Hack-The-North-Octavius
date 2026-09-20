import rpi_gpio as GPIO
import time

# Sign that TIGHTENS each spool
DIRECTION = {
    "1": -1,
    "2": 1,
    "3": 1,
}

MOTORS = {
    "1": [17, 18, 27, 22],
    "2": [23, 24, 25, 5],
    "3": [6, 13, 19, 26],
}

SEQUENCE = [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
]

STEP_DELAY = 0.004
SPIN_TIME = 0.5

# Remember each motor's phase between commands so it doesn't
# jump back to phase 0 every time
phases = {m: 0 for m in MOTORS}

GPIO.setmode(GPIO.BCM)
for pins in MOTORS.values():
    for pin in pins:
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, GPIO.LOW)


def set_motor(pins, state):
    for pin, value in zip(pins, state):
        GPIO.output(pin, GPIO.HIGH if value else GPIO.LOW)


def stop_motor(pins):
    for pin in pins:
        GPIO.output(pin, GPIO.LOW)


def spin_motors(commands):
    """
    commands: dict like {"1": +1, "2": -1, "3": -1}
        +1 = tighten, -1 = loosen
    All listed motors step together in one loop.
    """
    # Convert tighten/loosen into physical stepping direction
    steps = {m: c * DIRECTION[m] for m, c in commands.items()}

    for m, c in commands.items():
        print(f"Motor {m}: {'TIGHTEN' if c > 0 else 'LOOSEN'}")

    # Energize current phase on every motor so rotors align
    for m in steps:
        set_motor(MOTORS[m], SEQUENCE[phases[m]])
    time.sleep(0.05)

    start = time.monotonic()
    while time.monotonic() - start < SPIN_TIME:
        for m, d in steps.items():
            phases[m] = (phases[m] + d) % len(SEQUENCE)
            set_motor(MOTORS[m], SEQUENCE[phases[m]])
        time.sleep(STEP_DELAY)

    for m in steps:
        stop_motor(MOTORS[m])
    print("STOPPED")


def tighten_one(motor):
    """Tighten `motor`, loosen all the others."""
    spin_motors({m: (1 if m == motor else -1) for m in MOTORS})


def loosen_one(motor):
    """Loosen `motor`, tighten all the others."""
    spin_motors({m: (-1 if m == motor else 1) for m in MOTORS})


try:
    print("""
================================
       3 MOTOR CONTROL
================================
 1, 2, 3     = tighten that motor, loosen the others
-1, -2, -3   = loosen that motor, tighten the others
 q           = quit
================================
""")

    while True:
        command = input("> ").strip()

        if command.lower() == "q":
            break
        elif command in MOTORS:
            tighten_one(command)
        elif command.startswith("-") and command[1:] in MOTORS:
            loosen_one(command[1:])
        else:
            print("Enter 1, 2, 3, -1, -2, -3, or q")

finally:
    print("Shutting down motors...")
    for pins in MOTORS.values():
        stop_motor(pins)
    GPIO.cleanup()

print("Done.")