import subprocess

PINS = [17, 18, 27, 22]

SEQUENCE = [
    [1, 0, 0, 1],
    [1, 1, 0, 0],
    [0, 1, 1, 0],
    [0, 0, 1, 1],
]

current_state = [0, 0, 0, 0]


def gpio(pin, value):
    subprocess.run(
        [
            "gpio-rp1",
            "set",
            str(pin),
            "dh" if value else "dl",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def set_motor(new_state):
    global current_state

    for i in range(4):

        # Only touch GPIO if it actually changed
        if new_state[i] != current_state[i]:
            gpio(PINS[i], new_state[i])

    current_state = new_state.copy()


# Configure outputs once
for pin in PINS:
    subprocess.run(
        ["gpio-rp1", "set", str(pin), "op", "pn", "dl"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


# Run motor
for _ in range(500):

    for state in SEQUENCE:
        set_motor(state)


# Shut motor off
set_motor([0, 0, 0, 0])