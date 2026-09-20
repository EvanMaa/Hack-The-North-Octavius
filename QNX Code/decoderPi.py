import rpi_gpio as GPIO
import socket
import threading
import time

# ============================================================
# CONFIG
# ============================================================

MOTORS = [
    [17, 18, 27, 22],  # Motor 1
    [23, 24, 25, 5],   # Motor 2
    [6, 13, 19, 26],   # Motor 3
]

SEQUENCE = [
    [1, 0, 0, 1],
    [1, 1, 0, 0],
    [0, 1, 1, 0],
    [0, 0, 1, 1],
]

STEP_DELAY = 0.01

UDP_PORT = 5005

# Stop robot if laptop disappears for this long
WATCHDOG_TIMEOUT = 0.5


# ============================================================
# SHARED STATE
# ============================================================

# -1 = backward
#  0 = stop
# +1 = forward
motor_commands = [0, 0, 0]

last_packet_time = time.monotonic()

running = True


# ============================================================
# GPIO
# ============================================================

GPIO.setmode(GPIO.BCM)

for motor in MOTORS:
    for pin in motor:
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, GPIO.LOW)


def set_motor(pins, state):
    for pin, value in zip(pins, state):
        GPIO.output(
            pin,
            GPIO.HIGH if value else GPIO.LOW
        )


def stop_motor(pins):
    for pin in pins:
        GPIO.output(pin, GPIO.LOW)


# ============================================================
# UDP RECEIVER
# ============================================================

def udp_receiver():
    global motor_commands
    global last_packet_time
    global running

    sock = socket.socket(
        socket.AF_INET,
        socket.SOCK_DGRAM
    )

    sock.bind(("0.0.0.0", UDP_PORT))

    print(f"Listening on UDP port {UDP_PORT}")

    while running:

        try:
            data, addr = sock.recvfrom(1024)

            # Expected:
            # "1,0,-1"

            message = data.decode().strip()

            values = [
                int(x)
                for x in message.split(",")
            ]

            # Validate packet
            if (
                len(values) == 3
                and all(x in (-1, 0, 1) for x in values)
            ):
                motor_commands = values
                last_packet_time = time.monotonic()

                print("Command:", motor_commands)

            else:
                print("Invalid packet:", message)

        except Exception as e:
            print("UDP error:", e)


# ============================================================
# MOTOR LOOP
# ============================================================

def motor_loop():
    global running

    # Each motor keeps its own phase
    phases = [0, 0, 0]

    while running:

        # Take current command
        commands = motor_commands.copy()

        # --------------------------------------------
        # WATCHDOG
        # --------------------------------------------

        if (
            time.monotonic() - last_packet_time
            > WATCHDOG_TIMEOUT
        ):
            commands = [0, 0, 0]

        # --------------------------------------------
        # UPDATE MOTORS
        # --------------------------------------------

        for i in range(3):

            command = commands[i]
            pins = MOTORS[i]

            # STOP
            if command == 0:
                stop_motor(pins)
                continue

            # FORWARD / BACKWARD
            phases[i] = (
                phases[i] + command
            ) % len(SEQUENCE)

            set_motor(
                pins,
                SEQUENCE[phases[i]]
            )

        time.sleep(STEP_DELAY)


# ============================================================
# START
# ============================================================

receiver_thread = threading.Thread(
    target=udp_receiver,
    daemon=True
)

motor_thread = threading.Thread(
    target=motor_loop,
    daemon=True
)

receiver_thread.start()
motor_thread.start()


print()
print("3-motor controller running")
print("Expected UDP packets:")
print("  1,0,-1")
print("  0,0,0")
print(" -1,1,0")
print()

try:

    while True:
        time.sleep(1)

except KeyboardInterrupt:

    print("\nStopping...")

finally:

    running = False

    motor_commands = [0, 0, 0]

    for motor in MOTORS:
        stop_motor(motor)

    GPIO.cleanup()

    print("Done.")