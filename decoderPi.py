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

# Motor 1 is physically reversed
MOTOR_DIRECTION = [-1, 1, 1]

SEQUENCE = [
    [1, 0, 0, 1],
    [1, 1, 0, 0],
    [0, 1, 1, 0],
    [0, 0, 1, 1],
]

# AUTO mode speeds
SPEED_DELAYS = {
    1: 0.020,  # slow
    2: 0.010,  # medium
    3: 0.005,  # fast
}

# Manual mode settings
MANUAL_STEP_DELAY = 0.010
MANUAL_SPIN_TIME = 1.0

UDP_PORT = 5005
WATCHDOG_TIMEOUT = 0.5


# ============================================================
# SHARED STATE
# ============================================================

# Start in MANUAL mode
mode = "MANUAL"

# Commands actually consumed by motor loop
motor_commands = [0, 0, 0]

# Latest command received from laptop
auto_commands = [0, 0, 0]

last_packet_time = time.monotonic()

running = True

state_lock = threading.Lock()


# ============================================================
# GPIO SETUP
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


def stop_all():
    for motor in MOTORS:
        stop_motor(motor)


# ============================================================
# UDP RECEIVER
# ============================================================

def udp_receiver():
    global auto_commands
    global last_packet_time
    global running

    sock = socket.socket(
        socket.AF_INET,
        socket.SOCK_DGRAM
    )

    sock.bind(("0.0.0.0", UDP_PORT))
    sock.settimeout(0.5)

    print(f"[UDP] Listening on port {UDP_PORT}")

    while running:

        try:
            data, addr = sock.recvfrom(1024)

            message = data.decode().strip()

            values = [
                int(x)
                for x in message.split(",")
            ]

            if (
                len(values) == 3
                and all(-3 <= x <= 3 for x in values)
            ):

                with state_lock:
                    auto_commands = values
                    last_packet_time = time.monotonic()

                # Don't spam terminal constantly
                if mode == "AUTO":
                    print(f"\r[AUTO] {values}     ", end="")

            else:
                print("\nInvalid UDP packet:", message)

        except socket.timeout:
            pass

        except Exception as e:
            if running:
                print("\nUDP error:", e)

    sock.close()


# ============================================================
# MOTOR LOOP
# ============================================================

def motor_loop():
    global running

    phases = [0, 0, 0]

    last_steps = [
        time.monotonic(),
        time.monotonic(),
        time.monotonic(),
    ]

    motor_active = [False, False, False]

    while running:

        now = time.monotonic()

        with state_lock:

            current_mode = mode

            if current_mode == "AUTO":

                # Watchdog
                if now - last_packet_time <= WATCHDOG_TIMEOUT:
                    commands = auto_commands.copy()
                else:
                    commands = [0, 0, 0]

            else:
                # Manual commands are directly stored here
                commands = motor_commands.copy()

        # ----------------------------------------------------
        # UPDATE EACH MOTOR
        # ----------------------------------------------------

        for i in range(3):

            raw_command = commands[i]

            # STOP
            if raw_command == 0:

                if motor_active[i]:
                    stop_motor(MOTORS[i])
                    motor_active[i] = False

                continue

            # ------------------------------------------------
            # SPEED
            # ------------------------------------------------

            if current_mode == "AUTO":
                speed_level = abs(raw_command)
                step_delay = SPEED_DELAYS[speed_level]

            else:
                # Manual always runs at medium speed
                step_delay = MANUAL_STEP_DELAY

            # ------------------------------------------------
            # DIRECTION
            # ------------------------------------------------

            direction = 1 if raw_command > 0 else -1

            # Fix reversed Motor 1
            direction *= MOTOR_DIRECTION[i]

            # ------------------------------------------------
            # STEP
            # ------------------------------------------------

            if now - last_steps[i] >= step_delay:

                phases[i] = (
                    phases[i] + direction
                ) % len(SEQUENCE)

                set_motor(
                    MOTORS[i],
                    SEQUENCE[phases[i]]
                )

                last_steps[i] = now
                motor_active[i] = True

        time.sleep(0.0005)


# ============================================================
# MANUAL COMMAND
# ============================================================

def manual_spin(motor_number, direction):
    global motor_commands

    index = motor_number - 1

    # Stop everything first
    with state_lock:
        motor_commands = [0, 0, 0]

    time.sleep(0.02)

    # Command selected motor
    command = [0, 0, 0]
    command[index] = direction

    with state_lock:
        motor_commands = command

    direction_name = (
        "FORWARD" if direction > 0 else "BACKWARD"
    )

    print(
        f"Motor {motor_number} "
        f"{direction_name} for {MANUAL_SPIN_TIME}s"
    )

    time.sleep(MANUAL_SPIN_TIME)

    # Stop after test
    with state_lock:
        motor_commands = [0, 0, 0]


# ============================================================
# START THREADS
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


# ============================================================
# TERMINAL INTERFACE
# ============================================================

print("""
========================================
       SPIRAL ROBOT CONTROLLER
========================================

Starting mode: MANUAL

MODE CONTROL
------------

m  = toggle MANUAL / AUTO
s  = stop all motors
q  = quit


MANUAL MODE
-----------

 1 = Motor 1 forward
 2 = Motor 2 forward
 3 = Motor 3 forward

-1 = Motor 1 backward
-2 = Motor 2 backward
-3 = Motor 3 backward


AUTO MODE
---------

Commands come from UDP:

-3 = fast backward
-2 = medium backward
-1 = slow backward
 0 = stop
 1 = slow forward
 2 = medium forward
 3 = fast forward

Motor 1 physical direction correction enabled.

========================================
""")


# ============================================================
# MAIN COMMAND LOOP
# ============================================================

try:

    while True:

        command = input(f"\n[{mode}] > ").strip().lower()

        # ----------------------------------------------------
        # QUIT
        # ----------------------------------------------------

        if command == "q":
            break

        # ----------------------------------------------------
        # STOP
        # ----------------------------------------------------

        elif command == "s":

            with state_lock:
                motor_commands = [0, 0, 0]
                auto_commands = [0, 0, 0]

            stop_all()

            print("ALL MOTORS STOPPED")

        # ----------------------------------------------------
        # TOGGLE MODE
        # ----------------------------------------------------

        elif command == "m":

            with state_lock:

                # Stop before changing mode
                motor_commands = [0, 0, 0]

                if mode == "MANUAL":
                    mode = "AUTO"
                else:
                    mode = "MANUAL"

            stop_all()

            print()
            print("========================")
            print(f" MODE -> {mode}")
            print("========================")

        # ----------------------------------------------------
        # MANUAL MOTOR CONTROL
        # ----------------------------------------------------

        elif mode == "MANUAL":

            if command in ("1", "2", "3"):

                manual_spin(
                    motor_number=int(command),
                    direction=1
                )

            elif command in ("-1", "-2", "-3"):

                manual_spin(
                    motor_number=abs(int(command)),
                    direction=-1
                )

            else:
                print(
                    "Manual commands: "
                    "1 2 3 -1 -2 -3"
                )

        # ----------------------------------------------------
        # AUTO MODE
        # ----------------------------------------------------

        else:

            print(
                "AUTO mode is controlled by UDP. "
                "Press m for MANUAL."
            )


except KeyboardInterrupt:

    print("\nCTRL+C")


# ============================================================
# CLEANUP
# ============================================================

finally:

    print("\nShutting down...")

    running = False

    with state_lock:
        motor_commands = [0, 0, 0]
        auto_commands = [0, 0, 0]

    stop_all()

    GPIO.cleanup()

    print("Done.")