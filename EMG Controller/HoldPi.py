import socket
import threading
from pynput import keyboard


HOST = "192.168.137.238"
PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

held_keys = set()
lock = threading.Lock()


KEY_VALUES = {
    "1": (3, -1, -1),
    "2": (-1, 3, -1),
    "3": (-1, -1, 3),

    "q": (-3, 0, 0),
    "w": (0, -3, 0),
    "e": (0, 0, -3),
}


def send_loop():
    while True:
        with lock:
            x = 0
            y = 0
            z = 0

            for key in held_keys:
                dx, dy, dz = KEY_VALUES[key]

                x += dx
                y += dy
                z += dz

            value = (x, y, z)

        print(value)
        packet = ",".join(str(x) for x in value)
        sock.sendto(
            packet.encode(),
            (HOST, PORT)
        )

        threading.Event().wait(0.05)  # 20 messages/second


def on_press(key):
    try:
        key = key.char.lower()
    except AttributeError:
        return

    if key in KEY_VALUES:
        with lock:
            held_keys.add(key)


def on_release(key):
    try:
        key = key.char.lower()
    except AttributeError:
        return

    if key in KEY_VALUES:
        with lock:
            held_keys.discard(key)


# Start the continuous sender
threading.Thread(
    target=send_loop,
    daemon=True
).start()


# Listen for keyboard events
with keyboard.Listener(
    on_press=on_press,
    on_release=on_release
) as listener:
    listener.join()
