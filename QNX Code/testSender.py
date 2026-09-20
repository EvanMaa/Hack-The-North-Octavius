import socket
import time

PI_IP = "192.168.137.248"
PORT = 5005

sock = socket.socket(
    socket.AF_INET,
    socket.SOCK_DGRAM
)

command = (1, 0, -1)

while True:

    message = ",".join(
        str(x) for x in command
    )

    sock.sendto(
        message.encode(),
        (PI_IP, PORT)
    )

    print("Sent:", command)

    # 10 Hz
    time.sleep(1)
