import socket

PI_IP = "192.168.137.248"
PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

while True:
    message = input("Command: ")

    sock.sendto(
        message.encode("utf-8"),
        (PI_IP, PORT)
    )

    if message.lower() == "quit":
        break

sock.close()