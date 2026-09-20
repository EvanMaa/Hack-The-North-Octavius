import socket

PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Listen on all Pi network interfaces
sock.bind(("0.0.0.0", PORT))

print(f"Listening for UDP packets on port {PORT}...")

while True:
    data, address = sock.recvfrom(1024)

    message = data.decode("utf-8").strip()

    print(f"Received from {address}: {message}")