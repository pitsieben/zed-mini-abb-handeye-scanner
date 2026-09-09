import socket
import sys
import time

HOST = '0.0.0.0'  # Listen on all available network adapters
PORT = 5000       # Port number matching RAPID code

def main():
    print("=" * 60)
    print("       ABB ROBOT <-> PC TCP/IP CONNECTION TESTER")
    print("=" * 60)
    print(f"[*] Starting TCP Server on all interfaces (Port {PORT})...")
    
    # Try to display local IP addresses for convenience
    try:
        hostname = socket.gethostname()
        local_ips = socket.gethostbyname_ex(hostname)[2]
        print(f"[*] Available PC IP addresses to enter in RAPID:")
        for ip in local_ips:
            if not ip.startswith("127."):
                print(f"    -> {ip}")
    except Exception:
        pass
        
    print(f"\n[*] Listening... Run the RAPID program on your ABB controller now.")
    print("=" * 60)

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Allow socket to be reused immediately if restarted
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    try:
        server_socket.bind((HOST, PORT))
        server_socket.listen(1)
    except Exception as e:
        print(f"\n[!] Failed to bind to port {PORT}: {e}")
        print("    Make sure no other test script or dashboard is currently running.")
        sys.exit(1)

    try:
        conn, addr = server_socket.accept()
        print(f"\n[+] CONNECTION ESTABLISHED!")
        print(f"[+] Robot connected from IP: {addr[0]} (Port: {addr[1]})")
        print("-" * 60)

        packet_count = 0
        with conn:
            while True:
                data = conn.recv(1024)
                if not data:
                    print("\n[-] Robot closed the connection.")
                    break
                
                packet_count += 1
                msg = data.decode('utf-8', errors='ignore').strip()
                
                print(f"\n[Packet #{packet_count}] Received: {msg}")
                
                # Check if it's position coordinates (7 comma-separated numbers)
                parts = msg.split(',')
                if len(parts) == 7:
                    try:
                        x, y, z, q1, q2, q3, q4 = [float(p.strip()) for p in parts]
                        print(f"    -> Parsed Position: X={x:.2f} mm, Y={y:.2f} mm, Z={z:.2f} mm")
                        print(f"    -> Parsed Quat:     Q1={q1:.4f}, Q2={q2:.4f}, Q3={q3:.4f}, Q4={q4:.4f}")
                    except ValueError:
                        pass
                elif msg == "DONE":
                    print("    -> [Signal] Robot indicated test sequence is DONE.")
                
                # Send Acknowledgment back to robot
                ack_reply = f"ACK_{packet_count}"
                conn.sendall(ack_reply.encode('utf-8'))
                print(f"    -> Sent response to Robot: '{ack_reply}'")

    except KeyboardInterrupt:
        print("\n[*] Stopped by user.")
    finally:
        server_socket.close()
        print("\n[*] Socket closed. Test completed.")

if __name__ == "__main__":
    main()
