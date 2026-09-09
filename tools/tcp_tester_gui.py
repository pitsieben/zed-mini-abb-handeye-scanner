import socket
import threading
import tkinter as tk
from tkinter import ttk

class TCPTesterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Robot TCP Listener Test")
        self.root.geometry("400x350")
        
        # UI Elements
        self.status_var = tk.StringVar(value="Status: Waiting to start server...")
        ttk.Label(root, textvariable=self.status_var, font=("Arial", 12, "bold")).pack(pady=10)
        
        self.port_var = tk.StringVar(value="5000")
        frame_port = ttk.Frame(root)
        frame_port.pack(pady=5)
        ttk.Label(frame_port, text="Port: ").pack(side=tk.LEFT)
        ttk.Entry(frame_port, textvariable=self.port_var, width=10).pack(side=tk.LEFT)
        
        self.start_btn = ttk.Button(root, text="Start Server", command=self.start_server)
        self.start_btn.pack(pady=10)
        
        # Data Display
        frame_data = ttk.LabelFrame(root, text="Latest Received Pose", padding=10)
        frame_data.pack(pady=10, fill="x", padx=20)
        
        self.pos_var = tk.StringVar(value="X: 0.0, Y: 0.0, Z: 0.0")
        ttk.Label(frame_data, textvariable=self.pos_var, font=("Courier", 10)).pack(anchor="w")
        
        self.quat_var = tk.StringVar(value="Q1: 0.0, Q2: 0.0, Q3: 0.0, Q4: 0.0")
        ttk.Label(frame_data, textvariable=self.quat_var, font=("Courier", 10)).pack(anchor="w")
        
        self.log_text = tk.Text(root, height=5, width=40, font=("Courier", 8))
        self.log_text.pack(pady=10, padx=20)
        
        self.server_thread = None
        self.running = False
        
    def log(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        
    def start_server(self):
        if self.running:
            return
            
        port = int(self.port_var.get())
        self.running = True
        self.start_btn.config(state="disabled")
        
        self.server_thread = threading.Thread(target=self.run_tcp_server, args=(port,), daemon=True)
        self.server_thread.start()
        
    def run_tcp_server(self, port):
        host = '0.0.0.0'
        
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((host, port))
                s.listen()
                
                self.root.after(0, self.status_var.set, f"Status: Listening on port {port}...")
                self.root.after(0, self.log, f"Server started on {host}:{port}")
                
                # Accept connection
                conn, addr = s.accept()
                with conn:
                    self.root.after(0, self.status_var.set, f"Status: Connected to {addr}")
                    self.root.after(0, self.log, f"Client connected: {addr}")
                    
                    while self.running:
                        data = conn.recv(1024)
                        if not data:
                            self.root.after(0, self.status_var.set, "Status: Client disconnected.")
                            break
                            
                        msg = data.decode('utf-8').strip()
                        self.root.after(0, self.log, f"Received: {msg}")
                        
                        if msg == "DONE":
                            self.root.after(0, self.status_var.set, "Status: Calibration sequence DONE.")
                            break
                            
                        # Parse and display data
                        parts = msg.split(',')
                        if len(parts) == 7:
                            tx, ty, tz, q1, q2, q3, q4 = parts
                            self.root.after(0, self.pos_var.set, f"X: {tx:>8}, Y: {ty:>8}, Z: {tz:>8}")
                            self.root.after(0, self.quat_var.set, f"Q1: {q1:>6}, Q2: {q2:>6}, Q3: {q3:>6}, Q4: {q4:>6}")
                            
                        # Send ACK
                        conn.sendall(b"ACK")
                        self.root.after(0, self.log, "Sent: ACK")
                        
        except Exception as e:
            self.root.after(0, self.status_var.set, f"Status: Error - {str(e)}")
            self.root.after(0, self.log, f"Error: {str(e)}")
            
        finally:
            self.running = False
            self.root.after(0, self.start_btn.config, {"state": "normal"})

if __name__ == "__main__":
    root = tk.Tk()
    app = TCPTesterApp(root)
    root.mainloop()
