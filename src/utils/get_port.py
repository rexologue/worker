import socket
import random

def get_safe_free_port() -> int:
    for _ in range(100): 
        port = random.randint(49152, 65535)

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('127.0.0.1', port))
                return port
            
            except OSError:
                continue

    raise RuntimeError("Failed to find a free port in the range 49152–65535")
