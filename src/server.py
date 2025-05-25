import socket
import threading
import os
import time
import logging

# --- Configuration ---
HOST = '0.0.0.0'  # Listen on all available network interfaces
PORT = 8080
# Make sure this points to the directory structure like:
# BASE_SEGMENTS_DIR/
# └── your_video_name/
#     ├── 1080p-8000k/
#     │   ├── segment-000.ts
#     │   └── ...
#     └── 720p-4000k/
#         ├── segment-000.ts
#         └── ...
BASE_SEGMENTS_DIR = "video_segments"  # 修改为你实际存储分片的根目录
LOG_FILE = "transmission_log.txt"
BUFFER_SIZE = 4096 # For sending file chunks

# --- Logger Setup ---
logger = logging.getLogger('StreamingServer')
logger.setLevel(logging.INFO)
# File handler
fh = logging.FileHandler(LOG_FILE, encoding='utf-8') # Specify encoding for log file
fh.setLevel(logging.INFO)
# Console handler (optional)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# Formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(client_ip)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)

# Adapter to add client_ip to log records if available
class ClientIPLogAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        if 'extra' not in kwargs:
            kwargs['extra'] = {}
        kwargs['extra'].update(self.extra)
        return msg, kwargs

def get_bitrate_from_filename(filename):
    """Extracts bitrate like '8000k' from 'video_name-1080p-8000k-000.ts'"""
    try:
        parts = filename.split('-')
        # Iterate backwards to find the bitrate part more reliably if video_name also has hyphens
        for part in reversed(parts):
            if part.endswith('k') and part[:-1].isdigit(): # Check if it's like '8000k'
                return part
    except Exception:
        return "unknown_bitrate"
    return "unknown_bitrate"

def handle_client(client_socket, client_address_tuple):
    client_ip_port = f"{client_address_tuple[0]}:{client_address_tuple[1]}"
    # Use adapter for logging with client IP
    adapter = ClientIPLogAdapter(logger, {'client_ip': client_ip_port})
    
    adapter.info(f"New connection established.")
    try:
        while True:
            request_data = client_socket.recv(1024).decode('utf-8')
            if not request_data:
                adapter.info(f"Connection closed gracefully by client.")
                break

            adapter.info(f"Received request: {request_data.strip()}")

            if request_data.startswith("GET "):
                parts = request_data.split(" ", 1) # Split only on the first space
                if len(parts) < 2:
                    client_socket.sendall(b"ERROR 400 Invalid request format\n")
                    adapter.error(f"Invalid request format: {request_data.strip()}")
                    continue
                
                # requested_path is like "video_name/quality_suffix/segment_file.ts"
                # e.g., "bbb_sunflower_1080p_30fps_normal/1080p-8000k/bbb_sunflower_1080p_30fps_normal-1080p-8000k-000.ts"
                requested_path = parts[1].strip() 
                
                # Security: Prevent directory traversal attacks
                if ".." in requested_path:
                    client_socket.sendall(b"ERROR 400 Invalid path\n")
                    adapter.warning(f"Directory traversal attempt blocked for path: {requested_path}")
                    continue

                segment_filename = os.path.basename(requested_path)
                full_segment_path = os.path.join(BASE_SEGMENTS_DIR, requested_path)
                # Normalize path to prevent issues with mixed slashes or redundant parts
                full_segment_path = os.path.normpath(full_segment_path)

                # Ensure the path is still within the base directory after normalization
                if not full_segment_path.startswith(os.path.normpath(BASE_SEGMENTS_DIR)):
                    client_socket.sendall(b"ERROR 400 Invalid path (outside base dir)\n")
                    adapter.warning(f"Path traversal attempt (normalized) blocked for path: {full_segment_path}")
                    continue
                
                if os.path.exists(full_segment_path) and os.path.isfile(full_segment_path):
                    try:
                        file_size = os.path.getsize(full_segment_path)
                        
                        # Send OK, file size, then a newline.
                        # Example: "OK 12345\n"
                        response_header = f"OK {file_size}\n"
                        client_socket.sendall(response_header.encode('utf-8'))
                        adapter.info(f"Sending header for {segment_filename}: {response_header.strip()}")

                        send_start_time = time.time()
                        sent_bytes = 0
                        with open(full_segment_path, 'rb') as f:
                            while True:
                                chunk = f.read(BUFFER_SIZE)
                                if not chunk:
                                    break
                                client_socket.sendall(chunk)
                                sent_bytes += len(chunk)
                        send_end_time = time.time()
                        
                        if sent_bytes != file_size:
                             adapter.warning(f"File size mismatch for {segment_filename}. Expected {file_size}, sent {sent_bytes}")

                        bitrate = get_bitrate_from_filename(segment_filename)
                        send_duration = send_end_time - send_start_time
                        adapter.info(
                            f"SENT {segment_filename} ({file_size} bytes) "
                            f"Bitrate(file): {bitrate}, SendTime: {send_duration:.4f}s"
                        )
                    except ConnectionError as e: # More specific for network issues during send
                        adapter.error(f"Connection error while sending file {segment_filename}: {e}")
                        break # Break from while True loop for this client
                    except Exception as e:
                        adapter.error(f"Error sending file {segment_filename}: {e}")
                        try:
                            # Try to inform client, but socket might be broken
                            client_socket.sendall(b"ERROR 500 Server error during file send\n")
                        except:
                            pass 
                        break # Break from while True loop for this client
                else:
                    adapter.warning(f"File not found: {full_segment_path} (requested: {requested_path})")
                    client_socket.sendall(b"ERROR 404 File not found\n")
            
            elif request_data.strip().upper() == "QUIT":
                 adapter.info(f"Received QUIT command. Closing connection.")
                 break
            else:
                adapter.warning(f"Unknown command received: {request_data.strip()}")
                client_socket.sendall(b"ERROR 400 Unknown command\n")

    except ConnectionResetError:
        adapter.warning(f"Connection reset by peer.")
    except UnicodeDecodeError:
        adapter.warning(f"Received non-UTF-8 data, possibly binary. Closing connection.")
    except Exception as e:
        adapter.error(f"Unexpected error in client handler: {e}")
    finally:
        client_socket.close()
        adapter.info(f"Connection finalized.")


def start_server():
    # Use the adapter for server-level logs too, with a default 'client_ip'
    server_adapter = ClientIPLogAdapter(logger, {'client_ip': 'SERVER_MAIN'})
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) 
    try:
        server_socket.bind((HOST, PORT))
        server_socket.listen(10) # Increased backlog slightly
        server_adapter.info(f"Server listening on {HOST}:{PORT}")
        server_adapter.info(f"Serving segments from: {os.path.abspath(BASE_SEGMENTS_DIR)}")

        while True:
            try:
                client_socket, client_address_tuple = server_socket.accept()
                thread = threading.Thread(target=handle_client, args=(client_socket, client_address_tuple))
                thread.daemon = True 
                thread.start()
            except Exception as e: # Catch errors in accept loop if any
                server_adapter.error(f"Error accepting connection: {e}")
                time.sleep(0.1) # Avoid busy-looping on repeated accept errors

    except OSError as e:
        server_adapter.error(f"Server failed to start: {e} (Hint: Is port {PORT} already in use?)")
    except KeyboardInterrupt:
        server_adapter.info("Server is shutting down due to KeyboardInterrupt...")
    finally:
        server_socket.close()
        server_adapter.info("Server shut down successfully.")

if __name__ == "__main__":
    # Use the adapter for initial checks too
    main_adapter = ClientIPLogAdapter(logger, {'client_ip': 'SERVER_INIT'})
    if not os.path.exists(BASE_SEGMENTS_DIR) or not os.path.isdir(BASE_SEGMENTS_DIR):
        main_adapter.error(f"Base segments directory '{BASE_SEGMENTS_DIR}' not found or is not a directory.")
        main_adapter.error("Please create it and ensure it contains the segmented video subdirectories.")
    else:
        start_server()