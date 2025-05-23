import socket
import threading
import os
import time
import logging

# --- Configuration ---
HOST = '0.0.0.0'  # Listen on all available network interfaces
PORT = 8080
BASE_SEGMENTS_DIR = "video_segments"  # Directory where segmented TS files are stored
LOG_FILE = "transmission_log.txt"
BUFFER_SIZE = 4096 # For sending file chunks

# --- Logger Setup ---
logger = logging.getLogger('StreamingServer')
logger.setLevel(logging.INFO)
# File handler
fh = logging.FileHandler(LOG_FILE)
fh.setLevel(logging.INFO)
# Console handler (optional)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# Formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)


def get_bitrate_from_filename(filename):
    """Extracts bitrate like '8000k' from 'ocean-1080p-8000k-0.ts'"""
    try:
        parts = filename.split('-')
        for part in parts:
            if part.endswith('k'):
                return part
    except Exception:
        return "unknown_bitrate"
    return "unknown_bitrate"

def handle_client(client_socket, client_address):
    logger.info(f"[NEW CONNECTION] {client_address} connected.")
    try:
        while True:
            # Client request format: "GET video_name/quality_suffix/segment_name.ts"
            # e.g., "GET your_long_video/1080p-8000k/your_long_video-1080p-8000k-000.ts"
            request_data = client_socket.recv(1024).decode('utf-8')
            if not request_data:
                logger.info(f"[CONNECTION CLOSED] {client_address} disconnected gracefully.")
                break

            logger.info(f"[{client_address}] Received request: {request_data}")

            if request_data.startswith("GET "):
                parts = request_data.split(" ")
                if len(parts) < 2:
                    client_socket.sendall(b"ERROR Invalid request format\n")
                    logger.error(f"[{client_address}] Invalid request: {request_data}")
                    continue
                
                requested_path = parts[1].strip() # e.g., your_long_video/1080p-8000k/your_long_video-1080p-8000k-000.ts
                segment_filename = os.path.basename(requested_path) # e.g., your_long_video-1080p-8000k-000.ts
                
                # Construct full path to the segment file
                # The client sends the relative path starting from BASE_SEGMENTS_DIR
                # e.g. "your_long_video/1080p-8000k/your_long_video-1080p-8000k-000.ts"
                full_segment_path = os.path.join(BASE_SEGMENTS_DIR, requested_path)
                
                if os.path.exists(full_segment_path):
                    try:
                        file_size = os.path.getsize(full_segment_path)
                        # Send file size first (optional, but good for client to know)
                        # client_socket.sendall(f"SIZE {file_size}\n".encode('utf-8'))
                        # For simplicity in basic version, let's just send OK then data
                        client_socket.sendall(b"OK\n") # Acknowledge that file exists and will be sent

                        send_start_time = time.time()
                        with open(full_segment_path, 'rb') as f:
                            while True:
                                chunk = f.read(BUFFER_SIZE)
                                if not chunk:
                                    break
                                client_socket.sendall(chunk)
                        send_end_time = time.time()
                        
                        bitrate = get_bitrate_from_filename(segment_filename)
                        logger.info(
                            f"[{client_address}] SENT {segment_filename} ({file_size} bytes) "
                            f"Bitrate: {bitrate}, SendTime: {send_end_time - send_start_time:.4f}s"
                        )
                    except Exception as e:
                        logger.error(f"[{client_address}] Error sending file {segment_filename}: {e}")
                        try:
                            client_socket.sendall(b"ERROR Server error during file send\n")
                        except:
                            pass # Client might have already disconnected
                        break 
                else:
                    logger.warning(f"[{client_address}] File not found: {full_segment_path}")
                    client_socket.sendall(b"ERROR File not found\n")
            elif request_data.strip().upper() == "QUIT":
                 logger.info(f"[{client_address}] Received QUIT command. Closing connection.")
                 break
            else:
                logger.warning(f"[{client_address}] Unknown command: {request_data}")
                client_socket.sendall(b"ERROR Unknown command\n")

    except ConnectionResetError:
        logger.warning(f"[CONNECTION RESET] {client_address} connection was reset.")
    except Exception as e:
        logger.error(f"[UNEXPECTED ERROR] Client {client_address}: {e}")
    finally:
        client_socket.close()
        logger.info(f"[CONNECTION CLOSED] {client_address} connection finalized.")


def start_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) # Reuse address
    try:
        server_socket.bind((HOST, PORT))
        server_socket.listen(5) # Max 5 queued connections
        logger.info(f"Server listening on {HOST}:{PORT}")
        logger.info(f"Serving segments from: {os.path.abspath(BASE_SEGMENTS_DIR)}")

        while True:
            client_socket, client_address = server_socket.accept()
            thread = threading.Thread(target=handle_client, args=(client_socket, client_address))
            thread.daemon = True # Allows main program to exit even if threads are running
            thread.start()
    except OSError as e:
        logger.error(f"Server failed to start: {e}")
    except KeyboardInterrupt:
        logger.info("Server is shutting down...")
    finally:
        server_socket.close()
        logger.info("Server shut down successfully.")

if __name__ == "__main__":
    if not os.path.exists(BASE_SEGMENTS_DIR):
        logger.error(f"Base segments directory '{BASE_SEGMENTS_DIR}' not found. Please create it and segment videos first.")
    else:
        start_server()