import socket
import os
import subprocess
import time
import logging # Added for better client-side logging

# --- Configuration ---
SERVER_HOST = '127.0.0.1'  # Server's IP address
SERVER_PORT = 8080
DOWNLOAD_DIR = "download"  # Temporary storage for segments
PLAYER_PATH = "ffplay"     # Path to ffplay or vlc. Examples:
                           # "ffplay" (if in PATH)
                           # "/usr/bin/ffplay" (Linux)
                           # "/Applications/VLC.app/Contents/MacOS/VLC" --play-and-exit (macOS VLC example)
                           # "C:\\path\\to\\vlc\\vlc.exe" --play-and-exit (Windows VLC example)
BUFFER_SIZE = 4096 # For receiving file chunks

# --- Logger Setup ---
# Basic client-side logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()]) # Log to console
logger = logging.getLogger('StreamingClient')


def play_segment(segment_path):
    """Calls an external player to play the segment."""
    if not os.path.exists(segment_path):
        logger.error(f"Segment {segment_path} not found for playing.")
        return False # Indicate failure
    
    logger.info(f"Attempting to play segment: {segment_path}...")
    try:
        if "ffplay" in PLAYER_PATH.lower():
            # -autoexit: ffplay closes when playback finishes
            # -nodisp: (optional) disable video display, useful for testing without GUI
            # -loglevel quiet: suppress ffplay's own console output unless error
            cmd = [PLAYER_PATH, '-autoexit', '-loglevel', 'error', segment_path]
        elif "vlc" in PLAYER_PATH.lower():
             cmd = [PLAYER_PATH, '--play-and-exit', segment_path] # VLC specific
        else: # Generic attempt, might not auto-exit or be quiet
            cmd = [PLAYER_PATH, segment_path]

        process = subprocess.Popen(cmd) # Use Popen for non-blocking if needed, or run for blocking
        process.wait() # Wait for the player to finish
        
        if process.returncode == 0:
            logger.info(f"Finished playing {segment_path} successfully.")
            return True
        else:
            logger.error(f"Player for {segment_path} exited with code {process.returncode}.")
            return False

    except subprocess.CalledProcessError as e: # .run(check=True) raises this
        logger.error(f"Player error for {segment_path}: {e}")
    except FileNotFoundError:
        logger.error(f"Player executable '{PLAYER_PATH}' not found. Please install it or check PLAYER_PATH.")
        logger.error("Exiting due to player misconfiguration.")
        exit(1) # Critical error, exit
    except Exception as e:
        logger.error(f"An unexpected error occurred during playback of {segment_path}: {e}")
    return False # Indicate failure


def receive_exact_bytes(sock, num_bytes):
    """Receives exactly num_bytes from the socket."""
    data = b''
    while len(data) < num_bytes:
        remaining_bytes = num_bytes - len(data)
        chunk = sock.recv(min(BUFFER_SIZE, remaining_bytes))
        if not chunk:
            raise ConnectionError("Socket connection broken while receiving data.")
        data += chunk
    return data

def start_streaming_session(client_socket, video_name, quality_suffix):
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    segment_index = 0

    while True:
        # Construct segment name based on convention from segment_video.py
        # e.g., bbb_sunflower_1080p_30fps_normal-1080p-8000k-000.ts
        segment_filename_on_server = f"{video_name}-{quality_suffix}-{segment_index:03d}.ts"
        # Client requests path relative to server's BASE_SEGMENTS_DIR
        # e.g., "bbb_sunflower_1080p_30fps_normal/1080p-8000k/bbb_sunflower_1080p_30fps_normal-1080p-8000k-000.ts"
        request_path_on_server = f"{video_name}/{quality_suffix}/{segment_filename_on_server}"
        
        logger.info(f"Requesting segment: {request_path_on_server}")
        request_message = f"GET {request_path_on_server}\n"
        try:
            client_socket.sendall(request_message.encode('utf-8'))

            # Receive the header line (e.g., "OK 12345\n" or "ERROR ...\n")
            # Read up to a reasonable header size, look for newline
            header_data = b""
            while not header_data.endswith(b"\n"):
                byte = client_socket.recv(1)
                if not byte:
                    raise ConnectionError("Connection closed by server while waiting for header.")
                header_data += byte
            
            header_str = header_data.decode('utf-8').strip()
            logger.info(f"Server response header: {header_str}")

            if header_str.startswith("OK "):
                try:
                    parts = header_str.split(" ", 1)
                    expected_size = int(parts[1])
                except (IndexError, ValueError) as e:
                    logger.error(f"Invalid OK response format from server: {header_str} - Error: {e}")
                    break # Stop trying to stream

                local_segment_filename = f"temp_{video_name}_{quality_suffix}_{segment_index:03d}.ts"
                local_segment_path = os.path.join(DOWNLOAD_DIR, local_segment_filename)
                
                logger.info(f"Receiving segment data ({expected_size} bytes) into {local_segment_path}...")
                
                try:
                    segment_data = receive_exact_bytes(client_socket, expected_size)
                    with open(local_segment_path, 'wb') as f:
                        f.write(segment_data)
                    
                    logger.info(f"Received {len(segment_data)} bytes for segment {segment_index}.")

                    if len(segment_data) == expected_size:
                        play_segment(local_segment_path)
                    else:
                        logger.error(f"File size mismatch for {local_segment_path}. Expected {expected_size}, got {len(segment_data)}.")
                
                finally: # Ensure cleanup even if play_segment fails or other errors
                    if os.path.exists(local_segment_path):
                        logger.info(f"Cleaning up {local_segment_path}")
                        try:
                            os.remove(local_segment_path)
                        except OSError as e:
                            logger.error(f"Error deleting segment {local_segment_path}: {e}")
                
                segment_index += 1

            elif header_str.startswith("ERROR 404"):
                logger.warning(f"Segment {request_path_on_server} not found on server (404). End of video or invalid request.")
                break # Stop trying to stream
            elif header_str.startswith("ERROR"):
                logger.error(f"Server returned an error: {header_str}. Stopping stream.")
                break # Stop trying to stream
            else:
                logger.error(f"Unknown server response: {header_str}. Stopping stream.")
                break # Stop trying to stream
        
        except ConnectionError as e:
            logger.error(f"Connection error: {e}. Stopping stream.")
            break
        except Exception as e:
            logger.error(f"An unexpected error occurred in streaming loop: {e}")
            break
        
        time.sleep(0.1) # Small delay before requesting next segment

    logger.info("Streaming session finished.")

def main():
    # --- Configuration for Client ---
    # Update VIDEO_TO_STREAM to match the folder name on the server (derived from original video filename)
    VIDEO_TO_STREAM = "bbb_sunflower_1080p_30fps_normal" 
    QUALITY_TO_STREAM = "1080p-8000k"   # Choose one of the available qualities

    client_socket = None # Initialize to None
    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        logger.info(f"Connecting to server {SERVER_HOST}:{SERVER_PORT}...")
        client_socket.connect((SERVER_HOST, SERVER_PORT))
        logger.info("Connected to server.")

        start_streaming_session(client_socket, VIDEO_TO_STREAM, QUALITY_TO_STREAM)

    except socket.timeout:
        logger.error(f"Connection to server {SERVER_HOST}:{SERVER_PORT} timed out.")
    except ConnectionRefusedError:
        logger.error(f"Connection refused. Is the server running at {SERVER_HOST}:{SERVER_PORT}?")
    except Exception as e:
        logger.error(f"An error occurred in main: {e}")
    finally:
        if client_socket:
            logger.info("Sending QUIT to server (if possible) and closing connection.")
            try:
                # Try to send QUIT, but don't fail hard if socket is already broken
                client_socket.sendall(b"QUIT\n")
            except OSError: # Catch errors like "socket is already closed" or "broken pipe"
                logger.warning("Could not send QUIT command; socket may already be closed.")
            except Exception as e_quit:
                logger.warning(f"Error sending QUIT command: {e_quit}")
            finally:
                client_socket.close()
                logger.info("Client socket closed.")

if __name__ == "__main__":
    main()