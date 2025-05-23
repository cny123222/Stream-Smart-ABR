import socket
import os
import subprocess
import time

# --- Configuration ---
SERVER_HOST = '127.0.0.1'  # Server's IP address
SERVER_PORT = 8080
DOWNLOAD_DIR = "download"  # Temporary storage for segments
PLAYER_PATH = "ffplay"     # Path to ffplay or vlc (e.g., "vlc --play-and-exit") 
                           # On Windows, ffplay might be 'ffplay.exe'
                           # Ensure ffplay is in your PATH or provide the full path.
                           # ffplay arguments for auto-exit: -autoexit -nodisp (for no display if testing)
                           # For actual playing: ffplay -autoexit <file>
BUFFER_SIZE = 4096

def play_segment(segment_path):
    """Calls an external player to play the segment."""
    if not os.path.exists(segment_path):
        print(f"Error: Segment {segment_path} not found for playing.")
        return
    
    print(f"Playing segment: {segment_path}...")
    try:
        # For ffplay, -autoexit makes it close after playing.
        # You can add -nodisp if you don't want the video window during testing.
        if PLAYER_PATH.startswith("ffplay"):
            cmd = [PLAYER_PATH, '-autoexit', '-loglevel', 'quiet', segment_path]
        elif PLAYER_PATH.startswith("vlc"): # Example for VLC
             cmd = [PLAYER_PATH, '--play-and-exit', segment_path]
        else:
            cmd = [PLAYER_PATH, segment_path] # Generic, might not auto-exit

        subprocess.run(cmd, check=True)
        print(f"Finished playing {segment_path}.")
    except subprocess.CalledProcessError as e:
        print(f"Player error for {segment_path}: {e}")
    except FileNotFoundError:
        print(f"Error: Player '{PLAYER_PATH}' not found. Please install it or check the path.")
        print("Exiting. Please configure PLAYER_PATH correctly.")
        exit(1) # Exit if player is not found, as it's critical.


def start_streaming_session(client_socket, video_name, quality_suffix):
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    segment_index = 0

    while True:
        # Construct segment name based on convention from segment_video.py
        # e.g., your_long_video-1080p-8000k-000.ts
        segment_filename_on_server = f"{video_name}-{quality_suffix}-{segment_index:03d}.ts"
        # Client requests path relative to server's BASE_SEGMENTS_DIR
        request_path_on_server = f"{video_name}/{quality_suffix}/{segment_filename_on_server}"
        
        print(f"Requesting segment: {request_path_on_server}")
        client_socket.sendall(f"GET {request_path_on_server}\n".encode('utf-8'))

        response = client_socket.recv(1024).decode('utf-8').strip()

        if response == "OK":
            local_segment_path = os.path.join(DOWNLOAD_DIR, f"temp_segment_{segment_index:03d}.ts")
            print(f"Receiving segment data into {local_segment_path}...")
            
            try:
                with open(local_segment_path, 'wb') as f:
                    while True: # This loop needs a way to know when the file ends
                                # This simple implementation relies on server closing connection
                                # or sending a specific "end of file" marker if we enhance protocol.
                                # For now, if chunk is small, it might mean end of file if server sends in one go
                                # OR we can read until server stops sending or an explicit EOF marker.
                                # A common way is to get file size first. Let's assume for now small files or
                                # just keep reading until recv returns 0 bytes (which means server closed this segment stream)
                                # THIS PART IS CRITICAL AND NEEDS ROBUSTNESS for partial sends.
                                # A better way: server sends file size, client reads that many bytes.
                                # For now, trying to read until socket seems to deliver no more (can be tricky)
                                
                                # Let's refine this to read in chunks until no more data for this segment
                                # The server sends "OK\n" then the raw file data.
                                # The client needs to know when the file data ends.
                                # One simple approach is to try to read a large chunk. If it's less than
                                # buffer, assume it's the last. This is not robust.
                                # The server current implementation just sends all bytes and that's it.
                                # The client needs to handle this.
                                # A simple way for now: server sends all data for one file, then waits for next GET.
                                # Client reads until it can't for a short timeout or specific amount.
                                # For now, a fixed-size buffer read will be used, but this is an area for improvement.
                                # (e.g., server sends file size first)

                                data_received_this_segment = 0
                                while True: # Loop to receive one segment
                                    # This blocking recv will wait for data.
                                    # If server sends file and then stops (waiting for next GET),
                                    # this recv will eventually timeout or return 0 if connection closes.
                                    # This is a simplified receive loop.
                                    try:
                                        # Set a timeout to avoid blocking indefinitely if server doesn't send EOT
                                        client_socket.settimeout(2.0) # 2 seconds timeout for chunk
                                        chunk = client_socket.recv(BUFFER_SIZE)
                                        client_socket.settimeout(None) # Reset timeout
                                    except socket.timeout:
                                        # print("Socket timeout waiting for chunk, assuming end of segment.")
                                        break # Assume end of segment if timeout
                                    
                                    if not chunk:
                                        # print("No more data for this segment (or connection closed).")
                                        break # No more data for this segment or server closed
                                    f.write(chunk)
                                    data_received_this_segment += len(chunk)
                                    # A more robust way: server sends expected size, client reads until size met.
                                    # Or server sends an explicit "END_OF_SEGMENT" message.
                                    # For basic: if chunk < BUFFER_SIZE, assume it's the last chunk. This is NOT robust.
                                    # The current server sends all data then waits. The client will read until recv is empty.
                                    # The break conditions (timeout, no chunk) should handle this.
                                
                                if data_received_this_segment > 0:
                                    print(f"Received {data_received_this_segment} bytes for segment {segment_index}.")
                                    play_segment(local_segment_path)
                                else:
                                    print(f"No data received for segment {segment_index}. Stopping.")
                                    os.remove(local_segment_path) # Clean up empty file
                                    return # Stop if no data received

                                # Clean up segment after playing
                                if os.path.exists(local_segment_path):
                                    print(f"Cleaning up {local_segment_path}")
                                    os.remove(local_segment_path)
                                
                                segment_index += 1

            except Exception as e:
                print(f"Error during segment reception or playback: {e}")
                if os.path.exists(local_segment_path): # Clean up if error
                    os.remove(local_segment_path)
                break # Exit loop on error

        elif response.startswith("ERROR File not found"):
            print(f"Segment {request_path_on_server} not found on server. End of video or error.")
            break
        elif response.startswith("ERROR"):
            print(f"Server error: {response}")
            break
        else:
            print(f"Unknown server response: {response}")
            break
        
        time.sleep(0.1) # Small delay before requesting next segment

    print("Streaming session finished.")

def main():
    # --- Configuration for Client ---
    # For the basic version, let's assume we know the video name and one quality.
    # In a more advanced version, client might first ask server for available videos/qualities.
    VIDEO_TO_STREAM = "your_long_video" # Should match the name used in segmentation
    QUALITY_TO_STREAM = "1080p-8000k"   # Should match one of the segmented qualities

    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print(f"Connecting to server {SERVER_HOST}:{SERVER_PORT}...")
        client_socket.connect((SERVER_HOST, SERVER_PORT))
        print("Connected to server.")

        start_streaming_session(client_socket, VIDEO_TO_STREAM, QUALITY_TO_STREAM)

    except ConnectionRefusedError:
        print(f"Connection refused. Is the server running at {SERVER_HOST}:{SERVER_PORT}?")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if 'client_socket' in locals() and client_socket:
            print("Sending QUIT to server.")
            try:
                client_socket.sendall(b"QUIT\n")
            except: # Ignore errors if socket already closed
                pass
            client_socket.close()
            print("Connection closed.")

if __name__ == "__main__":
    main()