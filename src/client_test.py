import socket
import os
import time
import logging
import vlc # 导入 python-vlc 库
import threading # 用于在单独线程中处理播放器事件，避免阻塞主下载循环

# --- Configuration ---
SERVER_HOST = '127.0.0.1'
SERVER_PORT = 8080
DOWNLOAD_DIR = "download"
BUFFER_SIZE = 4096

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger('StreamingClientVLC')

# --- VLC Player Setup ---
# 全局变量来持有VLC实例和播放器相关对象
vlc_instance = None
media_list_player = None
media_list = None
media_player = None # MediaPlayer instance used by MediaListPlayer
played_segments_to_delete = [] # List to keep track of segments that can be deleted
player_event_manager = None
keep_streaming = True # Flag to control streaming loop

def initialize_player():
    global vlc_instance, media_list_player, media_list, media_player, player_event_manager
    if vlc_instance is None: # Ensure only one instance
        vlc_instance = vlc.Instance()
        media_list = vlc_instance.media_list_new()
        media_list_player = vlc_instance.media_list_player_new()
        media_player = vlc_instance.media_player_new() # Create a media player instance
        media_list_player.set_media_player(media_player) # Assign it to the list player
        media_list_player.set_media_list(media_list)
        
        # Setup event handling for media changes (to delete played files)
        player_event_manager = media_player.event_manager()
        player_event_manager.event_attach(vlc.EventType.MediaPlayerMediaChanged, on_media_changed_callback)
        player_event_manager.event_attach(vlc.EventType.MediaPlayerEndReached, on_end_reached_callback)
        logger.info("VLC Player initialized.")

# --- Event Callbacks for VLC ---
current_playing_media_path = None # Track what's currently set to play

def on_media_changed_callback(event):
    global current_playing_media_path, played_segments_to_delete
    # This event fires when the media player starts playing a new item.
    # The *previously* playing item (if any) can now be considered for deletion.
    
    # The event itself doesn't directly give the path of the *previous* media easily in all contexts.
    # We'll manage deletion based on a queue or by identifying the new media.
    
    new_media = media_player.get_media()
    if new_media:
        new_media_path = new_media.get_mrl()
        logger.info(f"MediaPlayerMediaChanged: Now playing {new_media_path}")
        
        # If there was a previously playing segment, add it to the delete queue
        # This logic needs to be robust. A simpler way might be to delete based on
        # the order in the media_list once we know an item has *finished*.
        # For now, let's use a simpler approach: delete segments once they are no longer
        # in the immediate upcoming part of the playlist or a buffer.
        # For this example, we will delete after an item is *popped* by the player.
        # A robust way is needed to identify *which* segment just finished.
        # 'MediaPlayerMediaChanged' indicates a *new* media started.
        # So, the 'current_playing_media_path' *before* this change was the one that finished.
        if current_playing_media_path and os.path.exists(current_playing_media_path):
            logger.info(f"Adding to delete queue (was playing): {current_playing_media_path}")
            # Schedule for deletion (actual deletion might happen slightly later to avoid race conditions)
            # For simplicity here, let's just delete it. In a more robust system, use a queue.
            if current_playing_media_path.startswith(os.path.join(DOWNLOAD_DIR, "temp_")):
                try:
                    os.remove(current_playing_media_path)
                    logger.info(f"Cleaned up (after media change): {current_playing_media_path}")
                except OSError as e:
                    logger.error(f"Error deleting segment (on media change) {current_playing_media_path}: {e}")
            
        current_playing_media_path = new_media_path
    else:
        logger.info("MediaPlayerMediaChanged: New media is None (possibly end of list).")


def on_end_reached_callback(event):
    global keep_streaming
    logger.info("MediaPlayerEndReached: Playlist finished or player stopped.")
    # We might want to stop fetching new segments if the playlist ends and no new ones are being added.
    # Or, if we expect more segments, this might be a signal for ABR or re-buffering.
    # For now, this could indicate the end if no more segments are being queued.
    # keep_streaming = False # Optionally stop fetching if playlist truly ends


def add_segment_to_playlist(segment_path):
    global media_list, media_list_player, vlc_instance
    if not os.path.exists(segment_path):
        logger.error(f"Segment {segment_path} not found for adding to playlist.")
        return

    media = vlc_instance.media_new(segment_path)
    media_list.add_media(media)
    logger.info(f"Added to playlist: {segment_path}")

    if media_list_player.get_state() != vlc.State.Playing:
        logger.info("Playlist not playing. Starting playback...")
        media_list_player.play()
        # Small delay to allow playback to start
        time.sleep(0.5) 
        if media_list_player.get_state() != vlc.State.Playing:
            logger.warning("Failed to start playback for the playlist.")


def receive_exact_bytes(sock, num_bytes):
    """Receives exactly num_bytes from the socket."""
    data = b''
    while len(data) < num_bytes:
        remaining_bytes = num_bytes - len(data)
        # Set a timeout for individual recv calls to prevent indefinite blocking
        sock.settimeout(5.0) # 5 seconds timeout for a chunk
        try:
            chunk = sock.recv(min(BUFFER_SIZE, remaining_bytes))
        except socket.timeout:
            logger.error("Socket timeout while receiving segment data.")
            raise ConnectionError("Socket timeout during data reception.")
        finally:
            sock.settimeout(None) # Reset timeout

        if not chunk:
            raise ConnectionError("Socket connection broken while receiving data.")
        data += chunk
    return data

def start_streaming_session(client_socket, video_name, quality_suffix):
    global keep_streaming, current_playing_media_path
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    segment_index = 0
    keep_streaming = True # Reset flag for new session

    # Buffer a few segments ahead on disk before starting playback or for smoother transitions
    # For this example, we add to playlist immediately.
    # A more advanced version would manage a disk buffer size.

    # Initialize player if not already done
    initialize_player()
    current_playing_media_path = None # Reset

    while keep_streaming:
        segment_filename_on_server = f"{video_name}-{quality_suffix}-{segment_index:03d}.ts"
        request_path_on_server = f"{video_name}/{quality_suffix}/{segment_filename_on_server}"
        
        logger.info(f"Requesting segment: {request_path_on_server}")
        request_message = f"GET {request_path_on_server}\n"
        try:
            client_socket.sendall(request_message.encode('utf-8'))

            header_data = b""
            # Set a timeout for reading the header
            client_socket.settimeout(10.0) # 10 seconds for header
            try:
                while not header_data.endswith(b"\n"):
                    byte = client_socket.recv(1)
                    if not byte:
                        raise ConnectionError("Connection closed by server while waiting for header.")
                    header_data += byte
            except socket.timeout:
                logger.error("Timeout waiting for server response header.")
                keep_streaming = False
                break
            finally:
                client_socket.settimeout(None) # Reset timeout

            header_str = header_data.decode('utf-8').strip()
            logger.info(f"Server response header: {header_str}")

            if header_str.startswith("OK "):
                try:
                    parts = header_str.split(" ", 1)
                    expected_size = int(parts[1])
                except (IndexError, ValueError) as e:
                    logger.error(f"Invalid OK response format from server: {header_str} - Error: {e}")
                    keep_streaming = False; break

                local_segment_filename = f"temp_{video_name}_{quality_suffix}_{segment_index:03d}.ts"
                local_segment_path = os.path.join(DOWNLOAD_DIR, local_segment_filename)
                
                logger.info(f"Receiving segment data ({expected_size} bytes) into {local_segment_path}...")
                
                segment_data = receive_exact_bytes(client_socket, expected_size) # Can raise ConnectionError
                with open(local_segment_path, 'wb') as f:
                    f.write(segment_data)
                
                logger.info(f"Received {len(segment_data)} bytes for segment {segment_index}.")

                if len(segment_data) == expected_size:
                    add_segment_to_playlist(local_segment_path)
                    # Note: Cleanup of 'local_segment_path' is now handled by on_media_changed_callback (approximately)
                else:
                    logger.error(f"File size mismatch. Expected {expected_size}, got {len(segment_data)}.")
                    if os.path.exists(local_segment_path): os.remove(local_segment_path) # Clean up partial
                
                segment_index += 1

            elif header_str.startswith("ERROR 404"):
                logger.warning(f"Segment {request_path_on_server} not found (404). Assuming end of video.")
                keep_streaming = False; break 
            elif header_str.startswith("ERROR"):
                logger.error(f"Server error: {header_str}. Stopping.")
                keep_streaming = False; break
            else:
                logger.error(f"Unknown server response: {header_str}. Stopping.")
                keep_streaming = False; break
        
        except ConnectionError as e:
            logger.error(f"Connection error during streaming: {e}.")
            keep_streaming = False; break
        except Exception as e:
            logger.error(f"Unexpected error in streaming loop: {e}")
            keep_streaming = False; break
        
        # Add a small delay or manage download speed for ABR later
        # For now, just a minimal delay if not aggressively fetching.
        # If player is buffering well, this might not be needed.
        # time.sleep(0.1) 

    logger.info("Streaming session finished fetching segments.")
    # Wait for the playlist to finish if it's still playing
    if media_list_player:
        while media_list_player.get_state() in [vlc.State.Playing, vlc.State.Opening, vlc.State.Buffering]:
            time.sleep(0.5)
    logger.info("Playback has likely ended or player stopped.")


def main():
    global vlc_instance, media_list_player # Allow main to stop them
    VIDEO_TO_STREAM = "bbb_sunflower" # Ensure this matches server folder
    QUALITY_TO_STREAM = "480p-1500k" # Or other quality

    client_socket = None
    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        logger.info(f"Connecting to server {SERVER_HOST}:{SERVER_PORT}...")
        client_socket.connect((SERVER_HOST, SERVER_PORT))
        logger.info("Connected to server.")

        start_streaming_session(client_socket, VIDEO_TO_STREAM, QUALITY_TO_STREAM)

    except socket.timeout:
        logger.error(f"Connection to server {SERVER_HOST}:{SERVER_PORT} timed out.")
    except ConnectionRefusedError:
        logger.error(f"Connection refused. Server at {SERVER_HOST}:{SERVER_PORT} might not be running.")
    except Exception as e:
        logger.error(f"An error occurred in main: {e}")
    finally:
        if client_socket:
            logger.info("Sending QUIT to server and closing connection.")
            try:
                client_socket.sendall(b"QUIT\n")
            except OSError: 
                logger.warning("Could not send QUIT; socket likely closed.")
            finally:
                client_socket.close()
                logger.info("Client socket closed.")
        
        if media_list_player:
            media_list_player.stop()
            logger.info("VLC MediaListPlayer stopped.")
        # Release VLC resources if they were initialized
        # This part is tricky as vlc_instance might be used by threads from events.
        # A full robust cleanup of VLC might need more careful handling of its event loop.
        # For simplicity, we stop the player. Full instance release might be for app exit.


if __name__ == "__main__":
    # Create download directory if it doesn't exist
    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)
    main()
    logger.info("Client application finished.")