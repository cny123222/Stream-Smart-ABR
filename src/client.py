import os
import time
import logging
import vlc
import threading
import requests # For HTTP requests
import tempfile # For temporary M3U8 file
from urllib.parse import urlparse, urljoin, quote # For URL manipulation
import urllib.parse # For parsing query parameters
import http.server
import socketserver

import AES # Your AES decryption module

# --- Configuration ---
SERVER_HOST = '127.0.0.1' # Your HLS server host
SERVER_PORT = 8081        # Your HLS server port
LOCAL_PROXY_HOST = '127.0.0.1'
LOCAL_PROXY_PORT = 8082   # Port for the local decryption proxy
DOWNLOAD_DIR = "download" # Directory for storing temp modified m3u8
SOCKET_TIMEOUT_SECONDS = 10
# RETRY_DOWNLOAD_DELAY = 2 # Not currently used in this HLS model

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger('HLSClientVLC')

# --- VLC Player Setup & State ---
vlc_instance = None
media_player = None
player_event_manager = None
player_reached_end = threading.Event() # Event to signal when playback has ended or errored

# --- Local Decryption Proxy ---
g_local_proxy_server_instance = None # Global reference to the proxy server instance
g_proxy_runner_thread = None         # Global reference to the thread running the proxy server

class DecryptionProxyHandler(http.server.BaseHTTPRequestHandler):
    """
    Handles requests from VLC for TS segments.
    Fetches encrypted segments from the main server, decrypts them, and serves them to VLC.
    """
    def do_GET(self):
        # Using a clean adapter with no extra that could conflict with standard log record attributes
        log_adapter = logging.LoggerAdapter(logger, {}) 
        request_log_tag = f"[ProxyRequest URI: {self.path}]" # Tag for this specific request

        try:
            parsed_url = urlparse(self.path)
            query_params = urllib.parse.parse_qs(parsed_url.query)
            
            original_segment_url = query_params.get('url', [None])[0]

            if not original_segment_url:
                log_adapter.error(f"{request_log_tag} Proxy GET request missing 'url' parameter.")
                self.send_error(400, "Bad Request: Missing 'url' parameter")
                return

            log_adapter.info(f"{request_log_tag} Received request for original URL: {original_segment_url}")

            # Fetch the encrypted segment from the main server
            try:
                response = requests.get(original_segment_url, timeout=SOCKET_TIMEOUT_SECONDS, stream=True)
                response.raise_for_status() # Raise an exception for HTTP errors
            except requests.exceptions.RequestException as e:
                log_adapter.error(f"{request_log_tag} Proxy failed to fetch segment {original_segment_url} from main server: {e}")
                self.send_error(502, f"Bad Gateway: Could not fetch from origin: {e}")
                return

            encrypted_data = response.content 

            if not encrypted_data:
                log_adapter.warning(f"{request_log_tag} Proxy received empty content for {original_segment_url}")
                self.send_error(502, "Bad Gateway: Empty content from origin")
                return
            
            # Decrypt the segment
            try:
                decrypted_data = AES.aes_decrypt_cbc(encrypted_data, AES.AES_KEY)
            except Exception as e:
                log_adapter.error(f"{request_log_tag} Proxy failed to decrypt segment from {original_segment_url}: {e}", exc_info=True)
                self.send_error(500, "Internal Server Error: Decryption failed")
                return

            log_adapter.info(f"{request_log_tag} Successfully decrypted segment from {original_segment_url}. Decrypted size: {len(decrypted_data)} bytes.")

            # Send the decrypted segment to VLC
            self.send_response(200)
            self.send_header('Content-type', 'video/MP2T')
            self.send_header('Content-Length', str(len(decrypted_data)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(decrypted_data)
            log_adapter.debug(f"{request_log_tag} Finished sending decrypted segment for {original_segment_url} to VLC.")

        except ConnectionResetError:
            log_adapter.warning(f"{request_log_tag} Connection reset by VLC.")
        except BrokenPipeError:
            log_adapter.warning(f"{request_log_tag} Broken pipe while writing to VLC (VLC might have closed connection).")
        except Exception as e:
            log_adapter.error(f"{request_log_tag} Error handling GET request: {e}", exc_info=True)
            if not self.wfile.closed:
                try:
                    self.send_error(500, f"Internal Server Error: {e}")
                except Exception:
                    pass


class ThreadingLocalProxyServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True

def _run_proxy_server_target():
    """Target function for the proxy server thread."""
    global g_local_proxy_server_instance
    # Local variable for the server instance within this thread's scope
    current_thread_server_instance = None
    try:
        current_thread_server_instance = ThreadingLocalProxyServer(
            (LOCAL_PROXY_HOST, LOCAL_PROXY_PORT), 
            DecryptionProxyHandler
        )
        # Assign to global only after successful creation
        g_local_proxy_server_instance = current_thread_server_instance 
        logger.info(f"PROXY_THREAD: Local decryption proxy starting on http://{LOCAL_PROXY_HOST}:{LOCAL_PROXY_PORT}")
        current_thread_server_instance.serve_forever() # This blocks until shutdown() is called
    except OSError as e:
        logger.error(f"PROXY_THREAD: Could not start local decryption proxy (OSError): {e} (Port {LOCAL_PROXY_PORT} busy?)")
        g_local_proxy_server_instance = None # Ensure global is None if creation failed
    except Exception as e:
        logger.error(f"PROXY_THREAD: An unexpected error in proxy server run: {e}", exc_info=True)
        # If current_thread_server_instance was created but serve_forever failed,
        # it might need cleanup, but stop_proxy_server() should handle the global ref.
        g_local_proxy_server_instance = None # Ensure global is None on significant error
    finally:
        # This block executes when serve_forever() returns.
        logger.info(f"PROXY_THREAD: Local decryption proxy server loop ({threading.current_thread().name}) has finished.")
        # Do not set g_local_proxy_server_instance to None here;
        # stop_proxy_server() manages the lifecycle of the global reference.

def start_proxy_server():
    """Starts the local decryption proxy server in a new thread."""
    global g_proxy_runner_thread, g_local_proxy_server_instance

    if g_proxy_runner_thread and g_proxy_runner_thread.is_alive():
        logger.warning("PROXY_MAIN: Proxy server thread is already running.")
        # Check if the instance is valid; it might be running but instance became None due to error
        return bool(g_local_proxy_server_instance) 

    # Clear any previous instance before starting a new one
    g_local_proxy_server_instance = None 
    
    g_proxy_runner_thread = threading.Thread(target=_run_proxy_server_target, daemon=True, name="ProxyServerThread")
    g_proxy_runner_thread.start()
    
    time.sleep(0.5) # Allow time for the server to start or fail during initialization

    if g_proxy_runner_thread.is_alive() and g_local_proxy_server_instance:
        logger.info("PROXY_MAIN: Proxy server thread started and server instance seems successfully created.")
        return True
    else:
        logger.error("PROXY_MAIN: Proxy server thread failed to start properly or server instance not created.")
        if g_proxy_runner_thread and g_proxy_runner_thread.is_alive():
            logger.info("PROXY_MAIN: Proxy thread is alive but instance is None. Attempting to join failed proxy runner thread.")
            # If instance is None, _run_proxy_server_target likely hit an exception.
            # The thread might exit shortly. shutdown/close on None is not possible.
            g_proxy_runner_thread.join(timeout=1.0) # Give it a chance to exit
        g_local_proxy_server_instance = None # Ensure it's None
        g_proxy_runner_thread = None
        return False

def stop_proxy_server():
    """Stops the local decryption proxy server."""
    global g_local_proxy_server_instance, g_proxy_runner_thread
    
    server_instance_to_stop = g_local_proxy_server_instance # Capture the global reference

    if server_instance_to_stop:
        logger.info(f"PROXY_MAIN: Attempting to stop proxy server instance: {server_instance_to_stop}")
        try:
            # 1. Signal the serve_forever() loop to stop
            if hasattr(server_instance_to_stop, 'shutdown'):
                logger.debug("PROXY_MAIN: Calling server_instance_to_stop.shutdown()")
                server_instance_to_stop.shutdown()
            else:
                logger.warning("PROXY_MAIN: Captured proxy server instance for shutdown is invalid or has no 'shutdown' method.")

            # 2. Wait for the thread running serve_forever() to terminate
            if g_proxy_runner_thread and g_proxy_runner_thread.is_alive():
                logger.debug(f"PROXY_MAIN: Waiting for proxy server thread ({g_proxy_runner_thread.name}) to join...")
                g_proxy_runner_thread.join(timeout=3.0)
                if g_proxy_runner_thread.is_alive():
                    logger.warning("PROXY_MAIN: Proxy server thread did not join cleanly.")
            
            # 3. Close the server's listening socket
            # This should be done after the thread has joined (i.e., serve_forever has returned)
            if hasattr(server_instance_to_stop, 'server_close'):
                logger.debug("PROXY_MAIN: Calling server_instance_to_stop.server_close()")
                server_instance_to_stop.server_close()
            else:
                logger.warning("PROXY_MAIN: Captured proxy server instance for server_close is invalid or has no 'server_close' method.")
        
        except Exception as e_shutdown:
            logger.error(f"PROXY_MAIN: Error during proxy server shutdown/close sequence: {e_shutdown}", exc_info=True)
        finally:
            # Clear global references after attempting to stop and clean up
            g_local_proxy_server_instance = None 
            g_proxy_runner_thread = None
            logger.info("PROXY_MAIN: Proxy server stop sequence finished.")
    else:
        logger.info("PROXY_MAIN: Proxy server instance (g_local_proxy_server_instance) was already None or not started.")


def fetch_and_rewrite_m3u8(original_m3u8_url):
    logger.info(f"Fetching original M3U8 from: {original_m3u8_url}")
    try:
        response = requests.get(original_m3u8_url, timeout=SOCKET_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch M3U8 playlist {original_m3u8_url}: {e}")
        return None

    original_m3u8_content = response.text
    modified_lines = []
    m3u8_base_url = urljoin(original_m3u8_url, '.') # More robust way to get base URL

    for line in original_m3u8_content.splitlines():
        line_stripped = line.strip()
        if line_stripped and not line_stripped.startswith("#") and (line_stripped.endswith(".ts") or ".ts?" in line_stripped): # Handle .ts with query params
            original_segment_url = line_stripped
            # Resolve original_segment_url to absolute if it's relative
            original_segment_url_absolute = urljoin(m3u8_base_url, original_segment_url)
            
            encoded_original_url = quote(original_segment_url_absolute, safe='')
            rewritten_segment_url = f"http://{LOCAL_PROXY_HOST}:{LOCAL_PROXY_PORT}/proxy?url={encoded_original_url}"
            modified_lines.append(rewritten_segment_url)
            logger.debug(f"Rewrote segment URL: '{original_segment_url}' -> '{rewritten_segment_url}'")
        else:
            modified_lines.append(line)
            
    return "\n".join(modified_lines)


def initialize_vlc_player():
    global vlc_instance, media_player, player_event_manager
    if vlc_instance is None:
        instance_args = ['--no-video-title-show'] # Removed --quiet for more VLC logs if needed
        # For HLS, VLC's internal network caching is more relevant.
        # You can adjust this value; larger might mean longer startup but more resilience to network hiccups.
        instance_args.append(f'--network-caching=6000') # e.g., 6 seconds VLC internal network buffer
        # instance_args.append('--verbose=2') # Uncomment for verbose VLC logs
        vlc_instance = vlc.Instance(instance_args)
        media_player = vlc_instance.media_player_new()
        player_event_manager = media_player.event_manager()
        player_event_manager.event_attach(vlc.EventType.MediaPlayerEndReached, on_player_end_reached_callback)
        player_event_manager.event_attach(vlc.EventType.MediaPlayerEncounteredError, on_player_error_callback)
        # Optional: More events for debugging
        # player_event_manager.event_attach(vlc.EventType.MediaPlayerBuffering, lambda e: logger.info(f"VLC Buffering: {media_player.get_media().get_stats()}"))
        # player_event_manager.event_attach(vlc.EventType.MediaPlayerPlaying, lambda e: logger.info("VLC Playing"))
        logger.info("VLC Player initialized for HLS playback.")

def on_player_end_reached_callback(event):
    logger.info("VLC_EVENT: MediaPlayerEndReached - Playback finished or stream ended.")
    player_reached_end.set()

def on_player_error_callback(event):
    logger.error("VLC_EVENT: MediaPlayerEncounteredError - An error occurred during VLC playback.")
    player_reached_end.set()

def play_hls_stream(video_name_on_server, quality_on_server):
    global player_reached_end

    initialize_vlc_player()
    player_reached_end.clear()

    # Construct the original M3U8 URL based on typical HLS server structure
    # (server serves files from quality_suffix directory)
    m3u8_filename = f"{video_name_on_server}-{quality_on_server}.m3u8" # As per your segment_video.py output
    original_m3u8_url = f"http://{SERVER_HOST}:{SERVER_PORT}/{video_name_on_server}/{quality_on_server}/{m3u8_filename}"
    logger.info(f"Attempting to play HLS stream. Original M3U8 URL: {original_m3u8_url}")

    modified_m3u8_content = fetch_and_rewrite_m3u8(original_m3u8_url)

    if not modified_m3u8_content:
        logger.error("Could not get modified M3U8 content. Aborting playback.")
        return

    temp_m3u8_file_path = None # Initialize to ensure it's defined for finally
    try:
        if not os.path.exists(DOWNLOAD_DIR):
            os.makedirs(DOWNLOAD_DIR)
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".m3u8", delete=False, dir=DOWNLOAD_DIR, encoding='utf-8') as tmp_file:
            tmp_file.write(modified_m3u8_content)
            temp_m3u8_file_path = tmp_file.name
        
        logger.info(f"Modified M3U8 saved to temporary file: {temp_m3u8_file_path}")
        
        if os.name == 'nt':
            mrl = 'file:///' + temp_m3u8_file_path.replace('\\', '/')
        else:
            mrl = f'file://{os.path.abspath(temp_m3u8_file_path)}' # Ensure absolute path for non-Windows MRLs
            
        logger.info(f"VLC will play MRL: {mrl}")
        
        media = vlc_instance.media_new(mrl)
        if not media:
            logger.error(f"Failed to create VLC media object for MRL: {mrl}")
            return
            
        media_player.set_media(media)
        media.release() # Media object can be released after set_media

        if media_player.play() == -1:
            logger.error("Failed to start VLC playback.")
            return
        
        logger.info("Playback command issued. Waiting for end or error...")
        player_reached_end.wait() 
        logger.info("Playback wait loop in play_hls_stream finished.")

    except Exception as e:
        logger.error(f"Error during HLS stream playback setup or execution: {e}", exc_info=True)
    finally:
        if media_player and media_player.is_playing():
            media_player.stop()
            logger.info("VLC player stopped in play_hls_stream finally block.")
        if temp_m3u8_file_path and os.path.exists(temp_m3u8_file_path):
            try:
                os.remove(temp_m3u8_file_path)
                logger.info(f"Cleaned up temporary M3U8 file: {temp_m3u8_file_path}")
            except OSError as e_rem:
                logger.warning(f"Could not remove temporary M3U8 file {temp_m3u8_file_path}: {e_rem}")


def main():
    # --- Windows DLL 加载辅助 ---
    if os.name == 'nt':
        vlc_install_dir = None
        common_paths = [
            os.path.join(os.environ.get("ProgramFiles", "C:\\Program Files"), "VideoLAN", "VLC"),
            os.path.join(os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)"), "VideoLAN", "VLC")
        ]
        for p in common_paths:
            if os.path.isdir(p) and os.path.exists(os.path.join(p, "libvlc.dll")):
                vlc_install_dir = p; break
        if vlc_install_dir:
            logger.info(f"VLC DLL directory hint: {vlc_install_dir}. Ensure python-vlc can find it.")
            # os.add_dll_directory(vlc_install_dir) # Call before 'import vlc' if needed and Python version supports
        else:
             logger.warning("VLC installation not found in common paths. Ensure VLC is in PATH or VLC_SDK_PATH is set.")

    VIDEO_TO_STREAM = "bbb_sunflower" 
    QUALITY_TO_STREAM = "480p-1500k"  

    try:
        if not hasattr(AES, 'AES_KEY') or not AES.AES_KEY:
            logger.error("AES.AES_KEY is not defined or is empty in AES.py. Cannot proceed.")
            return
        if not callable(getattr(AES, 'aes_decrypt_cbc', None)):
            logger.error("AES.aes_decrypt_cbc function is not defined in AES.py. Cannot proceed.")
            return
        logger.info("AES module loaded and key/function seem present.")

        if not start_proxy_server(): # Call new start function
            # Error already logged by start_proxy_server
            return

        play_hls_stream(VIDEO_TO_STREAM, QUALITY_TO_STREAM)

    except KeyboardInterrupt:
        logger.info("Application interrupted by user (Ctrl+C).")
    except Exception as e:
        logger.error(f"An error occurred in main: {e}", exc_info=True)
    finally:
        logger.info("Main: Initiating cleanup...")
        
        # Stop VLC player first, if it's playing
        if media_player and media_player.is_playing():
            media_player.stop()
            logger.info("Main: VLC media player stopped.")
        if vlc_instance: # Release VLC instance if it was created
            vlc_instance.release()
            logger.info("Main: VLC instance released.")

        # Then stop the proxy server
        stop_proxy_server() # Call new stop function
        
        if os.path.exists(DOWNLOAD_DIR):
            try:
                # Clean up any remaining .m3u8 files in the temp download dir
                for f_name in os.listdir(DOWNLOAD_DIR):
                    if f_name.endswith(".m3u8"):
                        os.remove(os.path.join(DOWNLOAD_DIR, f_name))
                # If directory is empty, remove it
                if not os.listdir(DOWNLOAD_DIR):
                    os.rmdir(DOWNLOAD_DIR)
                else: # Or log if other files are present (should not happen with NamedTemporaryFile usually)
                    logger.info(f"Directory '{DOWNLOAD_DIR}' not empty after M3U8 cleanup.")
            except Exception as e_clean:
                logger.warning(f"Error during final cleanup of {DOWNLOAD_DIR}: {e_clean}")

        logger.info("Client application finished.")

if __name__ == "__main__":
    main()