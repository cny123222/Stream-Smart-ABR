import os
import time
import logging
import threading
import requests # For HTTP requests
import tempfile # For temporary M3U8 file (less used now)
from urllib.parse import urlparse, urljoin, quote, unquote, parse_qs # For URL manipulation
import http.server
import socketserver
import re
import webbrowser # To open the browser

import AES # Your AES decryption module

# --- Configuration ---
SERVER_HOST = '127.0.0.1' # Your HLS server host
SERVER_PORT = 8081        # Your HLS server port
LOCAL_PROXY_HOST = '127.0.0.1'
LOCAL_PROXY_PORT = 8082   # Port for the local proxy server
DOWNLOAD_DIR = "download" # Directory for any temp files if needed
SOCKET_TIMEOUT_SECONDS = 10

VIDEO_TO_STREAM_NAME = "bbb_sunflower" # Base name for the video stream

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger('HLSJS_Client')

# --- Local Proxy Server Globals ---
g_local_proxy_server_instance = None
g_proxy_runner_thread = None

# --- ABR State (Shared between ABR logic and Proxy for stats) ---
abr_lock = threading.Lock()
# This now primarily reflects the ABR *algorithm's* chosen quality, HLS.js makes its own choices.
current_abr_algorithm_selected_media_m3u8_url_on_server = None

# --- HTML Content for the Player ---
HTML_PLAYER_CONTENT = """
<!DOCTYPE html>
<html>
<head>
    <title>HLS.js Player</title>
    <script src="https://cdn.jsdelivr.net/npm/hls.js@latest"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f4f4f4; color: #333; }}
        #videoContainer {{ width: 80%; max-width: 800px; margin: 0 auto; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
        video {{ width: 100%; background-color: #000; }}
        h1 {{ text-align: center; color: #555; }}
        .controls {{ text-align: center; margin-top: 10px; }}
        button {{ padding: 8px 15px; margin: 5px; cursor: pointer; }}
    </style>
</head>
<body>
    <h1>HLS.js Streaming Client</h1>
    <div id="videoContainer">
        <video id="videoPlayer" controls></video>
    </div>
    <div class="controls">
        <p>Master Playlist URL: <span id="masterUrlSpan"></span></p>
    </div>

    <script>
        document.addEventListener('DOMContentLoaded', function () {
            const video = document.getElementById('videoPlayer');
            const masterM3u8Url = `http://{LOCAL_PROXY_HOST}:{LOCAL_PROXY_PORT}/{VIDEO_TO_STREAM_NAME}/master.m3u8`;
            document.getElementById('masterUrlSpan').textContent = masterM3u8Url;
            
            if (Hls.isSupported()) {
                const hls = new Hls({
                    // debug: true, // Enable for more detailed HLS.js logs in browser console
                    // --- Configuration for ABR (HLS.js internal ABR) ---
                    // capLevelToPlayerSize: true, // Match quality to player size
                    // abrEwmaDefaultEstimate: 500000, // Initial bandwidth estimate in bits/s (optional)
                    // abrMaxWithRealBitrate: true, // Use actual segment bitrate for ABR decisions
                });
                hls.loadSource(masterM3u8Url);
                hls.attachMedia(video);
                hls.on(Hls.Events.MANIFEST_PARSED, function () {
                    console.log("Manifest parsed, playing video.");
                    video.play();
                });
                hls.on(Hls.Events.ERROR, function (event, data) {
                    if (data.fatal) {
                        switch (data.type) {
                            case Hls.ErrorTypes.NETWORK_ERROR:
                                console.error('Fatal network error encountered:', data);
                                // Try to recover from network errors
                                // hls.startLoad(); // or hls.loadSource(masterM3u8Url);
                                break;
                            case Hls.ErrorTypes.MEDIA_ERROR:
                                console.error('Fatal media error encountered:', data);
                                // hls.recoverMediaError(); // Attempt to recover
                                break;
                            default:
                                console.error('Fatal HLS error:', data);
                                hls.destroy(); // Cannot recover
                                break;
                        }
                    } else {
                        console.warn('Non-fatal HLS error:', data);
                    }
                });
                 hls.on(Hls.Events.LEVEL_SWITCHED, function(event, data) {
                    console.log(`HLS.js switched to level: ${data.level}, Bitrate: ${hls.levels[data.level].bitrate/1000} Kbps, Resolution: ${hls.levels[data.level].width}x${hls.levels[data.level].height}`);
                });
            } else if (video.canPlayType('application/vnd.apple.mpegurl')) {
                // Native HLS support (e.g., Safari)
                video.src = masterM3u8Url;
                video.addEventListener('loadedmetadata', function () {
                    video.play();
                });
            } else {
                alert("HLS is not supported in your browser.");
            }
        });
    </script>
</body>
</html>
""".replace("{LOCAL_PROXY_HOST}", LOCAL_PROXY_HOST) \
 .replace("{LOCAL_PROXY_PORT}", str(LOCAL_PROXY_PORT)) \
 .replace("{VIDEO_TO_STREAM_NAME}", VIDEO_TO_STREAM_NAME)


class DecryptionProxyHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        log_adapter = logging.LoggerAdapter(logger, {'path': self.path})
        request_log_tag = f"[ProxyRequest URI: {self.path}]"
        parsed_url = urlparse(self.path)
        path_components = parsed_url.path.strip('/').split('/')

        try:
            # 1. Serve player.html
            if parsed_url.path == '/' or parsed_url.path == '/player.html':
                log_adapter.info(f"{request_log_tag} Serving player.html")
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(HTML_PLAYER_CONTENT.encode('utf-8'))
                return

            # 2. Serve Master M3U8 (e.g., /bbb_sunflower/master.m3u8)
            # Assumes master playlist is always named "master.m3u8"
            # and is at /{VIDEO_NAME}/master.m3u8
            if len(path_components) == 2 and path_components[0] == VIDEO_TO_STREAM_NAME and path_components[1] == "master.m3u8":
                video_name_from_url = path_components[0]
                original_master_m3u8_url = f"http://{SERVER_HOST}:{SERVER_PORT}/{video_name_from_url}/master.m3u8"
                log_adapter.info(f"{request_log_tag} Request for master M3U8. Fetching from: {original_master_m3u8_url}")

                try:
                    response = requests.get(original_master_m3u8_url, timeout=SOCKET_TIMEOUT_SECONDS)
                    response.raise_for_status()
                    master_content = response.text
                    modified_master_content = self._rewrite_master_playlist(master_content, original_master_m3u8_url)
                    
                    self.send_response(200)
                    self.send_header('Content-type', 'application/vnd.apple.mpegurl') # or 'application/x-mpegURL'
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(modified_master_content.encode('utf-8'))
                    log_adapter.info(f"{request_log_tag} Served modified master M3U8.")
                except requests.exceptions.RequestException as e:
                    log_adapter.error(f"{request_log_tag} Failed to fetch/process master M3U8 {original_master_m3u8_url}: {e}")
                    self.send_error(502, f"Bad Gateway: Could not fetch master M3U8 from origin: {e}")
                return

            # 3. Serve Media M3U8 (e.g., /bbb_sunflower/720p-4000k/bbb_sunflower-720p-4000k.m3u8)
            # Assumes path structure: /{VIDEO_NAME}/{QUALITY_DIR}/{PLAYLIST_NAME}.m3u8
            if len(path_components) == 3 and path_components[0] == VIDEO_TO_STREAM_NAME and path_components[2].endswith(".m3u8"):
                video_name_from_url = path_components[0]
                quality_dir_from_url = path_components[1]
                playlist_filename_from_url = path_components[2]
                
                original_media_m3u8_url = f"http://{SERVER_HOST}:{SERVER_PORT}/{video_name_from_url}/{quality_dir_from_url}/{playlist_filename_from_url}"
                log_adapter.info(f"{request_log_tag} Request for media M3U8. Fetching from: {original_media_m3u8_url}")

                try:
                    response = requests.get(original_media_m3u8_url, timeout=SOCKET_TIMEOUT_SECONDS)
                    response.raise_for_status()
                    media_content = response.text
                    modified_media_content = self._rewrite_media_playlist(media_content, original_media_m3u8_url)

                    self.send_response(200)
                    self.send_header('Content-type', 'application/vnd.apple.mpegurl')
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(modified_media_content.encode('utf-8'))
                    log_adapter.info(f"{request_log_tag} Served modified media M3U8 for quality {quality_dir_from_url}.")
                except requests.exceptions.RequestException as e:
                    log_adapter.error(f"{request_log_tag} Failed to fetch/process media M3U8 {original_media_m3u8_url}: {e}")
                    self.send_error(502, f"Bad Gateway: Could not fetch media M3U8 from origin: {e}")
                return

            # 4. Serve Decrypted TS Segment (e.g., /decrypt_segment?url=ENCODED_ORIGINAL_TS_URL)
            if parsed_url.path == '/decrypt_segment':
                query_params = parse_qs(parsed_url.query)
                original_ts_url_on_server_encoded = query_params.get('url', [None])[0]

                if not original_ts_url_on_server_encoded:
                    log_adapter.error(f"{request_log_tag} Decrypt request missing 'url' parameter.")
                    self.send_error(400, "Bad Request: Missing 'url' parameter")
                    return
                
                original_ts_url_on_server = unquote(original_ts_url_on_server_encoded)
                log_adapter.info(f"{request_log_tag} HLS.js requested TS: {original_ts_url_on_server}")
                
                try:
                    fetch_start_time = time.time()
                    response_ts = requests.get(original_ts_url_on_server, timeout=SOCKET_TIMEOUT_SECONDS, stream=True)
                    response_ts.raise_for_status()
                    encrypted_data = response_ts.content
                    fetch_end_time = time.time()

                    if ABRManager.instance:
                        download_duration = fetch_end_time - fetch_start_time
                        segment_size = len(encrypted_data)
                        ABRManager.instance.add_segment_download_stat(original_ts_url_on_server, segment_size, download_duration)

                    if not encrypted_data:
                        log_adapter.warning(f"{request_log_tag} Proxy received empty content for {original_ts_url_on_server}")
                        if ABRManager.instance: ABRManager.instance.report_download_error(original_ts_url_on_server)
                        self.send_error(502, "Bad Gateway: Empty content from origin for TS segment")
                        return
                    
                    decrypted_data = AES.aes_decrypt_cbc(encrypted_data, AES.AES_KEY)
                    log_adapter.debug(f"{request_log_tag} Successfully decrypted segment from {original_ts_url_on_server}. Decrypted size: {len(decrypted_data)} bytes.")

                    self.send_response(200)
                    self.send_header('Content-type', 'video/MP2T')
                    self.send_header('Content-Length', str(len(decrypted_data)))
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(decrypted_data)
                    log_adapter.debug(f"{request_log_tag} Finished sending decrypted segment for {original_ts_url_on_server} to HLS.js.")

                except requests.exceptions.RequestException as e_req:
                    log_adapter.error(f"{request_log_tag} Proxy failed to fetch segment {original_ts_url_on_server} from main server: {e_req}")
                    if ABRManager.instance: ABRManager.instance.report_download_error(original_ts_url_on_server)
                    self.send_error(502, f"Bad Gateway: Could not fetch TS from origin: {e_req}")
                except Exception as e_dec:
                    log_adapter.error(f"{request_log_tag} Proxy failed to decrypt segment from {original_ts_url_on_server}: {e_dec}", exc_info=True)
                    self.send_error(500, "Internal Server Error: Decryption failed")
                return

            # If no route matched
            log_adapter.warning(f"{request_log_tag} Unknown path requested.")
            self.send_error(404, "Not Found")

        except ConnectionResetError:
            log_adapter.warning(f"{request_log_tag} Connection reset by client.")
        except BrokenPipeError:
            log_adapter.warning(f"{request_log_tag} Broken pipe while writing to client.")
        except Exception as e:
            log_adapter.error(f"{request_log_tag} Error handling GET request: {e}", exc_info=True)
            if not self.wfile.closed:
                try: self.send_error(500, f"Internal Server Error: {e}")
                except Exception: pass
    
    def _rewrite_master_playlist(self, master_content, original_master_url):
        """Rewrites media playlist URLs in master M3U8 to point to this proxy."""
        log_adapter = logging.LoggerAdapter(logger, {})
        lines = master_content.splitlines()
        modified_lines = []
        # Base for resolving relative URLs in the master playlist if they are relative *to the master playlist itself*
        # For media playlists, we want them to be proxied.
        # Example: if master has "../qualityX/playlist.m3u8", we need to make it "http://proxy/video_name/qualityX/playlist.m3u8"

        for i in range(len(lines)):
            line = lines[i].strip()
            if line.startswith("#EXT-X-STREAM-INF:"):
                modified_lines.append(line) # Keep the info line
                if i + 1 < len(lines):
                    media_playlist_relative_or_absolute_url_in_master = lines[i+1].strip()
                    # Resolve the original media playlist URL against the original master URL
                    original_media_playlist_absolute_url = urljoin(original_master_url, media_playlist_relative_or_absolute_url_in_master)
                    
                    # Now, create a new URL that points to our proxy, maintaining the path structure relative to SERVER_HOST:SERVER_PORT
                    # e.g. http://S_HOST:S_PORT/video/qual/play.m3u8 -> http://L_PROXY_HOST:L_PROXY_PORT/video/qual/play.m3u8
                    parsed_original_media_playlist_url = urlparse(original_media_playlist_absolute_url)
                    proxied_media_playlist_url = f"http://{LOCAL_PROXY_HOST}:{LOCAL_PROXY_PORT}{parsed_original_media_playlist_url.path}"
                    
                    modified_lines.append(proxied_media_playlist_url)
                    log_adapter.debug(f"Master Rewrite: '{media_playlist_relative_or_absolute_url_in_master}' -> '{proxied_media_playlist_url}'")
                    i += 1 # Skip the original URL line as we've processed it
            else:
                modified_lines.append(line)
        return "\n".join(modified_lines)

    def _rewrite_media_playlist(self, media_content, original_media_url):
        """Rewrites TS segment URLs in media M3U8 to point to the proxy's decryption endpoint."""
        log_adapter = logging.LoggerAdapter(logger, {})
        lines = media_content.splitlines()
        modified_lines = []
        media_m3u8_base_url = urljoin(original_media_url, '.') # Base for resolving relative TS paths

        for line in lines:
            line_stripped = line.strip()
            if line_stripped and not line_stripped.startswith("#") and \
               (line_stripped.endswith(".ts") or ".ts?" in line_stripped):
                original_segment_relative_url = line_stripped
                original_segment_absolute_url_on_server = urljoin(media_m3u8_base_url, original_segment_relative_url)
                
                encoded_original_url = quote(original_segment_absolute_url_on_server, safe='')
                rewritten_segment_url = f"http://{LOCAL_PROXY_HOST}:{LOCAL_PROXY_PORT}/decrypt_segment?url={encoded_original_url}"
                modified_lines.append(rewritten_segment_url)
                log_adapter.debug(f"Media Rewrite: Segment '{original_segment_relative_url}' -> '{rewritten_segment_url}'")
            else:
                modified_lines.append(line)
        return "\n".join(modified_lines)

    def log_message(self, format, *args): # Quieter logging for proxy
        logger.debug(f"Proxy HTTP Log: {self.address_string()} - {args[0]} {args[1]}")


class ThreadingLocalProxyServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True

def _run_proxy_server_target():
    global g_local_proxy_server_instance
    current_thread_server_instance = None
    try:
        current_thread_server_instance = ThreadingLocalProxyServer(
            (LOCAL_PROXY_HOST, LOCAL_PROXY_PORT), DecryptionProxyHandler)
        g_local_proxy_server_instance = current_thread_server_instance 
        logger.info(f"PROXY_THREAD: Local proxy server starting on http://{LOCAL_PROXY_HOST}:{LOCAL_PROXY_PORT}")
        logger.info(f"PROXY_THREAD: Player available at http://{LOCAL_PROXY_HOST}:{LOCAL_PROXY_PORT}/player.html")
        current_thread_server_instance.serve_forever() 
    except OSError as e:
        logger.error(f"PROXY_THREAD: Could not start local proxy server (OSError): {e}")
        g_local_proxy_server_instance = None
    except Exception as e:
        logger.error(f"PROXY_THREAD: An unexpected error in proxy server run: {e}", exc_info=True)
        g_local_proxy_server_instance = None
    finally:
        logger.info(f"PROXY_THREAD: Local proxy server loop ({threading.current_thread().name}) has finished.")

def start_proxy_server():
    global g_proxy_runner_thread, g_local_proxy_server_instance
    if g_proxy_runner_thread and g_proxy_runner_thread.is_alive():
        logger.warning("PROXY_MAIN: Proxy server thread is already running.")
        return bool(g_local_proxy_server_instance) 
    g_local_proxy_server_instance = None 
    g_proxy_runner_thread = threading.Thread(target=_run_proxy_server_target, daemon=True, name="ProxyServerThread")
    g_proxy_runner_thread.start()
    time.sleep(0.5) # Give server a moment to start
    if g_proxy_runner_thread.is_alive() and g_local_proxy_server_instance:
        logger.info("PROXY_MAIN: Proxy server thread started and server instance successfully created.")
        return True
    else:
        logger.error("PROXY_MAIN: Proxy server thread failed to start or server instance not created.")
        if g_proxy_runner_thread and g_proxy_runner_thread.is_alive():
            # Try to tell it to shutdown if instance wasn't formed but thread is alive
            if g_local_proxy_server_instance: # Should be None here based on logic, but defensive
                 g_local_proxy_server_instance.shutdown()
            g_proxy_runner_thread.join(timeout=1.0)
        g_local_proxy_server_instance = None
        g_proxy_runner_thread = None
        return False

def stop_proxy_server():
    global g_local_proxy_server_instance, g_proxy_runner_thread
    server_instance_to_stop = g_local_proxy_server_instance 
    if server_instance_to_stop:
        logger.info(f"PROXY_MAIN: Attempting to stop proxy server instance: {server_instance_to_stop}")
        try:
            if hasattr(server_instance_to_stop, 'shutdown'):
                server_instance_to_stop.shutdown() # This will break the serve_forever loop
            # The thread should then exit due to serve_forever ending
            if g_proxy_runner_thread and g_proxy_runner_thread.is_alive():
                g_proxy_runner_thread.join(timeout=3.0) # Wait for thread to finish
                if g_proxy_runner_thread.is_alive(): 
                    logger.warning("PROXY_MAIN: Proxy thread did not join cleanly after shutdown.")
            # server_close is typically called after shutdown and join
            if hasattr(server_instance_to_stop, 'server_close'):
                server_instance_to_stop.server_close()
        except Exception as e_shutdown:
            logger.error(f"PROXY_MAIN: Error during proxy shutdown: {e_shutdown}", exc_info=True)
        finally:
            g_local_proxy_server_instance = None 
            g_proxy_runner_thread = None
            logger.info("PROXY_MAIN: Proxy server stop sequence finished.")
    elif g_proxy_runner_thread and g_proxy_runner_thread.is_alive():
        # This case might happen if server instance creation failed but thread is lingering
        logger.warning("PROXY_MAIN: Proxy server instance was None, but thread is alive. Attempting to join thread.")
        g_proxy_runner_thread.join(timeout=1.0)
        g_proxy_runner_thread = None
        logger.info("PROXY_MAIN: Proxy thread join attempt finished.")
    else:
        logger.info("PROXY_MAIN: Proxy server instance was None or not started.")


def parse_m3u8_attributes(attr_string): # Copied from your original, seems fine
    attributes = {}
    try:
        for match in re.finditer(r'([A-Z0-9-]+)=("([^"]*)"|([^,"]*))', attr_string):
            key = match.group(1)
            value = match.group(3) if match.group(3) is not None else match.group(4)
            if value.isdigit(): 
                attributes[key] = int(value)
            else: 
                attributes[key] = value
    except Exception as e:
        logger.error(f"Error parsing M3U8 attributes string '{attr_string}': {e}")
    return attributes

def fetch_master_m3u8_for_abr_init(master_m3u8_url_on_server): # For ABRManager init
    logger.info(f"ABR_INIT: Fetching master M3U8 from: {master_m3u8_url_on_server}")
    try:
        response = requests.get(master_m3u8_url_on_server, timeout=SOCKET_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.error(f"ABR_INIT: Failed to fetch master M3U8 playlist {master_m3u8_url_on_server}: {e}")
        return None

    content = response.text
    lines = content.splitlines()
    available_streams = []
    master_m3u8_base_url = urljoin(master_m3u8_url_on_server, '.')

    for i in range(len(lines)):
        line = lines[i].strip()
        if line.startswith("#EXT-X-STREAM-INF:"):
            attributes_str = line.split(":", 1)[1]
            attributes = parse_m3u8_attributes(attributes_str)
            if i + 1 < len(lines):
                media_playlist_relative_url = lines[i+1].strip()
                # IMPORTANT: For ABRManager, we need the *original* server URLs for its internal logic
                media_playlist_absolute_url_on_origin = urljoin(master_m3u8_base_url, media_playlist_relative_url)
                stream_info = {
                    'url': media_playlist_absolute_url_on_origin, 
                    'bandwidth': attributes.get('BANDWIDTH'),
                    'resolution': attributes.get('RESOLUTION'),
                    'codecs': attributes.get('CODECS'),
                    'attributes_str': attributes_str
                }
                available_streams.append(stream_info)
                logger.debug(f"ABR_INIT: Found stream in master M3U8: {stream_info}")
    
    if not available_streams:
        logger.warning(f"ABR_INIT: No #EXT-X-STREAM-INF tags found in master M3U8 at {master_m3u8_url_on_server}")
        return None
    return available_streams


class ABRManager: # Largely same as your provided, but its direct control is lessened
    instance = None 

    def __init__(self, available_streams_from_master):
        if ABRManager.instance is not None:
            # Allow re-initialization for simplicity in script restarts, though true singleton might be stricter
            logger.warning("ABRManager re-initialized. Previous instance overwritten.")
        ABRManager.instance = self
        
        # Ensure streams are sorted by bandwidth (lowest to highest)
        self.available_streams = sorted(
            [s for s in available_streams_from_master if s.get('bandwidth') is not None], 
            key=lambda s: s['bandwidth']
        )
        if not self.available_streams:
             logger.error("ABRManager initialized with no available streams with bandwidth info!")
             # Add a dummy stream to prevent crashes if all streams lack bandwidth
             self.available_streams = [{'url': 'dummy', 'bandwidth': 0, 'resolution': 'N/A'}]


        self.current_stream_index_by_abr = 0 # Index for self.available_streams based on ABR algorithm
        self.segment_download_stats = [] 
        self.max_stats_history = 20 
        self.estimated_bandwidth_bps = 0
        
        self.safety_factor = 0.8 # Example: Target 80% of estimated bandwidth
        self.abr_thread = None
        self.stop_abr_event = threading.Event()
        
        logger.info(f"ABRManager initialized. Available streams (sorted by bandwidth):")
        for i, s in enumerate(self.available_streams):
            logger.info(f"  [{i}] BW: {s.get('bandwidth', 'N/A')}, Res: {s.get('resolution', 'N/A')}, URL: {s['url']}")
        
        self._update_current_abr_selected_url() # Initialize global based on starting index

    def _update_current_abr_selected_url(self):
        global current_abr_algorithm_selected_media_m3u8_url_on_server
        with abr_lock:
            if self.available_streams and 0 <= self.current_stream_index_by_abr < len(self.available_streams):
                current_abr_algorithm_selected_media_m3u8_url_on_server = self.available_streams[self.current_stream_index_by_abr]['url']
            else:
                logger.warning("ABR: Could not update current selected URL due to invalid index or no streams.")


    def add_segment_download_stat(self, url, size_bytes, duration_seconds):
        # url is the original URL on the server, useful for identifying quality
        if duration_seconds > 0.001: 
            self.segment_download_stats.append({'url': url, 'size': size_bytes, 'duration': duration_seconds})
            if len(self.segment_download_stats) > self.max_stats_history:
                self.segment_download_stats.pop(0)
            # logger.debug(f"ABR: Added stat: {size_bytes} bytes in {duration_seconds:.3f}s for segment from {url.split('/')[-1]}")

    def report_download_error(self, url):
        logger.warning(f"ABR: Reported download error for segment from URL: {url}")
        # Could implement logic here, e.g., temporary bandwidth reduction

    def _estimate_bandwidth(self):
        if not self.segment_download_stats:
            return self.estimated_bandwidth_bps # Return last estimate if no new data
        
        # Using harmonic mean for more robust estimation against outliers
        # Consider last N segments, e.g., last 5
        relevant_stats = self.segment_download_stats[-5:]
        if not relevant_stats: return self.estimated_bandwidth_bps

        sum_of_time_per_byte = 0
        total_bytes_for_hm = 0
        for stat in relevant_stats:
            if stat['size'] > 0:
                sum_of_time_per_byte += stat['duration'] / stat['size']
                total_bytes_for_hm += stat['size']
        
        if total_bytes_for_hm == 0 or sum_of_time_per_byte == 0:
            # Fallback to simple average if harmonic mean calculation is problematic
            total_bytes_simple = sum(s['size'] for s in relevant_stats)
            total_time_simple = sum(s['duration'] for s in relevant_stats)
            if total_time_simple == 0: return self.estimated_bandwidth_bps
            self.estimated_bandwidth_bps = (total_bytes_simple * 8) / total_time_simple
        else:
            # Harmonic mean of rates = N / sum(1/rate_i) = N / sum(time_i / byte_i)
            # Average rate = total_bytes / total_time
            # Effective rate here related to total_bytes_for_hm / sum_of_time_weighted_by_bytes
            # Simplified: (total_bytes * 8) / total_time remains a common approach
            total_bytes = sum(s['size'] for s in relevant_stats)
            total_time = sum(s['duration'] for s in relevant_stats)
            if total_time == 0: return self.estimated_bandwidth_bps
            self.estimated_bandwidth_bps = (total_bytes * 8) / total_time

        logger.info(f"ABR: New estimated bandwidth: {self.estimated_bandwidth_bps / 1000:.0f} Kbps (based on last {len(relevant_stats)} segments)")
        return self.estimated_bandwidth_bps

    def _abr_decision_logic(self):
        if not self.available_streams or len(self.available_streams) <=1: # No decision if 0 or 1 stream
            return

        estimated_bw = self._estimate_bandwidth()
        if estimated_bw == 0: return 

        current_actual_bw_of_selected_stream = self.available_streams[self.current_stream_index_by_abr].get('bandwidth', 0)
        
        # Rate-based selection: Find the highest quality sustainable
        next_best_index = 0 # Default to lowest
        for i in range(len(self.available_streams) -1, -1, -1): # Highest to lowest
            stream_bw = self.available_streams[i].get('bandwidth', 0)
            if estimated_bw * self.safety_factor > stream_bw:
                next_best_index = i
                break
        
        if next_best_index != self.current_stream_index_by_abr:
            old_stream_info = self.available_streams[self.current_stream_index_by_abr]
            self.current_stream_index_by_abr = next_best_index
            new_stream_info = self.available_streams[self.current_stream_index_by_abr]
            self._update_current_abr_selected_url() # Update the global reflecting ABR algorithm's choice
            
            logger.info(f"ABR ALGORITHM DECISION: Switch from {old_stream_info['url'].split('/')[-2]} (idx {self.available_streams.index(old_stream_info)}, BW {old_stream_info.get('bandwidth',0)/1000:.0f} Kbps) "
                        f"to {new_stream_info['url'].split('/')[-2]} (idx {self.current_stream_index_by_abr}, BW {new_stream_info.get('bandwidth',0)/1000:.0f} Kbps). "
                        f"(Estimated Net BW: {estimated_bw/1000:.0f} Kbps)")
            # NOTE: This decision is by the Python ABR. HLS.js makes its own separate decision.
            # You would compare HLS.js behavior (seen in its logs/events) with this algorithm's ideal choice.

    def abr_loop(self):
        logger.info("ABR monitoring thread started (for stats and algorithm decisions).")
        # Initial decision after a short delay to gather first segment stats
        time.sleep(5) # Wait for a few segments to be downloaded by HLS.js
        while not self.stop_abr_event.is_set():
            try:
                self._abr_decision_logic()
            except Exception as e:
                logger.error(f"Error in ABR decision loop: {e}", exc_info=True)
            
            # Decision interval (e.g., every few seconds or after N segments)
            # HLS.js will be making its own decisions more frequently.
            # This loop is for *your* ABR algorithm's simulation.
            for _ in range(5): # Check stop event frequently during sleep
                if self.stop_abr_event.is_set(): break
                time.sleep(1)
        logger.info("ABR monitoring thread (Python algorithm) stopped.")

    def start(self):
        self.stop_abr_event.clear()
        self.abr_thread = threading.Thread(target=self.abr_loop, daemon=True, name="PythonABRLogicThread")
        self.abr_thread.start()

    def stop(self):
        if self.abr_thread and self.abr_thread.is_alive():
            logger.info("Stopping Python ABR monitoring thread...")
            self.stop_abr_event.set()
            self.abr_thread.join(timeout=2.0) # Adjusted timeout
            if self.abr_thread.is_alive():
                logger.warning("Python ABR monitoring thread did not stop cleanly.")
            else:
                logger.info("Python ABR monitoring thread stopped.")
        ABRManager.instance = None # Clear singleton


def main():
    abr_manager = None
    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)

    try:
        if not hasattr(AES, 'AES_KEY') or not AES.AES_KEY:
            logger.error("AES.AES_KEY not defined/empty in AES.py. Please define it."); return
        if not callable(getattr(AES, 'aes_decrypt_cbc', None)):
            logger.error("AES.aes_decrypt_cbc function not defined in AES.py."); return
        logger.info("AES module checks passed.")

        if not start_proxy_server(): 
            logger.error("Failed to start the local proxy server. Aborting.")
            return

        # For ABRManager initialization, fetch the original master M3U8
        master_m3u8_url_on_origin = f"http://{SERVER_HOST}:{SERVER_PORT}/{VIDEO_TO_STREAM_NAME}/master.m3u8"
        available_streams = fetch_master_m3u8_for_abr_init(master_m3u8_url_on_origin)
        
        if not available_streams:
            logger.error(f"Could not fetch or parse master M3U8 from {master_m3u8_url_on_origin} for ABR init. Aborting.")
            # Attempt to continue without ABR manager if streams couldn't be fetched
            # This allows proxy to still work if user wants to test non-ABR parts.
        else:
             abr_manager = ABRManager(available_streams)
             abr_manager.start() # Start the Python ABR algorithm thread

        # Open the HTML player page in the default web browser
        player_page_url = f"http://{LOCAL_PROXY_HOST}:{LOCAL_PROXY_PORT}/player.html"
        logger.info(f"Attempting to open player page in browser: {player_page_url}")
        webbrowser.open(player_page_url)

        logger.info("Client setup complete. Proxy is running. Player should open in browser.")
        logger.info("Press Ctrl+C to stop the client and proxy server.")
        
        while True: # Keep main thread alive until Ctrl+C
            time.sleep(1)

    except KeyboardInterrupt:
        logger.info("Application interrupted by user (Ctrl+C). Cleaning up...")
    except Exception as e:
        logger.error(f"An error occurred in main: {e}", exc_info=True)
    finally:
        logger.info("Main: Initiating cleanup...")
        if abr_manager:
            abr_manager.stop()
        stop_proxy_server()
        
        # Clean up download directory (optional, if it was used)
        # if os.path.exists(DOWNLOAD_DIR):
        #     try:
        #         for f in os.listdir(DOWNLOAD_DIR): os.remove(os.path.join(DOWNLOAD_DIR, f))
        #         os.rmdir(DOWNLOAD_DIR)
        #         logger.info(f"Cleaned up {DOWNLOAD_DIR}")
        #     except OSError as e_rem:
        #         logger.warning(f"Could not fully cleanup {DOWNLOAD_DIR}: {e_rem}")
        logger.info("Client application finished.")

if __name__ == "__main__":
    main()