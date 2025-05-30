import os
import time
import logging
import threading
import requests
from urllib.parse import urlparse, urljoin, quote, unquote, parse_qs
import http.server
import socketserver
import re
import webbrowser
import json # For WebSocket messages

# --- WebSocket and AsyncIO ---
import asyncio
import websockets

import AES # Your AES decryption module

# --- Configuration ---
SERVER_HOST = '127.0.0.1'
SERVER_PORT = 8081
LOCAL_PROXY_HOST = '127.0.0.1'
LOCAL_PROXY_PORT = 8082
WEBSOCKET_PORT = 8083 # Port for WebSocket server
DOWNLOAD_DIR = "download_cache"
SOCKET_TIMEOUT_SECONDS = 10
VIDEO_TO_STREAM_NAME = "bbb_sunflower"

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger('HLSJS_Client_With_ABR_Control')

# --- Local Proxy Server Globals ---
g_local_proxy_server_instance = None
g_proxy_runner_thread = None

# --- WebSocket Server Globals ---
g_connected_websocket_clients = set()
g_websocket_server_thread = None
g_asyncio_loop_for_websocket = None # Will store the asyncio loop for thread-safe calls

# --- ABR State ---
abr_lock = threading.Lock()
current_abr_algorithm_selected_media_m3u8_url_on_server = None # Still useful for logging

# --- HTML Content (Placeholder for WebSocket Client JS) ---
HTML_PLAYER_CONTENT = f"""
<!DOCTYPE html>
<html>
<head>
    <title>HLS.js Player (Python ABR Controlled)</title>
    <script src="https://cdn.jsdelivr.net/npm/hls.js@latest"></script>
    <style> /* ... Your CSS ... */ </style>
</head>
<body>
    <h1>HLS.js Streaming Client (Python ABR Controlled)</h1>
    <div id="videoContainer">
        <video id="videoPlayer" controls></video>
    </div>
    <div class="controls">
        <p>Master Playlist URL: <span id="masterUrlSpan"></span></p>
        <p>ABR Control: Python Backend</p>
    </div>

    <script>
        document.addEventListener('DOMContentLoaded', function () {{
            const video = document.getElementById('videoPlayer');
            const masterM3u8Url = `http://{LOCAL_PROXY_HOST}:{LOCAL_PROXY_PORT}/{VIDEO_TO_STREAM_NAME}/master.m3u8`;
            document.getElementById('masterUrlSpan').textContent = masterM3u8Url;
            let hlsInstance = null;

            if (Hls.isSupported()) {{
                hlsInstance = new Hls({{
                    // debug: true,
                }});
                hlsInstance.loadSource(masterM3u8Url);
                hlsInstance.attachMedia(video);

                hlsInstance.on(Hls.Events.MANIFEST_PARSED, function (event, data) {{
                    console.log("Manifest parsed. Levels available:", hlsInstance.levels);
                    video.play();
                    // ** IMPORTANT: Disable HLS.js internal ABR **
                    if (hlsInstance.levels && hlsInstance.levels.length > 1) {{
                        hlsInstance.currentLevel = 0; // Start at lowest quality (index 0)
                        hlsInstance.autoLevelCapping = -1; // Remove any HLS.js internal capping
                        hlsInstance.autoLevelEnabled = false; // Disable internal ABR
                        console.log("HLS.js auto ABR disabled. Python ABR will control levels.");
                    }}
                }});

                hlsInstance.on(Hls.Events.LEVEL_SWITCHED, function(event, data) {{
                    console.log(`HLS.js ACTUALLY switched to level: ${{data.level}}, Bitrate: ${{hlsInstance.levels[data.level].bitrate/1000}} Kbps`);
                }});
                
                hlsInstance.on(Hls.Events.ERROR, function (event, data) {{ /* ... error handling ... */ }});

            }} else if (video.canPlayType('application/vnd.apple.mpegurl')) {{
                video.src = masterM3u8Url;
                video.addEventListener('loadedmetadata', function () {{ video.play(); }});
            }} else {{
                alert("HLS is not supported in your browser.");
            }}

            // WebSocket Client Logic
            const ws = new WebSocket(`ws://{LOCAL_PROXY_HOST}:{WEBSOCKET_PORT}`);

            ws.onopen = function() {{
                console.log("WebSocket connection established with Python backend.");
            }};

            ws.onmessage = function(event) {{
                try {{
                    const message = JSON.parse(event.data);
                    if (message.type === "SET_LEVEL" && hlsInstance) {{
                        const newLevelIndex = parseInt(message.levelIndex);
                        if (hlsInstance.levels && newLevelIndex >= 0 && newLevelIndex < hlsInstance.levels.length) {{
                            if (hlsInstance.currentLevel !== newLevelIndex || hlsInstance.nextLevel !== newLevelIndex) {{ // Avoid redundant sets
                                console.log(`Python ABR COMMAND: Switch to level index: ${{newLevelIndex}} (current: ${{hlsInstance.currentLevel}}, next: ${{hlsInstance.nextLevel}})`);
                                // Using nextLevel for smoother transition (switches after current segment)
                                // If HLS.js is already trying to switch to this level, this might not be necessary
                                // but it reinforces the decision from Python.
                                hlsInstance.nextLevel = newLevelIndex; 
                            }}
                        }} else {{
                            console.warn("Invalid level index received from Python ABR:", newLevelIndex);
                        }}
                    }}
                }} catch (e) {{
                    console.error("Error processing WebSocket message:", e, "Raw data:", event.data);
                }}
            }};

            ws.onclose = function() {{
                console.log("WebSocket connection closed.");
            }};

            ws.onerror = function(error) {{
                console.error("WebSocket error:", error);
            }};
        }});
    </script>
</body>
</html>
""".replace("{LOCAL_PROXY_HOST}", LOCAL_PROXY_HOST) \
 .replace("{LOCAL_PROXY_PORT}", str(LOCAL_PROXY_PORT)) \
 .replace("{WEBSOCKET_PORT}", str(WEBSOCKET_PORT)) \
 .replace("{VIDEO_TO_STREAM_NAME}", VIDEO_TO_STREAM_NAME)


# --- WebSocket Server Functions ---
async def handle_websocket_client(websocket): # Remove 'path' from arguments
    global g_connected_websocket_clients
    
    client_identifier = getattr(websocket, 'path', None)
    if client_identifier is None:
        # websocket.remote_address is a tuple (host, port)
        client_identifier = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
            
    logger.info(f"WebSocket client connected: {client_identifier}")
    g_connected_websocket_clients.add(websocket)
    try:
        async for message in websocket: 
            logger.info(f"Received message from WebSocket client {client_identifier} (not expected for ABR control): {message}")
    except websockets.exceptions.ConnectionClosedOK:
        logger.info(f"WebSocket client {client_identifier} disconnected gracefully.")
    except websockets.exceptions.ConnectionClosedError as e:
        logger.warning(f"WebSocket client {client_identifier} connection closed with error: {e}")
    except Exception as e:
        logger.error(f"Error in WebSocket client handler for {client_identifier}: {e}", exc_info=True)
    finally:
        g_connected_websocket_clients.remove(websocket)
        logger.info(f"WebSocket client {client_identifier} removed from connected set.")

async def run_websocket_server_async():
    global g_asyncio_loop_for_websocket
    g_asyncio_loop_for_websocket = asyncio.get_event_loop()
    logger.info(f"Starting WebSocket server on ws://{LOCAL_PROXY_HOST}:{WEBSOCKET_PORT}")
    async with websockets.serve(handle_websocket_client, LOCAL_PROXY_HOST, WEBSOCKET_PORT):
        await asyncio.Future() # Run forever

def start_websocket_server_in_thread():
    global g_websocket_server_thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(run_websocket_server_async())
    except KeyboardInterrupt: # Allow Ctrl+C to stop this thread's loop if it's the main focus
        logger.info("WebSocket server thread received KeyboardInterrupt.")
    finally:
        loop.close()
        logger.info("WebSocket server asyncio loop closed.")


async def broadcast_message_async(message_str):
    if g_connected_websocket_clients:
        # Create a list of tasks for sending messages to avoid issues if the set changes during iteration
        clients_to_send_to = list(g_connected_websocket_clients)
        tasks = [client.send(message_str) for client in clients_to_send_to]
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    client_repr = str(clients_to_send_to[i].remote_address) if hasattr(clients_to_send_to[i], 'remote_address') else "Unknown Client"
                    logger.error(f"Error sending message to WebSocket client {client_repr}: {result}")


def schedule_abr_broadcast(level_index):
    if g_asyncio_loop_for_websocket and g_asyncio_loop_for_websocket.is_running():
        message = json.dumps({"type": "SET_LEVEL", "levelIndex": level_index})
        # Schedule the async broadcast function to run in the WebSocket's asyncio loop
        asyncio.run_coroutine_threadsafe(broadcast_message_async(message), g_asyncio_loop_for_websocket)
    else:
        logger.warning("Cannot schedule ABR broadcast: WebSocket asyncio loop not available or not running.")

class DecryptionProxyHandler(http.server.BaseHTTPRequestHandler):
    # ... (do_GET, _rewrite_master_playlist, _rewrite_media_playlist remain largely the same) ...
    # Ensure _rewrite_master_playlist provides ALL variants to HLS.js
    # so HLS.js knows all available levels and their original order.

    def do_GET(self):
        log_adapter = logging.LoggerAdapter(logger, {'path': self.path})
        request_log_tag = f"[ProxyRequest URI: {self.path}]"
        parsed_url = urlparse(self.path)
        path_components = parsed_url.path.strip('/').split('/')

        try:
            if parsed_url.path == '/' or parsed_url.path == '/player.html':
                log_adapter.info(f"{request_log_tag} Serving player.html")
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(HTML_PLAYER_CONTENT.encode('utf-8'))
                return

            if len(path_components) == 2 and path_components[0] == VIDEO_TO_STREAM_NAME and path_components[1] == "master.m3u8":
                video_name_from_url = path_components[0]
                original_master_m3u8_url = f"http://{SERVER_HOST}:{SERVER_PORT}/{video_name_from_url}/master.m3u8"
                log_adapter.info(f"{request_log_tag} Request for master M3U8. Fetching from: {original_master_m3u8_url}")
                try:
                    response = requests.get(original_master_m3u8_url, timeout=SOCKET_TIMEOUT_SECONDS)
                    response.raise_for_status()
                    master_content = response.text
                    # IMPORTANT: This rewrite MUST ensure all original variants are present,
                    # so HLS.js knows all level indices. The URLs to media playlists are proxied.
                    modified_master_content = self._rewrite_master_playlist(master_content, original_master_m3u8_url)
                    
                    self.send_response(200)
                    self.send_header('Content-type', 'application/vnd.apple.mpegurl')
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(modified_master_content.encode('utf-8'))
                except requests.exceptions.RequestException as e:
                    self.send_error(502, f"Bad Gateway: Could not fetch master M3U8: {e}")
                return

            if len(path_components) == 3 and path_components[0] == VIDEO_TO_STREAM_NAME and path_components[2].endswith(".m3u8"):
                # ... (media playlist serving logic - same as before, rewrites TS to /decrypt_segment)
                video_name_from_url = path_components[0]
                quality_dir_from_url = path_components[1]
                playlist_filename_from_url = path_components[2]
                original_media_m3u8_url = f"http://{SERVER_HOST}:{SERVER_PORT}/{video_name_from_url}/{quality_dir_from_url}/{playlist_filename_from_url}"
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
                except requests.exceptions.RequestException as e:
                    self.send_error(502, f"Bad Gateway: Could not fetch media M3U8: {e}")
                return

            if parsed_url.path == '/decrypt_segment':
                # ... (TS segment decryption logic - same as before, calls ABRManager.add_segment_download_stat)
                query_params = parse_qs(parsed_url.query)
                original_ts_url_on_server_encoded = query_params.get('url', [None])[0]
                if not original_ts_url_on_server_encoded:
                    self.send_error(400, "Bad Request: Missing 'url' parameter"); return
                original_ts_url_on_server = unquote(original_ts_url_on_server_encoded)
                log_adapter.info(f"{request_log_tag} - Proxy serving TS segment, original URL: {original_ts_url_on_server}")
                try:
                    fetch_start_time = time.time()
                    response_ts = requests.get(original_ts_url_on_server, timeout=SOCKET_TIMEOUT_SECONDS, stream=True)
                    response_ts.raise_for_status()
                    encrypted_data = response_ts.content
                    fetch_end_time = time.time()
                    if ABRManager.instance:
                        ABRManager.instance.add_segment_download_stat(original_ts_url_on_server, len(encrypted_data), fetch_end_time - fetch_start_time)
                    if not encrypted_data: self.send_error(502, "Bad Gateway: Empty TS content"); return
                    decrypted_data = AES.aes_decrypt_cbc(encrypted_data, AES.AES_KEY)
                    self.send_response(200)
                    self.send_header('Content-type', 'video/MP2T')
                    self.send_header('Content-Length', str(len(decrypted_data)))
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(decrypted_data)
                except requests.exceptions.RequestException as e_req:
                    if ABRManager.instance: ABRManager.instance.report_download_error(original_ts_url_on_server)
                    self.send_error(502, f"Bad Gateway: Could not fetch TS: {e_req}")
                except Exception as e_dec:
                    self.send_error(500, f"Internal Server Error: Decryption failed: {e_dec}")
                return
            
            self.send_error(404, "Not Found")
        except Exception as e:
            # ... error handling ...
            pass

    def _rewrite_master_playlist(self, master_content, original_master_url):
        # This function should ensure all variants are present and their media playlist URLs
        # are rewritten to point to this proxy.
        # The logic should be similar to your previous version that passed all variants.
        log_adapter = logging.LoggerAdapter(logger, {})
        lines = master_content.splitlines()
        modified_lines = []
        for i in range(len(lines)):
            line = lines[i].strip()
            if line.startswith("#EXT-X-STREAM-INF:"):
                modified_lines.append(line) 
                if i + 1 < len(lines):
                    media_playlist_relative_or_absolute_url_in_master = lines[i+1].strip()
                    original_media_playlist_absolute_url = urljoin(original_master_url, media_playlist_relative_or_absolute_url_in_master)
                    parsed_original_media_playlist_url = urlparse(original_media_playlist_absolute_url)
                    proxied_media_playlist_url = f"http://{LOCAL_PROXY_HOST}:{LOCAL_PROXY_PORT}{parsed_original_media_playlist_url.path}"
                    modified_lines.append(proxied_media_playlist_url)
                    i += 1 
            else:
                modified_lines.append(line)
        return "\n".join(modified_lines)


    def _rewrite_media_playlist(self, media_content, original_media_url):
        # This function rewrites TS segment URLs to point to /decrypt_segment
        # Logic is same as your previous version.
        log_adapter = logging.LoggerAdapter(logger, {})
        lines = media_content.splitlines()
        modified_lines = []
        media_m3u8_base_url = urljoin(original_media_url, '.')
        for line in lines:
            line_stripped = line.strip()
            if line_stripped and not line_stripped.startswith("#") and \
               (line_stripped.endswith(".ts") or ".ts?" in line_stripped):
                original_segment_relative_url = line_stripped
                original_segment_absolute_url_on_server = urljoin(media_m3u8_base_url, original_segment_relative_url)
                encoded_original_url = quote(original_segment_absolute_url_on_server, safe='')
                rewritten_segment_url = f"http://{LOCAL_PROXY_HOST}:{LOCAL_PROXY_PORT}/decrypt_segment?url={encoded_original_url}"
                modified_lines.append(rewritten_segment_url)
            else:
                modified_lines.append(line)
        return "\n".join(modified_lines)

    def log_message(self, format, *args):
        logger.debug(f"Proxy HTTP Log: {self.address_string()} - {args[0]} {args[1]}")


class ThreadingLocalProxyServer(socketserver.ThreadingMixIn, http.server.HTTPServer): # Same
    daemon_threads = True
    allow_reuse_address = True

def _run_proxy_server_target(): # Same
    global g_local_proxy_server_instance
    try:
        g_local_proxy_server_instance = ThreadingLocalProxyServer(
            (LOCAL_PROXY_HOST, LOCAL_PROXY_PORT), DecryptionProxyHandler)
        logger.info(f"PROXY_THREAD: Local proxy server starting on http://{LOCAL_PROXY_HOST}:{LOCAL_PROXY_PORT}")
        g_local_proxy_server_instance.serve_forever() 
    except Exception as e: logger.error(f"PROXY_THREAD: Error: {e}", exc_info=True); g_local_proxy_server_instance = None
    finally: logger.info("PROXY_THREAD: Proxy server loop finished.")


def start_proxy_server(): # Same, ensure it returns True/False
    # ... (same as your last working version) ...
    global g_proxy_runner_thread, g_local_proxy_server_instance
    if g_proxy_runner_thread and g_proxy_runner_thread.is_alive(): return bool(g_local_proxy_server_instance) 
    g_local_proxy_server_instance = None 
    g_proxy_runner_thread = threading.Thread(target=_run_proxy_server_target, daemon=True, name="ProxyServerThread")
    g_proxy_runner_thread.start()
    time.sleep(0.5) 
    return bool(g_proxy_runner_thread.is_alive() and g_local_proxy_server_instance)


def stop_proxy_server(): # Same
    # ... (same as your last working version) ...
    global g_local_proxy_server_instance, g_proxy_runner_thread
    if g_local_proxy_server_instance:
        logger.info("PROXY_MAIN: Stopping proxy server instance...")
        g_local_proxy_server_instance.shutdown()
        if g_proxy_runner_thread: g_proxy_runner_thread.join(timeout=2.0)
        g_local_proxy_server_instance.server_close()
        g_local_proxy_server_instance = None
        g_proxy_runner_thread = None
        logger.info("PROXY_MAIN: Proxy server stopped.")


# --- ABRManager (Modified to broadcast decisions) ---
class ABRManager:
    instance = None 

    def __init__(self, available_streams_from_master):
        ABRManager.instance = self
        self.available_streams = sorted(
            [s for s in available_streams_from_master if s.get('bandwidth') is not None], 
            key=lambda s: s['bandwidth']
        )
        if not self.available_streams: # Add a dummy to prevent crashes if all lack bandwidth
             self.available_streams = [{'url': 'dummy', 'bandwidth': 0, 'resolution': 'N/A', 'attributes_str': ''}]
        
        # current_stream_index_by_abr is the index into self.available_streams
        # This index will be sent to HLS.js
        self.current_stream_index_by_abr = 0 
        self.segment_download_stats = [] 
        self.max_stats_history = 20 
        self.estimated_bandwidth_bps = 0
        self.safety_factor = 0.8
        self.abr_thread = None
        self.stop_abr_event = threading.Event()
        
        logger.info(f"ABRManager initialized. Available streams (sorted by bandwidth for indexing):")
        for i, s in enumerate(self.available_streams):
            logger.info(f"  Level Index [{i}] BW: {s.get('bandwidth', 'N/A')}, Res: {s.get('resolution', 'N/A')}, URL: {s['url']}")
        self._update_current_abr_selected_url_logging() # For logging selected URL
        # Send initial decision (e.g., lowest quality)
        schedule_abr_broadcast(self.current_stream_index_by_abr)


    def _update_current_abr_selected_url_logging(self): # For logging only
        global current_abr_algorithm_selected_media_m3u8_url_on_server
        with abr_lock:
            if self.available_streams and 0 <= self.current_stream_index_by_abr < len(self.available_streams):
                current_abr_algorithm_selected_media_m3u8_url_on_server = self.available_streams[self.current_stream_index_by_abr]['url']

    def add_segment_download_stat(self, url, size_bytes, duration_seconds): # Same
        if duration_seconds > 0.001: 
            self.segment_download_stats.append({'url': url, 'size': size_bytes, 'duration': duration_seconds})
            if len(self.segment_download_stats) > self.max_stats_history:
                self.segment_download_stats.pop(0)

    def report_download_error(self, url):  # Same
        logger.warning(f"ABR: Reported download error for segment from URL: {url}")

    def _estimate_bandwidth(self): # Same
        # ... (your preferred bandwidth estimation logic) ...
        if not self.segment_download_stats: return self.estimated_bandwidth_bps
        relevant_stats = self.segment_download_stats[-5:]
        if not relevant_stats: return self.estimated_bandwidth_bps
        total_bytes = sum(s['size'] for s in relevant_stats)
        total_time = sum(s['duration'] for s in relevant_stats)
        if total_time == 0: return self.estimated_bandwidth_bps
        self.estimated_bandwidth_bps = (total_bytes * 8) / total_time
        logger.info(f"ABR Python Algo: Estimated BW: {self.estimated_bandwidth_bps / 1000:.0f} Kbps")
        return self.estimated_bandwidth_bps

    def _abr_decision_logic(self):
        if not self.available_streams or len(self.available_streams) <=1: return

        estimated_bw = self._estimate_bandwidth()
        if estimated_bw == 0 and not self.segment_download_stats : # No data yet, stick to initial
            logger.info("ABR Python Algo: No stats yet, sticking to initial level.")
            # schedule_abr_broadcast(self.current_stream_index_by_abr) # Re-affirm if needed
            return

        next_best_index = 0 
        for i in range(len(self.available_streams) -1, -1, -1):
            stream_bw = self.available_streams[i].get('bandwidth', 0)
            if estimated_bw * self.safety_factor > stream_bw:
                next_best_index = i
                break
        
        if next_best_index != self.current_stream_index_by_abr:
            old_stream_info = self.available_streams[self.current_stream_index_by_abr]
            self.current_stream_index_by_abr = next_best_index
            new_stream_info = self.available_streams[self.current_stream_index_by_abr]
            self._update_current_abr_selected_url_logging()
            
            logger.info(f"ABR PYTHON ALGO DECISION: Switch from level index {self.available_streams.index(old_stream_info)} (BW {old_stream_info.get('bandwidth',0)/1000:.0f} Kbps) "
                        f"to level index {self.current_stream_index_by_abr} (BW {new_stream_info.get('bandwidth',0)/1000:.0f} Kbps).")
            
            # **BROADCAST DECISION TO WEBSOCKET CLIENTS**
            schedule_abr_broadcast(self.current_stream_index_by_abr)
        # else:
            # logger.debug(f"ABR Python Algo: No change in decision, current index {self.current_stream_index_by_abr}")


    def abr_loop(self):
        logger.info("ABR Python Algo monitoring thread started.")
        # --- TEST: Hardcode level switch after 10 seconds ---
        # time.sleep(10) 
        # hardcoded_level_index = 1 # 假设你想切换到 level index 1 (e.g., 720p)
        # logger.info(f"TESTING: Hardcoding switch to level index: {hardcoded_level_index}")
        # schedule_abr_broadcast(hardcoded_level_index)
        # --- END TEST ---
        time.sleep(5) 
        while not self.stop_abr_event.is_set():
            try:
                self._abr_decision_logic()
            except Exception as e:
                logger.error(f"Error in ABR decision loop: {e}", exc_info=True)
            
            for _ in range(3): # Check stop event frequently during sleep (e.g. decision every 3s)
                if self.stop_abr_event.is_set(): break
                time.sleep(1)
        logger.info("ABR Python Algo monitoring thread stopped.")

    def start(self): # Same
        self.stop_abr_event.clear()
        self.abr_thread = threading.Thread(target=self.abr_loop, daemon=True, name="PythonABRLogicThread")
        self.abr_thread.start()

    def stop(self): # Same
        if self.abr_thread and self.abr_thread.is_alive():
            self.stop_abr_event.set()
            self.abr_thread.join(timeout=2.0)
        ABRManager.instance = None


# Function to fetch initial master m3u8 for ABRManager setup
def fetch_master_m3u8_for_abr_init(master_m3u8_url_on_server):
    # ... (Same as your previous version - this gets original URLs and attributes for ABR init) ...
    logger.info(f"ABR_INIT: Fetching master M3U8 from: {master_m3u8_url_on_server}")
    try:
        response = requests.get(master_m3u8_url_on_server, timeout=SOCKET_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.exceptions.RequestException as e: return None
    content = response.text; lines = content.splitlines(); available_streams = []
    master_m3u8_base_url = urljoin(master_m3u8_url_on_server, '.')
    for i in range(len(lines)):
        line = lines[i].strip()
        if line.startswith("#EXT-X-STREAM-INF:"):
            attributes_str = line.split(":", 1)[1]
            attributes = parse_m3u8_attributes(attributes_str) # Use your parse_m3u8_attributes
            if i + 1 < len(lines):
                media_playlist_relative_url = lines[i+1].strip()
                media_playlist_absolute_url_on_origin = urljoin(master_m3u8_base_url, media_playlist_relative_url)
                available_streams.append({
                    'url': media_playlist_absolute_url_on_origin, 
                    'bandwidth': attributes.get('BANDWIDTH'),
                    'resolution': attributes.get('RESOLUTION'),
                    'codecs': attributes.get('CODECS'),
                    'attributes_str': attributes_str
                })
    return available_streams if available_streams else None

def parse_m3u8_attributes(attr_string): # Ensure this is defined
    attributes = {}
    try:
        for match in re.finditer(r'([A-Z0-9-]+)=("([^"]*)"|([^,"]*))', attr_string):
            key = match.group(1)
            value = match.group(3) if match.group(3) is not None else match.group(4)
            if value.isdigit(): attributes[key] = int(value)
            else: attributes[key] = value
    except Exception as e: logger.error(f"Error parsing M3U8 attributes: {e}")
    return attributes


def main():
    global g_websocket_server_thread, g_asyncio_loop_for_websocket # Ensure global scope for loop if accessed elsewhere for shutdown
    abr_manager = None
    # ... (AES checks, DOWNLOAD_DIR creation) ...
    if not hasattr(AES, 'AES_KEY') or not AES.AES_KEY: logger.error("AES_KEY not defined!"); return
    if not callable(getattr(AES, 'aes_decrypt_cbc', None)): logger.error("aes_decrypt_cbc not defined!"); return

    # 1. Start the HTTP Proxy Server
    if not start_proxy_server(): 
        logger.error("Failed to start the local HTTP proxy server. Aborting.")
        return

    # 2. Start the WebSocket Server in a separate thread
    logger.info("Starting WebSocket server thread...")
    g_websocket_server_thread = threading.Thread(target=start_websocket_server_in_thread, daemon=True, name="WebSocketServerThread")
    g_websocket_server_thread.start()
    
    # Wait a moment for the WebSocket server's asyncio loop to be assigned
    time.sleep(1) 
    if not (g_asyncio_loop_for_websocket and g_asyncio_loop_for_websocket.is_running()):
        logger.error("WebSocket server asyncio loop did not start correctly. ABR control might fail.")
        # Decide if you want to abort or continue without WebSocket ABR control
        # For now, we'll log and continue, ABRManager might log errors when trying to broadcast

    # 3. Initialize ABR Manager (after proxy and WebSocket are ready)
    master_m3u8_url_on_origin = f"http://{SERVER_HOST}:{SERVER_PORT}/{VIDEO_TO_STREAM_NAME}/master.m3u8"
    available_streams = fetch_master_m3u8_for_abr_init(master_m3u8_url_on_origin)
    if available_streams:
        abr_manager = ABRManager(available_streams) # ABRManager constructor now sends initial level
        abr_manager.start() # Start ABR decision loop
    else:
        logger.error(f"Could not fetch master M3U8 for ABR init from {master_m3u8_url_on_origin}. ABR will not function.")

    # 4. Open browser
    player_page_url = f"http://{LOCAL_PROXY_HOST}:{LOCAL_PROXY_PORT}/player.html"
    logger.info(f"Attempting to open player page in browser: {player_page_url}")
    webbrowser.open(player_page_url)

    logger.info("Client setup complete. Press Ctrl+C to stop.")
    try:
        while True: time.sleep(1) # Keep main thread alive
    except KeyboardInterrupt:
        logger.info("Application interrupted by user (Ctrl+C).")
    finally:
        logger.info("Main: Initiating cleanup...")
        if abr_manager: abr_manager.stop()
        
        # Stop WebSocket server's asyncio loop (if it's running)
        if g_asyncio_loop_for_websocket and g_asyncio_loop_for_websocket.is_running():
            logger.info("Stopping WebSocket server asyncio loop...")
            g_asyncio_loop_for_websocket.call_soon_threadsafe(g_asyncio_loop_for_websocket.stop)
            # The run_websocket_server_async will then exit its await asyncio.Future()
        
        if g_websocket_server_thread and g_websocket_server_thread.is_alive():
            logger.info("Waiting for WebSocket server thread to join...")
            g_websocket_server_thread.join(timeout=2.0)
            if g_websocket_server_thread.is_alive(): logger.warning("WebSocket server thread did not join cleanly.")
        
        stop_proxy_server() # Stop HTTP proxy
        logger.info("Client application finished.")

if __name__ == "__main__":
    main()