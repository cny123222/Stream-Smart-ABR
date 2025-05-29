import os
import time
import logging
import vlc
import threading
import requests # For HTTP requests
import tempfile # For temporary M3U8 file
from urllib.parse import urlparse, urljoin, quote, unquote # For URL manipulation
import urllib.parse # For parsing query parameters
import http.server
import socketserver
import re

import AES

# --- Configuration ---
SERVER_HOST = '127.0.0.1' # Your HLS server host
SERVER_PORT = 8081        # Your HLS server port
LOCAL_PROXY_HOST = '127.0.0.1'
LOCAL_PROXY_PORT = 8082   # Port for the local decryption proxy
DOWNLOAD_DIR = "download_temp_m3u8" # Directory for storing temp modified m3u8
SOCKET_TIMEOUT_SECONDS = 10

VIDEO_TO_STREAM_NAME = "bbb_sunflower" # Just the video name, master.m3u8 will be appended

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger('HLSClientVLC')

# --- VLC Player Setup & State ---
vlc_instance = None
media_player = None
player_event_manager = None
player_reached_end = threading.Event()

# --- Local Decryption Proxy ---
g_local_proxy_server_instance = None
g_proxy_runner_thread = None

# --- ABR State (Shared between ABR logic and Proxy if needed) ---
abr_lock = threading.Lock()
current_selected_media_m3u8_url_on_server = None # Full URL on the original server for the current quality
current_segment_urls_relative_to_media_m3u8 = [] # List of relative segment URLs for the current media m3u8

# DecryptionProxyHandler and ThreadingLocalProxyServer classes remain the same as your last provided version
# _run_proxy_server_target, start_proxy_server, stop_proxy_server remain the same

class DecryptionProxyHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        log_adapter = logging.LoggerAdapter(logger, {}) 
        request_log_tag = f"[ProxyRequest URI: {self.path}]"

        try:
            parsed_url_from_vlc = urlparse(self.path) # path is like /proxy?url=ENCODED_ORIGINAL_TS_URL
            query_params = urllib.parse.parse_qs(parsed_url_from_vlc.query)
            original_ts_url_on_server_encoded = query_params.get('url', [None])[0]

            if not original_ts_url_on_server_encoded:
                log_adapter.error(f"{request_log_tag} Proxy GET request missing 'url' parameter.")
                self.send_error(400, "Bad Request: Missing 'url' parameter")
                return
            
            original_ts_url_on_server = unquote(original_ts_url_on_server_encoded)
            log_adapter.info(f"{request_log_tag} VLC requested (based on its M3U8): {original_ts_url_on_server}")

            # --- ABR逻辑介入，决定实际要获取的分片URL ---
            actual_url_to_fetch_from_server = original_ts_url_on_server # 默认值

            try:
                # 1. 从VLC请求的URL中解析出视频名和分片序号/基本名
                # original_ts_url_on_server 格式: http://{S_HOST}:{S_PORT}/{VID_NAME}/{ORIG_QUAL}/{VID_NAME}-{ORIG_QUAL}-{INDEX}.ts
                path_parts = urlparse(original_ts_url_on_server).path.strip('/').split('/')
                
                if len(path_parts) >= 3:
                    # path_parts[0] is video_name, path_parts[1] is original_quality_suffix
                    # path_parts[-1] is the full segment filename
                    requested_video_name_from_url = path_parts[0] 
                    segment_filename_from_vlc_m3u8 = path_parts[-1]

                    # 从分片文件名中提取纯粹的序号部分，例如 "00016"
                    # 假设分片名格式为: {video_name}-{any_quality_suffix}-{index_str}.ts
                    # 我们需要捕获 video_name 和 index_str
                    # 例如: bbb_sunflower-480p-1500k-00016.ts -> video_name="bbb_sunflower", index="00016"
                    
                    # 更通用的方式是找到最后一个'-'和'.ts'之间的部分作为序号
                    match_index = re.search(r'-(\d+)\.ts$', segment_filename_from_vlc_m3u8)
                    if match_index:
                        segment_index_str = match_index.group(1) # 例如 "00016" or "16" (取决于你的 %0xd)
                                                                # 需要确保你的 %05d 和这里的正则匹配
                        
                        # 2. 获取ABR当前选择的媒体M3U8 URL
                        with abr_lock:
                            abr_selected_media_m3u8_url = current_selected_media_m3u8_url_on_server
                        
                        if abr_selected_media_m3u8_url:
                            # 3. 从ABR选择的媒体M3U8 URL中解析出新的目标质量后缀和视频名
                            # 例如: http://127.0.0.1:8081/bbb_sunflower/720p-4000k/bbb_sunflower-720p-4000k.m3u8
                            abr_path_parts = urlparse(abr_selected_media_m3u8_url).path.strip('/').split('/')
                            if len(abr_path_parts) >= 3: # video_name/quality_suffix/playlist.m3u8
                                abr_video_name = abr_path_parts[0]
                                abr_selected_quality_suffix = abr_path_parts[1]

                                # 4. 构造新的分片文件名和完整URL
                                # 使用从VLC请求中解析出的视频名和序号，但使用ABR选择的质量后缀
                                new_segment_filename = f"{requested_video_name_from_url}-{abr_selected_quality_suffix}-{segment_index_str}.ts"
                                # 如果你的序号格式是固定的（比如%05d），确保segment_index_str符合
                                # segment_index_str_padded = segment_index_str.zfill(5) # 如果需要补零
                                # new_segment_filename = f"{requested_video_name_from_url}-{abr_selected_quality_suffix}-{segment_index_str_padded}.ts"


                                actual_url_to_fetch_from_server = \
                                    f"http://{SERVER_HOST}:{SERVER_PORT}/{requested_video_name_from_url}/{abr_selected_quality_suffix}/{new_segment_filename}"
                                
                                if actual_url_to_fetch_from_server != original_ts_url_on_server:
                                    log_adapter.info(f"{request_log_tag} ABR Override: Fetching '{new_segment_filename}' from quality '{abr_selected_quality_suffix}' "
                                                     f"instead of segment from '{path_parts[1]}'. Target: {actual_url_to_fetch_from_server}")
                                else:
                                    log_adapter.debug(f"{request_log_tag} ABR: Sticking with current quality segment: {actual_url_to_fetch_from_server}")
                            else:
                                log_adapter.warning(f"{request_log_tag} Could not parse ABR selected media M3U8 URL structure: {abr_selected_media_m3u8_url}")
                        else:
                            log_adapter.warning(f"{request_log_tag} No ABR selected media M3U8 URL available yet (current_selected_media_m3u8_url_on_server is None). Using VLC's requested URL.")
                    else:
                        log_adapter.warning(f"{request_log_tag} Could not parse segment index from filename: {segment_filename_from_vlc_m3u8}. Using VLC's requested URL.")
                else:
                    log_adapter.warning(f"{request_log_tag} Could not parse original TS URL structure: {original_ts_url_on_server}. Using VLC's requested URL.")
            except Exception as e_abr_url_logic:
                log_adapter.error(f"{request_log_tag} Error in ABR URL construction logic: {e_abr_url_logic}. Defaulting to VLC's requested URL.", exc_info=True)
                actual_url_to_fetch_from_server = original_ts_url_on_server # Fallback

            # --- 从最终确定的URL获取数据 (actual_url_to_fetch_from_server) ---
            log_adapter.info(f"{request_log_tag} Final URL to fetch from server: {actual_url_to_fetch_from_server}")
            
            try:
                fetch_start_time = time.time()
                response = requests.get(actual_url_to_fetch_from_server, timeout=SOCKET_TIMEOUT_SECONDS, stream=True)
                # ... (后续的 response.raise_for_status(), encrypted_data = response.content, ABRManager.add_segment_download_stat 调用不变) ...
                response.raise_for_status() 
                encrypted_data = response.content 
                fetch_end_time = time.time()
                
                if ABRManager.instance: 
                    download_duration = fetch_end_time - fetch_start_time
                    segment_size = len(encrypted_data)
                    # 传递 actual_url_to_fetch_from_server 给统计，因为它反映了实际下载的码率
                    ABRManager.instance.add_segment_download_stat(actual_url_to_fetch_from_server, segment_size, download_duration)

            except requests.exceptions.RequestException as e:
                log_adapter.error(f"{request_log_tag} Proxy failed to fetch segment {actual_url_to_fetch_from_server} from main server: {e}")
                if ABRManager.instance: ABRManager.instance.report_download_error(actual_url_to_fetch_from_server)
                self.send_error(502, f"Bad Gateway: Could not fetch from origin: {e}")
                return
            # ... (后续的解密和发送逻辑不变) ...
            if not encrypted_data: # Check after fetch
                log_adapter.warning(f"{request_log_tag} Proxy received empty content for {actual_url_to_fetch_from_server}")
                if ABRManager.instance: ABRManager.instance.report_download_error(actual_url_to_fetch_from_server)
                self.send_error(502, "Bad Gateway: Empty content from origin")
                return
            
            try:
                decrypted_data = AES.aes_decrypt_cbc(encrypted_data, AES.AES_KEY)
            except Exception as e:
                log_adapter.error(f"{request_log_tag} Proxy failed to decrypt segment from {actual_url_to_fetch_from_server}: {e}", exc_info=True)
                self.send_error(500, "Internal Server Error: Decryption failed")
                return

            log_adapter.info(f"{request_log_tag} Successfully decrypted segment from {actual_url_to_fetch_from_server}. Decrypted size: {len(decrypted_data)} bytes.")

            self.send_response(200)
            self.send_header('Content-type', 'video/MP2T')
            self.send_header('Content-Length', str(len(decrypted_data)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(decrypted_data)
            log_adapter.debug(f"{request_log_tag} Finished sending decrypted segment for {actual_url_to_fetch_from_server} to VLC.")

        except ConnectionResetError:
            log_adapter.warning(f"{request_log_tag} Connection reset by VLC.")
        except BrokenPipeError:
            log_adapter.warning(f"{request_log_tag} Broken pipe while writing to VLC.")
        except Exception as e:
            log_adapter.error(f"{request_log_tag} Error handling GET request: {e}", exc_info=True)
            if not self.wfile.closed:
                try: self.send_error(500, f"Internal Server Error: {e}")
                except Exception: pass

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
        logger.info(f"PROXY_THREAD: Local decryption proxy starting on http://{LOCAL_PROXY_HOST}:{LOCAL_PROXY_PORT}")
        current_thread_server_instance.serve_forever() 
    except OSError as e:
        logger.error(f"PROXY_THREAD: Could not start local decryption proxy (OSError): {e}")
        g_local_proxy_server_instance = None
    except Exception as e:
        logger.error(f"PROXY_THREAD: An unexpected error in proxy server run: {e}", exc_info=True)
        g_local_proxy_server_instance = None
    finally:
        logger.info(f"PROXY_THREAD: Local decryption proxy server loop ({threading.current_thread().name}) has finished.")

def start_proxy_server():
    global g_proxy_runner_thread, g_local_proxy_server_instance
    if g_proxy_runner_thread and g_proxy_runner_thread.is_alive():
        logger.warning("PROXY_MAIN: Proxy server thread is already running.")
        return bool(g_local_proxy_server_instance) 
    g_local_proxy_server_instance = None 
    g_proxy_runner_thread = threading.Thread(target=_run_proxy_server_target, daemon=True, name="ProxyServerThread")
    g_proxy_runner_thread.start()
    time.sleep(0.5) 
    if g_proxy_runner_thread.is_alive() and g_local_proxy_server_instance:
        logger.info("PROXY_MAIN: Proxy server thread started and server instance successfully created.")
        return True
    else:
        logger.error("PROXY_MAIN: Proxy server thread failed to start or server instance not created.")
        if g_proxy_runner_thread and g_proxy_runner_thread.is_alive():
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
                server_instance_to_stop.shutdown()
            if g_proxy_runner_thread and g_proxy_runner_thread.is_alive():
                g_proxy_runner_thread.join(timeout=3.0)
                if g_proxy_runner_thread.is_alive(): logger.warning("PROXY_MAIN: Proxy thread did not join cleanly.")
            if hasattr(server_instance_to_stop, 'server_close'):
                server_instance_to_stop.server_close()
        except Exception as e_shutdown:
            logger.error(f"PROXY_MAIN: Error during proxy shutdown: {e_shutdown}", exc_info=True)
        finally:
            g_local_proxy_server_instance = None 
            g_proxy_runner_thread = None
            logger.info("PROXY_MAIN: Proxy server stop sequence finished.")
    else:
        logger.info("PROXY_MAIN: Proxy server instance was None or not started.")

def parse_m3u8_attributes(attr_string):
    """Parses attributes from #EXT-X-STREAM-INF or similar tags."""
    attributes = {}
    try:
        # Regex to find KEY=VALUE or KEY="VALUE" pairs
        # Handles values that are quoted or unquoted
        # Corrected regex to handle simple non-string values like BANDWIDTH=12345
        for match in re.finditer(r'([A-Z0-9-]+)=("([^"]*)"|([^,"]*))', attr_string):
            key = match.group(1)
            # value is either in group 3 (quoted) or group 4 (unquoted)
            value = match.group(3) if match.group(3) is not None else match.group(4)
            if value.isdigit(): # Convert to int if it's all digits
                attributes[key] = int(value)
            else: # Otherwise, keep as string
                attributes[key] = value
    except Exception as e:
        logger.error(f"Error parsing M3U8 attributes string '{attr_string}': {e}")
    return attributes

def fetch_master_m3u8(master_m3u8_url_on_server):
    """Fetches and parses the master M3U8 playlist."""
    logger.info(f"Fetching master M3U8 from: {master_m3u8_url_on_server}")
    try:
        response = requests.get(master_m3u8_url_on_server, timeout=SOCKET_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch master M3U8 playlist {master_m3u8_url_on_server}: {e}")
        return None

    content = response.text
    lines = content.splitlines()
    
    available_streams = []
    # Base URL for resolving relative media playlist URLs in the master M3U8
    master_m3u8_base_url = urljoin(master_m3u8_url_on_server, '.')

    for i in range(len(lines)):
        line = lines[i].strip()
        if line.startswith("#EXT-X-STREAM-INF:"):
            attributes_str = line.split(":", 1)[1]
            attributes = parse_m3u8_attributes(attributes_str)
            if i + 1 < len(lines):
                media_playlist_relative_url = lines[i+1].strip()
                media_playlist_absolute_url = urljoin(master_m3u8_base_url, media_playlist_relative_url)
                stream_info = {
                    'url': media_playlist_absolute_url, # Absolute URL to the media M3U8 on the server
                    'bandwidth': attributes.get('BANDWIDTH'),
                    'resolution': attributes.get('RESOLUTION'),
                    'codecs': attributes.get('CODECS'),
                    'attributes_str': attributes_str # Store original for reference
                }
                available_streams.append(stream_info)
                logger.debug(f"Found stream in master M3U8: {stream_info}")
    
    if not available_streams:
        logger.warning(f"No #EXT-X-STREAM-INF tags found in master M3U8 at {master_m3u8_url_on_server}")
        return None
        
    return available_streams

def fetch_media_m3u8_and_rewrite(media_m3u8_url_on_server):
    """
    Fetches a specific media M3U8, rewrites its TS segment URLs to point to the local proxy,
    and returns the content of the modified M3U8 and list of original segment URLs.
    """
    global current_segment_urls_relative_to_media_m3u8 # For ABR to know what segments exist
    logger.info(f"Fetching media M3U8 from: {media_m3u8_url_on_server}")
    try:
        response = requests.get(media_m3u8_url_on_server, timeout=SOCKET_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch media M3U8 playlist {media_m3u8_url_on_server}: {e}")
        return None, []

    original_m3u8_content = response.text
    modified_lines = []
    # Base URL for resolving relative segment URLs within this media M3U8
    media_m3u8_base_url = urljoin(media_m3u8_url_on_server, '.')
    
    segment_urls_for_abr = []

    for line in original_m3u8_content.splitlines():
        line_stripped = line.strip()
        if line_stripped and not line_stripped.startswith("#") and \
           (line_stripped.endswith(".ts") or ".ts?" in line_stripped):
            original_segment_relative_url = line_stripped # URL as it appears in media M3U8
            segment_urls_for_abr.append(original_segment_relative_url)

            original_segment_absolute_url_on_server = urljoin(media_m3u8_base_url, original_segment_relative_url)
            
            # Encode the original absolute URL for the proxy's query string
            encoded_original_url = quote(original_segment_absolute_url_on_server, safe='')
            rewritten_segment_url = f"http://{LOCAL_PROXY_HOST}:{LOCAL_PROXY_PORT}/proxy?url={encoded_original_url}"
            modified_lines.append(rewritten_segment_url)
            logger.debug(f"Rewrote media segment URL: '{original_segment_relative_url}' -> '{rewritten_segment_url}' (orig_abs: {original_segment_absolute_url_on_server})")
        else:
            modified_lines.append(line)
    
    with abr_lock: # Update global list of segments for current quality (ABR might use this)
        current_segment_urls_relative_to_media_m3u8 = segment_urls_for_abr

    return "\n".join(modified_lines)

class ABRManager:
    instance = None # Singleton instance

    def __init__(self, available_streams_from_master):
        if ABRManager.instance is not None:
            raise Exception("ABRManager is a singleton!")
        ABRManager.instance = self
        
        self.available_streams = sorted(available_streams_from_master, key=lambda s: s.get('bandwidth', 0) or 0) # Sort by bandwidth
        self.current_stream_index = 0  # Start with the lowest quality
        self.segment_download_stats = [] # List to store (url, size_bytes, duration_seconds)
        self.max_stats_history = 20 # Keep history of last N segments for bandwidth estimation
        self.estimated_bandwidth_bps = 0
        
        # ABR algorithm parameters (example)
        self.safety_factor = 0.8
        self.abr_thread = None
        self.stop_abr_event = threading.Event()

        if not self.available_streams:
            logger.error("ABRManager initialized with no available streams!")
            return
        
        logger.info(f"ABRManager initialized. Available streams (sorted by bandwidth):")
        for i, s in enumerate(self.available_streams):
            logger.info(f"  [{i}] BW: {s.get('bandwidth', 'N/A')}, Res: {s.get('resolution', 'N/A')}, URL: {s['url']}")

    def get_initial_media_m3u8_url(self):
        if not self.available_streams:
            return None
        # Start with the lowest quality, or a pre-defined initial index
        self.current_stream_index = 0 # Or choose based on some initial probe
        selected_stream = self.available_streams[self.current_stream_index]
        logger.info(f"ABR: Initial stream selected: BW {selected_stream['bandwidth']}, URL {selected_stream['url']}")
        with abr_lock:
            global current_selected_media_m3u8_url_on_server
            current_selected_media_m3u8_url_on_server = selected_stream['url']
        return selected_stream['url']

    def add_segment_download_stat(self, url, size_bytes, duration_seconds):
        if duration_seconds > 0.001: # Avoid division by zero or tiny durations
            self.segment_download_stats.append({'url': url, 'size': size_bytes, 'duration': duration_seconds})
            if len(self.segment_download_stats) > self.max_stats_history:
                self.segment_download_stats.pop(0) # Keep history size limited
            # logger.debug(f"ABR: Added stat: {size_bytes} bytes in {duration_seconds:.3f}s for {url.split('/')[-1]}")

    def report_download_error(self, url):
        logger.warning(f"ABR: Reported download error for segment from URL related to {url}")
        # Could use this to penalize current bandwidth or trigger faster downshift

    def _estimate_bandwidth(self):
        if not self.segment_download_stats:
            return 0 # Not enough data
        
        # Example: Weighted moving average (more weight to recent segments)
        # Or Harmonic Mean for robustness
        total_bytes = 0
        total_time = 0
        # Using last 5 segments or all if fewer than 5
        relevant_stats = self.segment_download_stats[-5:]
        if not relevant_stats: return self.estimated_bandwidth_bps # Keep last estimate if no new data

        for stat in relevant_stats:
            total_bytes += stat['size']
            total_time += stat['duration']
        
        if total_time == 0: return self.estimated_bandwidth_bps # Avoid division by zero

        self.estimated_bandwidth_bps = (total_bytes * 8) / total_time # Convert to bits per second
        logger.info(f"ABR: New estimated bandwidth: {self.estimated_bandwidth_bps / 1000:.0f} Kbps (based on last {len(relevant_stats)} segments)")
        return self.estimated_bandwidth_bps

    def _abr_decision_logic(self):
        """This is where the core ABR algorithm would reside."""
        estimated_bw = self._estimate_bandwidth()
        if estimated_bw == 0: # Not enough data or error
            return # No change

        current_bw = self.available_streams[self.current_stream_index].get('bandwidth', 0) or 0
        
        # Simple rate-based selection with safety factor
        next_best_index = self.current_stream_index
        
        # Try to find the highest quality we can sustain
        for i in range(len(self.available_streams) - 1, -1, -1): # Iterate from highest to lowest
            stream_bw = self.available_streams[i].get('bandwidth', 0) or 0
            if estimated_bw * self.safety_factor > stream_bw:
                next_best_index = i
                break # Found the best fit from high down
        else: # If loop completed without break, means even lowest quality is too high (or list empty)
            next_best_index = 0 # Default to lowest if no suitable found (or if only one stream)

        if next_best_index != self.current_stream_index:
            # Implement stability: e.g., don't switch too often, or only if significant diff
            # For now, just switch if different
            old_stream_url = self.available_streams[self.current_stream_index]['url']
            self.current_stream_index = next_best_index
            new_stream_info = self.available_streams[self.current_stream_index]
            
            with abr_lock:
                global current_selected_media_m3u8_url_on_server
                current_selected_media_m3u8_url_on_server = new_stream_info['url']
            
            logger.info(f"ABR DECISION: Switch from {old_stream_url.split('/')[-2]} (idx {self.available_streams.index(next(s for s in self.available_streams if s['url']==old_stream_url))}) "
                        f"to {new_stream_info['url'].split('/')[-2]} (idx {self.current_stream_index}), "
                        f"Target BW: {new_stream_info['bandwidth']/1000:.0f} Kbps. (Estimated BW: {estimated_bw/1000:.0f} Kbps)")
            
            # How to make VLC switch?
            # Option 1: Stop current, fetch new media M3U8 for new_stream_info['url'], rewrite, play. (Not seamless)
            # Option 2: Proxy-assisted (more complex if VLC is playing one media M3U8).
            #    If the proxy is ABR-aware, it can start fetching segments from the new quality.
            #    The `DecryptionProxyHandler` would need to know the currently selected media M3U8 base URL
            #    from `current_selected_media_m3u8_url_on_server` and construct segment URLs accordingly.
            #    This requires segment names (e.g., segment-001.ts) to be consistent across qualities.
            #    VLC will continue requesting segments based on the M3U8 it *initially* loaded.
            #    So the proxy must be smart.
            #
            #    For now, this ABR logic just updates `current_selected_media_m3u8_url_on_server`.
            #    The proxy in its current form does not use this global to change source yet.
            #    It uses the `original_segment_url` from the query param, which comes from the M3U8 VLC has.
            #
            #    To make proxy-assisted ABR work:
            #    1. VLC loads an initial media M3U8 (e.g., lowest quality).
            #    2. ABR runs, changes `current_selected_media_m3u8_url_on_server`.
            #    3. Proxy's do_GET needs to:
            #       a. Get segment name from VLC's request (e.g., "segment-001.ts").
            #       b. Get base URL for the *current ABR-selected quality* from `current_selected_media_m3u8_url_on_server`.
            #       c. Construct the true target URL: urljoin(ABR_selected_media_m3u8_base_url, "segment-001.ts").
            #       d. Fetch, decrypt, serve.
            #    This change needs to be made in DecryptionProxyHandler's do_GET.
            
            # For now, we'll just log the decision. Actual switching mechanism needs implementation.
            # One way is to regenerate the temp M3U8 VLC is playing from, but that's very tricky with VLC.
            # The simplest (non-seamless) is to stop and restart VLC with the new media M3U8.
            # For this assignment, demonstrating the ABR *decision logic* and how the proxy *could* act on it is key.

    def abr_loop(self):
        logger.info("ABR monitoring thread started.")
        while not self.stop_abr_event.is_set():
            try:
                self._abr_decision_logic()
            except Exception as e:
                logger.error(f"Error in ABR decision loop: {e}", exc_info=True)
            
            # Wait for a certain interval before next decision
            # This interval should be related to segment duration or a fixed time
            # For example, check every segment or every few seconds
            time.sleep(3) # Example: make a decision every 3 seconds
        logger.info("ABR monitoring thread stopped.")

    def start(self):
        self.stop_abr_event.clear()
        self.abr_thread = threading.Thread(target=self.abr_loop, daemon=True, name="ABRLogicThread")
        self.abr_thread.start()

    def stop(self):
        if self.abr_thread and self.abr_thread.is_alive():
            logger.info("Stopping ABR monitoring thread...")
            self.stop_abr_event.set()
            self.abr_thread.join(timeout=2.0)
            if self.abr_thread.is_alive():
                logger.warning("ABR monitoring thread did not stop cleanly.")
            logger.info("ABR monitoring thread stopped.")
        ABRManager.instance = None


def initialize_vlc_player(): # Remains the same
    global vlc_instance, media_player, player_event_manager
    if vlc_instance is None:
        instance_args = [
            '--no-video-title-show', 
            f'--network-caching=6000', # 你可以调整这个缓冲值
            # '--verbose=2', # 如果需要更详细的VLC日志，可以取消注释
            '--avcodec-hw=none'  # <--- 添加这个参数尝试禁用所有硬件解码
            # '--no-d3d11va'
        ]
        vlc_instance = vlc.Instance(instance_args)
        media_player = vlc_instance.media_player_new()
        player_event_manager = media_player.event_manager()
        player_event_manager.event_attach(vlc.EventType.MediaPlayerEndReached, on_player_end_reached_callback)
        player_event_manager.event_attach(vlc.EventType.MediaPlayerEncounteredError, on_player_error_callback)
        logger.info("VLC Player initialized for HLS playback.")

def on_player_end_reached_callback(event): # Remains the same
    logger.info("VLC_EVENT: MediaPlayerEndReached - Playback finished or stream ended.")
    player_reached_end.set()

def on_player_error_callback(event): # Remains the same
    logger.error("VLC_EVENT: MediaPlayerEncounteredError - An error occurred during VLC playback.")
    player_reached_end.set()

def play_hls_stream(video_name_on_server, initial_abr_manager=None): # Pass ABR manager
    global player_reached_end, current_selected_media_m3u8_url_on_server

    initialize_vlc_player()
    player_reached_end.clear()

    if not initial_abr_manager:
        logger.error("ABRManager instance not provided to play_hls_stream. Cannot determine initial stream.")
        return

    # ABR manager selects the initial media M3U8 URL
    selected_media_m3u8_url = initial_abr_manager.get_initial_media_m3u8_url()
    if not selected_media_m3u8_url:
        logger.error("ABRManager could not provide an initial media M3U8 URL.")
        return
    
    with abr_lock: # Ensure current_selected_media_m3u8_url_on_server is set
        current_selected_media_m3u8_url_on_server = selected_media_m3u8_url
        logger.info(f"Initial media M3U8 URL selected by ABR: {current_selected_media_m3u8_url_on_server}")


    # Fetch and rewrite this specific media M3U8
    modified_media_m3u8_content = fetch_media_m3u8_and_rewrite(selected_media_m3u8_url)

    if not modified_media_m3u8_content:
        logger.error("Could not get modified media M3U8 content for initial playback. Aborting.")
        return

    temp_m3u8_file_path = None
    try:
        if not os.path.exists(DOWNLOAD_DIR):
            os.makedirs(DOWNLOAD_DIR)
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".m3u8", delete=False, dir=DOWNLOAD_DIR, encoding='utf-8') as tmp_file:
            tmp_file.write(modified_media_m3u8_content)
            temp_m3u8_file_path = tmp_file.name
        
        logger.info(f"Modified MEDIA M3U8 for initial playback saved to: {temp_m3u8_file_path}")
        
        mrl = ('file:///' + temp_m3u8_file_path.replace('\\', '/')) if os.name == 'nt' else f'file://{os.path.abspath(temp_m3u8_file_path)}'
        logger.info(f"VLC will play MRL (from selected media M3U8): {mrl}")
        
        media = vlc_instance.media_new(mrl)
        if not media: logger.error(f"Failed to create VLC media object for MRL: {mrl}"); return
            
        media_player.set_media(media)
        media.release()

        if media_player.play() == -1: logger.error("Failed to start VLC playback."); return
        
        logger.info("Playback command issued. Waiting for end or error...")
        if initial_abr_manager: initial_abr_manager.start() # Start ABR decisions after playback starts

        player_reached_end.wait() 
        logger.info("Playback wait loop in play_hls_stream finished.")

    except Exception as e:
        logger.error(f"Error during HLS stream playback: {e}", exc_info=True)
    finally:
        if initial_abr_manager: initial_abr_manager.stop() # Stop ABR thread
        if media_player and media_player.is_playing():
            media_player.stop()
        if temp_m3u8_file_path and os.path.exists(temp_m3u8_file_path):
            try:
                os.remove(temp_m3u8_file_path)
            except OSError as e_rem:
                logger.warning(f"Could not remove temp M3U8 {temp_m3u8_file_path}: {e_rem}")

def main():
    # ... (VLC DLL hint remains the same) ...

    abr_manager = None # Initialize ABR manager variable

    try:
        if not hasattr(AES, 'AES_KEY') or not AES.AES_KEY: # AES checks remain
            logger.error("AES.AES_KEY not defined/empty."); return
        if not callable(getattr(AES, 'aes_decrypt_cbc', None)):
            logger.error("AES.aes_decrypt_cbc not defined."); return
        logger.info("AES module loaded.")

        if not start_proxy_server(): return # Start proxy first

        # 1. Construct URL for the master M3U8 playlist on the server
        master_m3u8_url = f"http://{SERVER_HOST}:{SERVER_PORT}/{VIDEO_TO_STREAM_NAME}/master.m3u8"
        
        # 2. Fetch and parse the master M3U8 to get available streams
        available_streams = fetch_master_m3u8(master_m3u8_url)
        if not available_streams:
            logger.error(f"Could not fetch or parse master M3U8 from {master_m3u8_url}. Aborting.")
            return # Stop if master M3U8 cannot be processed
        
        # 3. Initialize ABR Manager with available streams
        abr_manager = ABRManager(available_streams)

        # 4. Start HLS Playback, passing the ABR manager
        # play_hls_stream will use the ABR manager to get the initial media M3U8 URL
        play_hls_stream(VIDEO_TO_STREAM_NAME, initial_abr_manager=abr_manager)

    except KeyboardInterrupt:
        logger.info("Application interrupted by user (Ctrl+C).")
    except Exception as e:
        logger.error(f"An error occurred in main: {e}", exc_info=True)
    finally:
        logger.info("Main: Initiating cleanup...")
        if abr_manager: # Stop ABR manager if it was started
            abr_manager.stop()
        if media_player and media_player.is_playing():
            media_player.stop()
        if vlc_instance:
            vlc_instance.release()
        stop_proxy_server()
        # ... (DOWNLOAD_DIR cleanup remains) ...
        logger.info("Client application finished.")

if __name__ == "__main__":
    main()