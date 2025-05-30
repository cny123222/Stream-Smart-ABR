import threading
import time
import logging
import re
import requests
from urllib.parse import urljoin

logger = logging.getLogger(__name__) # Use module-specific logger

SOCKET_TIMEOUT_SECONDS = 10

# ABR-specific globals or make them instance members
current_abr_algorithm_selected_media_m3u8_url_on_server = None # Better as instance var or property

def parse_m3u8_attributes(attr_string):
    attributes = {}
    try:
        for match in re.finditer(r'([A-Z0-9-]+)=("([^"]*)"|([^,"]*))', attr_string):
            key = match.group(1)
            value = match.group(3) if match.group(3) is not None else match.group(4)
            if value.isdigit(): attributes[key] = int(value)
            else: attributes[key] = value
    except Exception as e: logger.error(f"Error parsing M3U8 attributes: {e}")
    return attributes

# Function to fetch initial master m3u8 for ABRManager setup
def fetch_master_m3u8_for_abr_init(master_m3u8_url_on_server):
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

class ABRManager:
    instance = None 

    def __init__(self, available_streams_from_master, broadcast_abr_decision_callback):
        ABRManager.instance = self
        self.available_streams = sorted(
            [s for s in available_streams_from_master if s.get('bandwidth') is not None], 
            key=lambda s: s['bandwidth']
        )
        if not self.available_streams: # Add a dummy to prevent crashes if all lack bandwidth
             self.available_streams = [{'url': 'dummy', 'bandwidth': 0, 'resolution': 'N/A', 'attributes_str': ''}]
        
        # current_stream_index_by_abr is the index into self.available_streams
        # This index will be sent to HLS.js
        self.broadcast_abr_decision = broadcast_abr_decision_callback
        self.current_stream_index_by_abr = 0 
        self.segment_download_stats = [] 
        self.max_stats_history = 20 
        self.estimated_bandwidth_bps = 0
        self.safety_factor = 0.8
        self.abr_thread = None
        self.stop_abr_event = threading.Event()
        self._internal_lock = threading.Lock()
        self._current_selected_url_for_logging = None
        
        logger.info(f"ABRManager initialized. Available streams (sorted by bandwidth for indexing):")
        for i, s in enumerate(self.available_streams):
            logger.info(f"  Level Index [{i}] BW: {s.get('bandwidth', 'N/A')}, Res: {s.get('resolution', 'N/A')}, URL: {s['url']}")
        if self.available_streams:
            self._update_current_abr_selected_url_logging()
        # Send initial decision (e.g., lowest quality)
        self.broadcast_abr_decision(self.current_stream_index_by_abr)


    def _update_current_abr_selected_url_logging(self): # For logging only
        with self._internal_lock:
            if self.available_streams and 0 <= self.current_stream_index_by_abr < len(self.available_streams):
                self._current_selected_url_for_logging = self.available_streams[self.current_stream_index_by_abr]['url']
            else:
                self._current_selected_url_for_logging = None
                
                
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
            self.broadcast_abr_decision(self.current_stream_index_by_abr)
        # else:
            # logger.debug(f"ABR Python Algo: No change in decision, current index {self.current_stream_index_by_abr}")
            
    def get_current_abr_decision_url(self):
        """Provides thread-safe access to the current ABR decided URL for logging/display."""
        with self._internal_lock:
            return self._current_selected_url_for_logging

    def abr_loop(self):
        logger.info("ABR Python Algo monitoring thread started.")
        # --- TEST: Hardcode level switch after 10 seconds ---
        # time.sleep(10) 
        # hardcoded_level_index = 1 # 假设你想切换到 level index 1 (e.g., 720p)
        # logger.info(f"TESTING: Hardcoding switch to level index: {hardcoded_level_index}")
        # self.broadcast_abr_decision(hardcoded_level_index)
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