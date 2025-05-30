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
        self.current_player_buffer_s = 0.0
        
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
                
    def update_player_buffer_level(self, buffer_seconds):
        with self._internal_lock:
            self.current_player_buffer_s = buffer_seconds
        logger.debug(f"ABR: Player buffer level updated to {buffer_seconds:.2f}s") # client.py中已记录debug信息
                
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
        if not self.available_streams or len(self.available_streams) <= 1:
            # logger.debug("ABR: Not enough streams to make a decision or no streams available.")
            return

        # 1. 获取当前状态
        estimated_bw_bps = self._estimate_bandwidth() # 获取估算带宽
        with self._internal_lock: # 确保线程安全地读取缓冲区大小
            current_buffer_s = self.current_player_buffer_s # 获取当前播放器缓冲时长，假设已由 client.py 更新
        current_level_index = self.current_stream_index_by_abr # 获取当前码率等级索引

        logger.info(
            f"ABR LOGIC: Est. BW: {estimated_bw_bps / 1000:.0f} Kbps, "
            f"Buffer: {current_buffer_s:.2f}s, "
            f"Current Level Idx: {current_level_index}"
        )

        # 如果完全没有下载统计数据（通常在刚开始时），并且估算带宽为0，则保持初始码率
        if estimated_bw_bps == 0 and not self.segment_download_stats:
            logger.info("ABR LOGIC: No bandwidth stats yet, sticking to initial/current level.")
            # 不需要广播，因为ABRManager在初始化时已经广播过一次初始码率
            return

        # 2. 定义缓冲区阈值 (这些值可以根据经验调整)
        BUFFER_THRESHOLD_LOW = 8.0  # s, 低缓冲区阈值，低于此值可能考虑降码率
        BUFFER_THRESHOLD_HIGH = 25.0 # s, 高缓冲区阈值，高于此值且带宽允许时可考虑升码率
        BUFFER_THRESHOLD_EMERGENCY = 3.0 # s, 紧急阈值，应强烈考虑降至最低码率

        # 3. 定义码率选择参数
        # safety_factor 用于在选择码率时留出一些余量，应对网络波动
        # 你原有的 safety_factor = 0.8
        # 我们可以根据缓冲区动态调整这个安全系数
        if current_buffer_s < BUFFER_THRESHOLD_LOW:
            dynamic_safety_factor = 0.7 # 缓冲区低，更保守
        elif current_buffer_s > BUFFER_THRESHOLD_HIGH:
            dynamic_safety_factor = 0.9 # 缓冲区高，可以更激进一些
        else:
            dynamic_safety_factor = 0.8 # 缓冲区健康，使用标准值

        target_bitrate_bps = estimated_bw_bps * dynamic_safety_factor
        logger.debug(f"ABR LOGIC: Dynamic Safety Factor: {dynamic_safety_factor:.2f}, Target BW for selection: {target_bitrate_bps / 1000:.0f} Kbps")

        # 4. 决策逻辑
        next_best_index = current_level_index # 默认为当前等级

        # 4.1 紧急情况：缓冲区过低，且不是已在最低码率
        if current_buffer_s < BUFFER_THRESHOLD_EMERGENCY and current_level_index > 0:
            next_best_index = 0 # 立即切换到最低码率
            logger.warning(f"ABR LOGIC: EMERGENCY! Buffer at {current_buffer_s:.2f}s. Switching to lowest quality (index 0).")
        
        # 4.2 尝试提升码率 (向上选择)
        # 条件：缓冲区较高，且当前不是最高码率
        elif current_buffer_s > BUFFER_THRESHOLD_HIGH and current_level_index < len(self.available_streams) - 1:
            # 从当前等级的下一个等级开始查找，看哪个等级的码率小于等于目标码率
            # 我们希望选择不超过 target_bitrate_bps 的最高可用码率
            potential_upgrade_index = current_level_index
            for i in range(len(self.available_streams) - 1, current_level_index, -1): # 从最高码率向下检查到当前码率的下一个
                stream_bw = self.available_streams[i].get('bandwidth', 0)
                if target_bitrate_bps >= stream_bw: # 如果目标带宽能支撑这个更高码率
                    # 额外检查：确保不是跳跃太多级，除非带宽非常充裕
                    # （这个简单版本暂时不加跳级限制，但实际中可以考虑）
                    potential_upgrade_index = i
                    break # 找到了可以升级到的最高等级
            if potential_upgrade_index > current_level_index :
                 logger.info(f"ABR LOGIC: Condition for UPGRADE met (buffer {current_buffer_s:.2f}s > {BUFFER_THRESHOLD_HIGH:.1f}s). Potential index: {potential_upgrade_index}")
                 next_best_index = potential_upgrade_index
            else:
                 logger.info(f"ABR LOGIC: Buffer high, but target BW ({target_bitrate_bps/1000:.0f}Kbps) doesn't support higher levels than current ({self.available_streams[current_level_index].get('bandwidth',0)/1000:.0f}Kbps).")


        # 4.3 尝试降低码率 (向下选择)
        # 条件：缓冲区较低 (但非紧急)，或估算带宽无法支撑当前码率
        elif current_buffer_s < BUFFER_THRESHOLD_LOW or target_bitrate_bps < self.available_streams[current_level_index].get('bandwidth', 0):
            if current_level_index > 0 : # 确保不是已经在最低码率
                # 从当前等级的下一个更低等级开始查找，选择一个最接近但不超过目标码率的
                # 如果所有更低等级都超过目标码率（不太可能），或者目标码率非常低，则选择最低的
                potential_downgrade_index = 0 # 默认降到最低以防万一
                for i in range(current_level_index -1, -1, -1): # 从当前等级的下一个更低等级开始向下找
                    stream_bw = self.available_streams[i].get('bandwidth', 0)
                    if target_bitrate_bps >= stream_bw : # 如果目标带宽能支撑这个（更低的）码率
                         potential_downgrade_index = i
                         break # 找到了最合适的较低码率
                    # 如果循环结束都没找到（即target_bitrate_bps比所有更低码率都低），则会降到索引0
                
                logger.info(f"ABR LOGIC: Condition for DOWNGRADE met (buffer {current_buffer_s:.2f}s < {BUFFER_THRESHOLD_LOW:.1f}s or target_bw too low). Potential index: {potential_downgrade_index}")
                next_best_index = potential_downgrade_index
            else:
                logger.info(f"ABR LOGIC: Condition for downgrade met, but already at lowest level (index 0).")
        
        # 4.4 如果不需要升降级，但当前带宽也无法支持当前码率，也应考虑降级（确保稳定性）
        # 这个条件部分被 4.3 覆盖，但可以再明确一下
        elif target_bitrate_bps < self.available_streams[current_level_index].get('bandwidth', 0) and current_level_index > 0:
            logger.info(f"ABR LOGIC: Target BW ({target_bitrate_bps/1000:.0f}Kbps) cannot sustain current level ({self.available_streams[current_level_index].get('bandwidth',0)/1000:.0f}Kbps). Looking for lower.")
            # 逻辑与4.3类似，寻找能支撑的最低码率
            temp_idx = 0
            for i in range(current_level_index -1, -1, -1):
                if target_bitrate_bps >= self.available_streams[i].get('bandwidth', 0):
                    temp_idx = i
                    break
            next_best_index = temp_idx

        # 5. 应用决策并广播
        if next_best_index != current_level_index:
            old_stream_info = self.available_streams[current_level_index]
            self.current_stream_index_by_abr = next_best_index # 更新当前ABR选择的码率索引
            new_stream_info = self.available_streams[self.current_stream_index_by_abr]
            self._update_current_abr_selected_url_logging() # 更新日志记录的URL
            
            logger.info(f"ABR DECISION: Switch from level index {self.available_streams.index(old_stream_info)} "
                        f"(BW {old_stream_info.get('bandwidth',0)/1000:.0f} Kbps) "
                        f"to level index {self.current_stream_index_by_abr} "
                        f"(BW {new_stream_info.get('bandwidth',0)/1000:.0f} Kbps). "
                        f"Reason: Est.BW={estimated_bw_bps/1000:.0f}Kbps, Buf={current_buffer_s:.2f}s")
            
            self.broadcast_abr_decision(self.current_stream_index_by_abr) # 通过WebSocket广播决策给客户端
        else:
            logger.info(f"ABR DECISION: No change from current level index {current_level_index}. Est.BW={estimated_bw_bps/1000:.0f}Kbps, Buf={current_buffer_s:.2f}s")

            
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