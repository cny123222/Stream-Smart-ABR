import threading
import time
import logging
import re
import requests
from urllib.parse import urljoin

logger = logging.getLogger(__name__) # 使用模块特定的日志记录器

SOCKET_TIMEOUT_SECONDS = 10

# ABR特定的全局变量
current_abr_algorithm_selected_media_m3u8_url_on_server = None

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

# 用于ABRManager设置的获取初始主m3u8的函数
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
            attributes = parse_m3u8_attributes(attributes_str) # 使用parse_m3u8_attributes
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

    # --- 定义决策逻辑的枚举或常量 ---
    LOGIC_TYPE_BANDWIDTH_ONLY = "bandwidth_only"
    LOGIC_TYPE_BANDWIDTH_BUFFER = "bandwidth_buffer"
    LOGIC_TYPE_ENHANCED_BUFFER_RESPONSE = "enhanced_buffer_response" # 积极响应缓冲区的逻辑

    def __init__(self, available_streams_from_master, broadcast_abr_decision_callback,
                 broadcast_bw_estimate_callback=None,
                 logic_type=LOGIC_TYPE_BANDWIDTH_BUFFER): # 默认使用带宽+缓冲区逻辑
        ABRManager.instance = self
        self.available_streams = sorted(
            [s for s in available_streams_from_master if s.get('bandwidth') is not None],
            key=lambda s: s['bandwidth']
        )
        if not self.available_streams:
            self.available_streams = [{'url': 'dummy', 'bandwidth': 0, 'resolution': 'N/A', 'attributes_str': ''}]

        self.broadcast_abr_decision = broadcast_abr_decision_callback
        self.broadcast_bw_estimate = broadcast_bw_estimate_callback
        self.current_stream_index_by_abr = 0
        self.segment_download_stats = []
        self.max_stats_history = 5 
        self.estimated_bandwidth_bps = 0
        # self.safety_factor = 0.8 # 各个逻辑内部可以有自己的安全系数

        self.abr_thread = None
        self.stop_abr_event = threading.Event()
        self._internal_lock = threading.Lock()
        self._current_selected_url_for_logging = None
        self.current_player_buffer_s = 0.0

        # --- 存储选择的决策逻辑类型 ---
        self.logic_type = logic_type
        logger.info(f"ABRManager initialized with logic type: {self.logic_type}")

        if self.available_streams: #
            self._update_current_abr_selected_url_logging()
        self.broadcast_abr_decision(self.current_stream_index_by_abr)


    def _update_current_abr_selected_url_logging(self):
        with self._internal_lock:
            if self.available_streams and 0 <= self.current_stream_index_by_abr < len(self.available_streams):
                self._current_selected_url_for_logging = self.available_streams[self.current_stream_index_by_abr]['url']
            else:
                self._current_selected_url_for_logging = None

    def add_segment_download_stat(self, url, size_bytes, duration_seconds):
        if duration_seconds > 0.0001:
            # --- 记录下载开始和结束时间，用于更高级的带宽估计 ---
            download_end_time = time.time()
            download_start_time = download_end_time - duration_seconds
            self.segment_download_stats.append({
                'url': url,
                'size': size_bytes,
                'duration': duration_seconds,
                'start_time': download_start_time, # 用于判断是否"最近"
                'end_time': download_end_time,     # 用于判断是否"最近"
                'throughput_bps': (size_bytes * 8) / duration_seconds
            })

            if len(self.segment_download_stats) > self.max_stats_history: # 根据需要调整这里的max_stats_history
                self.segment_download_stats.pop(0)

    def report_download_error(self, url):
        logger.warning(f"ABR: Reported download error for segment from URL: {url}")
        # --- 错误报告加入统计，用于某些决策逻辑 ---
        self.segment_download_stats.append({
            'url': url,
            'error': True,
            'time': time.time()
        })
        if len(self.segment_download_stats) > self.max_stats_history:
            self.segment_download_stats.pop(0)

    def _estimate_bandwidth_simple_average(self): # 带宽估计方法
        if not self.segment_download_stats: return self.estimated_bandwidth_bps # 返回上一次的值或0
        
        # 只考虑成功的下载
        successful_downloads = [s for s in self.segment_download_stats if not s.get('error') and 'size' in s and 'duration' in s]
        if not successful_downloads: return self.estimated_bandwidth_bps

        # 选择最近的N条
        # relevant_stats = successful_downloads[-self.max_stats_history:] # 用全部记录的来平均
        relevant_stats = successful_downloads # 使用max_stats_history限制的总数
        
        if not relevant_stats: return self.estimated_bandwidth_bps

        total_bytes = sum(s['size'] for s in relevant_stats)
        total_time = sum(s['duration'] for s in relevant_stats)

        if total_time == 0: return self.estimated_bandwidth_bps # 避免除以0
        
        self.estimated_bandwidth_bps = (total_bytes * 8) / total_time
        # logger.info(f"ABR SimpleAvg BW Est: {self.estimated_bandwidth_bps / 1000:.0f} Kbps") # 日志由具体决策逻辑打印
        if self.broadcast_bw_estimate and self.estimated_bandwidth_bps > 0: # 仅当有有效估算时发送
            self.broadcast_bw_estimate(self.estimated_bandwidth_bps / 1_000_000) # 发送Mbps
        return self.estimated_bandwidth_bps

    # --- 增强的带宽估计 ---
    def _estimate_bandwidth_enhanced(self):
        # 对最近的片段赋予更高权重
        # 如果最近一个片段下载速度远低于平均，则临时拉低平均值
        
        current_avg_bps = self._estimate_bandwidth_simple_average() # 先获取简单平均

        # 只考虑成功的下载
        successful_downloads = [s for s in self.segment_download_stats if not s.get('error') and 'throughput_bps' in s]
        if not successful_downloads:
            return current_avg_bps # 没有成功下载的统计，返回简单平均

        last_segment_info = successful_downloads[-1]
        last_segment_throughput_bps = last_segment_info['throughput_bps']

        # 如果最近一次下载速度显著低于当前平均，且平均值大于0
        if current_avg_bps > 0 and last_segment_throughput_bps < current_avg_bps * 0.5: # 低于平均一半
            logger.warning(f"ABR Enhanced BW: Last segment throughput ({last_segment_throughput_bps/1000:.0f} Kbps) "
                           f"is much lower than average ({current_avg_bps/1000:.0f} Kbps). Adjusting estimate downwards.")
            # 更激进地降低估算，取最近一次和平均值的一个较小比例的组合
            adjusted_bps = (last_segment_throughput_bps * 0.7) + (current_avg_bps * 0.3)
            self.estimated_bandwidth_bps = adjusted_bps # 更新主估算值
            if self.broadcast_bw_estimate and adjusted_bps > 0:
                self.broadcast_bw_estimate(adjusted_bps / 1_000_000) # 发送Mbps
            return adjusted_bps
        
        # 否则，正常返回简单平均值
        return current_avg_bps


    def update_player_buffer_level(self, buffer_seconds):
        with self._internal_lock:
            self.current_player_buffer_s = buffer_seconds

    def get_current_abr_decision_url(self):
        with self._internal_lock:
            return self._current_selected_url_for_logging

    # --- 决策逻辑的主分发方法 ---
    def _abr_decision_logic(self):
        if self.logic_type == self.LOGIC_TYPE_BANDWIDTH_ONLY:
            self._logic_bandwidth_only()
        elif self.logic_type == self.LOGIC_TYPE_BANDWIDTH_BUFFER:
            self._logic_bandwidth_buffer()
        elif self.logic_type == self.LOGIC_TYPE_ENHANCED_BUFFER_RESPONSE:
            self._logic_enhanced_buffer_response()
        else:
            logger.warning(f"Unknown ABR logic type: {self.logic_type}. Defaulting to bandwidth_buffer.")
            self._logic_bandwidth_buffer()

    # --- 决策逻辑: 只看带宽 ---
    def _logic_bandwidth_only(self):
        if not self.available_streams or len(self.available_streams) <= 1: return

        estimated_bw_bps = self._estimate_bandwidth_simple_average() # 使用简单平均带宽
        current_level_index = self.current_stream_index_by_abr
        safety_factor = 0.8 # 此逻辑固定的安全系数

        logger.info(
            f"ABR LOGIC (BW_ONLY): Est. BW: {estimated_bw_bps / 1000:.0f} Kbps, "
            f"Current Level Idx: {current_level_index}"
        )

        if estimated_bw_bps == 0 and not self.segment_download_stats:
            logger.info("ABR LOGIC (BW_ONLY): No stats, sticking to current.")
            return

        target_bitrate_bps = estimated_bw_bps * safety_factor
        next_best_index = 0 # 默认最低
        
        # 从最高码率往下找，找到第一个能被目标带宽支持的
        for i in range(len(self.available_streams) - 1, -1, -1):
            stream_bw = self.available_streams[i].get('bandwidth', 0)
            if target_bitrate_bps >= stream_bw:
                next_best_index = i
                break
        
        if next_best_index != current_level_index:
            # ... 广播决策的代码 ...
            old_stream_info = self.available_streams[current_level_index]
            self.current_stream_index_by_abr = next_best_index
            new_stream_info = self.available_streams[self.current_stream_index_by_abr]
            self._update_current_abr_selected_url_logging()
            logger.info(f"ABR DECISION (BW_ONLY): Switch from level {current_level_index} "
                        f"(~{old_stream_info.get('bandwidth',0)/1000:.0f}Kbps) to {next_best_index} "
                        f"(~{new_stream_info.get('bandwidth',0)/1000:.0f}Kbps). Target_BW={target_bitrate_bps/1000:.0f}Kbps")
            self.broadcast_abr_decision(self.current_stream_index_by_abr)
        else:
            logger.info(f"ABR DECISION (BW_ONLY): No change from level {current_level_index}.")


    # --- 决策逻辑: 看带宽和缓冲区 ---
    def _logic_bandwidth_buffer(self):
        if not self.available_streams or len(self.available_streams) <= 1: return

        estimated_bw_bps = self._estimate_bandwidth_simple_average() # 简单平均
        with self._internal_lock:
            current_buffer_s = self.current_player_buffer_s
        current_level_index = self.current_stream_index_by_abr

        logger.info(
            f"ABR LOGIC (BW_BUFFER): Est. BW: {estimated_bw_bps / 1000:.0f} Kbps, "
            f"Buffer: {current_buffer_s:.2f}s, "
            f"Current Level Idx: {current_level_index}"
        )

        if estimated_bw_bps == 0 and not self.segment_download_stats:
            logger.info("ABR LOGIC (BW_BUFFER): No bandwidth stats yet, sticking to current level.")
            return

        BUFFER_THRESHOLD_LOW = 8.0
        BUFFER_THRESHOLD_HIGH = 25.0
        BUFFER_THRESHOLD_EMERGENCY = 3.0

        if current_buffer_s < BUFFER_THRESHOLD_LOW: dynamic_safety_factor = 0.7
        elif current_buffer_s > BUFFER_THRESHOLD_HIGH: dynamic_safety_factor = 0.9
        else: dynamic_safety_factor = 0.8

        target_bitrate_bps = estimated_bw_bps * dynamic_safety_factor
        logger.debug(f"ABR LOGIC (BW_BUFFER): Dyn Safety: {dynamic_safety_factor:.2f}, Target Sel. BW: {target_bitrate_bps / 1000:.0f} Kbps")

        next_best_index = current_level_index

        if current_buffer_s < BUFFER_THRESHOLD_EMERGENCY and current_level_index > 0:
            next_best_index = 0
            logger.warning(f"ABR LOGIC (BW_BUFFER): EMERGENCY! Buffer {current_buffer_s:.2f}s. Switching to lowest (idx 0).")
        elif current_buffer_s > BUFFER_THRESHOLD_HIGH and current_level_index < len(self.available_streams) - 1:
            potential_upgrade_index = current_level_index
            for i in range(len(self.available_streams) - 1, current_level_index, -1):
                if target_bitrate_bps >= self.available_streams[i].get('bandwidth', 0):
                    potential_upgrade_index = i; break
            if potential_upgrade_index > current_level_index:
                logger.info(f"ABR LOGIC (BW_BUFFER): UPGRADE condition met (buf {current_buffer_s:.2f}s > {BUFFER_THRESHOLD_HIGH:.1f}s). Potential idx: {potential_upgrade_index}")
                next_best_index = potential_upgrade_index
            # else: logger.info(f"ABR LOGIC (BW_BUFFER): Buffer high, but target BW no support higher.")
        elif current_buffer_s < BUFFER_THRESHOLD_LOW or target_bitrate_bps < self.available_streams[current_level_index].get('bandwidth', 0):
            if current_level_index > 0:
                potential_downgrade_index = 0
                for i in range(current_level_index - 1, -1, -1):
                    if target_bitrate_bps >= self.available_streams[i].get('bandwidth', 0):
                        potential_downgrade_index = i; break
                logger.info(f"ABR LOGIC (BW_BUFFER): DOWNGRADE condition met. Potential idx: {potential_downgrade_index}")
                next_best_index = potential_downgrade_index
            # else: logger.info(f"ABR LOGIC (BW_BUFFER): Downgrade condition, but already at lowest.")
        elif target_bitrate_bps < self.available_streams[current_level_index].get('bandwidth', 0) and current_level_index > 0: # 再次检查稳定性
            logger.info(f"ABR LOGIC (BW_BUFFER): Target BW cannot sustain current. Looking lower.")
            temp_idx = 0
            for i in range(current_level_index - 1, -1, -1):
                if target_bitrate_bps >= self.available_streams[i].get('bandwidth', 0): temp_idx = i; break
            next_best_index = temp_idx
        
        if next_best_index != current_level_index:
            # ... (广播决策的代码) ...
            old_stream_info = self.available_streams[current_level_index]
            self.current_stream_index_by_abr = next_best_index
            new_stream_info = self.available_streams[self.current_stream_index_by_abr]
            self._update_current_abr_selected_url_logging()
            logger.info(f"ABR DECISION (BW_BUFFER): Switch from level {current_level_index} "
                        f"(~{old_stream_info.get('bandwidth',0)/1000:.0f}Kbps) to {next_best_index} "
                        f"(~{new_stream_info.get('bandwidth',0)/1000:.0f}Kbps). Est.BW={estimated_bw_bps/1000:.0f}Kbps, Buf={current_buffer_s:.2f}s")
            self.broadcast_abr_decision(self.current_stream_index_by_abr)
        else:
            logger.info(f"ABR DECISION (BW_BUFFER): No change from level {current_level_index}. Est.BW={estimated_bw_bps/1000:.0f}Kbps, Buf={current_buffer_s:.2f}s")


    # --- 决策逻辑: 增强的缓冲区响应和带宽估计 ---
    def _logic_enhanced_buffer_response(self):
        if not self.available_streams or len(self.available_streams) <= 1: return

        estimated_bw_bps = self._estimate_bandwidth_enhanced() # 使用增强的带宽估计
        with self._internal_lock:
            current_buffer_s = self.current_player_buffer_s
        current_level_index = self.current_stream_index_by_abr
        
        logger.info(
            f"ABR LOGIC (ENHANCED): Est. BW: {estimated_bw_bps / 1000:.0f} Kbps, "
            f"Buffer: {current_buffer_s:.2f}s, "
            f"Current Level Idx: {current_level_index}"
        )

        # 定义阈值 (可以与 BW_BUFFER 逻辑中的不同)
        BUFFER_LOW_WARNING = 10.0  # s, 警告阈值，低于此值应更积极考虑降级
        BUFFER_LOW_CRITICAL = 5.0  # s, 危险阈值，强烈建议降级，甚至跳级
        BUFFER_HIGH_SAFE = 20.0    # s, 安全阈值，高于此值且带宽允许时可考虑升级
        BUFFER_STABLE_TARGET = 15.0 # s, 期望的稳定缓冲目标 (用于更细致的调整)

        # 带宽利用率和安全系数
        # safety_factor_normal = 0.85
        # safety_factor_aggressive_upgrade = 0.90 # 缓冲区很高时
        # safety_factor_conservative_downgrade = 0.75 # 缓冲区很低时
        
        next_best_index = current_level_index # 默认保持当前

        # 1. 处理下载错误 (示例：如果最近有下载错误，则更保守)
        #    在 self.segment_download_stats 中检查 'error': True 的记录
        recent_errors = [s for s in self.segment_download_stats if s.get('error') and time.time() - s.get('time', 0) < 10] # 例如10秒内的错误
        if recent_errors:
            logger.warning(f"ABR LOGIC (ENHANCED): Recent download errors detected. Being more conservative.")
            # 此处可以临时降低安全系数或强制检查降级


        # 2. 缓冲区过低时的紧急/关键处理
        if current_buffer_s < BUFFER_LOW_CRITICAL and current_level_index > 0:
            # 降到最低或者能维持的最低（基于一个非常保守的带宽估计）
            # 此时，我们甚至可以忽略当前的 estimated_bw_bps，因为它可能滞后
            # 直接尝试降一级，或者如果缓冲区非常非常低，直接降到0
            if current_buffer_s < BUFFER_LOW_CRITICAL / 2: # 例如，小于危险阈值的一半
                 next_best_index = 0
                 logger.warning(f"ABR LOGIC (ENHANCED): CRITICALLY LOW BUFFER ({current_buffer_s:.2f}s)! Forcing to lowest quality (idx 0).")
            else:
                 next_best_index = max(0, current_level_index - 1) # 至少降一级
                 logger.warning(f"ABR LOGIC (ENHANCED): Low buffer ({current_buffer_s:.2f}s). Considering downgrade to {next_best_index}.")

        # 3. 尝试升级 (缓冲区安全，带宽允许)
        elif current_buffer_s > BUFFER_HIGH_SAFE and current_level_index < len(self.available_streams) - 1:
            # 使用一个相对积极的安全系数来判断能否升级
            target_upgrade_bw = estimated_bw_bps * 0.90 # 例如用90%的估计带宽
            potential_upgrade_index = current_level_index
            # 从当前等级的下一个开始，找到能支撑的最高等级
            for i in range(current_level_index + 1, len(self.available_streams)):
                if target_upgrade_bw >= self.available_streams[i].get('bandwidth', 0):
                    potential_upgrade_index = i # 继续尝试更高的
                else:
                    break # 这个等级无法支撑，更高级别也不行
            if potential_upgrade_index > current_level_index:
                logger.info(f"ABR LOGIC (ENHANCED): UPGRADE condition met. Buffer ({current_buffer_s:.2f}s), TargetBW ({target_upgrade_bw/1000:.0f}Kbps). Potential idx: {potential_upgrade_index}")
                next_best_index = potential_upgrade_index

        # 4. 尝试降级 (缓冲区警告，或带宽不足以维持当前)
        #    (确保这个条件不会与上面的紧急处理冲突或重复太多)
        elif (current_buffer_s < BUFFER_LOW_WARNING and current_level_index > 0 and next_best_index == current_level_index) or \
             (estimated_bw_bps * 0.80 < self.available_streams[current_level_index].get('bandwidth', 0) and current_level_index > 0 and next_best_index == current_level_index) :
            # 使用一个相对保守的带宽估计来选择降级目标
            target_downgrade_bw = estimated_bw_bps * 0.80 # 用80%的估计带宽，或更低
            
            # 从当前等级往下找，找到第一个能被target_downgrade_bw稳定支持的
            # 如果没有，则选择最低（索引0）
            new_idx = 0 # 默认降到最低
            for i in range(current_level_index - 1, -1, -1):
                if target_downgrade_bw >= self.available_streams[i].get('bandwidth', 0):
                    new_idx = i
                    break 
            logger.info(f"ABR LOGIC (ENHANCED): DOWNGRADE condition met. Buffer ({current_buffer_s:.2f}s), EstBW ({estimated_bw_bps/1000:.0f}Kbps). Potential idx: {new_idx}")
            next_best_index = new_idx
            
        # 5. 避免过于频繁的切换：可以加入一个计时器，两次切换之间至少间隔多久
        #     if time.time() - self.last_switch_time < MIN_INTERVAL_BETWEEN_SWITCHES: return

        if next_best_index != current_level_index:
            # ... (广播决策的代码) ...
            old_stream_info = self.available_streams[current_level_index]
            self.current_stream_index_by_abr = next_best_index
            new_stream_info = self.available_streams[self.current_stream_index_by_abr]
            self._update_current_abr_selected_url_logging()
            logger.info(f"ABR DECISION (ENHANCED): Switch from level {current_level_index} "
                        f"(~{old_stream_info.get('bandwidth',0)/1000:.0f}Kbps) to {next_best_index} "
                        f"(~{new_stream_info.get('bandwidth',0)/1000:.0f}Kbps). Est.BW={estimated_bw_bps/1000:.0f}Kbps, Buf={current_buffer_s:.2f}s")
            self.broadcast_abr_decision(self.current_stream_index_by_abr)
        else:
            logger.info(f"ABR DECISION (ENHANCED): No change from level {current_level_index}. Est.BW={estimated_bw_bps/1000:.0f}Kbps, Buf={current_buffer_s:.2f}s")


    def abr_loop(self):
        logger.info(f"ABR Python Algo ({self.logic_type}) monitoring thread started.")
        time.sleep(3) # 初始等待，让播放器先缓冲一些
        while not self.stop_abr_event.is_set():
            try:
                self._abr_decision_logic() # 调用主分发方法
            except Exception as e:
                logger.error(f"Error in ABR decision loop ({self.logic_type}): {e}", exc_info=True)
            
            # 决策频率 (例如3秒)
            # 更频繁的决策可能导致振荡，太慢则响应不及时
            sleep_interval = 3.0 
            for _ in range(int(sleep_interval)): # 允许更早地被stop_event打断
                if self.stop_abr_event.is_set(): break
                time.sleep(1)
        logger.info(f"ABR Python Algo ({self.logic_type}) monitoring thread stopped.")

    def start(self):
        self.stop_abr_event.clear()
        self.abr_thread = threading.Thread(target=self.abr_loop, daemon=True, name="PythonABRLogicThread")
        self.abr_thread.start()

    def stop(self):
        if self.abr_thread and self.abr_thread.is_alive():
            self.stop_abr_event.set()
            self.abr_thread.join(timeout=2.0) # 给线程时间干净地退出
        ABRManager.instance = None # 清理实例