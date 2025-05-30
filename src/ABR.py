import threading
import time
import logging
import re
import requests
from urllib.parse import urljoin

import os
import numpy as np
import random
import json

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

    # --- 新增：定义决策逻辑的枚举或常量 ---
    LOGIC_TYPE_BANDWIDTH_ONLY = "bandwidth_only"
    LOGIC_TYPE_BANDWIDTH_BUFFER = "bandwidth_buffer"
    LOGIC_TYPE_ENHANCED_BUFFER_RESPONSE = "enhanced_buffer_response"
    LOGIC_TYPE_Q_LEARNING = "q_learning"

    def __init__(self, available_streams_from_master, broadcast_abr_decision_callback,
                 logic_type=LOGIC_TYPE_Q_LEARNING, # 可以将Q学习设为默认或通过参数选择
                 # --- Q学习相关参数 ---
                 q_learning_rate=0.1,
                 q_discount_factor=0.9,
                 q_epsilon=0.1, # 初始探索率
                 q_epsilon_decay=0.995, # 探索率衰减
                 q_epsilon_min=0.01,
                 q_table_save_path="q_table.json"):
        ABRManager.instance = self
        self.available_streams = sorted(
            [s for s in available_streams_from_master if s.get('bandwidth') is not None],
            key=lambda s: s['bandwidth']
        )
        if not self.available_streams:
            self.available_streams = [{'url': 'dummy', 'bandwidth': 0, 'resolution': 'N/A', 'attributes_str': ''}]

        self.broadcast_abr_decision = broadcast_abr_decision_callback
        self.current_stream_index_by_abr = 0
        self.segment_download_stats = []
        self.max_stats_history = 5 # 你之前用的是20，但对于快速响应，可以考虑减小，或根据逻辑调整
        self.estimated_bandwidth_bps = 0
        # self.safety_factor = 0.8 # 各个逻辑内部可以有自己的安全系数

        self.abr_thread = None
        self.stop_abr_event = threading.Event()
        self._internal_lock = threading.Lock()
        self._current_selected_url_for_logging = None
        self.current_player_buffer_s = 0.0

        # --- 新增：存储选择的决策逻辑类型 ---
        self.logic_type = logic_type
        logger.info(f"ABRManager initialized with logic type: {self.logic_type}")
        # ---
        
        if self.logic_type == self.LOGIC_TYPE_Q_LEARNING:
            self.lr = q_learning_rate
            self.gamma = q_discount_factor
            self.epsilon = q_epsilon
            self.epsilon_decay = q_epsilon_decay
            self.epsilon_min = q_epsilon_min
            self.q_table_save_path = q_table_save_path
            
            # --- 初始化Q表 ---
            # 状态空间维度: (带宽等级数, 缓冲等级数, 质量等级数)
            # 动作空间维度: 质量等级数
            # 这些离散化的函数和维度大小需要你根据实际情况定义
            self.bw_levels = self._discretize_bandwidth_levels() # 返回带宽等级定义
            self.buffer_levels = self._discretize_buffer_levels() # 返回缓冲等级定义
            self.num_quality_levels = len(self.available_streams)

            # Q表初始化为0或小的随机数
            self.q_table = np.zeros((len(self.bw_levels) + 1, # +1 用于处理超出范围的情况或作为特殊标记
                                     len(self.buffer_levels) + 1,
                                     self.num_quality_levels, # 当前质量作为状态的一部分
                                     self.num_quality_levels)) # 动作是选择下一个质量

            self._load_q_table() # 尝试加载已保存的Q表

            self.last_state_action_reward = None # 用于存储 (s, a, R, s') 中的 s, a
            self.last_played_quality_index = self.current_stream_index_by_abr # 用于计算切换惩罚


        # ... (其余初始化代码，如日志、广播初始决策等保持不变) ...
        if self.available_streams:
            self._update_current_abr_selected_url_logging()
        # Q学习的初始决策可能随机或基于一个简单的策略，直到Q表有意义
        if self.logic_type != self.LOGIC_TYPE_Q_LEARNING:
            self.broadcast_abr_decision(self.current_stream_index_by_abr)
        else:
            # Q学习的初始动作可以在第一次 _logic_q_learning 调用时决定
            pass


    def _update_current_abr_selected_url_logging(self): # 和你之前的一样
        with self._internal_lock:
            if self.available_streams and 0 <= self.current_stream_index_by_abr < len(self.available_streams):
                self._current_selected_url_for_logging = self.available_streams[self.current_stream_index_by_abr]['url']
            else:
                self._current_selected_url_for_logging = None

    def add_segment_download_stat(self, url, size_bytes, duration_seconds): # 和你之前的一样
        if duration_seconds > 0.001:
            # --- 新增：记录下载开始和结束时间，用于更高级的带宽估计 ---
            download_end_time = time.time()
            download_start_time = download_end_time - duration_seconds
            self.segment_download_stats.append({
                'url': url,
                'size': size_bytes,
                'duration': duration_seconds,
                'start_time': download_start_time, # 用于判断是否“最近”
                'end_time': download_end_time,     # 用于判断是否“最近”
                'throughput_bps': (size_bytes * 8) / duration_seconds
            })
            # ---
            if len(self.segment_download_stats) > self.max_stats_history: # 你可以根据需要调整这里的max_stats_history
                self.segment_download_stats.pop(0)

    def report_download_error(self, url): # 和你之前的一样
        logger.warning(f"ABR: Reported download error for segment from URL: {url}")
        # --- 新增：错误报告也可以加入统计，用于某些决策逻辑 ---
        self.segment_download_stats.append({
            'url': url,
            'error': True,
            'time': time.time()
        })
        if len(self.segment_download_stats) > self.max_stats_history:
            self.segment_download_stats.pop(0)
        # ---

    def _estimate_bandwidth_simple_average(self): # 你之前的带宽估计方法
        if not self.segment_download_stats: return self.estimated_bandwidth_bps # 返回上一次的值或0
        
        # 只考虑成功的下载
        successful_downloads = [s for s in self.segment_download_stats if not s.get('error') and 'size' in s and 'duration' in s]
        if not successful_downloads: return self.estimated_bandwidth_bps

        # 可以选择最近的N条，或者全部 (这里用全部，因为max_stats_history控制了数量)
        # relevant_stats = successful_downloads[-self.max_stats_history:] # 如果想用全部记录的来平均
        relevant_stats = successful_downloads # 使用max_stats_history限制的总数
        
        if not relevant_stats: return self.estimated_bandwidth_bps

        total_bytes = sum(s['size'] for s in relevant_stats)
        total_time = sum(s['duration'] for s in relevant_stats)

        if total_time == 0: return self.estimated_bandwidth_bps # 避免除以0
        
        self.estimated_bandwidth_bps = (total_bytes * 8) / total_time
        # logger.info(f"ABR SimpleAvg BW Est: {self.estimated_bandwidth_bps / 1000:.0f} Kbps") # 日志由具体决策逻辑打印
        return self.estimated_bandwidth_bps

    def _estimate_bandwidth_enhanced(self):
        # 这个方法可以更复杂，例如使用谐波平均值，或者对最近的片段赋予更高权重
        # 这里我们简单实现一个：如果最近一个片段下载速度远低于平均，则临时拉低平均值
        
        current_avg_bps = self._estimate_bandwidth_simple_average() # 先获取简单平均

        # 只考虑成功的下载
        successful_downloads = [s for s in self.segment_download_stats if not s.get('error') and 'throughput_bps' in s]
        if not successful_downloads:
            return current_avg_bps # 没有成功下载的统计，返回简单平均

        last_segment_info = successful_downloads[-1]
        last_segment_throughput_bps = last_segment_info['throughput_bps']

        # 如果最近一次下载速度显著低于当前平均，且平均值大于0
        if current_avg_bps > 0 and last_segment_throughput_bps < current_avg_bps * 0.5: # 例如，低于平均一半
            logger.warning(f"ABR Enhanced BW: Last segment throughput ({last_segment_throughput_bps/1000:.0f} Kbps) "
                           f"is much lower than average ({current_avg_bps/1000:.0f} Kbps). Adjusting estimate downwards.")
            # 更激进地降低估算，例如取最近一次和平均值的一个较小比例的组合
            adjusted_bps = (last_segment_throughput_bps * 0.7) + (current_avg_bps * 0.3)
            self.estimated_bandwidth_bps = adjusted_bps # 更新主估算值
            return adjusted_bps
        
        # 否则，正常返回简单平均值 (或者可以加入对持续高速的奖励等)
        return current_avg_bps


    def update_player_buffer_level(self, buffer_seconds): # 和你之前的一样
        with self._internal_lock:
            self.current_player_buffer_s = buffer_seconds

    def get_current_abr_decision_url(self): # 和你之前的一样
        with self._internal_lock:
            return self._current_selected_url_for_logging
        
    # --- Q学习辅助方法 ---
    def _discretize_bandwidth(self, current_bandwidth_bps):
        # 示例：将实际带宽值映射到离散等级索引
        # self.bw_levels = [(0, 1e6), (1e6, 3e6), (3e6, 5e6), (5e6, 10e6), (10e6, float('inf'))]
        for i, (low, high) in enumerate(self.bw_levels):
            if low <= current_bandwidth_bps < high:
                return i
        return len(self.bw_levels) # 超出定义的最高等级

    def _discretize_buffer(self, current_buffer_s):
        # 示例：将实际缓冲时长映射到离散等级索引
        # self.buffer_levels = [(0, 5), (5, 10), (10, 20), (20, 30), (30, float('inf'))]
        for i, (low, high) in enumerate(self.buffer_levels):
            if low <= current_buffer_s < high:
                return i
        return len(self.buffer_levels) # 超出定义的最高等级

    def _get_current_q_state(self):
        # 获取并离散化当前状态
        # 注意：带宽估计和缓冲区获取需要是最新的
        estimated_bw_bps = self._estimate_bandwidth_enhanced() # 或者你选择的其他估计方法
        with self._internal_lock:
            current_buffer_s = self.current_player_buffer_s
        
        # current_quality_index 应该是上一个决策/实际播放的质量等级
        # 但在做决策时，我们通常用的是 *导致* 当前缓冲和带宽观察值的那个质量
        # self.current_stream_index_by_abr 是 ABR *想要* 设置的等级
        # self.last_played_quality_index 是 QoE 反馈回来 *实际* 播放的等级 (或者由 LEVEL_SWITCHED 更新)
        # 为简单起见，我们先用 self.current_stream_index_by_abr (即上一次决策的结果)
        # 更准确的做法是，QoE模块上报实际播放的码率和该码率下的体验，然后用于Q表更新
        
        discrete_bw = self._discretize_bandwidth(estimated_bw_bps)
        discrete_buf = self._discretize_buffer(current_buffer_s)
        # current_quality_idx_for_state = self.current_stream_index_by_abr # 或者 self.last_played_quality_index
        current_quality_idx_for_state = self.last_played_quality_index # 使用上一个实际播放的质量作为当前状态的一部分

        return (discrete_bw, discrete_buf, current_quality_idx_for_state)

    def _save_q_table(self):
        try:
            # Q表是numpy数组，直接用json存可能不方便，可以转成list
            # 或者使用 numpy.save / numpy.load
            with open(self.q_table_save_path, 'w') as f:
                json.dump(self.q_table.tolist(), f)
            logger.info(f"Q-Table saved to {self.q_table_save_path}")
        except Exception as e:
            logger.error(f"Error saving Q-Table: {e}")

    def _load_q_table(self):
        try:
            if os.path.exists(self.q_table_save_path):
                with open(self.q_table_save_path, 'r') as f:
                    q_list = json.load(f)
                    self.q_table = np.array(q_list)
                logger.info(f"Q-Table loaded from {self.q_table_save_path}")
        except Exception as e:
            logger.error(f"Error loading Q-Table or Q-Table not found, starting fresh: {e}")
            # 如果加载失败，确保Q表维度正确并初始化
            self.q_table = np.zeros((len(self.bw_levels) + 1,
                                     len(self.buffer_levels) + 1,
                                     self.num_quality_levels,
                                     self.num_quality_levels))


    # --- 示例：你需要定义这些离散化级别 ---
    def _discretize_bandwidth_levels(self): # 定义带宽桶
        # 例如: levels in Mbps, then convert to bps
        # [(min_bps, max_bps), ...]
        return [(0, 1*1024*1024), (1*1024*1024, 3*1024*1024), (3*1024*1024, 6*1024*1024), (6*1024*1024, float('inf'))]

    def _discretize_buffer_levels(self): # 定义缓冲桶
        # 例如: levels in seconds
        return [(0, 5.0), (5.0, 10.0), (10.0, 20.0), (20.0, float('inf'))]

    # --- 决策逻辑的主分发方法 ---
    def _abr_decision_logic(self):
        if self.logic_type == self.LOGIC_TYPE_BANDWIDTH_ONLY:
            self._logic_bandwidth_only()
        elif self.logic_type == self.LOGIC_TYPE_BANDWIDTH_BUFFER:
            self._logic_bandwidth_buffer()
        elif self.logic_type == self.LOGIC_TYPE_ENHANCED_BUFFER_RESPONSE:
            self._logic_enhanced_buffer_response()
        elif self.logic_type == self.LOGIC_TYPE_Q_LEARNING:
            self._logic_q_learning() # 调用Q学习决策逻辑
        else:
            logger.warning(f"Unknown ABR logic type: {self.logic_type}. Defaulting to bandwidth_buffer.")
            self._logic_bandwidth_buffer()

    # --- 决策逻辑1: 只看带宽 (简化版，类似你最初的) ---
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
            # ... (广播决策的代码，与你之前版本类似，此处省略以保持简洁，但你需要实现它) ...
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


    # --- 决策逻辑2: 看带宽和缓冲区 (你上一轮测试的逻辑) ---
    def _logic_bandwidth_buffer(self):
        # 这个方法基本就是你上一轮测试时使用的 _abr_decision_logic 的内容
        # 我会直接将那份逻辑粘贴过来，并做少量调整以适应新结构
        if not self.available_streams or len(self.available_streams) <= 1: return

        estimated_bw_bps = self._estimate_bandwidth_simple_average() # 这个逻辑用简单平均
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


    # --- 决策逻辑3: 增强的缓冲区响应和带宽估计 (新的) ---
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

        # 定义阈值 (可以与 BW_BUFFER 逻辑中的不同，或者更细化)
        # 例如，更敏感的低缓冲区阈值，和用于“耐心等待”的阈值
        BUFFER_LOW_WARNING = 10.0  # s, 警告阈值，低于此值应更积极考虑降级
        BUFFER_LOW_CRITICAL = 5.0  # s, 危险阈值，强烈建议降级，甚至跳级
        BUFFER_HIGH_SAFE = 20.0    # s, 安全阈值，高于此值且带宽允许时可考虑升级
        BUFFER_STABLE_TARGET = 15.0 # s, 期望的稳定缓冲目标 (用于更细致的调整，此版本可能不用)

        # 带宽利用率和安全系数
        # safety_factor_normal = 0.85
        # safety_factor_aggressive_upgrade = 0.90 # 缓冲区很高时
        # safety_factor_conservative_downgrade = 0.75 # 缓冲区很低时
        
        next_best_index = current_level_index # 默认保持当前

        # 1. 处理下载错误 (示例：如果最近有下载错误，则更保守)
        #    你可以在 self.segment_download_stats 中检查 'error': True 的记录
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
            
        # 5. （可选）避免过于频繁的切换：可以加入一个计时器，两次切换之间至少间隔多久
        #     例如: if time.time() - self.last_switch_time < MIN_INTERVAL_BETWEEN_SWITCHES: return

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

    # --- Q学习的决策和更新逻辑 ---
    # 这个方法会在ABR的循环中被调用
    def _logic_q_learning(self):
        current_s = self._get_current_q_state() # 获取当前离散化状态 (s)
        
        # 1. 根据epsilon-greedy策略选择动作 (a)
        if random.uniform(0, 1) < self.epsilon:
            action_index = random.randint(0, self.num_quality_levels - 1) # 探索：随机选择一个质量等级
            logger.info(f"Q-LEARNING: Exploring action: {action_index}")
        else:
            # 利用：选择当前状态下Q值最大的动作
            # Q表维度: (bw_idx, buf_idx, current_quality_idx, action_quality_idx)
            action_index = np.argmax(self.q_table[current_s[0], current_s[1], current_s[2], :])
            logger.info(f"Q-LEARNING: Exploiting action: {action_index} (Q-values: {self.q_table[current_s[0], current_s[1], current_s[2], :]})")

        # 选定的动作 (action_index) 就是下一个要请求的码率等级
        chosen_quality_index = int(action_index)

        # 2. 执行动作 (通过广播给播放器) 并获取奖励 (R) 和新状态 (s')
        #    - 动作的执行: 广播 chosen_quality_index
        #    - 奖励的获取: 这个比较复杂。奖励不是立即获得的。
        #      它是在下一个分片下载完成、播放器状态更新（可能发生卡顿、质量切换完成）后才能计算。
        #      你需要一个机制来收集这些QoE事件，并计算出一个综合奖励。

        # 简化处理：我们将动作的执行（广播）和Q表的更新分离开。
        # _logic_q_learning 负责决策 (选择动作)。
        # 另一个方法 (例如 on_segment_played_or_stall) 将负责在收到反馈后更新Q表。

        # --- 决策部分 ---
        if chosen_quality_index != self.current_stream_index_by_abr: # 如果决策与当前设置不同
            old_level_idx_for_log = self.current_stream_index_by_abr
            self.current_stream_index_by_abr = chosen_quality_index # 更新 ABR 管理器想要设置的级别
            
            # 更新 self.last_state_action_reward 以便后续Q表更新
            # 此时我们还不知道 Reward 和 next_state
            # Reward 和 next_state 需要在 client.py 收集到 QoE 事件后计算和确定
            # 然后再调用 ABRManager 的一个新方法来更新Q表
            self.last_state_action_reward = {
                "state": current_s, # (s)
                "action": chosen_quality_index, # (a)
                "timestamp": time.time() # 记录决策时间，用于后续匹配反馈
            }

            logger.info(f"Q-LEARNING DECISION: Switch from level {old_level_idx_for_log} "
                        f"to {chosen_quality_index}. State: {current_s}")
            self.broadcast_abr_decision(chosen_quality_index)
        else:
            logger.info(f"Q-LEARNING DECISION: No change from level {self.current_stream_index_by_abr}. State: {current_s}")
            # 即使不切换，也可能需要记录状态和动作，以便在之后计算奖励并更新Q表 (如果奖励是基于持续状态的)
            # 为简单起见，我们主要在切换时或周期性地更新Q表
            # 如果没有切换，可以考虑不立即更新 self.last_state_action_reward，除非你的奖励机制也考虑“保持”的奖励

        # 探索率衰减
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


    # --- 新增：用于在收到反馈后更新Q表的方法 ---
    # 这个方法应该由 client.py 在收集到足够信息计算出奖励 R 和新状态 s' 后调用
    def update_q_table_after_feedback(self, reward, next_actual_played_quality_index):
        if self.last_state_action_reward is None:
            logger.warning("Q-LEARNING UPDATE: last_state_action_reward is None. Cannot update Q-table.")
            return

        s = self.last_state_action_reward["state"]      # (bw_idx, buf_idx, prev_quality_idx)
        a = self.last_state_action_reward["action"]    # chosen_quality_idx for that state
        R = reward                                     # 计算得到的奖励

        # 获取新状态 s'
        # 新状态应该是 *执行动作 a 并获得奖励 R 之后* 的状态
        # 这里的 next_actual_played_quality_index 是执行动作 a 后，播放器实际切换到的质量等级
        # 我们需要用这个实际播放的质量等级，以及 *在那之后* 观察到的新带宽和缓冲区来构成 s'
        # 这部分是最tricky的，因为状态和奖励的对应关系需要明确
        
        # 假设 client.py 会在计算完奖励 R 后，立即获取最新的带宽和缓冲区情况
        # 并将 next_actual_played_quality_index 也传递过来
        # (或者，ABRManager 自己在更新Q表前重新获取最新的离散状态作为 s_prime 的一部分)

        # 重新获取执行动作a并收到奖励R后的状态 s'
        # (这需要 client.py 协调：收集QoE -> 计算R -> 获取当前bw/buf -> 调用此方法)
        with self._internal_lock: # 获取最新缓冲区
            s_prime_buffer_s = self.current_player_buffer_s
        s_prime_bw_bps = self._estimate_bandwidth_enhanced() # 获取最新带宽

        s_prime_discrete_bw = self._discretize_bandwidth(s_prime_bw_bps)
        s_prime_discrete_buf = self._discretize_buffer(s_prime_buffer_s)
        
        # s' 中的质量等级应该是动作 a 对应的质量等级 (即 next_actual_played_quality_index)
        s_prime = (s_prime_discrete_bw, s_prime_discrete_buf, next_actual_played_quality_index)

        logger.info(f"Q-LEARNING UPDATE: s={s}, a={a}, R={R:.2f}, s'={s_prime}")

        # Q学习更新规则
        q_predict = self.q_table[s[0], s[1], s[2], a]
        q_target = R + self.gamma * np.max(self.q_table[s_prime[0], s_prime[1], s_prime[2], :]) # max_a' Q(s', a')
        
        self.q_table[s[0], s[1], s[2], a] += self.lr * (q_target - q_predict)
        
        logger.debug(f"Q-LEARNING UPDATE: Q-value for (s:{s}, a:{a}) changed from {q_predict:.3f} to {self.q_table[s[0], s[1], s[2], a]:.3f}")

        # 更新上一个实际播放的质量，用于下一个状态S的计算
        self.last_played_quality_index = next_actual_played_quality_index
        self.last_state_action_reward = None # 清理，等待下一次决策

    def abr_loop(self):
        logger.info(f"ABR Python Algo ({self.logic_type}) monitoring thread started.")
        # 初始等待时间，等播放器缓冲一些
        # 对于Q学习，第一次决策可能需要一些初始状态信息
        time.sleep(5) 
        
        if self.logic_type == self.LOGIC_TYPE_Q_LEARNING:
            # Q学习的第一次决策可能在循环外或循环内处理
            # 确保 self.last_played_quality_index 初始化正确
            self.last_played_quality_index = self.current_stream_index_by_abr # 假设初始播放的是这个

        while not self.stop_abr_event.is_set():
            try:
                self._abr_decision_logic()
            except Exception as e:
                logger.error(f"Error in ABR decision loop ({self.logic_type}): {e}", exc_info=True)
            
            sleep_interval = 3.0 # 决策频率
            for _ in range(int(sleep_interval)):
                if self.stop_abr_event.is_set(): break
                time.sleep(1)
        
        if self.logic_type == self.LOGIC_TYPE_Q_LEARNING:
            self._save_q_table() # 训练结束时保存Q表

        logger.info(f"ABR Python Algo ({self.logic_type}) monitoring thread stopped.")


    def start(self): # 和你之前的一样
        self.stop_abr_event.clear()
        self.abr_thread = threading.Thread(target=self.abr_loop, daemon=True, name="PythonABRLogicThread")
        self.abr_thread.start()

    def stop(self): # 和你之前的一样
        if self.abr_thread and self.abr_thread.is_alive():
            self.stop_abr_event.set()
            self.abr_thread.join(timeout=2.0) # 给线程一点时间干净地退出
        ABRManager.instance = None # 清理实例