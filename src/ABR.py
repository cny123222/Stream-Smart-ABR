import threading
import time
import logging
import re
import requests
import numpy as np
from urllib.parse import urljoin
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import os
import json

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
    logger.info(f"ABR_INIT: Fetching master M3U8 from: {master_m3u8_url_on_server}") # 日志: 获取主M3U8
    try:
        response = requests.get(master_m3u8_url_on_server, timeout=SOCKET_TIMEOUT_SECONDS) # SOCKET_TIMEOUT_SECONDS 应在某处定义
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.error(f"ABR_INIT: Failed to fetch master M3U8: {e}") # 日志: 获取主M3U8失败
        return None
    
    content = response.text
    lines = content.splitlines()
    available_streams = []
    master_m3u8_base_url = urljoin(master_m3u8_url_on_server, '.') # 用于解析相对路径

    for i in range(len(lines)):
        line = lines[i].strip()
        if line.startswith("#EXT-X-STREAM-INF:"):
            attributes_str = line.split(":", 1)[1]
            attributes = parse_m3u8_attributes(attributes_str) # 假设 parse_m3u8_attributes 已定义
            
            if i + 1 < len(lines):
                media_playlist_relative_url = lines[i+1].strip()
                media_playlist_absolute_url_on_origin = urljoin(master_m3u8_base_url, media_playlist_relative_url)
                
                # --- 从媒体播放列表的相对URL中提取suffix ---
                # 假设 media_playlist_relative_url 的格式是 "quality_suffix/playlist_name.m3u8"
                # 例如 "2160p-16000k/bbb_sunflower-2160p-16000k.m3u8"
                # 那么 quality_suffix 就是 "2160p-16000k"
                derived_suffix = None
                try:
                    # os.path.normpath 处理掉可能的 './' 或 '../' (虽然这里通常是直接的相对路径)
                    # os.path.dirname 会给出 "2160p-16000k" (如果路径是 "2160p-16000k/file.m3u8")
                    # 如果 media_playlist_relative_url 直接就是 "2160p-16000k.m3u8" 且没有目录结构，
                    # 你可能需要不同的解析方式，或者确保你的URL结构是包含目录的。
                    # 假设你的 master M3U8 中列出的媒体播放列表URL包含了质量目录，如下所示：
                    # 2160p-16000k/bbb_sunflower-2160p-16000k.m3u8
                    path_parts = media_playlist_relative_url.replace('\\', '/').split('/')
                    if len(path_parts) > 1: # 期望至少有 "directory/file.m3u8"
                        derived_suffix = path_parts[0]
                    elif len(path_parts) == 1 and attributes.get('RESOLUTION'): # 如果没有目录，尝试从分辨率等属性构造一个
                        # 这是一个备选方案，如果URL不包含可识别的suffix目录
                        # 你可以根据你的实际命名规则来构造，或者从其他属性推断
                        # 例如，你可以基于分辨率和带宽构建一个类似 "2160p-16000k" 的字符串
                        # 为了简单，如果URL没有目录，我们暂时不生成suffix，依赖于后续的匹配逻辑
                        logger.warning(f"Media playlist URL '{media_playlist_relative_url}' does not seem to have a quality directory prefix. Suffix might be unreliable.")
                        # derived_suffix = f"{attributes.get('RESOLUTION', 'unknownres')}_{attributes.get('BANDWIDTH', 0)//1000}k" # 示例构造
                    else:
                        logger.warning(f"Cannot derive suffix from media playlist URL: {media_playlist_relative_url}")
                except Exception as e_suffix:
                    logger.error(f"Error deriving suffix from '{media_playlist_relative_url}': {e_suffix}")


                available_streams.append({
                    'url': media_playlist_absolute_url_on_origin, 
                    'bandwidth': attributes.get('BANDWIDTH'),
                    'resolution': attributes.get('RESOLUTION'),
                    'codecs': attributes.get('CODECS'),
                    'attributes_str': attributes_str,
                    'suffix': derived_suffix # <--- 存储提取或生成的suffix
                })
    
    if not available_streams:
        logger.warning("ABR_INIT: No streams found in master M3U8.") # 日志: 主M3U8中未找到流
        return None
    return available_streams

class DQNNetwork(nn.Module):
    """DQN神经网络"""
    def __init__(self, state_size, action_size, hidden_size=128):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, action_size)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

class ReplayBuffer:
    """经验回放缓冲区"""
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

class ABRManager:
    instance = None

    # --- 定义决策逻辑的枚举或常量 ---
    LOGIC_TYPE_SLBW = "slbw"  # (Simple Last Bandwidth) - 只看最近一次带宽估计
    LOGIC_TYPE_SWMA = "swma"  # (Simple Moving Window Average) - 最近N个带宽的简单平均
    LOGIC_TYPE_EWMA = "ewma"  # (Exponentially Weighted Moving Average) - 最近N个带宽的指数加权平均
    LOGIC_TYPE_BUFFER_ONLY = "buffer_only" # 只看缓冲区
    LOGIC_TYPE_BANDWIDTH_BUFFER = "bandwidth_buffer" # 当前的带宽+缓冲区组合逻辑 (可以作为基准或一种选项)
    LOGIC_TYPE_COMPREHENSIVE = "comprehensive_rules" # 你设想的基于趋势、历史等的综合规则算法
    # LOGIC_TYPE_DQN = "dqn" # 为DQN预留，但具体实现由外部控制或不同方式集成
    LOGIC_TYPE_DQN = "dqn_rl" # DQN强化学习逻辑

    def __init__(self, available_streams_from_master, broadcast_abr_decision_callback,
                 broadcast_bw_estimate_callback=None,
                 logic_type=LOGIC_TYPE_BANDWIDTH_BUFFER, # 默认使用带宽+缓冲区逻辑
                 # --- 带宽估计相关参数 --- #
                 max_bw_stats_history=4,  # SWMA和EWMA等可能用到的历史窗口大小
                 ewma_alpha=0.7,          # EWMA的平滑因子alpha
                 # --- 切换抑制参数 --- #
                 min_switch_interval_seconds=5.0, # 最小切换间隔
                 # --- 综合逻辑特定参数 ---
                 nominal_segment_duration_s=5.0, # 标准分片时长（秒）
                 comp_params=None # 允许外部传入综合逻辑的参数字典
                ):
        ABRManager.instance = self
        self.available_streams = sorted(
            [s for s in available_streams_from_master if s.get('bandwidth') is not None],
            key=lambda s: s['bandwidth']
        )
        if not self.available_streams: # 若列表为空，添加一个虚拟项防止后续代码出错
            self.available_streams = [{'url': 'dummy', 'bandwidth': 0, 'resolution': 'N/A', 'attributes_str': '', 'height':0}]
            self.num_quality_levels = 0
        else:
            self.num_quality_levels = len(self.available_streams)

        self.broadcast_abr_decision = broadcast_abr_decision_callback
        self.broadcast_bw_estimate = broadcast_bw_estimate_callback
        self.current_stream_index_by_abr = 0 # 当前ABR选择的码率等级索引
        
        # 分片下载统计: 存储 {'url': str, 'size': bytes, 'duration': sec, 'throughput_bps': bps, 'timestamp': time.time()}
        self.segment_download_stats = [] 
        self.max_bw_stats_history = max_bw_stats_history 

        # 带宽估计值
        self.estimated_bandwidth_bps = 0.0      # 通用带宽估计值
        self.swma_bandwidth_bps = 0.0          # SWMA带宽估计值
        self.ewma_bandwidth_bps = 0.0           # EWMA专用带宽估计值
        self.ewma_alpha = ewma_alpha            # EWMA平滑因子

        # 内部状态和锁
        self.abr_thread = None
        self.stop_abr_event = threading.Event()
        self._internal_lock = threading.Lock()
        self._current_selected_url_for_logging = None # 用于日志记录的当前选择的媒体播放列表URL
        self.current_player_buffer_s = 0.0      # 当前播放器缓冲时长 (秒)
        
        # 切换控制
        self.last_switch_time = 0               # 上次切换时间戳
        self.MIN_SWITCH_INTERVAL = min_switch_interval_seconds # 最小切换间隔
        
        # 综合逻辑 
        self.NOMINAL_SEGMENT_DURATION_S = nominal_segment_duration_s
        self.bandwidth_history_for_trend = [] # 存储 (timestamp, throughput_bps)
        self.MAX_BW_HISTORY_FOR_TREND = 7     # 用于趋势分析的独立历史窗口大小
        
        self.last_segment_download_start_time = 0.0
        self.current_segment_target_bitrate_bps = 0.0 # 当前下载分片的目标比特率
        self.current_segment_url_being_downloaded = None
        self.consecutive_segment_timeouts = 0 # 连续分片“卡住”计数

        self.recent_rebuffering_info = {'count': 0, 'last_timestamp': 0, 'total_duration_s': 0.0}
        self.RECOVERY_HOLD_UNTIL_TIMESTAMP = 0 # 从严重卡顿或骤降中恢复时的升档锁定时间戳

        # 综合规则算法的参数字典 (如果外部没传，则使用默认值)
        default_comp_params = {
            "safety_factor_stable": 0.85, "safety_factor_volatile": 0.7,
            "safety_factor_recovering": 0.65, "safety_factor_critical": 0.5,
            "buffer_target_ideal": 20.0, "buffer_low_threshold": 12.0,
            "buffer_emergency_threshold": 5.0, "buffer_high_for_upgrade": 25.0,
            "bw_drop_major_ratio": 0.5, "bw_drop_minor_ratio": 0.3, # 下降百分比
            "bw_trend_significant_change_bps_per_s": 500 * 1024, # 带宽变化率阈值
            "volatility_coeff_variation_threshold": 0.35,
            "segment_stuck_timeout_factor": 2.0, # 预期下载时间的X倍
            "max_consecutive_stuck_segments": 2,
            "recent_stall_period_s": 45, # 考察最近X秒内的卡顿
            "stall_count_for_conservative_mode": 1, # 只要有1次近期卡顿就进入保守
            "recovery_hold_duration_s": 15.0, # 发生骤降或严重卡顿后，升档锁定的持续时间
            "expectation_ewma_cap_factor": 3.0,       # 用于限制预期计算中EWMA的因子 (例如，EWMA不会被认为超过目标码率的3倍)
            "segment_stuck_fallback_duration_factor": 2.5, # 如果EWMA过低，预期下载时间为 名义时长 * 此因子
            "abs_min_expected_time_s": 1.0,           # 绝对最小预期下载时间 (例如 1.0秒)
            "min_expected_time_nominal_ratio": 0.25,  # 最小预期时间与名义分片时长的比例 (例如 0.25 代表1/4时长)
        }
        self.COMP_PARAMS = comp_params if isinstance(comp_params, dict) else default_comp_params

        # 存储选择的决策逻辑类型
        self.logic_type = logic_type
        logger.info(f"ABRManager initialized with logic type: {self.logic_type}")

        # --- DQN相关初始化 ---
        if self.logic_type == self.LOGIC_TYPE_DQN:
            self._init_dqn()

        if self.available_streams and self.num_quality_levels > 0:
            self._update_current_abr_selected_url_logging()
            self.broadcast_abr_decision(self.current_stream_index_by_abr) # 广播初始决策
        else:
            logger.warning("ABRManager initialized with no available streams. ABR decisions might be affected.") # 日志: 无可用流

    def _init_dqn(self):
        """初始化DQN相关组件"""
        # 状态空间维度: [当前带宽(Mbps), 缓冲区长度(s), 当前比特率等级(归一化), 最近下载时间(s), 下载成功率]
        self.state_size = 5
        # 动作空间: 选择比特率等级
        self.action_size = len(self.available_streams)
        
        # 神经网络
        self.dqn = DQNNetwork(self.state_size, self.action_size)
        self.target_dqn = DQNNetwork(self.state_size, self.action_size)
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=0.001)
        
        # 经验回放
        self.replay_buffer = ReplayBuffer(capacity=10000)
        
        # DQN超参数
        self.epsilon = 0.9  # 探索率
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1
        self.gamma = 0.95  # 折扣因子
        self.batch_size = 32
        self.target_update_freq = 100  # 目标网络更新频率
        
        # 训练相关
        self.training_step = 0
        self.last_state = None
        self.last_action = None
        self.last_qoe_score = 0
        
        # 模型保存路径
        self.model_save_path = "dqn_abr_model.pth"
        self.load_model_if_exists()
        
        logger.info(f"DQN ABR initialized: state_size={self.state_size}, action_size={self.action_size}")

    def load_model_if_exists(self):
        """加载已保存的模型"""
        if os.path.exists(self.model_save_path):
            try:
                checkpoint = torch.load(self.model_save_path)
                self.dqn.load_state_dict(checkpoint['dqn_state_dict'])
                self.target_dqn.load_state_dict(checkpoint['target_dqn_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.epsilon = checkpoint.get('epsilon', self.epsilon)
                self.training_step = checkpoint.get('training_step', 0)
                logger.info(f"DQN model loaded from {self.model_save_path}")
            except Exception as e:
                logger.warning(f"Failed to load DQN model: {e}")

    def save_model(self):
        """保存模型"""
        if self.logic_type == self.LOGIC_TYPE_DQN:
            try:
                torch.save({
                    'dqn_state_dict': self.dqn.state_dict(),
                    'target_dqn_state_dict': self.target_dqn.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epsilon': self.epsilon,
                    'training_step': self.training_step
                }, self.model_save_path)
                logger.info(f"DQN model saved to {self.model_save_path}")
            except Exception as e:
                logger.error(f"Failed to save DQN model: {e}")

    def get_current_state(self):
        """获取当前状态向量"""
        # 当前带宽 (Mbps)
        bandwidth_mbps = max(0, self.estimated_bandwidth_bps / 1_000_000)
        
        # 缓冲区长度 (秒)
        buffer_level = max(0, self.current_player_buffer_s)
        
        # 当前比特率等级 (归一化到0-1)
        current_bitrate_level = self.current_stream_index_by_abr / max(1, len(self.available_streams) - 1)
        
        # 最近下载时间 (秒)
        recent_download_time = 0
        if self.segment_download_stats:
            successful_downloads = [s for s in self.segment_download_stats if not s.get('error') and 'duration' in s]
            if successful_downloads:
                recent_download_time = successful_downloads[-1]['duration']
        
        # 下载成功率 (最近5个片段)
        success_rate = 1.0
        if self.segment_download_stats:
            recent_stats = self.segment_download_stats[-5:]
            if recent_stats:
                successful_count = len([s for s in recent_stats if not s.get('error')])
                success_rate = successful_count / len(recent_stats)
        
        state = np.array([
            bandwidth_mbps,
            buffer_level,
            current_bitrate_level,
            recent_download_time,
            success_rate
        ], dtype=np.float32)
        
        return state

    def calculate_qoe_reward(self, old_state, action, new_state):
        """计算QoE奖励函数"""
        # 获取选择的比特率
        selected_bitrate_mbps = self.available_streams[action]['bandwidth'] / 1_000_000
        
        # 1. 质量奖励 (比特率越高越好)
        max_bitrate_mbps = max(s['bandwidth'] for s in self.available_streams) / 1_000_000
        quality_reward = (selected_bitrate_mbps / max_bitrate_mbps) * 10
        
        # 2. 缓冲区健康度奖励
        buffer_target = 15.0  # 目标缓冲区长度
        buffer_level = new_state[1]
        
        if buffer_level < 2.0:
            # 严重卡顿惩罚
            buffer_reward = -50.0
        elif buffer_level < 5.0:
            # 低缓冲区惩罚
            buffer_reward = -10.0
        elif 10.0 <= buffer_level <= 20.0:
            # 理想缓冲区奖励
            buffer_reward = 5.0
        elif buffer_level > 30.0:
            # 过高缓冲区轻微惩罚
            buffer_reward = -2.0
        else:
            buffer_reward = 0
        
        # 3. 切换惩罚 (避免频繁切换)
        switch_penalty = 0
        if action != self.current_stream_index_by_abr:
            # 计算切换幅度
            switch_magnitude = abs(action - self.current_stream_index_by_abr)
            switch_penalty = -switch_magnitude * 2.0
        
        # 4. 带宽效率奖励
        bandwidth_efficiency = 0
        if old_state[0] > 0:  # 如果有带宽估计
            utilization = selected_bitrate_mbps / old_state[0]
            if 0.7 <= utilization <= 0.9:
                bandwidth_efficiency = 3.0  # 良好的带宽利用率
            elif utilization > 1.0:
                bandwidth_efficiency = -5.0  # 超出带宽能力
        
        # 5. 下载成功率奖励
        success_rate_reward = new_state[4] * 2.0  # 成功率越高越好
        
        total_reward = (quality_reward + buffer_reward + switch_penalty + 
                       bandwidth_efficiency + success_rate_reward)
        
        logger.debug(f"DQN Reward: Quality={quality_reward:.2f}, Buffer={buffer_reward:.2f}, "
                    f"Switch={switch_penalty:.2f}, BWEff={bandwidth_efficiency:.2f}, "
                    f"Success={success_rate_reward:.2f}, Total={total_reward:.2f}")
        
        return total_reward

    def select_action_dqn(self, state):
        """使用ε-贪婪策略选择动作"""
        if random.random() < self.epsilon:
            # 随机探索
            action = random.randint(0, self.action_size - 1)
            logger.debug(f"DQN: Random action selected: {action}")
        else:
            # 利用当前策略
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.dqn(state_tensor)
                action = q_values.argmax().item()
            logger.debug(f"DQN: Policy action selected: {action}")
        
        return action

    def train_dqn(self):
        """训练DQN网络"""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # 采样经验
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones)
        
        # 计算当前Q值
        current_q_values = self.dqn(states).gather(1, actions.unsqueeze(1))
        
        # 计算目标Q值
        with torch.no_grad():
            next_q_values = self.target_dqn(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # 计算损失
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.dqn.parameters(), 1.0)  # 梯度裁剪
        self.optimizer.step()
        
        # 更新目标网络
        self.training_step += 1
        if self.training_step % self.target_update_freq == 0:
            self.target_dqn.load_state_dict(self.dqn.state_dict())
            logger.info(f"DQN: Target network updated at step {self.training_step}")
        
        # 衰减探索率
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        logger.debug(f"DQN Training: Loss={loss.item():.4f}, Epsilon={self.epsilon:.3f}")

    def _update_current_abr_selected_url_logging(self):
        with self._internal_lock:
            if self.available_streams and 0 <= self.current_stream_index_by_abr < len(self.available_streams):
                self._current_selected_url_for_logging = self.available_streams[self.current_stream_index_by_abr]['url']
            else:
                self._current_selected_url_for_logging = None

    def add_segment_download_stat(self, url, size_bytes, duration_seconds):
        # 当接收到新的分片下载完成的统计信息时调用此方法
        logger.info(f"ADD_STAT_INPUT - URL: {url}, Size: {size_bytes} bytes, Duration: {duration_seconds:.4f}s") # 日志：记录输入参数
        if duration_seconds > 1e-5:
            throughput_bps = (size_bytes * 8) / duration_seconds
            timestamp = time.time()
            
            with self._internal_lock:
                old_ewma_for_log = self.ewma_bandwidth_bps # 保存更新前的值用于日志
                current_throughput_for_log = throughput_bps # 保存当前样本用于日志
                current_alpha_for_log = self.ewma_alpha # 保存当前alpha用于日志
                
                # 更新通用下载统计 (用于SWMA等)
                self.segment_download_stats.append({
                    'url': url, 'size': size_bytes, 'duration': duration_seconds,
                    'throughput_bps': throughput_bps, 'timestamp': timestamp
                })
                if len(self.segment_download_stats) > self.max_bw_stats_history:
                    self.segment_download_stats.pop(0)
                
                # 更新趋势分析用的带宽历史
                self.bandwidth_history_for_trend.append((timestamp, throughput_bps))
                if len(self.bandwidth_history_for_trend) > self.MAX_BW_HISTORY_FOR_TREND:
                    self.bandwidth_history_for_trend.pop(0)
                    
                # 更新SWMA带宽估计值
                if self.segment_download_stats:
                    throughputs = [s['throughput_bps'] for s in self.segment_download_stats if 'throughput_bps' in s]
                    if not throughputs:
                        return 0.0  
                    self.swma_bandwidth_bps = sum(throughputs) / len(throughputs)
                
                # 更新EWMA带宽估计值
                if self.ewma_bandwidth_bps == 0.0:
                    self.ewma_bandwidth_bps = throughput_bps
                else:
                    term1 = self.ewma_alpha * throughput_bps
                    term2_factor = (1 - self.ewma_alpha)
                    term2 = term2_factor * self.ewma_bandwidth_bps
                
                # 如果成功下载的是当前追踪的“卡住风险”分片，则重置相关状态
                if url == self.current_segment_url_being_downloaded:
                    logger.debug(f"ComprehensiveABR: Successfully downloaded tracked segment {url}. Resetting stuck-segment states.") # 日志: 成功下载追踪分片
                    self.current_segment_url_being_downloaded = None
                    self.last_segment_download_start_time = 0
                    self.current_segment_target_bitrate_bps = 0
                    self.consecutive_segment_timeouts = 0 # 成功下载，重置连续超时计数
        else:
            logger.warning(f"Segment download duration {duration_seconds:.4f}s too short or invalid for stat, URL: {url}") # 日志: 分片下载时长过短
            
    def notify_segment_download_started(self, segment_url, target_level_idx):
        """
        当代理开始下载一个新分片时由 client.py 调用。
        用于“提前预警”机制。

        参数:
            segment_url (str): 正在下载的分片的URL。
            target_level_idx (int): 客户端期望加载的这个分片所属的码率等级索引。
        """
        with self._internal_lock:
            self.current_segment_url_being_downloaded = segment_url
            self.last_segment_download_start_time = time.time()
            
            target_bitrate = 0
            if 0 <= target_level_idx < self.num_quality_levels:
                target_bitrate = self.available_streams[target_level_idx].get('bandwidth', 0)
            
            if target_bitrate > 0:
                self.current_segment_target_bitrate_bps = target_bitrate
            else: # 如果无法获取目标码率 (例如 target_level_idx 无效)
                logger.warning(f"ComprehensiveABR: Could not determine target bitrate for segment {segment_url} (level {target_level_idx}). Using lowest available bitrate for expectation.") # 日志: 无法确定目标码率
                self.current_segment_target_bitrate_bps = self.available_streams[0].get('bandwidth', 500000) # 默认用最低码率

            logger.debug(
                f"ComprehensiveABR: Segment download initiated: {segment_url} (TargetLvl: {target_level_idx}, "
                f"TargetRate: {self.current_segment_target_bitrate_bps/1000:.0f}Kbps). "
                f"Monitoring for slow download."
            ) # 日志: 分片下载开始监控

    def notify_rebuffering_event(self, timestamp_ms):
        """当 client.py 从前端接收到 REBUFFERING_START 事件时调用。"""
        with self._internal_lock:
            current_time_s = timestamp_ms / 1000.0
            self.recent_rebuffering_info['count'] += 1
            self.recent_rebuffering_info['last_timestamp'] = current_time_s
            # 可以选择清理过老的卡顿次数记录，例如只保留最近X分钟内的
            logger.info(f"ComprehensiveABR: Rebuffering event notified at {current_time_s:.2f}. Recent stall count: {self.recent_rebuffering_info['count']}.") # 日志:收到卡顿通知

    def report_download_error(self, url):
        logger.warning(f"ABR: Reported download error for segment from URL: {url}")
        # --- 错误报告加入统计，用于某些决策逻辑 ---
        self.segment_download_stats.append({
            'url': url,
            'error': True,
            'time': time.time()
        })
        self.consecutive_segment_timeouts += 1 # 增加连续超时计数
        if len(self.segment_download_stats) > self.max_bw_stats_history:
            self.segment_download_stats.pop(0)
            
    def update_player_buffer_level(self, buffer_seconds):
        with self._internal_lock:
            self.current_player_buffer_s = buffer_seconds

    def get_current_abr_decision_url(self):
        with self._internal_lock:
            return self._current_selected_url_for_logging
    
    # --- 带宽估计算法 ---
    def _get_last_segment_throughput(self):
        # 获取最近一次成功下载的分片的吞吐量 (用于SLBW)
        with self._internal_lock:
            if self.segment_download_stats:
                # 确保最后一个统计是有效的吞吐量数据
                for stat in reversed(self.segment_download_stats):
                    if 'throughput_bps' in stat:
                        estimated_throughput_bps = stat['throughput_bps']
                        if self.broadcast_bw_estimate and estimated_throughput_bps > 0:
                            self.broadcast_bw_estimate(estimated_throughput_bps / 1_000_000)
                        return estimated_throughput_bps
        return 0.0 # 如果没有有效数据则返回0
    
    def _get_swma_throughput(self):
        # 计算最近N个分片吞吐量的简单移动平均 (用于SWMA)
        with self._internal_lock:
            if self.broadcast_bw_estimate and self.swma_bandwidth_bps > 0:
                self.broadcast_bw_estimate(self.swma_bandwidth_bps / 1_000_000)
            return self.swma_bandwidth_bps
        
    def _get_ewma_throughput(self):
        # 获取指数加权移动平均带宽 (用于EWMA)
        # EWMA值在 add_segment_download_stat 中已实时更新
        with self._internal_lock:
            if self.broadcast_bw_estimate and self.ewma_bandwidth_bps > 0:
                self.broadcast_bw_estimate(self.ewma_bandwidth_bps / 1_000_000)
            return self.ewma_bandwidth_bps
        
    def get_network_analysis_features(self):
        """
        分析 self.bandwidth_history_for_trend 来判断网络趋势和波动性。
        返回一个包含分析特征的字典。
        """
        features = {
            'mean_bw_bps': 0.0, 'std_dev_bw_bps': 0.0, 'coeff_variation': 0.0,
            'slope_bps_per_s': 0.0, 'trend': "UNKNOWN", # STABLE, INCREASING, DECREASING, VOLATILE, SUDDEN_DROP
            'is_sudden_drop': False, 'is_volatile': False,
            'latest_bw_bps': 0.0
        }
        with self._internal_lock:
            if len(self.bandwidth_history_for_trend) < 2: # 需要至少两个点才能分析趋势
                if self.bandwidth_history_for_trend:
                    features['latest_bw_bps'] = self.bandwidth_history_for_trend[-1][1]
                    features['mean_bw_bps'] = features['latest_bw_bps']
                    features['trend'] = "STABLE" # 数据太少，假设稳定
                return features

            timestamps = np.array([t for t, bw in self.bandwidth_history_for_trend])
            bitrates = np.array([bw for t, bw in self.bandwidth_history_for_trend])

            features['latest_bw_bps'] = bitrates[-1]
            features['mean_bw_bps'] = np.mean(bitrates)
            features['std_dev_bw_bps'] = np.std(bitrates)

            if features['mean_bw_bps'] > 0:
                features['coeff_variation'] = features['std_dev_bw_bps'] / features['mean_bw_bps']

            # 简化的斜率计算 (最近两点，或线性回归)
            if len(timestamps) >= 2:
                dt = timestamps[-1] - timestamps[0] # 使用整个窗口的时间差
                if dt > 0.1: # 避免除以过小的时间
                    # 使用窗口内所有点的线性回归斜率可能更稳健
                    # (这里简化为使用首尾两个主要数据点，或最近两个点)
                    slope_calc_points = min(len(timestamps), 3) # 使用最近3个点计算斜率
                    if len(timestamps) >= slope_calc_points :
                        idx_start = -slope_calc_points
                        slope_dt = timestamps[-1] - timestamps[idx_start]
                        slope_dbw = bitrates[-1] - bitrates[idx_start]
                        if slope_dt > 0.1:
                            features['slope_bps_per_s'] = slope_dbw / slope_dt
                    if len(timestamps) >=3 : # 使用更稳定的趋势判断，比如基于N点线性回归的斜率
                        # 为了简单，我们用一个简化的方法：最近值与平均值的比较，以及最近几个值的趋势
                        # 使用Numpy的polyfit进行线性回归获得斜率
                        # (需要 pip install numpy scipy)
                        try:
                            # 确保时间戳是相对的，以避免大的浮点数问题 (或者直接用原始时间戳)
                            # relative_timestamps = timestamps - timestamps[0]
                            # slope, intercept = np.polyfit(relative_timestamps, bitrates, 1)
                            # features['slope_bps_per_s'] = slope

                            # 更简单的：比较最近一半和之前一半的平均值
                            mid_point = len(bitrates) // 2
                            if mid_point > 0:
                                avg_first_half = np.mean(bitrates[:mid_point])
                                avg_second_half = np.mean(bitrates[mid_point:])
                                if avg_second_half > avg_first_half * (1 + self.COMP_PARAMS["volatility_coeff_variation_threshold"]*0.5): # 显著增加
                                    features['trend'] = "INCREASING"
                                elif avg_second_half < avg_first_half * (1 - self.COMP_PARAMS["volatility_coeff_variation_threshold"]*0.5): # 显著减少
                                    features['trend'] = "DECREASING"
                                else:
                                    features['trend'] = "STABLE"
                        except Exception as e_fit:
                            logger.debug(f"Could not perform polyfit for trend: {e_fit}")
                            features['trend'] = "STABLE" # 出错则认为稳定

            # 判断骤降和波动
            if features['trend'] == "DECREASING" and \
               bitrates[-1] < features['mean_bw_bps'] * (1 - self.COMP_PARAMS["bw_drop_minor_ratio"]):
                if bitrates[-1] < features['mean_bw_bps'] * (1 - self.COMP_PARAMS["bw_drop_major_ratio"]):
                    logger.warning(f"Network Analysis: Potential MAJOR sudden drop detected. Current: {bitrates[-1]/1e6:.2f} Mbps, Mean: {features['mean_bw_bps']/1e6:.2f} Mbps")
                    features['is_sudden_drop'] = True
                    features['trend'] = "SUDDEN_DROP"
                else:
                    logger.info(f"Network Analysis: Potential MINOR sudden drop detected. Trend: {features['trend']}")
                    # features['trend'] 仍然是 DECREASING

            if features['coeff_variation'] > self.COMP_PARAMS["volatility_coeff_variation_threshold"]:
                logger.info(f"Network Analysis: VOLATILE network detected. CoV: {features['coeff_variation']:.2f}")
                features['is_volatile'] = True
                if features['trend'] == "STABLE": # 如果趋势稳定但波动大，则标记为波动
                    features['trend'] = "VOLATILE"
            
            if features['trend'] == "UNKNOWN" and not features['is_sudden_drop'] and not features['is_volatile']:
                 features['trend'] = "STABLE"


        logger.debug(f"Network Analysis Features: {features}") # 日志: 网络分析特征
        return features

    # --- 决策逻辑的主分发方法 ---
    def _abr_decision_logic(self):
        # 根据 self.logic_type 调用相应的决策逻辑函数
        if self.num_quality_levels == 0: # 如果没有可用的码率等级
            logger.warning("ABR: No quality levels available to make a decision.") # 日志: 无可用码率等级
            return

        if self.logic_type == self.LOGIC_TYPE_SLBW:
            self._logic_slbw()
        elif self.logic_type == self.LOGIC_TYPE_SWMA:
            self._logic_swma()
        elif self.logic_type == self.LOGIC_TYPE_EWMA:
            self._logic_ewma()
        elif self.logic_type == self.LOGIC_TYPE_BUFFER_ONLY:
            self._logic_buffer_only()
        elif self.logic_type == self.LOGIC_TYPE_BANDWIDTH_BUFFER:
            self._logic_bandwidth_buffer()
        elif self.logic_type == self.LOGIC_TYPE_DQN:
            self._logic_dqn()
        elif self.logic_type == self.LOGIC_TYPE_COMPREHENSIVE:
            self._logic_comprehensive_rules()
        else:
            logger.warning(f"Unknown ABR logic type: '{self.logic_type}'. Defaulting to LOGIC_TYPE_BANDWIDTH_BUFFER.") # 日志: 未知逻辑类型
            self.logic_type = self.LOGIC_TYPE_BANDWIDTH_BUFFER # 重置为默认以防后续错误
            self._logic_bandwidth_buffer()
            
    def _logic_dqn(self):
        """基于DQN的ABR决策逻辑"""
        if not self.available_streams or len(self.available_streams) <= 1:
            return

        # 更新带宽估计
        estimated_bw_bps = self._get_ewma_throughput()
        
        # 获取当前状态
        current_state = self.get_current_state()
        
        # 如果有上一步的经验，添加到经验回放缓冲区
        if self.last_state is not None and self.last_action is not None:
            reward = self.calculate_qoe_reward(self.last_state, self.last_action, current_state)
            done = False  # 流媒体场景通常不会结束
            
            self.replay_buffer.push(
                self.last_state, self.last_action, reward, current_state, done
            )
            
            # 训练网络
            self.train_dqn()
        
        # 选择动作
        action = self.select_action_dqn(current_state)
        
        # 执行动作（切换码率）
        if action != self.current_stream_index_by_abr:
            old_stream_info = self.available_streams[self.current_stream_index_by_abr]
            self.current_stream_index_by_abr = action
            new_stream_info = self.available_streams[self.current_stream_index_by_abr]
            self._update_current_abr_selected_url_logging()
            
            logger.info(f"ABR DECISION (DQN): Switch from level {self.current_stream_index_by_abr} "
                       f"(~{old_stream_info.get('bandwidth',0)/1000:.0f}Kbps) to {action} "
                       f"(~{new_stream_info.get('bandwidth',0)/1000:.0f}Kbps). "
                       f"Est.BW={self.estimated_bandwidth_bps/1000:.0f}Kbps, "
                       f"Buf={self.current_player_buffer_s:.2f}s, "
                       f"Epsilon={self.epsilon:.3f}")
            
            self.broadcast_abr_decision(self.current_stream_index_by_abr)
        else:
            logger.info(f"ABR DECISION (DQN): No change from level {action}. "
                       f"Est.BW={self.estimated_bandwidth_bps/1000:.0f}Kbps, "
                       f"Buf={self.current_player_buffer_s:.2f}s")
        
        # 保存当前状态和动作用于下次计算奖励
        self.last_state = current_state.copy()
        self.last_action = action
        
        # 定期保存模型
        if self.training_step % 500 == 0 and self.training_step > 0:
            self.save_model()
            
    # --- 辅助方法：执行切换决策 ---
    def _execute_switch_decision(self, next_best_index, current_logic_name, decision_rationale_log="", is_emergency=False):
        # 封装通用的切换逻辑，包括日志记录和广播
        current_level_index = self.current_stream_index_by_abr
        
        if next_best_index == current_level_index:
            # logger.info(f"ABR DECISION ({current_logic_name}): No change from current level {current_level_index}. {decision_rationale_log}") # 这条日志可能太频繁，只在真决策时打
            return
          
        # 切换抑制逻辑
        current_min_switch_interval = self.MIN_SWITCH_INTERVAL
        if is_emergency: # 如果是紧急情况（例如，强制降到最低码率）
            current_min_switch_interval = self.COMP_PARAMS.get("min_switch_interval_emergency_override_s", 1.0) # 允许更快的紧急切换
            logger.warning(f"ABR: Emergency condition detected, using shorter switch interval: {current_min_switch_interval}s")

        if time.time() - self.last_switch_time < current_min_switch_interval:
            logger.info(f"ABR DECISION ({current_logic_name}): Switch to level {next_best_index} proposed, "
                        f"but it's too soon since last switch ({time.time() - self.last_switch_time:.1f}s ago, MinInterval: {current_min_switch_interval:.1f}s). "
                        f"Holding current level {current_level_index}. Rationale for proposed switch: {decision_rationale_log}")
            return 

        self.last_switch_time = time.time() 
        old_stream_info = self.available_streams[current_level_index]
        self.current_stream_index_by_abr = next_best_index
        new_stream_info = self.available_streams[next_best_index]
        self._update_current_abr_selected_url_logging()

        logger.info(f"ABR DECISION ({current_logic_name}): Switch from level {current_level_index} "
                    f"(~{old_stream_info.get('bandwidth',0)/1000:.0f}Kbps) to {next_best_index} "
                    f"(~{new_stream_info.get('bandwidth',0)/1000:.0f}Kbps). {decision_rationale_log}")
        self.broadcast_abr_decision(self.current_stream_index_by_abr)
        
    # --- 具体ABR决策逻辑实现 ---
    def _logic_slbw(self):
        # SLBW (Simple Last Bandwidth) - 只根据最近一次成功下载的分片吞吐量做决策
        current_level_index = self.current_stream_index_by_abr
        estimated_bw_bps = self._get_last_segment_throughput()
        safety_factor = 0.8 # SLBW 的安全系数

        # 日志记录当前状态
        logger.info(
            f"ABR LOGIC (SLBW): Est.BW (Last): {estimated_bw_bps / 1000:.0f} Kbps, "
            f"Current Level Idx: {current_level_index}"
        )

        if estimated_bw_bps == 0.0: # 如果最近一次吞吐量为0 (可能因为还没有统计数据)
            logger.info("ABR LOGIC (SLBW): No valid last segment throughput. Sticking to current level.")
            # 第一次或没有统计数据时，不主动切换，依赖初始设置或上一次决策
            return

        target_bitrate_bps = estimated_bw_bps * safety_factor
        next_best_index = 0 # 默认选择最低码率，以防万一

        # 从最高码率往下找，找到第一个能被目标带宽支持的
        for i in range(self.num_quality_levels - 1, -1, -1):
            stream_bw = self.available_streams[i].get('bandwidth', 0)
            logger.debug(f"ABR LOGIC (SLBW): Checking level {i} (BW={stream_bw/1000:.0f}Kbps) vs Target_Select_BW={target_bitrate_bps/1000:.0f}Kbps.") # 调试日志
            if target_bitrate_bps >= stream_bw:
                next_best_index = i
                break
        
        rationale = f"Est.BW (Last): {estimated_bw_bps/1000:.0f} Kbps, Target Sel. BW: {target_bitrate_bps/1000:.0f} Kbps."
        self._execute_switch_decision(next_best_index, "SLBW", rationale)
        
    def _logic_swma(self):
        # SWMA (Simple Moving Window Average) - 根据最近N个分片吞吐量的简单平均值做决策
        current_level_index = self.current_stream_index_by_abr
        estimated_bw_bps = self._get_swma_throughput()
        safety_factor = 0.85 # SWMA 的安全系数，平均值相对稳定，安全系数可略高

        logger.info(
            f"ABR LOGIC (SWMA): Est.BW (Avg): {estimated_bw_bps / 1000:.0f} Kbps, "
            f"Current Level Idx: {current_level_index}"
        )

        if estimated_bw_bps == 0.0:
            logger.info("ABR LOGIC (SWMA): Average bandwidth estimate is zero. Sticking to current level.")
            return

        target_bitrate_bps = estimated_bw_bps * safety_factor
        next_best_index = 0

        for i in range(self.num_quality_levels - 1, -1, -1):
            stream_bw = self.available_streams[i].get('bandwidth', 0)
            logger.debug(f"ABR LOGIC (SWMA): Checking level {i} (BW={stream_bw/1000:.0f}Kbps) vs Target_Select_BW={target_bitrate_bps/1000:.0f}Kbps.")
            if target_bitrate_bps >= stream_bw:
                next_best_index = i
                break
        
        rationale = f"Est.BW (SWMA): {estimated_bw_bps/1000:.0f} Kbps, Target Sel. BW: {target_bitrate_bps/1000:.0f} Kbps."
        self._execute_switch_decision(next_best_index, "SWMA", rationale)
        
    def _logic_ewma(self):
        # EWMA (Exponentially Weighted Moving Average) - 根据指数加权移动平均带宽做决策
        current_level_index = self.current_stream_index_by_abr
        estimated_bw_bps = self._get_ewma_throughput() # EWMA值在add_segment_download_stat中更新
        safety_factor = 0.9 # EWMA 通常更平滑，安全系数可以设置得相对较高

        logger.info(
            f"ABR LOGIC (EWMA): Est.BW (EWMA): {estimated_bw_bps / 1000:.0f} Kbps (alpha={self.ewma_alpha}), "
            f"Current Level Idx: {current_level_index}"
        )

        if estimated_bw_bps == 0.0:
            logger.info("ABR LOGIC (EWMA): EWMA bandwidth estimate is zero. Sticking to current level.")
            return

        target_bitrate_bps = estimated_bw_bps * safety_factor
        next_best_index = 0

        for i in range(self.num_quality_levels - 1, -1, -1):
            stream_bw = self.available_streams[i].get('bandwidth', 0)
            logger.debug(f"ABR LOGIC (EWMA): Checking level {i} (BW={stream_bw/1000:.0f}Kbps) vs Target_Select_BW={target_bitrate_bps/1000:.0f}Kbps.")
            if target_bitrate_bps >= stream_bw:
                next_best_index = i
                break
        
        rationale = f"Est.BW (EWMA): {estimated_bw_bps/1000:.0f} Kbps, Target Sel. BW: {target_bitrate_bps/1000:.0f} Kbps."
        self._execute_switch_decision(next_best_index, "EWMA", rationale)
        
    def _logic_buffer_only(self):
        # 只根据缓冲区占用情况决策
        with self._internal_lock:
            current_buffer_s = self.current_player_buffer_s
        current_level_index = self.current_stream_index_by_abr
        next_best_index = current_level_index

        # 定义缓冲区阈值 (这些值需要仔细调整，可以作为类的配置参数)
        BUFFER_LOW_FOR_DOWNGRADE = 10.0  # s, 低于此值考虑降级
        BUFFER_HIGH_FOR_UPGRADE = 25.0   # s, 高于此值考虑升级

        logger.info(
            f"ABR LOGIC (BUFFER_ONLY): Buffer: {current_buffer_s:.2f}s, "
            f"Current Level Idx: {current_level_index}"
        )
        rationale = f"Buffer: {current_buffer_s:.2f}s."

        # 缓冲区过高，且当前不是最高码率，尝试提升质量
        if current_buffer_s > BUFFER_HIGH_FOR_UPGRADE and current_level_index < (self.num_quality_levels - 1):
            next_best_index = current_level_index + 1 # 尝试升一级
            rationale += f" High buffer, attempting upgrade to {next_best_index}."
            logger.info(f"ABR LOGIC (BUFFER_ONLY): High buffer ({current_buffer_s:.2f}s), attempting upgrade to level {next_best_index}.")

        # 缓冲区过低，且当前不是最低码率，尝试降低质量
        elif current_buffer_s < BUFFER_LOW_FOR_DOWNGRADE and current_level_index > 0:
            next_best_index = current_level_index - 1 # 尝试降一级
            rationale += f" Low buffer, attempting downgrade to {next_best_index}."
            logger.info(f"ABR LOGIC (BUFFER_ONLY): Low buffer ({current_buffer_s:.2f}s), attempting downgrade to level {next_best_index}.")
        
        self._execute_switch_decision(next_best_index, "BUFFER_ONLY", rationale)
        
    def _logic_bandwidth_buffer(self):
        # 使用多个缓冲阈值动态调整安全系数，并结合带宽进行决策
        current_level_index = self.current_stream_index_by_abr
        estimated_bw_bps = self._get_ewma_throughput() # 示例：这里可以使用EWMA作为带宽估计，或SWMA
        with self._internal_lock:
            current_buffer_s = self.current_player_buffer_s

        logger.info(
            f"ABR LOGIC (BW_BUFFER): Est.BW: {estimated_bw_bps / 1000:.0f} Kbps, "
            f"Buffer: {current_buffer_s:.2f}s, Current Level Idx: {current_level_index}"
        )

        if estimated_bw_bps == 0.0 and not self.segment_download_stats: # 若无带宽统计数据
            logger.info("ABR LOGIC (BW_BUFFER): No bandwidth stats yet. Sticking to current level.")
            return

        # 缓冲区阈值和动态安全系数逻辑
        BUFFER_THRESHOLD_LOW = 10.0       #
        BUFFER_THRESHOLD_MEDIUM = 15.0    #
        BUFFER_THRESHOLD_HIGH = 25.0      #
        BUFFER_THRESHOLD_EMERGENCY = 5.0  #

        dynamic_safety_factor = 0.8 # 默认安全系数
        if current_buffer_s < BUFFER_THRESHOLD_EMERGENCY: # 紧急情况，最保守
            dynamic_safety_factor = 0.5 
        elif current_buffer_s < BUFFER_THRESHOLD_LOW: #
            dynamic_safety_factor = 0.6
        elif current_buffer_s < BUFFER_THRESHOLD_MEDIUM: #
            dynamic_safety_factor = 0.8
        elif current_buffer_s < BUFFER_THRESHOLD_HIGH: #
            dynamic_safety_factor = 0.9
        else: # 缓冲区非常高，可以更激进
            dynamic_safety_factor = 0.95 #

        target_bitrate_bps = estimated_bw_bps * dynamic_safety_factor
        logger.debug(f"ABR LOGIC (BW_BUFFER): DynamicSF: {dynamic_safety_factor:.2f}, TargetSelectBW: {target_bitrate_bps / 1000:.0f} Kbps")

        next_best_index = current_level_index # 默认保持当前等级

        # 决策优先级：紧急缓冲 -> 高缓冲升档 -> (低缓冲 或 带宽不足)降档 -> 带宽稳定性检查降档
        if current_buffer_s < BUFFER_THRESHOLD_EMERGENCY and current_level_index > 0: #
            next_best_index = 0 # 紧急情况，降至最低码率
            logger.warning(f"ABR LOGIC (BW_BUFFER): EMERGENCY! Buffer {current_buffer_s:.2f}s. Switching to lowest quality (index 0).")
        elif current_buffer_s > BUFFER_THRESHOLD_HIGH and current_level_index < (self.num_quality_levels - 1): #
            # 尝试升档：找到目标带宽能支持的最高码率（在当前之上）
            potential_upgrade_index = current_level_index
            for i in range(self.num_quality_levels - 1, current_level_index, -1): # 从最高往下找
                if target_bitrate_bps >= self.available_streams[i].get('bandwidth', 0):
                    potential_upgrade_index = i
                    break 
            if potential_upgrade_index > current_level_index:
                logger.info(f"ABR LOGIC (BW_BUFFER): UPGRADE condition (Buffer {current_buffer_s:.2f}s > {BUFFER_THRESHOLD_HIGH:.1f}s). Potential index: {potential_upgrade_index}")
                next_best_index = potential_upgrade_index
        elif (current_buffer_s < BUFFER_THRESHOLD_LOW or \
              target_bitrate_bps < self.available_streams[current_level_index].get('bandwidth', 0)) and \
             current_level_index > 0: #
            # 尝试降档：找到目标带宽能支持的最高码率（在当前之下，或最低）
            potential_downgrade_index = 0 # 默认降到最低
            for i in range(current_level_index - 1, -1, -1): # 从当前之下往下找
                if target_bitrate_bps >= self.available_streams[i].get('bandwidth', 0):
                    potential_downgrade_index = i
                    break
            logger.info(f"ABR LOGIC (BW_BUFFER): DOWNGRADE condition (Low Buffer or Insufficient BW). Potential index: {potential_downgrade_index}")
            next_best_index = potential_downgrade_index
        
        rationale = f"Est.BW: {estimated_bw_bps/1000:.0f} Kbps, Buf: {current_buffer_s:.2f}s, DynSF: {dynamic_safety_factor:.2f}, TargetSelBW: {target_bitrate_bps/1000:.0f} Kbps."
        self._execute_switch_decision(next_best_index, "BW_BUFFER", rationale)
        

    def _logic_comprehensive_rules(self):
        # 步骤 0: 基本信息
        if self.num_quality_levels == 0:
            logger.warning("ComprehensiveABR: No quality levels available.")
            return
        
        current_level_index = self.current_stream_index_by_abr
        next_best_index = current_level_index 

        with self._internal_lock:
            current_buffer_s = self.current_player_buffer_s
            consecutive_errors = self.consecutive_segment_timeouts 
            current_ewma_bps = self.ewma_bandwidth_bps 
            current_seg_url = self.current_segment_url_being_downloaded
            current_seg_start_time = self.last_segment_download_start_time
            current_seg_target_bps = self.current_segment_target_bitrate_bps

        logger.info(
            f"ABR LOGIC (COMPREHENSIVE): InitState - EWMA_BW: {current_ewma_bps / 1000:.0f} Kbps, "
            f"Buffer: {current_buffer_s:.2f}s, CurLvl: {current_level_index}, ConsecutiveErrors: {consecutive_errors}"
        )

        is_current_segment_stuck = False
        stuck_severity = "NONE" # "MODERATE", "SEVERE"
        effective_bw_for_decision = current_ewma_bps

        # --- 步骤 1: 卡顿预警 (你的核心Intuition之一) ---
        if current_seg_url and current_seg_start_time > 0 and current_seg_target_bps > 0:
            time_elapsed_downloading = time.time() - current_seg_start_time
            # ... (expected_download_time 的计算逻辑保持之前的优化版本)
            nominal_duration_s = self.NOMINAL_SEGMENT_DURATION_S
            nominal_segment_size_bits = current_seg_target_bps * nominal_duration_s
            capped_ewma_for_expectation = current_ewma_bps
            expectation_ewma_cap_factor = self.COMP_PARAMS.get("expectation_ewma_cap_factor", 3.0) 
            if current_ewma_bps > current_seg_target_bps * expectation_ewma_cap_factor:
                capped_ewma_for_expectation = current_seg_target_bps * expectation_ewma_cap_factor
            if capped_ewma_for_expectation > 1000:
                expected_time_based_on_ewma = nominal_segment_size_bits / capped_ewma_for_expectation
            else:
                fallback_factor = self.COMP_PARAMS.get("segment_stuck_fallback_duration_factor", 2.5)
                expected_time_based_on_ewma = nominal_duration_s * fallback_factor
            abs_min_expected_time_s = self.COMP_PARAMS.get("abs_min_expected_time_s", 1.0)
            min_expected_time_nominal_ratio = self.COMP_PARAMS.get("min_expected_time_nominal_ratio", 0.25)
            robust_min_expected_time_s = max(abs_min_expected_time_s, nominal_duration_s * min_expected_time_nominal_ratio)
            expected_download_time = max(expected_time_based_on_ewma, robust_min_expected_time_s)
            
            logger.debug(f"ComprehensiveABR: StuckCheck - Seg: {current_seg_url.split('/')[-1][-25:]}, Elapsed: {time_elapsed_downloading:.2f}s, Expected: {expected_download_time:.2f}s")

            stuck_moderate_threshold = expected_download_time * self.COMP_PARAMS.get("stuck_severity_threshold_factor", 1.5)
            stuck_severe_threshold = expected_download_time * self.COMP_PARAMS.get("stuck_severe_severity_threshold_factor", 2.5) # 比segment_stuck_timeout_factor更早触发惩罚

            if time_elapsed_downloading > stuck_severe_threshold :
                is_current_segment_stuck = True
                stuck_severity = "SEVERE"
                penalty_factor = self.COMP_PARAMS.get("stuck_penalty_factor_severe", 0.3)
                logger.warning(f"ComprehensiveABR: Segment SEVERELY STUCK ({stuck_severity})!")
                effective_bw_for_decision = current_ewma_bps * penalty_factor
            elif time_elapsed_downloading > stuck_moderate_threshold:
                is_current_segment_stuck = True # 标记为卡顿，但不一定是最高惩罚等级
                stuck_severity = "MODERATE"
                penalty_factor = self.COMP_PARAMS.get("stuck_penalty_factor_moderate", 0.6)
                logger.warning(f"ComprehensiveABR: Segment MODERATELY STUCK ({stuck_severity}).")
                effective_bw_for_decision = current_ewma_bps * penalty_factor
            
            if is_current_segment_stuck:
                 # 进一步结合目标码率进行惩罚 (可选，通过参数控制)
                if self.COMP_PARAMS.get("stuck_use_target_bitrate_penalty", True):
                    bw_based_on_stuck_target = current_seg_target_bps * self.COMP_PARAMS.get("stuck_target_penalty_factor", 0.5)
                    effective_bw_for_decision = min(effective_bw_for_decision, bw_based_on_stuck_target)

                effective_bw_for_decision = max(effective_bw_for_decision, self.available_streams[0]['bandwidth'] * 0.8) # 最低保障
                logger.info(f"ComprehensiveABR: STUCK ({stuck_severity}) - Effective BW adjusted to {effective_bw_for_decision / 1000:.0f} Kbps.")

        # --- 步骤 1b: 处理连续下载错误 (PANIC MODE) ---
        if consecutive_errors >= self.COMP_PARAMS.get("max_consecutive_error_for_panic", 3):
            logger.error(
                f"ComprehensiveABR: PANIC! {consecutive_errors} consecutive download errors. "
                f"Forcing to lowest quality and entering long recovery."
            )
            if current_level_index > 0 : # 只有当前不是最低才执行强制切换
                self._execute_switch_decision(0, "COMPREHENSIVE-Panic", f"{consecutive_errors} errors.")
            self.RECOVERY_HOLD_UNTIL_TIMESTAMP = time.time() + self.COMP_PARAMS.get("panic_recovery_hold_duration_s", 30.0)
            # 通常在成功下载后重置 consecutive_errors，这里恐慌后暂时不重置，等待成功或超时
            if self.broadcast_bw_estimate and effective_bw_for_decision > 0 : # 广播可能降低的带宽
                 self.broadcast_bw_estimate(effective_bw_for_decision / 1_000_000)
            return # 紧急情况，结束本轮决策
        
        if self.broadcast_bw_estimate: # 正常广播
            bw_to_broadcast = effective_bw_for_decision if effective_bw_for_decision > 0 else current_ewma_bps
            if bw_to_broadcast > 0:
                self.broadcast_bw_estimate(bw_to_broadcast / 1_000_000)

         # --- 步骤 2 & 3: 分析网络状态并调整安全系数 (你的核心Intuition之一) ---
        network_features = self.get_network_analysis_features()
        current_safety_factor = self.COMP_PARAMS["safety_factor_stable"]
        in_recovery_mode = time.time() < self.RECOVERY_HOLD_UNTIL_TIMESTAMP
        
        # --- 核心状态判断逻辑 (用于日志和决策) ---
        # base_network_condition: 主要网络趋势判断
        # derived_abr_state: 结合了恢复模式、卡顿等的最终ABR行动状态
        base_network_condition = network_features['trend'] # STABLE, INCREASING, DECREASING, VOLATILE, SUDDEN_DROP
        derived_abr_state = base_network_condition # 初始化为基础网络趋势

        if in_recovery_mode:
            current_safety_factor = self.COMP_PARAMS["safety_factor_recovering"]
            derived_abr_state = f"RECOVERING (from {base_network_condition})" # 更具体的恢复状态
            if network_features['is_sudden_drop'] or (is_current_segment_stuck and base_network_condition == "DECREASING"): # 恢复期内再次恶化
                self.RECOVERY_HOLD_UNTIL_TIMESTAMP = time.time() + self.COMP_PARAMS["recovery_hold_duration_s"] # 延长恢复期
                derived_abr_state += "+EXTENDED"
        elif network_features['is_sudden_drop']:
            current_safety_factor = self.COMP_PARAMS.get("safety_factor_sudden_drop", 0.5)
            # 强制将有效带宽拉向最新样本（如果骤降被检测到，最新样本通常较低）
            if network_features['latest_bw_bps'] > 0 and network_features['latest_bw_bps'] < effective_bw_for_decision :
                effective_bw_for_decision = min(effective_bw_for_decision, network_features['latest_bw_bps'] * self.COMP_PARAMS.get("suddendrop_ewma_discount_factor",0.8)) # 使用最新样本打折
            self.RECOVERY_HOLD_UNTIL_TIMESTAMP = time.time() + self.COMP_PARAMS.get("suddendrop_recovery_duration_s", self.COMP_PARAMS["recovery_hold_duration_s"] * 1.5) # 骤降恢复期可能更长
            in_recovery_mode = True
            derived_abr_state = "SUDDEN_DROP_TRIGGERED_RECOVERY"
        elif network_features['is_volatile']:
            current_safety_factor = self.COMP_PARAMS["safety_factor_volatile"]
            derived_abr_state = "VOLATILE"
        elif base_network_condition == "DECREASING":
            current_safety_factor = self.COMP_PARAMS.get("safety_factor_decreasing", 0.7)
            derived_abr_state = "DECREASING"
        elif base_network_condition == "INCREASING":
            current_safety_factor = self.COMP_PARAMS.get("safety_factor_increasing", 0.9)
            derived_abr_state = "INCREASING"
        
        if is_current_segment_stuck: # 如果当前分片卡住，这是一个强烈的信号
            # 即使不在恢复模式，卡顿也应该导致更保守的行为
            current_safety_factor = min(current_safety_factor, self.COMP_PARAMS.get("safety_factor_stuck_segment", self.COMP_PARAMS["safety_factor_recovering"])) # 新参数或用恢复期SF
            if derived_abr_state == base_network_condition : # 如果状态未被其他条件改变
                derived_abr_state = f"{base_network_condition}+STUCK({stuck_severity})"
            else: # 如果状态已被改变 (例如 RECOVERING)，附加STUCK信息
                derived_abr_state += f"+STUCK({stuck_severity})"
        
        target_selectable_bps = effective_bw_for_decision * current_safety_factor
        
        # --- 清晰打印ABR的状态判断 ---
        logger.info(f"ComprehensiveABR: State for decision - "
                    f"EffectiveBW: {effective_bw_for_decision/1e6:.2f}Mbps, "
                    f"SafetyFactor({derived_abr_state}): {current_safety_factor:.2f}, "
                    f"TargetSelectableBW: {target_selectable_bps/1e6:.2f}Mbps, "
                    f"Buffer: {current_buffer_s:.1f}s, "
                    f"StuckSeverity: {stuck_severity}, "
                    f"InRecoveryMode: {in_recovery_mode}")

        # --- 步骤 4: 基于带宽、缓冲区和网络状态的码率选择 ---
        
        # 紧急缓冲判断 (最高优先级)
        if current_buffer_s < self.COMP_PARAMS["buffer_emergency_threshold"] and current_level_index > 0:
            next_best_index = 0 
            logger.warning(f"ComprehensiveABR: EMERGENCY BUFFER ({current_buffer_s:.2f}s). Targeting level 0.")
            # 允许紧急降级覆盖部分切换抑制（在 _execute_switch_decision 中实现）
            # 这里传递一个特殊理由
            rationale_for_emergency = f"EMERGENCY BUFFER ({current_buffer_s:.2f}s)"
        
        # 恢复模式下的逻辑 (如果上面紧急缓冲没触发)
        elif in_recovery_mode:
            logger.info(f"ComprehensiveABR: In RECOVERY MODE ({derived_abr_state}).")
            # 降级：如果当前码率都无法维持
            if target_selectable_bps < self.available_streams[current_level_index]['bandwidth'] * 0.95 and current_level_index > 0: # 恢复期对维持当前也要求高一点
                temp_idx = 0
                for i in range(current_level_index - 1, -1, -1):
                    if target_selectable_bps >= self.available_streams[i]['bandwidth']:
                        temp_idx = i; break
                next_best_index = temp_idx
                logger.info(f"ComprehensiveABR: RECOVERY - BW insufficient. Downgrading to {next_best_index}.")
            # 升档（非常保守）
            elif current_buffer_s > self.COMP_PARAMS["buffer_high_for_upgrade"] and \
                 not is_current_segment_stuck and \
                 network_features['trend'] == "INCREASING" and \
                 current_level_index < (self.num_quality_levels - 1):
                next_level_bw = self.available_streams[current_level_index + 1]['bandwidth']
                if target_selectable_bps > next_level_bw * self.COMP_PARAMS.get("recovery_upgrade_margin_factor", 1.25):
                    next_best_index = current_level_index + 1
                    logger.info(f"ComprehensiveABR: RECOVERY - Cautious upgrade to level {next_best_index} (Trend: INCREASING).")
                else:
                    logger.info(f"ComprehensiveABR: RECOVERY - Conditions not met for cautious upgrade. Holding {current_level_index}.")
            else: # 其他恢复期情况，倾向于维持不变或等待上述条件触发降级
                logger.info(f"ComprehensiveABR: RECOVERY - Holding level {current_level_index} (Buffer: {current_buffer_s:.1f}s, Trend: {network_features['trend']}).")

        # 正常模式下的升降档逻辑
        else:
            # --- 升档逻辑 (强化缓冲区动态判断) ---
            can_consider_upgrade = (derived_abr_state in ["STABLE", "INCREASING"] and # 只有在稳定或上升趋势时
                                   not is_current_segment_stuck and # 当前没有卡顿
                                   current_buffer_s > self.COMP_PARAMS["buffer_high_for_upgrade"] and
                                   current_level_index < (self.num_quality_levels - 1))
            
            if can_consider_upgrade:
                logger.info(f"ComprehensiveABR: NORMAL ({derived_abr_state}) - Considering UPGRADE from level {current_level_index}.")
                potential_upgrade_idx = current_level_index
                # ... (寻找 potential_upgrade_idx 的逻辑保持不变) ...
                for i in range(self.num_quality_levels - 1, current_level_index, -1):
                    required_bw_for_this_level = self.available_streams[i]['bandwidth'] * self.COMP_PARAMS.get("normal_upgrade_selection_margin", 1.05)
                    if target_selectable_bps >= required_bw_for_this_level:
                        potential_upgrade_idx = i
                        break
                
                if potential_upgrade_idx > current_level_index:
                    new_level_bitrate_bps = self.available_streams[potential_upgrade_idx]['bandwidth']
                    # --- 详细的缓冲区可持续性检查 ---
                    # 假设切换会“消耗”或“清空”一部分当前缓冲的感知价值
                    # 例如，固定损失几秒，或者当前缓冲的一小部分，取更保守者
                    buffer_lost_on_switch_estimate_s = max(
                        self.COMP_PARAMS.get("buffer_fixed_loss_on_upgrade_s", 2.0),
                        current_buffer_s * self.COMP_PARAMS.get("buffer_percent_loss_on_upgrade_ratio", 0.1)
                    )
                    # 更为激进的假设是完全清空与新旧码率无关的缓冲部分，只保留能用于新码率的部分，但这个很难量化
                    # 简单处理：假设切换后，我们能依赖的有效缓冲起点更低
                    effective_buffer_for_projection_s = max(0, current_buffer_s - buffer_lost_on_switch_estimate_s)

                    # 预估播放一个新码率分片后，缓冲区的变化 (秒)
                    # 变化量 = D_nom * (下载能力 / 消耗速率 - 1)
                    # 下载能力用 effective_bw_for_decision (它已乘以安全系数)
                    # 消耗速率是 new_level_bitrate_bps
                    buffer_change_per_segment_s = self.NOMINAL_SEGMENT_DURATION_S * (effective_bw_for_decision / new_level_bitrate_bps - 1)
                    
                    projected_buffer_after_one_segment = effective_buffer_for_projection_s + buffer_change_per_segment_s
                    
                    # 升级后缓冲区必须满足的最低要求
                    min_buffer_after_upgrade_segment = self.COMP_PARAMS.get("buffer_min_after_upgrade_segment_s", self.COMP_PARAMS["buffer_low_threshold"])

                    logger.info(f"ComprehensiveABR: UPGRADE CHECK to Lvl {potential_upgrade_idx} (Rate:{new_level_bitrate_bps/1e6:.1f}M) - "
                                f"CurrentBuffer: {current_buffer_s:.1f}s, Est.LossOnSwitch: {buffer_lost_on_switch_estimate_s:.1f}s, EffectiveBufferForProj: {effective_buffer_for_projection_s:.1f}s. "
                                f"ProjectedBufferChangePerSegment: {buffer_change_per_segment_s:.1f}s. "
                                f"ProjectedBufferAfterOneSegment: {projected_buffer_after_one_segment:.1f}s (Need > {min_buffer_after_upgrade_segment:.1f}s).")

                    if projected_buffer_after_one_segment >= min_buffer_after_upgrade_segment:
                        # 进一步检查带宽余量是否足够（即使target_selectable_bps通过了初步筛选）
                        if effective_bw_for_decision >= new_level_bitrate_bps * (1 + self.COMP_PARAMS.get("normal_upgrade_sustain_headroom", 0.15)):
                            next_best_index = potential_upgrade_idx
                            logger.info("ComprehensiveABR: UPGRADE approved (good headroom and projected buffer OK).")
                        elif effective_buffer_for_projection_s > self.COMP_PARAMS.get("buffer_needed_for_low_headroom_upgrade", 30.0) : # 如果带宽余量不足，但有效缓冲非常高
                            next_best_index = potential_upgrade_idx
                            logger.info("ComprehensiveABR: UPGRADE approved (low headroom but very high effective buffer and projected buffer OK).")
                        else:
                            logger.warning("ComprehensiveABR: UPGRADE REJECTED (projected buffer OK, but insufficient headroom and effective buffer not high enough for risk).")
                    else:
                        logger.warning("ComprehensiveABR: UPGRADE REJECTED (projected buffer after one segment too low).")
                else:
                    logger.info("ComprehensiveABR: NORMAL - High buffer but target_selectable_bps not sufficient for any quality upgrade.")
            
            # --- 降档逻辑 (如果未触发升级或不满足升级条件) ---
            # (确保 next_best_index 仍然是 current_level_index，即尚未因升级而改变)
            if next_best_index == current_level_index and current_level_index > 0 : 
                needs_downgrade = False
                reason_for_downgrade = ""

                if current_buffer_s < self.COMP_PARAMS["buffer_low_threshold"]:
                    needs_downgrade = True
                    reason_for_downgrade = "Low Buffer"
                elif target_selectable_bps < self.available_streams[current_level_index]['bandwidth'] * self.COMP_PARAMS.get("sustain_current_level_margin", 0.95): # 新参数: 维持当前码率也需要一定余量（例如0.95表示目标可选带宽至少是当前码率的95%）
                    needs_downgrade = True
                    reason_for_downgrade = "Insufficient BW for current level"
                elif network_features['trend'] in ["DECREASING", "VOLATILE"] and not is_current_segment_stuck : # 如果网络趋势不好（且非卡顿导致）
                    needs_downgrade = True
                    reason_for_downgrade = f"Bad Network Trend ({network_features['trend']})"
                elif is_current_segment_stuck: # 如果是因为当前分片卡住，也强烈考虑降级
                    needs_downgrade = True
                    reason_for_downgrade = "Segment Stuck"

                if needs_downgrade:
                    potential_downgrade_index = 0 
                    for i in range(current_level_index - 1, -1, -1):
                        # 降级时，确保目标带宽能稳定支持所选码率
                        if target_selectable_bps >= self.available_streams[i]['bandwidth'] * self.COMP_PARAMS.get("downgrade_selection_margin", 1.0): 
                            potential_downgrade_index = i
                            break
                    next_best_index = potential_downgrade_index
                    logger.info(f"ComprehensiveABR: NORMAL - DOWNGRADE condition met ({reason_for_downgrade}). Target level {next_best_index}.")

        # --- 步骤 5: 应用决策 ---
        rationale = (f"EffBW: {effective_bw_for_decision/1e6:.2f}M, TargetSelBW: {target_selectable_bps/1e6:.2f}M, "
                     f"Buf: {current_buffer_s:.1f}s, ABRState: {derived_abr_state}, SF: {current_safety_factor:.2f}, "
                     f"ConsErr: {consecutive_errors}, Stuck: {stuck_severity}, Recovery: {in_recovery_mode}")
        
        # 考虑是否因为紧急情况而覆盖切换抑制
        is_emergency_decision = (next_best_index == 0 and current_buffer_s < self.COMP_PARAMS["buffer_emergency_threshold"])
        
        self._execute_switch_decision(next_best_index, "COMPREHENSIVE", rationale, is_emergency=is_emergency_decision)



    def abr_loop(self): # 与之前类似，只是日志中加入了logic_type
        logger.info(f"ABR Python Algo (Logic: {self.logic_type}) monitoring thread started.") # 日志: ABR监控线程启动
        # 初始等待，让播放器先缓冲一些，并收集一些初始的网络统计数据
        # 这个时间可以根据需要调整，太短可能导致早期决策基于不充分数据
        time.sleep(3) 
        
        while not self.stop_abr_event.is_set():
            try:
                self._abr_decision_logic() # 调用主分发方法
            except Exception as e:
                logger.error(f"Error in ABR decision loop (Logic: {self.logic_type}): {e}", exc_info=True) # 日志: ABR决策循环出错
            
            # 决策频率 (例如3秒)
            # 更频繁的决策可能导致振荡，太慢则响应不及时
            sleep_interval = 3.0 
            for _ in range(int(sleep_interval)): # 允许更早地被stop_event打断
                if self.stop_abr_event.is_set(): break
                time.sleep(1)
        logger.info(f"ABR Python Algo (Logic: {self.logic_type}) monitoring thread stopped.") # 日志: ABR监控线程停止

    def start(self):
        self.stop_abr_event.clear()
        self.abr_thread = threading.Thread(target=self.abr_loop, daemon=True, name="PythonABRLogicThread")
        self.abr_thread.start()

    def stop(self):
        if self.abr_thread and self.abr_thread.is_alive():
            self.stop_abr_event.set()
            self.abr_thread.join(timeout=2.0) # 给线程时间干净地退出
     
        # 保存DQN模型
        if self.logic_type == self.LOGIC_TYPE_DQN:
            self.save_model()

        ABRManager.instance = None # 清理实例