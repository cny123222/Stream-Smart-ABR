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
    # LOGIC_TYPE_COMPREHENSIVE = "comprehensive_rules" # 你设想的基于趋势、历史等的综合规则算法
    # LOGIC_TYPE_DQN = "dqn" # 为DQN预留，但具体实现由外部控制或不同方式集成
    LOGIC_TYPE_DQN = "dqn_rl" # DQN强化学习逻辑

    def __init__(self, available_streams_from_master, broadcast_abr_decision_callback,
                 broadcast_bw_estimate_callback=None,
                 logic_type=LOGIC_TYPE_BANDWIDTH_BUFFER, # 默认使用带宽+缓冲区逻辑
                 # --- 带宽估计相关参数 --- #
                 max_bw_stats_history=5,  # SWMA和EWMA等可能用到的历史窗口大小
                 ewma_alpha=0.3,          # EWMA的平滑因子alpha
                 # --- 切换抑制参数 --- #
                 min_switch_interval_seconds=5.0 # 最小切换间隔
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
        if duration_seconds > 1e-5: # 只有当下载时长有效时才记录
            throughput_bps = (size_bytes * 8) / duration_seconds
            timestamp = time.time()
            
            with self._internal_lock: # 保证线程安全地修改列表
                self.segment_download_stats.append({
                    'url': url,
                    'size': size_bytes,
                    'duration': duration_seconds,
                    'throughput_bps': throughput_bps,
                    'timestamp': timestamp
                })
                # 维持下载统计历史记录的窗口大小
                if len(self.segment_download_stats) > self.max_bw_stats_history:
                    self.segment_download_stats.pop(0)
                
                # --- EWMA 带宽估计更新 ---
                # 每次有新的吞吐量数据点时，更新EWMA估计值
                if self.ewma_bandwidth_bps == 0.0: # 第一次赋值
                    self.ewma_bandwidth_bps = throughput_bps
                else:
                    self.ewma_bandwidth_bps = self.ewma_alpha * throughput_bps + (1 - self.ewma_alpha) * self.ewma_bandwidth_bps
                # --- EWMA 结束 ---

        else:
            logger.warning(f"Segment download duration too short or invalid ({duration_seconds:.4f}s) for stat, URL: {url}") # 日志: 分片下载时长过短

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
            
    def update_player_buffer_level(self, buffer_seconds):
        with self._internal_lock:
            self.current_player_buffer_s = buffer_seconds

    def get_current_abr_decision_url(self):
        with self._internal_lock:
            return self._current_selected_url_for_logging

    # def _estimate_bandwidth_simple_average(self): # 带宽估计方法
    #     if not self.segment_download_stats: return self.estimated_bandwidth_bps # 返回上一次的值或0
        
    #     # 只考虑成功的下载
    #     successful_downloads = [s for s in self.segment_download_stats if not s.get('error') and 'size' in s and 'duration' in s]
    #     if not successful_downloads: return self.estimated_bandwidth_bps

    #     # 选择最近的N条
    #     # relevant_stats = successful_downloads[-self.max_stats_history:] # 用全部记录的来平均
    #     relevant_stats = successful_downloads # 使用max_stats_history限制的总数
        
    #     if not relevant_stats: return self.estimated_bandwidth_bps

    #     total_bytes = sum(s['size'] for s in relevant_stats)
    #     total_time = sum(s['duration'] for s in relevant_stats)

    #     if total_time == 0: return self.estimated_bandwidth_bps # 避免除以0
        
    #     self.estimated_bandwidth_bps = (total_bytes * 8) / total_time
    #     # logger.info(f"ABR SimpleAvg BW Est: {self.estimated_bandwidth_bps / 1000:.0f} Kbps") # 日志由具体决策逻辑打印
    #     if self.broadcast_bw_estimate and self.estimated_bandwidth_bps > 0: # 仅当有有效估算时发送
    #         self.broadcast_bw_estimate(self.estimated_bandwidth_bps / 1_000_000) # 发送Mbps
    #     return self.estimated_bandwidth_bps

    # # --- 增强的带宽估计 ---
    # def _estimate_bandwidth_enhanced(self):
    #     # 对最近的片段赋予更高权重
    #     # 如果最近一个片段下载速度远低于平均，则临时拉低平均值
        
    #     current_avg_bps = self._estimate_bandwidth_simple_average() # 先获取简单平均

    #     # 只考虑成功的下载
    #     successful_downloads = [s for s in self.segment_download_stats if not s.get('error') and 'throughput_bps' in s]
    #     if not successful_downloads:
    #         return current_avg_bps # 没有成功下载的统计，返回简单平均

    #     last_segment_info = successful_downloads[-1]
    #     last_segment_throughput_bps = last_segment_info['throughput_bps']

    #     # 如果最近一次下载速度显著低于当前平均，且平均值大于0
    #     if current_avg_bps > 0 and last_segment_throughput_bps < current_avg_bps * 0.5: # 低于平均一半
    #         logger.warning(f"ABR Enhanced BW: Last segment throughput ({last_segment_throughput_bps/1000:.0f} Kbps) "
    #                        f"is much lower than average ({current_avg_bps/1000:.0f} Kbps). Adjusting estimate downwards.")
    #         # 更激进地降低估算，取最近一次和平均值的一个较小比例的组合
    #         adjusted_bps = (last_segment_throughput_bps * 0.7) + (current_avg_bps * 0.3)
    #         self.estimated_bandwidth_bps = adjusted_bps # 更新主估算值
    #         if self.broadcast_bw_estimate and adjusted_bps > 0:
    #             self.broadcast_bw_estimate(adjusted_bps / 1_000_000) # 发送Mbps
    #         return adjusted_bps
        
    #     # 否则，正常返回简单平均值
    #     return current_avg_bps
    
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
            if not self.segment_download_stats:
                return 0.0
            
            # 只考虑包含有效吞吐量数据的统计项
            throughputs = [s['throughput_bps'] for s in self.segment_download_stats if 'throughput_bps' in s]
            if not throughputs:
                return 0.0
            
            estimated_throughput_bps = sum(throughputs) / len(throughputs)
            if self.broadcast_bw_estimate and estimated_throughput_bps > 0:
                self.broadcast_bw_estimate(estimated_throughput_bps / 1_000_000)
            return estimated_throughput_bps
        
    def _get_ewma_throughput(self):
        # 获取指数加权移动平均带宽 (用于EWMA)
        # EWMA值在 add_segment_download_stat 中已实时更新
        with self._internal_lock:
            if self.broadcast_bw_estimate and self.ewma_bandwidth_bps > 0:
                self.broadcast_bw_estimate(self.ewma_bandwidth_bps / 1_000_000)
            return self.ewma_bandwidth_bps

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
    def _execute_switch_decision(self, next_best_index, current_logic_name, decision_rationale_log=""):
        # 封装通用的切换逻辑，包括日志记录和广播
        current_level_index = self.current_stream_index_by_abr
        
        if next_best_index == current_level_index:
            logger.info(f"ABR DECISION ({current_logic_name}): No change from current level {current_level_index}. {decision_rationale_log}") # 日志: 无需切换
            return
          
        # 切换抑制逻辑
        if time.time() - self.last_switch_time < self.MIN_SWITCH_INTERVAL:
            logger.info(f"ABR DECISION ({current_logic_name}): Switch to level {next_best_index} proposed, "
                        f"but it's too soon since last switch ({time.time() - self.last_switch_time:.1f}s ago). "
                        f"Holding current level {current_level_index}.") # 日志: 切换过于频繁
            return # 保持当前等级，不进行切换

        # 执行切换
        self.last_switch_time = time.time() # 更新上次切换时间
        old_stream_info = self.available_streams[current_level_index]
        self.current_stream_index_by_abr = next_best_index
        new_stream_info = self.available_streams[next_best_index]
        self._update_current_abr_selected_url_logging()

        logger.info(f"ABR DECISION ({current_logic_name}): Switch from level {current_level_index} "
                    f"(~{old_stream_info.get('bandwidth',0)/1000:.0f}Kbps) to {next_best_index} "
                    f"(~{new_stream_info.get('bandwidth',0)/1000:.0f}Kbps). {decision_rationale_log}") # 日志: 执行切换
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
        # 这是为你设想的“最终版”综合算法预留的框架
        # 它会用到带宽趋势、卡顿历史、切换历史等
        logger.info("ABR LOGIC (COMPREHENSIVE): Not fully implemented yet. Using fallback or basic logic.") # 日志: 综合算法未实现
        
        # --- 1. 获取基础状态 ---
        estimated_bw_bps = self._get_ewma_throughput() # 可以选择 EWMA 作为基础带宽估计
        with self._internal_lock:
            current_buffer_s = self.current_player_buffer_s
        current_level_index = self.current_stream_index_by_abr
        
        # --- 2. 获取更高级的状态 (趋势、历史等) ---
        # network_state = self.get_network_state() # 假设你实现了这个方法
        # recent_stall = self.has_stalled_recently() # 假设你实现了这个方法
        # (这些方法的具体实现需要添加到 ABRManager 中)

        # --- 3. 决策逻辑 (高度依赖于你如何定义和使用上述状态) ---
        # 这里的决策会非常复杂，例如：
        # if network_state == "SUDDEN_DROP" or recent_stall:
        #     next_best_index = 0 # 紧急处理
        # elif network_state == "VOLATILE":
        #     # 保守策略
        #     pass
        # elif network_state == "INCREASING" and current_buffer_s > HIGH_BUFFER_FOR_COMPREHENSIVE:
        #     # 积极升档
        #     pass
        # else: // STABLE 或其他
        #     # 类似 bandwidth_buffer 的逻辑，但参数可能根据趋势微调
        #     pass
            
        # --- 临时：由于未完全实现，可以先退化为 bandwidth_buffer 逻辑 ---
        # 你需要将 _logic_bandwidth_buffer 的核心逻辑复制或重构到这里，
        # 或者直接调用它，或者设计一套全新的规则。
        # 为简单起见，暂时让它做一个非常基础的决策或不决策。
        next_best_index = current_level_index # 默认不切换
        rationale = f"Comprehensive logic (not fully implemented). Est.BW: {estimated_bw_bps/1000:.0f} Kbps, Buf: {current_buffer_s:.2f}s."
        
        # 作为一个占位符，可以先让它表现得像一个简单的带宽+缓冲区算法
        # （以下为复制BW_BUFFER的简化逻辑，实际应更复杂）
        BUFFER_EMERGENCY = 5.0
        BUFFER_LOW = 10.0
        BUFFER_HIGH = 25.0
        safety_factor = 0.8
        if current_buffer_s < BUFFER_EMERGENCY: safety_factor = 0.6
        elif current_buffer_s < BUFFER_LOW: safety_factor = 0.7
        elif current_buffer_s > BUFFER_HIGH: safety_factor = 0.9
        
        target_bitrate_bps = estimated_bw_bps * safety_factor

        if current_buffer_s < BUFFER_EMERGENCY and current_level_index > 0:
            next_best_index = 0
        elif current_buffer_s > BUFFER_HIGH and current_level_index < (self.num_quality_levels - 1):
            temp_idx = current_level_index
            for i in range(self.num_quality_levels - 1, current_level_index, -1):
                if target_bitrate_bps >= self.available_streams[i].get('bandwidth', 0):
                    temp_idx = i; break
            if temp_idx > current_level_index : next_best_index = temp_idx
        elif current_buffer_s < BUFFER_LOW or target_bitrate_bps < self.available_streams[current_level_index].get('bandwidth', 0):
             if current_level_index > 0:
                temp_idx = 0
                for i in range(current_level_index - 1, -1, -1):
                    if target_bitrate_bps >= self.available_streams[i].get('bandwidth', 0):
                        temp_idx = i; break
                next_best_index = temp_idx
        # --- 占位逻辑结束 ---

        self._execute_switch_decision(next_best_index, "COMPREHENSIVE", rationale)

    # # --- 决策逻辑: 只看带宽 ---
    # def _logic_bandwidth_only(self):
    #     if not self.available_streams or len(self.available_streams) <= 1: return

    #     estimated_bw_bps = self._estimate_bandwidth_simple_average() # 使用简单平均带宽
    #     current_level_index = self.current_stream_index_by_abr
    #     safety_factor = 0.8 # 此逻辑固定的安全系数

    #     logger.info(
    #         f"ABR LOGIC (BW_ONLY): Est. BW: {estimated_bw_bps / 1000:.0f} Kbps, "
    #         f"Current Level Idx: {current_level_index}"
    #     )

    #     if estimated_bw_bps == 0 and not self.segment_download_stats:
    #         logger.info("ABR LOGIC (BW_ONLY): No stats, sticking to current.")
    #         return

    #     target_bitrate_bps = estimated_bw_bps * safety_factor
    #     next_best_index = 0 # 默认最低
        
    #     # 从最高码率往下找，找到第一个能被目标带宽支持的
    #     for i in range(len(self.available_streams) - 1, -1, -1):
    #         stream_bw = self.available_streams[i].get('bandwidth', 0)
    #         # print(i, stream_bw, target_bitrate_bps)
    #         print(f"ABR LOGIC (BW_ONLY): Checking level {i} with BW={stream_bw/1000:.0f}Kbps, now we have {target_bitrate_bps/1000:.0f}Kbps.")
    #         if target_bitrate_bps >= stream_bw:
    #             next_best_index = i
    #             break
        
    #     if next_best_index != current_level_index:
    #         # ... 广播决策的代码 ...
    #         old_stream_info = self.available_streams[current_level_index]
    #         self.current_stream_index_by_abr = next_best_index
    #         new_stream_info = self.available_streams[self.current_stream_index_by_abr]
    #         self._update_current_abr_selected_url_logging()
    #         logger.info(f"ABR DECISION (BW_ONLY): Switch from level {current_level_index} "
    #                     f"(~{old_stream_info.get('bandwidth',0)/1000:.0f}Kbps) to {next_best_index} "
    #                     f"(~{new_stream_info.get('bandwidth',0)/1000:.0f}Kbps). Target_BW={target_bitrate_bps/1000:.0f}Kbps")
    #         self.broadcast_abr_decision(self.current_stream_index_by_abr)
    #     else:
    #         logger.info(f"ABR DECISION (BW_ONLY): No change from level {current_level_index}.")


    # # --- 决策逻辑: 看带宽和缓冲区 ---
    # def _logic_bandwidth_buffer(self):
    #     if not self.available_streams or len(self.available_streams) <= 1: return

    #     estimated_bw_bps = self._estimate_bandwidth_enhanced() # 稍微增强的带宽估计
    #     with self._internal_lock:
    #         current_buffer_s = self.current_player_buffer_s
    #     current_level_index = self.current_stream_index_by_abr

    #     logger.info(
    #         f"ABR LOGIC (BW_BUFFER): Est. BW: {estimated_bw_bps / 1000:.0f} Kbps, "
    #         f"Buffer: {current_buffer_s:.2f}s, "
    #         f"Current Level Idx: {current_level_index}"
    #     )

    #     if estimated_bw_bps == 0 and not self.segment_download_stats:
    #         logger.info("ABR LOGIC (BW_BUFFER): No bandwidth stats yet, sticking to current level.")
    #         return

    #     BUFFER_THRESHOLD_LOW = 10.0
    #     BUFFER_THRESHOLD_MEDIUM = 15.0
    #     BUFFER_THRESHOLD_HIGH = 25.0
    #     BUFFER_THRESHOLD_EMERGENCY = 5.0

    #     if current_buffer_s < BUFFER_THRESHOLD_LOW:
    #         dynamic_safety_factor = 0.6
    #     elif current_buffer_s < BUFFER_THRESHOLD_MEDIUM:
    #         dynamic_safety_factor = 0.8
    #     elif current_buffer_s < BUFFER_THRESHOLD_HIGH:
    #         dynamic_safety_factor = 0.9
    #     else:
    #         dynamic_safety_factor = 0.95

    #     target_bitrate_bps = estimated_bw_bps * dynamic_safety_factor
    #     logger.debug(f"ABR LOGIC (BW_BUFFER): Dyn Safety: {dynamic_safety_factor:.2f}, Target Sel. BW: {target_bitrate_bps / 1000:.0f} Kbps")

    #     next_best_index = current_level_index

    #     if current_buffer_s < BUFFER_THRESHOLD_EMERGENCY and current_level_index > 0:
    #         next_best_index = 0
    #         logger.warning(f"ABR LOGIC (BW_BUFFER): EMERGENCY! Buffer {current_buffer_s:.2f}s. Switching to lowest (idx 0).")
    #     elif current_buffer_s > BUFFER_THRESHOLD_HIGH and current_level_index < len(self.available_streams) - 1:
    #         potential_upgrade_index = current_level_index
    #         for i in range(len(self.available_streams) - 1, current_level_index, -1):
    #             if target_bitrate_bps >= self.available_streams[i].get('bandwidth', 0):
    #                 potential_upgrade_index = i; break
    #         if potential_upgrade_index > current_level_index:
    #             logger.info(f"ABR LOGIC (BW_BUFFER): UPGRADE condition met (buf {current_buffer_s:.2f}s > {BUFFER_THRESHOLD_HIGH:.1f}s). Potential idx: {potential_upgrade_index}")
    #             next_best_index = potential_upgrade_index
    #         # else: logger.info(f"ABR LOGIC (BW_BUFFER): Buffer high, but target BW no support higher.")
    #     elif current_buffer_s < BUFFER_THRESHOLD_LOW or target_bitrate_bps < self.available_streams[current_level_index].get('bandwidth', 0):
    #         if current_level_index > 0:
    #             potential_downgrade_index = 0
    #             for i in range(current_level_index - 1, -1, -1):
    #                 if target_bitrate_bps >= self.available_streams[i].get('bandwidth', 0):
    #                     potential_downgrade_index = i; break
    #             logger.info(f"ABR LOGIC (BW_BUFFER): DOWNGRADE condition met. Potential idx: {potential_downgrade_index}")
    #             next_best_index = potential_downgrade_index
    #         # else: logger.info(f"ABR LOGIC (BW_BUFFER): Downgrade condition, but already at lowest.")
    #     elif target_bitrate_bps < self.available_streams[current_level_index].get('bandwidth', 0) and current_level_index > 0: # 再次检查稳定性
    #         logger.info(f"ABR LOGIC (BW_BUFFER): Target BW cannot sustain current. Looking lower.")
    #         temp_idx = 0
    #         for i in range(current_level_index - 1, -1, -1):
    #             if target_bitrate_bps >= self.available_streams[i].get('bandwidth', 0): temp_idx = i; break
    #         next_best_index = temp_idx
        
    #     if next_best_index != current_level_index:
    #         # ... (广播决策的代码) ...
    #         if time.time() - self.last_switch_time < self.MIN_SWITCH_INTERVAL:
    #             logger.info(f"ABR LOGIC (BW_BUFFER): Too soon to switch. Waiting for {self.MIN_SWITCH_INTERVAL - (time.time() - self.last_switch_time):.2f}s.")
    #             next_best_index = current_level_index
    #         else:
    #             self.last_switch_time = time.time()
    #             old_stream_info = self.available_streams[current_level_index]
    #             self.current_stream_index_by_abr = next_best_index
    #             new_stream_info = self.available_streams[self.current_stream_index_by_abr]
    #             self._update_current_abr_selected_url_logging()
    #             logger.info(f"ABR DECISION (BW_BUFFER): Switch from level {current_level_index} "
    #                         f"(~{old_stream_info.get('bandwidth',0)/1000:.0f}Kbps) to {next_best_index} "
    #                         f"(~{new_stream_info.get('bandwidth',0)/1000:.0f}Kbps). Est.BW={estimated_bw_bps/1000:.0f}Kbps, Buf={current_buffer_s:.2f}s")
    #             self.broadcast_abr_decision(self.current_stream_index_by_abr)
    #     else:
    #         logger.info(f"ABR DECISION (BW_BUFFER): No change from level {current_level_index}. Est.BW={estimated_bw_bps/1000:.0f}Kbps, Buf={current_buffer_s:.2f}s")


    # # 决策逻辑3: 只看缓冲区
    # def _logic_buffer_only(self):
    #     if not self.available_streams or len(self.available_streams) <= 1:
    #         return

    #     with self._internal_lock:
    #         current_buffer_s = self.current_player_buffer_s
    #     current_level_index = self.current_stream_index_by_abr

    #     logger.info(
    #         f"ABR LOGIC (BUFFER_ONLY): Buffer: {current_buffer_s:.2f}s, "
    #         f"Current Level Idx: {current_level_index}"
    #     )

    #     # 定义缓冲区阈值
    #     BUFFER_THRESHOLD_LOW = 10.0
    #     BUFFER_THRESHOLD_HIGH = 25.0
    #     next_best_index = current_level_index

    #     # 缓冲区过高，尝试提升质量
    #     if current_buffer_s > BUFFER_THRESHOLD_HIGH and current_level_index < len(self.available_streams) - 1:
    #         next_best_index = current_level_index + 1
    #         logger.info(f"ABR LOGIC (BUFFER_ONLY): High buffer ({current_buffer_s:.2f}s), attempting upgrade to {next_best_index}.")

    #     # 缓冲区过低，尝试降低质量
    #     elif current_buffer_s < BUFFER_THRESHOLD_LOW and current_level_index > 0:
    #         next_best_index = current_level_index - 1
    #         logger.info(f"ABR LOGIC (BUFFER_ONLY): Low buffer ({current_buffer_s:.2f}s), attempting downgrade to {next_best_index}.")

    #     if next_best_index != current_level_index:
    #         if time.time() - self.last_switch_time < self.MIN_SWITCH_INTERVAL:
    #             logger.info(f"ABR LOGIC (BUFFER_ONLY): Too soon to switch. Waiting for {self.MIN_SWITCH_INTERVAL - (time.time() - self.last_switch_time):.2f}s.")
    #             next_best_index = current_level_index
    #         else:
    #             self.last_switch_time = time.time()
    #             old_stream_info = self.available_streams[current_level_index]
    #             self.current_stream_index_by_abr = next_best_index
    #             new_stream_info = self.available_streams[self.current_stream_index_by_abr]
    #             self._update_current_abr_selected_url_logging()
    #             logger.info(f"ABR DECISION (BUFFER_ONLY): Switch from level {current_level_index} "
    #                         f"(~{old_stream_info.get('bandwidth',0)/1000:.0f}Kbps) to {next_best_index} "
    #                         f"(~{new_stream_info.get('bandwidth',0)/1000:.0f}Kbps). Buf={current_buffer_s:.2f}s")
    #             self.broadcast_abr_decision(self.current_stream_index_by_abr)
    #     else:
    #         logger.info(f"ABR DECISION (BUFFER_ONLY): No change from level {current_level_index}. Buf={current_buffer_s:.2f}s")


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