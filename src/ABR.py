import threading
import time
import logging
import re
import requests
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
    LOGIC_TYPE_BANDWIDTH_ONLY = "bandwidth_only"
    LOGIC_TYPE_BANDWIDTH_BUFFER = "bandwidth_buffer"
    LOGIC_TYPE_ENHANCED_BUFFER_RESPONSE = "enhanced_buffer_response" # 积极响应缓冲区的逻辑
    LOGIC_TYPE_DQN = "dqn_rl" # DQN强化学习逻辑

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

        # --- DQN相关初始化 ---
        if self.logic_type == self.LOGIC_TYPE_DQN:
            self._init_dqn()

        if self.available_streams: #
            self._update_current_abr_selected_url_logging()
        self.broadcast_abr_decision(self.current_stream_index_by_abr)

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
        elif self.logic_type == self.LOGIC_TYPE_DQN:
            self._logic_dqn()
        else:
            logger.warning(f"Unknown ABR logic type: {self.logic_type}. Defaulting to bandwidth_buffer.")
            self._logic_bandwidth_buffer()

    def _logic_dqn(self):
        """基于DQN的ABR决策逻辑"""
        if not self.available_streams or len(self.available_streams) <= 1:
            return

        # 更新带宽估计
        self._estimate_bandwidth_simple_average()
        
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
     
        # 保存DQN模型
        if self.logic_type == self.LOGIC_TYPE_DQN:
            self.save_model()

        ABRManager.instance = None # 清理实例