import time
import threading
import logging
from typing import Dict, Optional
from collections import deque

logger = logging.getLogger('BufferManager')

class BufferInfo:
    """缓冲区信息"""
    def __init__(self, segment_idx: int, quality: str, segment_duration: float):
        self.segment_idx = segment_idx
        self.quality = quality
        self.segment_duration = segment_duration
        self.timestamp = time.time()

class BufferManager:
    """缓冲区管理器，用于跟踪播放缓冲区状态"""
    
    def __init__(self, target_buffer_seconds: float = 15.0):
        self.target_buffer_seconds = target_buffer_seconds
        self.lock = threading.Lock()
        
        # 缓冲区：存储已下载但未播放的分片信息
        self.buffer_queue = deque()  # BufferInfo对象队列
        
        # 播放状态跟踪
        self.currently_playing_idx = -1
        self.last_update_time = time.time()
        
        # 缓冲区水位统计
        self.buffer_level_history = deque(maxlen=100)
        
        logger.info(f"缓冲区管理器初始化，目标缓冲区大小: {target_buffer_seconds}秒")
    
    def add_segment_to_buffer(self, segment_idx: int, quality: str, segment_duration: float):
        """添加分片到缓冲区"""
        with self.lock:
            buffer_info = BufferInfo(segment_idx, quality, segment_duration)
            self.buffer_queue.append(buffer_info)
            logger.debug(f"分片 {segment_idx} ({quality}) 已添加到缓冲区")
            self._update_buffer_level()
    
    def update_playing_position(self, playing_idx: int):
        """更新当前播放位置"""
        with self.lock:
            old_idx = self.currently_playing_idx
            self.currently_playing_idx = playing_idx
            
            # 移除已播放的分片
            while self.buffer_queue and self.buffer_queue[0].segment_idx <= playing_idx:
                removed = self.buffer_queue.popleft()
                logger.debug(f"从缓冲区移除已播放分片: {removed.segment_idx}")
            
            self._update_buffer_level()
            
            if old_idx != playing_idx:
                logger.debug(f"播放位置更新: {old_idx} -> {playing_idx}, 缓冲区大小: {len(self.buffer_queue)}")
    
    def get_buffer_level_seconds(self) -> float:
        """获取当前缓冲区水位（秒）"""
        with self.lock:
            total_duration = sum(info.segment_duration for info in self.buffer_queue)
            return total_duration
    
    def get_buffer_level_segments(self) -> int:
        """获取当前缓冲区水位（分片数量）"""
        with self.lock:
            return len(self.buffer_queue)
    
    def is_buffer_healthy(self) -> bool:
        """检查缓冲区是否健康"""
        buffer_seconds = self.get_buffer_level_seconds()
        return buffer_seconds >= (self.target_buffer_seconds * 0.5)  # 至少50%目标水位
    
    def is_buffer_full(self) -> bool:
        """检查缓冲区是否已满"""
        buffer_seconds = self.get_buffer_level_seconds()
        return buffer_seconds >= self.target_buffer_seconds
    
    def get_buffer_status(self) -> Dict:
        """获取缓冲区状态信息"""
        with self.lock:
            buffer_seconds = self.get_buffer_level_seconds()
            buffer_segments = len(self.buffer_queue)
            
            status = {
                'buffer_seconds': buffer_seconds,
                'buffer_segments': buffer_segments,
                'target_seconds': self.target_buffer_seconds,
                'buffer_ratio': buffer_seconds / self.target_buffer_seconds if self.target_buffer_seconds > 0 else 0,
                'is_healthy': self.is_buffer_healthy(),
                'is_full': self.is_buffer_full(),
                'currently_playing_idx': self.currently_playing_idx
            }
            
            return status
    
    def _update_buffer_level(self):
        """更新缓冲区水位历史"""
        buffer_seconds = sum(info.segment_duration for info in self.buffer_queue)
        self.buffer_level_history.append((time.time(), buffer_seconds))
        self.last_update_time = time.time()
    
    def get_average_buffer_level(self, window_seconds: float = 30.0) -> float:
        """获取指定时间窗口内的平均缓冲区水位"""
        with self.lock:
            current_time = time.time()
            cutoff_time = current_time - window_seconds
            
            recent_levels = [level for timestamp, level in self.buffer_level_history 
                           if timestamp >= cutoff_time]
            
            if not recent_levels:
                return 0.0
            
            return sum(recent_levels) / len(recent_levels)
    
    def clear_buffer(self):
        """清空缓冲区"""
        with self.lock:
            self.buffer_queue.clear()
            self.currently_playing_idx = -1
            self.buffer_level_history.clear()
            logger.info("缓冲区已清空")