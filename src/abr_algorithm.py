import time
import logging
from collections import deque
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger('ABRAlgorithm')

class NetworkMetrics:
    """网络性能指标收集器"""
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.download_times = deque(maxlen=window_size)  # 下载时间序列
        self.segment_sizes = deque(maxlen=window_size)   # 分片大小序列
        self.throughputs = deque(maxlen=window_size)     # 吞吐量序列
        self.buffer_levels = deque(maxlen=window_size)   # 缓冲区水位序列
        
    def add_download_sample(self, segment_size_bytes: int, download_time_seconds: float):
        """添加下载样本"""
        if download_time_seconds <= 0:
            return
        
        throughput_bps = (segment_size_bytes * 8) / download_time_seconds  # bits per second
        
        self.download_times.append(download_time_seconds)
        self.segment_sizes.append(segment_size_bytes)
        self.throughputs.append(throughput_bps)
        
        logger.debug(f"网络样本: 大小={segment_size_bytes}字节, 时间={download_time_seconds:.3f}秒, "
                    f"吞吐量={throughput_bps/1000:.1f}kbps")
    
    def add_buffer_level(self, buffer_seconds: float):
        """添加缓冲区水位样本"""
        self.buffer_levels.append(buffer_seconds)
    
    def get_average_throughput(self) -> float:
        """获取平均吞吐量 (bps)"""
        if not self.throughputs:
            return 0.0
        return sum(self.throughputs) / len(self.throughputs)
    
    def get_recent_throughput(self, recent_count: int = 3) -> float:
        """获取最近几个样本的平均吞吐量"""
        if not self.throughputs:
            return 0.0
        recent_samples = list(self.throughputs)[-recent_count:]
        return sum(recent_samples) / len(recent_samples)
    
    def get_current_buffer_level(self) -> float:
        """获取当前缓冲区水位 (秒)"""
        if not self.buffer_levels:
            return 0.0
        return self.buffer_levels[-1]


class QualityLevel:
    """视频质量等级定义"""
    def __init__(self, suffix: str, bitrate_bps: int, resolution: str, priority: int):
        self.suffix = suffix          # 如 "480p-1500k"
        self.bitrate_bps = bitrate_bps  # 码率 (bps)
        self.resolution = resolution  # 如 "854x480"
        self.priority = priority      # 优先级，数值越小优先级越高
    
    def __repr__(self):
        return f"Quality({self.suffix}, {self.bitrate_bps/1000:.0f}kbps, {self.resolution})"


class ABRAlgorithm:
    """自适应码率算法"""
    
    # 预定义的质量等级 (与 segment_video.py 中的设置对应)
    QUALITY_LEVELS = [
        QualityLevel("480p-1500k", 1500000, "854x480", 3),
        QualityLevel("720p-4000k", 4000000, "1280x720", 2), 
        QualityLevel("1080p-8000k", 8000000, "1920x1080", 1)
    ]
    
    def __init__(self, initial_quality_suffix: str = "480p-1500k"):
        self.current_quality = self._find_quality_by_suffix(initial_quality_suffix)
        if not self.current_quality:
            self.current_quality = self.QUALITY_LEVELS[-1]  # 默认最低质量
        
        self.network_metrics = NetworkMetrics()
        
        # ABR 参数
        self.buffer_target_seconds = 15.0      # 目标缓冲区水位
        self.buffer_min_seconds = 5.0          # 最小缓冲区水位
        self.buffer_max_seconds = 30.0         # 最大缓冲区水位
        self.throughput_safety_factor = 1.5   # 吞吐量安全系数
        self.switch_up_threshold = 0.8         # 向上切换的吞吐量阈值比例
        self.switch_down_threshold = 1.2       # 向下切换的吞吐量阈值比例
        
        # 防抖动参数
        self.min_switch_interval = 10.0       # 最小切换间隔 (秒)
        self.last_switch_time = 0.0
        self.consecutive_switch_decisions = 0   # 连续相同切换决策计数
        self.min_consecutive_for_switch = 3     # 需要连续多少次相同决策才执行切换
        self.last_decision_quality = None
        
        logger.info(f"ABR算法初始化: 初始质量={self.current_quality}")
    
    def _find_quality_by_suffix(self, suffix: str) -> Optional[QualityLevel]:
        """根据后缀查找质量等级"""
        for quality in self.QUALITY_LEVELS:
            if quality.suffix == suffix:
                return quality
        return None
    
    def _get_next_higher_quality(self) -> Optional[QualityLevel]:
        """获取下一个更高质量等级"""
        current_priority = self.current_quality.priority
        candidates = [q for q in self.QUALITY_LEVELS if q.priority < current_priority]
        if candidates:
            return max(candidates, key=lambda q: q.priority)  # 选择优先级最高的（数值最小）
        return None
    
    def _get_next_lower_quality(self) -> Optional[QualityLevel]:
        """获取下一个更低质量等级"""
        current_priority = self.current_quality.priority
        candidates = [q for q in self.QUALITY_LEVELS if q.priority > current_priority]
        if candidates:
            return min(candidates, key=lambda q: q.priority)  # 选择优先级最低的（数值最大）
        return None
    
    def update_network_metrics(self, segment_size_bytes: int, download_time_seconds: float, 
                             buffer_level_seconds: float):
        """更新网络指标"""
        self.network_metrics.add_download_sample(segment_size_bytes, download_time_seconds)
        self.network_metrics.add_buffer_level(buffer_level_seconds)
    
    def decide_next_quality(self) -> Tuple[QualityLevel, str]:
        """
        决策下一个分片的质量等级
        返回: (质量等级, 决策原因)
        """
        current_time = time.time()
        
        # 检查是否在切换冷却期内
        if current_time - self.last_switch_time < self.min_switch_interval:
            return self.current_quality, f"切换冷却期内 ({self.min_switch_interval}s)"
        
        # 获取网络状态
        avg_throughput = self.network_metrics.get_average_throughput()
        recent_throughput = self.network_metrics.get_recent_throughput()
        buffer_level = self.network_metrics.get_current_buffer_level()
        
        if avg_throughput == 0:
            return self.current_quality, "无足够网络数据"
        
        # 使用最近吞吐量作为主要参考，平均吞吐量作为辅助
        effective_throughput = min(avg_throughput, recent_throughput * 1.2)
        
        logger.debug(f"ABR决策: 平均吞吐量={avg_throughput/1000:.1f}kbps, "
                    f"最近吞吐量={recent_throughput/1000:.1f}kbps, "
                    f"有效吞吐量={effective_throughput/1000:.1f}kbps, "
                    f"缓冲区={buffer_level:.1f}s")
        
        decision_quality = self.current_quality
        reason = "保持当前质量"
        
        # 缓冲区紧急情况：快速降级
        if buffer_level < self.buffer_min_seconds:
            lower_quality = self._get_next_lower_quality()
            if lower_quality:
                decision_quality = lower_quality
                reason = f"缓冲区过低 ({buffer_level:.1f}s < {self.buffer_min_seconds}s)"
        
        # 正常情况：基于吞吐量决策
        else:
            # 考虑向上切换
            higher_quality = self._get_next_higher_quality()
            if higher_quality:
                required_throughput = higher_quality.bitrate_bps * self.throughput_safety_factor
                if (effective_throughput > required_throughput and 
                    buffer_level > self.buffer_target_seconds):
                    decision_quality = higher_quality
                    reason = f"吞吐量充足升级 ({effective_throughput/1000:.1f}kbps > {required_throughput/1000:.1f}kbps)"
            
            # 考虑向下切换
            if decision_quality == self.current_quality:  # 如果没有向上切换
                current_required = self.current_quality.bitrate_bps * self.switch_down_threshold
                if effective_throughput < current_required:
                    lower_quality = self._get_next_lower_quality()
                    if lower_quality:
                        decision_quality = lower_quality
                        reason = f"吞吐量不足降级 ({effective_throughput/1000:.1f}kbps < {current_required/1000:.1f}kbps)"
        
        # 防抖动逻辑
        if decision_quality != self.current_quality:
            if self.last_decision_quality == decision_quality:
                self.consecutive_switch_decisions += 1
            else:
                self.consecutive_switch_decisions = 1
                self.last_decision_quality = decision_quality
            
            if self.consecutive_switch_decisions >= self.min_consecutive_for_switch:
                # 执行切换
                old_quality = self.current_quality
                self.current_quality = decision_quality
                self.last_switch_time = current_time
                self.consecutive_switch_decisions = 0
                self.last_decision_quality = None
                
                logger.info(f"质量切换: {old_quality.suffix} -> {self.current_quality.suffix} "
                           f"(原因: {reason})")
                return self.current_quality, f"切换: {reason}"
            else:
                return self.current_quality, f"防抖动: 需要{self.min_consecutive_for_switch - self.consecutive_switch_decisions}次确认"
        else:
            # 重置防抖动计数
            self.consecutive_switch_decisions = 0
            self.last_decision_quality = None
        
        return self.current_quality, reason
    
    def get_quality_list(self) -> List[QualityLevel]:
        """获取所有可用质量等级"""
        return self.QUALITY_LEVELS.copy()
    
    def get_current_quality(self) -> QualityLevel:
        """获取当前质量等级"""
        return self.current_quality