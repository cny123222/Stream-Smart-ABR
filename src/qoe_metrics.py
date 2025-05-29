import time
import logging
import json
import os
from collections import deque
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional

logger = logging.getLogger('QoEMetrics')

@dataclass
class QualitySwitchEvent:
    """质量切换事件"""
    timestamp: float
    from_quality: str
    to_quality: str
    reason: str
    buffer_level: float
    throughput_kbps: float

@dataclass
class BufferingEvent:
    """缓冲事件"""
    timestamp: float
    duration_seconds: float
    quality: str
    trigger_reason: str

@dataclass
class PlaybackSession:
    """播放会话统计"""
    session_start_time: float
    session_end_time: Optional[float]
    total_segments_downloaded: int
    total_bytes_downloaded: int
    average_download_speed_kbps: float
    quality_switches: List[QualitySwitchEvent]
    buffering_events: List[BufferingEvent]
    quality_time_distribution: Dict[str, float]  # 每个质量等级的播放时长
    startup_delay_seconds: float
    total_buffering_time: float
    mean_opinion_score: Optional[float]

class QoECalculator:
    """QoE (Quality of Experience) 计算器"""
    
    def __init__(self):
        self.session_start_time = time.time()
        self.session_end_time = None
        
        # 事件记录
        self.quality_switches: List[QualitySwitchEvent] = []
        self.buffering_events: List[BufferingEvent] = []
        
        # 播放统计
        self.total_segments_downloaded = 0
        self.total_bytes_downloaded = 0
        self.download_speeds = deque(maxlen=100)  # 保留最近100个速度样本
        
        # 质量分布统计
        self.quality_time_distribution: Dict[str, float] = {}
        self.current_quality = None
        self.last_quality_change_time = self.session_start_time
        
        # 缓冲相关
        self.startup_delay_seconds = 0.0
        self.total_buffering_time = 0.0
        self.playback_started = False
        self.is_buffering = False
        self.current_buffering_start = None
        
        logger.info("QoE统计器已初始化")
    
    def record_segment_download(self, segment_size_bytes: int, download_time_seconds: float, quality: str):
        """记录分片下载"""
        self.total_segments_downloaded += 1
        self.total_bytes_downloaded += segment_size_bytes
        
        if download_time_seconds > 0:
            speed_kbps = (segment_size_bytes * 8) / (download_time_seconds * 1000)
            self.download_speeds.append(speed_kbps)
        
        # 更新当前质量的播放时长
        if self.current_quality != quality:
            self._update_quality_time()
            self.current_quality = quality
            self.last_quality_change_time = time.time()
    
    def record_quality_switch(self, from_quality: str, to_quality: str, reason: str, 
                            buffer_level: float, throughput_kbps: float):
        """记录质量切换事件"""
        if from_quality == to_quality:
            return
        
        switch_event = QualitySwitchEvent(
            timestamp=time.time(),
            from_quality=from_quality,
            to_quality=to_quality,
            reason=reason,
            buffer_level=buffer_level,
            throughput_kbps=throughput_kbps
        )
        self.quality_switches.append(switch_event)
        
        logger.info(f"QoE: 质量切换记录 {from_quality} -> {to_quality} (原因: {reason})")
    
    def record_buffering_start(self, quality: str, trigger_reason: str = "网络"):
        """记录缓冲开始"""
        if not self.is_buffering:
            self.is_buffering = True
            self.current_buffering_start = time.time()
            logger.debug(f"QoE: 缓冲开始 (质量: {quality}, 原因: {trigger_reason})")
    
    def record_buffering_end(self):
        """记录缓冲结束"""
        if self.is_buffering and self.current_buffering_start:
            buffering_duration = time.time() - self.current_buffering_start
            
            buffering_event = BufferingEvent(
                timestamp=self.current_buffering_start,
                duration_seconds=buffering_duration,
                quality=self.current_quality or "unknown",
                trigger_reason="网络"
            )
            self.buffering_events.append(buffering_event)
            self.total_buffering_time += buffering_duration
            
            self.is_buffering = False
            self.current_buffering_start = None
            
            logger.info(f"QoE: 缓冲结束，持续时长 {buffering_duration:.2f}秒")
    
    def record_playback_start(self):
        """记录播放开始"""
        if not self.playback_started:
            self.startup_delay_seconds = time.time() - self.session_start_time
            self.playback_started = True
            logger.info(f"QoE: 播放开始，启动延迟 {self.startup_delay_seconds:.2f}秒")
    
    def _update_quality_time(self):
        """更新当前质量的播放时长"""
        if self.current_quality:
            duration = time.time() - self.last_quality_change_time
            if self.current_quality in self.quality_time_distribution:
                self.quality_time_distribution[self.current_quality] += duration
            else:
                self.quality_time_distribution[self.current_quality] = duration
    
    def calculate_mean_opinion_score(self) -> float:
        """
        计算平均主观评分 (MOS)
        基于ITU-T建议的QoE模型，考虑视频质量、切换频率、缓冲时间等因素
        返回1-5分的评分
        """
        if not self.playback_started:
            return 1.0  # 未开始播放
        
        session_duration = time.time() - self.session_start_time
        if session_duration <= 0:
            return 1.0
        
        # 基础分数：根据平均质量
        base_score = self._calculate_quality_score()
        
        # 切换频率惩罚
        switch_penalty = self._calculate_switch_penalty(session_duration)
        
        # 缓冲时间惩罚
        buffering_penalty = self._calculate_buffering_penalty(session_duration)
        
        # 启动延迟惩罚
        startup_penalty = self._calculate_startup_penalty()
        
        # 计算最终MOS
        mos = base_score - switch_penalty - buffering_penalty - startup_penalty
        mos = max(1.0, min(5.0, mos))  # 限制在1-5范围内
        
        logger.debug(f"MOS计算: 基础={base_score:.2f}, 切换惩罚={switch_penalty:.2f}, "
                    f"缓冲惩罚={buffering_penalty:.2f}, 启动惩罚={startup_penalty:.2f}, "
                    f"最终MOS={mos:.2f}")
        
        return mos
    
    def _calculate_quality_score(self) -> float:
        """计算基于视频质量的基础分数"""
        if not self.quality_time_distribution:
            return 3.0  # 默认中等分数
        
        total_time = sum(self.quality_time_distribution.values())
        if total_time == 0:
            return 3.0
        
        # 质量等级到分数的映射
        quality_scores = {
            "480p-1500k": 3.0,
            "720p-4000k": 4.0,
            "1080p-8000k": 5.0
        }
        
        weighted_score = 0.0
        for quality, duration in self.quality_time_distribution.items():
            weight = duration / total_time
            score = quality_scores.get(quality, 3.0)
            weighted_score += score * weight
        
        return weighted_score
    
    def _calculate_switch_penalty(self, session_duration: float) -> float:
        """计算质量切换惩罚"""
        if len(self.quality_switches) == 0:
            return 0.0
        
        # 切换频率：每分钟切换次数
        switches_per_minute = len(self.quality_switches) / (session_duration / 60)
        
        # 超过每分钟1次切换开始惩罚
        if switches_per_minute <= 1.0:
            return 0.0
        elif switches_per_minute <= 2.0:
            return 0.2
        elif switches_per_minute <= 3.0:
            return 0.5
        else:
            return 1.0
    
    def _calculate_buffering_penalty(self, session_duration: float) -> float:
        """计算缓冲时间惩罚"""
        if self.total_buffering_time == 0:
            return 0.0
        
        buffering_ratio = self.total_buffering_time / session_duration
        
        # 缓冲时间占比惩罚
        if buffering_ratio <= 0.01:  # ≤ 1%
            return 0.0
        elif buffering_ratio <= 0.05:  # ≤ 5%
            return 0.3
        elif buffering_ratio <= 0.10:  # ≤ 10%
            return 0.7
        else:
            return 1.5
    
    def _calculate_startup_penalty(self) -> float:
        """计算启动延迟惩罚"""
        if self.startup_delay_seconds <= 2.0:
            return 0.0
        elif self.startup_delay_seconds <= 5.0:
            return 0.1
        elif self.startup_delay_seconds <= 10.0:
            return 0.3
        else:
            return 0.5
    
    def finalize_session(self):
        """结束会话统计"""
        self.session_end_time = time.time()
        self._update_quality_time()
        if self.is_buffering:
            self.record_buffering_end()
        
        logger.info("QoE会话统计已结束")
    
    def get_session_summary(self) -> PlaybackSession:
        """获取会话摘要"""
        avg_speed = sum(self.download_speeds) / len(self.download_speeds) if self.download_speeds else 0.0
        
        return PlaybackSession(
            session_start_time=self.session_start_time,
            session_end_time=self.session_end_time,
            total_segments_downloaded=self.total_segments_downloaded,
            total_bytes_downloaded=self.total_bytes_downloaded,
            average_download_speed_kbps=avg_speed,
            quality_switches=self.quality_switches.copy(),
            buffering_events=self.buffering_events.copy(),
            quality_time_distribution=self.quality_time_distribution.copy(),
            startup_delay_seconds=self.startup_delay_seconds,
            total_buffering_time=self.total_buffering_time,
            mean_opinion_score=self.calculate_mean_opinion_score()
        )
    
    def export_to_json(self, filepath: str):
        """导出统计数据到JSON文件"""
        session = self.get_session_summary()
        
        # 转换为可序列化的字典
        data = asdict(session)
        
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"QoE统计数据已导出到: {filepath}")