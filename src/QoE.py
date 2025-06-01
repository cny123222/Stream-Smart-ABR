import time
import logging

logger = logging.getLogger(__name__) # 使用模块特定的日志记录器

# --- QoE 指标管理器 ---
class QoEMetricsManager:
    def __init__(self):
        self.startup_latency_ms = None
        self.rebuffering_events = [] # 存储 {'start_ts': timestamp_ms, 'duration_ms': duration_ms, 'end_ts': timestamp_ms}
        self.quality_switches_log = [] # 存储 {'timestamp': ms, 'from_level': idx, 'to_level': idx, 'to_bitrate': bps}
        
        self.session_active = False
        self.session_start_time_ms = 0
        self.total_session_duration_ms = 0 # 内容实际播放或应该播放的时间

        self.current_level_index = -1
        self.time_at_each_level = {} # {level_index: duration_ms}
        self.last_event_timestamp_ms = 0 # 用于计算切换/结束前在当前级别的持续时间
        logger.info("QoEMetricsManager initialized.")

    def start_session_if_needed(self, event_timestamp_ms):
        if not self.session_active:
            self.session_active = True
            self.session_start_time_ms = event_timestamp_ms # 会话从第一个重要事件开始
            self.last_event_timestamp_ms = event_timestamp_ms
            logger.info(f"QoE: Playback session started around {event_timestamp_ms}.")

    def update_time_at_level(self, event_timestamp_ms):
        if self.session_active and self.current_level_index != -1:
            duration_at_current_level_ms = event_timestamp_ms - self.last_event_timestamp_ms
            if duration_at_current_level_ms > 0:
                self.time_at_each_level[self.current_level_index] = \
                    self.time_at_each_level.get(self.current_level_index, 0) + duration_at_current_level_ms
        self.last_event_timestamp_ms = event_timestamp_ms

    def record_startup_latency(self, latency_ms, timestamp_ms):
        self.start_session_if_needed(timestamp_ms - latency_ms) # 会话在启动播放时开始
        if self.startup_latency_ms is None:
            self.startup_latency_ms = latency_ms
            logger.info(f"QoE Event: Startup Latency = {latency_ms} ms (at {timestamp_ms})")
            self.last_event_timestamp_ms = timestamp_ms # 启动后更新最后事件时间

    def record_rebuffering_start(self, timestamp_ms):
        self.start_session_if_needed(timestamp_ms)
        self.update_time_at_level(timestamp_ms)
        self.rebuffering_events.append({'start_ts': timestamp_ms, 'duration_ms': 0, 'end_ts': None}) # 持续时间将被更新
        logger.info(f"QoE Event: Rebuffering Started at {timestamp_ms}")

    def record_rebuffering_end(self, duration_ms, timestamp_ms):
        self.start_session_if_needed(timestamp_ms) # 应该已经激活
        # 查找最后一个未结束的卡顿事件
        for event in reversed(self.rebuffering_events):
            if event['end_ts'] is None:
                event['duration_ms'] = duration_ms
                event['end_ts'] = timestamp_ms
                logger.info(f"QoE Event: Rebuffering Ended. Duration = {duration_ms} ms (at {timestamp_ms})")
                break
        self.last_event_timestamp_ms = timestamp_ms # 卡顿后更新最后事件时间

    def record_quality_switch(self, from_level_index, to_level_index, to_bitrate, timestamp_ms):
        self.start_session_if_needed(timestamp_ms)
        self.update_time_at_level(timestamp_ms)

        # 如果 from_level_index 是 -1，这是初始级别设置。
        # current_level_index 帮助跟踪*实际*之前的播放级别。
        actual_from_level = self.current_level_index if from_level_index == -1 or self.current_level_index != -1 else from_level_index

        if actual_from_level != to_level_index : # 只记录实际切换或初始设置
            log_entry = {
                'timestamp': timestamp_ms,
                'from_level': actual_from_level,
                'to_level': to_level_index,
                'to_bitrate': to_bitrate
            }
            self.quality_switches_log.append(log_entry)
            if actual_from_level == -1:
                logger.info(f"QoE Event: Initial Level set to {to_level_index} (Bitrate: {to_bitrate/1000:.0f} Kbps) at {timestamp_ms}")
            else:
                logger.info(f"QoE Event: Quality Switch from level {actual_from_level} to {to_level_index} (Bitrate: {to_bitrate/1000:.0f} Kbps) at {timestamp_ms}")
        
        self.current_level_index = to_level_index

    def log_playback_session_end(self, timestamp_ms=None, available_abr_streams=None):
        if not self.session_active:
            logger.info("QoE: No active session to end or already ended.")
            return

        if timestamp_ms is None:
            timestamp_ms = time.time() * 1000
        
        self.update_time_at_level(timestamp_ms) # 计算自上次事件以来的时间
        self.total_session_duration_ms = timestamp_ms - self.session_start_time_ms
        self.session_active = False
        logger.info(f"QoE: Playback session ended at {timestamp_ms}. Total duration: {self.total_session_duration_ms:.0f} ms.")
        self.print_summary(available_abr_streams)

    def print_summary(self, available_abr_streams=None):
        logger.info("--- QoE Summary ---")
        if not self.session_start_time_ms: # 检查是否记录活动
            logger.info("  No playback activity recorded for QoE summary.")
            logger.info("--------------------")
            return

        if self.startup_latency_ms is not None:
            logger.info(f"  Startup Latency: {self.startup_latency_ms:.2f} ms")
        else:
            logger.info("  Startup Latency: Not recorded")
        
        num_stalls = len([e for e in self.rebuffering_events if e['duration_ms'] > 0])
        total_stall_duration = sum(e['duration_ms'] for e in self.rebuffering_events if e['duration_ms'] > 0)
        logger.info(f"  Rebuffering Events (Stalls): {num_stalls}")
        logger.info(f"  Total Rebuffering Duration: {total_stall_duration:.2f} ms")

        logger.info(f"  Quality Switches (logged): {len(self.quality_switches_log)}")
        # for i, switch in enumerate(self.quality_switches_log):
        #     logger.info(f"    Switch {i+1}: From {switch['from_level']} To {switch['to_level']} (Bitrate: {switch['to_bitrate']/1000:.0f} Kbps)")
        
        logger.info(f"  Time spent at each quality level (index: ms):")
        for level_idx, duration_ms in self.time_at_each_level.items():
            bitrate_str = "N/A"
            if available_abr_streams:
                if isinstance(level_idx, int) and 0 <= level_idx < len(available_abr_streams):
                    stream_info = available_abr_streams[level_idx]
                    if isinstance(stream_info, dict) and 'bandwidth' in stream_info:
                        bitrate_bps = stream_info.get('bandwidth', 0)
                        if bitrate_bps > 0:
                            bitrate_str = f"{bitrate_bps/1000:.0f} Kbps"
            logger.info(f"    Level {level_idx} ({bitrate_str}): {duration_ms:.0f} ms")

        # 平均播放比特率计算
        if available_abr_streams:
            total_weighted_bitrate_x_time = 0 # (比特率_bps * 在该级别的时间_秒) 的总和
            total_time_at_levels_seconds = 0
            for level_idx, duration_ms in self.time_at_each_level.items():
                if 0 <= level_idx < len(available_abr_streams):
                    bitrate_bps = available_abr_streams[level_idx].get('bandwidth', 0)
                    time_seconds = duration_ms / 1000.0
                    total_weighted_bitrate_x_time += bitrate_bps * time_seconds
                    total_time_at_levels_seconds += time_seconds
            
            if total_time_at_levels_seconds > 0:
                average_played_bitrate_kbps = (total_weighted_bitrate_x_time / total_time_at_levels_seconds) / 1000.0
                logger.info(f"  Average Played Bitrate (based on time at levels): {average_played_bitrate_kbps:.2f} Kbps")
            else:
                logger.info("  Average Played Bitrate: Not enough data.")

        logger.info(f"  Total Playback Session Duration (approx): {self.total_session_duration_ms:.2f} ms")
        if self.total_session_duration_ms > 0 :
            # 有效播放时间 = 总会话持续时间 - 总卡顿持续时间 - 启动延迟 (如果启动是会话的一部分)
            # 对于卡顿比率，通常是 总卡顿持续时间 / (总卡顿持续时间 + 实际播放时间)
            # 或者更简单: 总卡顿持续时间 / 总会话持续时间
            rebuffering_ratio = (total_stall_duration / self.total_session_duration_ms) * 100 if self.total_session_duration_ms > 0 else 0
            logger.info(f"  Rebuffering Ratio (approx): {rebuffering_ratio:.2f}%")
        logger.info("--------------------")