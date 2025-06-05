import time
import logging
import json

logger = logging.getLogger(__name__) # 使用模块特定的日志记录器

quality = [0, 0.287682, 0.693147, 1.098612, 1.791759]
psnr = [29.58124, 31.784398, 35.715179, 39.906998, 49.492304]

# --- QoE 指标管理器 ---
class QoEMetricsManager:
    def __init__(self):
        self.startup_latency_ms = None
        self.rebuffering_events = [] # 存储 {'start_ts': timestamp_ms, 'duration_ms': duration_ms, 'end_ts': timestamp_ms}
        self.quality_switches_log = [] # 存储 {'timestamp': ms, 'from_level': idx, 'to_level': idx, 'to_bitrate': bps}
        self.write_path = ''
        self.in_rebuffering = False
        
        self.session_active = False
        self.session_start_time_ms = 0
        self.total_session_duration_ms = 0 # 内容实际播放或应该播放的时间

        self.current_level_index = -1
        self.time_at_each_level = {} # {level_index: duration_ms}
        self.last_event_timestamp_ms = 0 # 用于计算切换/结束前在当前级别的持续时间
        logger.info("QoEMetricsManager initialized.")

    def set_write_path(self, path):
        self.write_path = path

    def start_session_if_needed(self, event_timestamp_ms):
        if not self.session_active:
            self.session_active = True
            self.session_start_time_ms = event_timestamp_ms # 会话从第一个重要事件开始
            self.last_event_timestamp_ms = event_timestamp_ms
            self.current_level_index = 0
            logger.info(f"QoE: Playback session started around {event_timestamp_ms}.")

    def update_time_at_level(self, event_timestamp_ms):
        if self.session_active and self.current_level_index != -1:
            duration_at_current_level_ms = event_timestamp_ms - self.last_event_timestamp_ms
            if duration_at_current_level_ms > 0:
                # print('oooooooo ', self.current_level_index, duration_at_current_level_ms, ' oooooooo')
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
        # print('wuwuwuwuwuwuwuwu', timestamp_ms)
        self.start_session_if_needed(timestamp_ms)
        self.update_time_at_level(timestamp_ms)
        self.rebuffering_events.append({'start_ts': timestamp_ms, 'duration_ms': 0, 'end_ts': None}) # 持续时间将被更新
        logger.info(f"QoE Event: Rebuffering Started at {timestamp_ms}")
        self.in_rebuffering = True

    def record_rebuffering_end(self, timestamp_ms):
        self.start_session_if_needed(timestamp_ms) ## 应该已经激活
        self.in_rebuffering = False
        # 查找最后一个未结束的卡顿事件
        for event in reversed(self.rebuffering_events):
            if event['end_ts'] is None:
                duration_ms = timestamp_ms - event['start_ts']
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
        if not self.in_rebuffering:
            self.update_time_at_level(timestamp_ms) # 计算自上次事件以来的时间
        else:
            self.record_rebuffering_end(timestamp_ms) # 确保最后一个卡顿事件被正确结束
        self.total_session_duration_ms = timestamp_ms - self.session_start_time_ms
        self.session_active = False
        logger.info(f"QoE: Playback session ended at {timestamp_ms}. Total duration: {self.total_session_duration_ms:.0f} ms.")
        self.print_summary(available_abr_streams)

    def print_summary(self, available_abr_streams=None,
                      # --- QoE 公式参数 ---
                      qoe_mu=4.0,          # 卡顿惩罚系数 mu
                      qoe_tau=1.0,           # 切换惩罚系数 tau
                      segment_duration_seconds=5.0 
                     ):
        logger.info("--- QoE Summary ---") # 日志: 开始打印QoE总结
        if not self.session_start_time_ms: # 检查是否有播放活动被记录
            logger.info("  No playback activity recorded for QoE summary.") # 日志: 没有播放活动记录
            logger.info("--------------------")
            # (你之前的默认返回 status 字典)
            return {
                'final_qoe_score': 'N/A', 'quality': 'N/A', 'effective_ratio': 'N/A', 
                'weighted_quality': 'N/A', 'PSNR': 'N/A', 'startup_latency': 'Not recorded',
                'rebuffering_counts': '0', 'rebuffering_duration': '0.00 ms',
                'rebuffering_ratio': '0.00%', 'switch_count': '0',
                'time_at_each_level_ms': {str(i): 0 for i in range(len(quality))},
                'average_played_bitrate_kbps': 'N/A',
                'total_playback_session_duration_ms': '0.00 ms',
                'qoe_formula_details': {
                    'term1_total_quality': 'N/A', 'term2_rebuffering_penalty': 'N/A',
                    'term3_switch_penalty': 'N/A', 'param_mu': qoe_mu, 'param_tau': qoe_tau,
                    'param_segment_duration_s': segment_duration_seconds
                },
                'switching_events': [], 'rebuffering_events': []
            }

        # --- 原有的统计数据计算 (保持不变) ---
        startup_latency_ms_val = self.startup_latency_ms if self.startup_latency_ms is not None else 0.0
        if self.startup_latency_ms is not None:
            logger.info(f"  Startup Latency: {self.startup_latency_ms:.2f} ms")
        else:
            logger.info("  Startup Latency: Not recorded")
        
        num_stalls = len([e for e in self.rebuffering_events if e['duration_ms'] > 0])
        total_stall_duration_ms = sum(e['duration_ms'] for e in self.rebuffering_events if e['duration_ms'] > 0)
        logger.info(f"  Rebuffering Events (Stalls): {num_stalls}")
        logger.info(f"  Total Rebuffering Duration: {total_stall_duration_ms:.2f} ms")

        logger.info(f"  Quality Switches (logged): {len(self.quality_switches_log)}")
        
        logger.info(f"  Time spent at each quality level (index: ms):")
        for level_idx, duration_ms_at_level in self.time_at_each_level.items():
            bitrate_str = "N/A"
            if available_abr_streams:
                if isinstance(level_idx, int) and 0 <= level_idx < len(available_abr_streams):
                    stream_info = available_abr_streams[level_idx]
                    if isinstance(stream_info, dict) and 'bandwidth' in stream_info:
                        bitrate_bps = stream_info.get('bandwidth', 0)
                        if bitrate_bps > 0:
                            bitrate_str = f"{bitrate_bps/1000:.0f} Kbps"
            logger.info(f"    Level {level_idx} ({bitrate_str}): {duration_ms_at_level:.0f} ms")

        average_played_bitrate_kbps_val = "N/A"
        if available_abr_streams:
            total_weighted_bitrate_x_time = 0 
            total_time_at_levels_seconds = 0
            for level_idx, duration_ms_at_level in self.time_at_each_level.items():
                if 0 <= level_idx < len(available_abr_streams):
                    bitrate_bps = available_abr_streams[level_idx].get('bandwidth', 0)
                    time_seconds_at_level = duration_ms_at_level / 1000.0
                    total_weighted_bitrate_x_time += bitrate_bps * time_seconds_at_level
                    total_time_at_levels_seconds += time_seconds_at_level
            if total_time_at_levels_seconds > 0:
                average_played_bitrate_kbps_val = (total_weighted_bitrate_x_time / total_time_at_levels_seconds) / 1000.0
                logger.info(f"  Average Played Bitrate (based on time at levels): {average_played_bitrate_kbps_val:.2f} Kbps")
            else:
                logger.info("  Average Played Bitrate: Not enough data.")

        logger.info(f"  Total Playback Session Duration (approx): {self.total_session_duration_ms:.2f} ms")
        rebuffering_ratio_val = 0.0
        if self.total_session_duration_ms > 0 :
            rebuffering_ratio_val = (total_stall_duration_ms / self.total_session_duration_ms) * 100
            logger.info(f"  Rebuffering Ratio (approx): {rebuffering_ratio_val:.2f}%")
        logger.info("--------------------")

        # --- QoE 公式计算 ---
        # 使用模块顶部的全局 quality 列表进行q(R_k)映射
        q_map_for_formula = quality # 直接引用全局 quality 列表

        # 1. 计算总图像质量项: sum_{所有播放时间} (q(R_k)_per_second * duration_at_R_k_seconds)
        qoe_total_quality_term = 0.0
        total_effective_played_duration_s = 0.0 # 记录用于计算质量项的有效播放总时长（秒）

        for level_idx, duration_ms_at_level in self.time_at_each_level.items():
            if 0 <= level_idx < len(q_map_for_formula): # 确保 level_idx 在 quality 列表的有效范围内
                quality_score_per_second = q_map_for_formula[level_idx] # 获取该等级的单位时间质量得分 q(R_k)
                duration_s_at_level = duration_ms_at_level / 1000.0 # 将毫秒转换为秒
                
                qoe_total_quality_term += quality_score_per_second * duration_s_at_level # 累加 (质量得分/秒 * 秒数)
                total_effective_played_duration_s += duration_s_at_level
            else:
                logger.warning(f"QoE Formula Calc: Level index {level_idx} out of bounds for quality map (len: {len(q_map_for_formula)}). Skipping for sum(q(R_k)).")
        
        # total_played_duration_s_for_quality_term 现在就是总的有效播放时长，用于日志或验证
        logger.info(f"QoE Metric - Term 1 (Total Time-Weighted Quality): {qoe_total_quality_term:.2f} (based on {total_effective_played_duration_s:.2f}s of effective playback)")

        # 2. 计算缓冲时间惩罚项: mu * sum(卡顿时长_k) (这部分不变)
        total_stall_duration_s = total_stall_duration_ms / 1000.0
        qoe_rebuffering_penalty_term = qoe_mu * total_stall_duration_s
        logger.info(f"QoE Metric - Term 2 (Rebuffering Penalty): {qoe_rebuffering_penalty_term:.2f} (mu={qoe_mu}, total_stall_s={total_stall_duration_s:.2f}s)")

        # 3. 计算码率切换惩罚项: tau * sum(|q(R_k+1) - q(R_k)|) (这部分不变, q(R_k)仍用quality列表的值)
        qoe_total_switch_abs_diff_sum = 0.0
        for switch_event in self.quality_switches_log:
            from_level = switch_event.get('from_level')
            to_level = switch_event.get('to_level')
            if from_level != -1 and from_level != to_level: 
                if 0 <= from_level < len(q_map_for_formula) and 0 <= to_level < len(q_map_for_formula):
                    q_from = q_map_for_formula[from_level]
                    q_to = q_map_for_formula[to_level]
                    qoe_total_switch_abs_diff_sum += abs(q_to - q_from)
                else:
                    logger.warning(f"QoE Formula Calc: Invalid level index in switch event for penalty: from={from_level}, to={to_level}.")
        
        qoe_switch_penalty_term = qoe_tau * qoe_total_switch_abs_diff_sum
        logger.info(f"QoE Metric - Term 3 (Switch Penalty): {qoe_switch_penalty_term:.2f} (tau={qoe_tau}, sum|q_diff|={qoe_total_switch_abs_diff_sum:.2f})")
        
        # 计算最终 QoE 得分 (这部分不变)
        final_qoe_score_val = qoe_total_quality_term - qoe_rebuffering_penalty_term - qoe_switch_penalty_term
        logger.info(f"QoE Metric - Final Calculated QoE Score: {final_qoe_score_val:.2f}")
        # --- QoE 公式计算结束 ---

        # --- 准备写入文件的status字典 (保持你之前的结构，只更新值) ---
        status = {
            'final_qoe_score': f'{final_qoe_score_val:.2f}',
            'quality': 'N/A', # 将通过下面的旧逻辑计算并更新
            'effective_ratio': (self.total_session_duration_ms - total_stall_duration_ms - startup_latency_ms_val) / self.total_session_duration_ms if self.total_session_duration_ms > 0 else 1.0,
            'weighted_quality': 'N/A', # 将通过下面的旧逻辑计算并更新
            'PSNR': 'N/A', # 将通过下面的旧逻辑计算并更新
            'switch_count': f'{len(self.quality_switches_log)}',
            'rebuffering_counts': f'{num_stalls}',
            'rebuffering_duration': f'{total_stall_duration_ms:.2f} ms',
            'total_duration': f'{self.total_session_duration_ms:.2f} ms',
            'rebuffering_ratio': f'{rebuffering_ratio_val:.2f}%',
            'startup_latency': f'{startup_latency_ms_val:.2f} ms' if self.startup_latency_ms is not None else 'Not recorded',
            'time_at_each_level_ms': {str(i): 0.0 for i in range(len(quality))},
            'average_played_bitrate_kbps': f'{average_played_bitrate_kbps_val:.2f} Kbps' if isinstance(average_played_bitrate_kbps_val, float) else average_played_bitrate_kbps_val,
            'qoe_formula_details': {
                'term1_total_time_weighted_quality': f'{qoe_total_quality_term:.2f}', # 修改了键名以反映计算方式
                'term2_rebuffering_penalty': f'{qoe_rebuffering_penalty_term:.2f}',
                'term3_switch_penalty': f'{qoe_switch_penalty_term:.2f}',
                'param_mu': qoe_mu,
                'param_tau': qoe_tau,
            },
            'switching_events': self.quality_switches_log,
            'rebuffering_events': self.rebuffering_events
        }
        
        # 填充 time_at_each_level_ms (保持不变)
        for level_idx, duration_ms in self.time_at_each_level.items():
            if str(level_idx) in status['time_at_each_level_ms']:
                 status['time_at_each_level_ms'][str(level_idx)] = float(f"{duration_ms:.0f}")

        # --- 保留并执行你之前的 'quality', 'PSNR', 和 'weighted_quality' 计算逻辑 (保持不变) ---
        total_duration_for_legacy_calc_ms = 0.0
        legacy_quality_val_sum_weighted = 0.0
        legacy_psnr_val_sum_weighted = 0.0

        for level_idx, duration_ms_at_level in self.time_at_each_level.items():
            total_duration_for_legacy_calc_ms += duration_ms_at_level
            if 0 <= level_idx < len(quality):
                legacy_quality_val_sum_weighted += quality[level_idx] * duration_ms_at_level
            if 0 <= level_idx < len(psnr):   
                legacy_psnr_val_sum_weighted += psnr[level_idx] * duration_ms_at_level
        
        if total_duration_for_legacy_calc_ms > 0:
            status['quality'] = f'{(legacy_quality_val_sum_weighted / total_duration_for_legacy_calc_ms):.4f}'
            status['PSNR'] = f'{(legacy_psnr_val_sum_weighted / total_duration_for_legacy_calc_ms):.2f}'
        else:
            status['quality'] = '0.0000'
            status['PSNR'] = '0.00'
        
        try:
            quality_float_for_weight = float(status['quality'])
            effective_ratio_float = float(status['effective_ratio'])
            status['weighted_quality'] = f'{(quality_float_for_weight + 1.5 * effective_ratio_float):.4f}'
        except ValueError:
            logger.warning("Could not calculate weighted_quality due to non-numeric 'quality' or 'effective_ratio'.")
            status['weighted_quality'] = 'N/A'
        # --- 结束保留的旧指标计算 ---
        
        # 写入文件 (保持不变)
        if self.write_path: 
            try:
                with open(self.write_path, 'w', encoding='utf-8') as f:
                    json.dump(status, f, ensure_ascii=False, indent=4)
                logger.info(f"QoE summary successfully written to: {self.write_path}")
            except IOError as e:
                logger.error(f"Failed to write QoE summary to {self.write_path}: {e}", exc_info=True)
        else:
            logger.warning("QoE summary not written: 'write_path' is not set in QoEMetricsManager.")
            
        logger.info("--- QoE Summary End ---")
        return status