# 在 QoE.py 的 QoEMetricsManager 类中的 print_summary 方法

    def print_summary(self, available_abr_streams=None, 
                      # --- 新增 QoE 公式参数 ---
                      qoe_quality_map=None, # 例如 {0: 1, 1: 2, 2: 3, 3: 4, 4: 5} 将level_idx映射到q(R_k)
                      qoe_mu=2.8,          # 卡顿惩罚系数 mu
                      qoe_tau=1.0,           # 切换惩罚系数 tau
                      segment_duration_seconds=5.0 # 每个视频分片的标准时长 (秒)
                     ):
        logger.info("--- QoE Summary ---")
        if not self.session_start_time_ms:
            logger.info("  No playback activity recorded for QoE summary.")
            logger.info("--------------------")
            # 返回一个包含默认值的status字典，或者None
            return { 
                'final_qoe_score': 'N/A', 
                'startup_latency': 'Not recorded',
                # ... 其他字段可以设为 'N/A' 或 0 ...
            }


        # --- 原有的统计数据计算 (保持不变) ---
        if self.startup_latency_ms is not None:
            logger.info(f"  Startup Latency: {self.startup_latency_ms:.2f} ms")
        else:
            logger.info("  Startup Latency: Not recorded")
        
        num_stalls = len([e for e in self.rebuffering_events if e['duration_ms'] > 0]) #
        total_stall_duration_ms = sum(e['duration_ms'] for e in self.rebuffering_events if e['duration_ms'] > 0) #
        logger.info(f"  Rebuffering Events (Stalls): {num_stalls}") #
        logger.info(f"  Total Rebuffering Duration: {total_stall_duration_ms:.2f} ms") #

        logger.info(f"  Quality Switches (logged): {len(self.quality_switches_log)}") #
        
        logger.info(f"  Time spent at each quality level (index: ms):") #
        # ... (你原有的打印 time_at_each_level 的代码) ...
        # ... (你原有的打印 Average Played Bitrate 的代码) ...
        logger.info(f"  Total Playback Session Duration (approx): {self.total_session_duration_ms:.2f} ms") #
        
        rebuffering_ratio = 0.0
        if self.total_session_duration_ms > 0 and total_stall_duration_ms > 0 : # 确保 self.total_session_duration_ms 大于0
            rebuffering_ratio = (total_stall_duration_ms / self.total_session_duration_ms) * 100 #
        logger.info(f"  Rebuffering Ratio (approx): {rebuffering_ratio:.2f}%") #
        logger.info("--------------------")

        # --- QoE 公式计算 ---
        final_qoe_score = 0.0
        
        # 1. 计算总图像质量: sum(q(R_k))
        # 我们需要一个从 level_idx 到质量得分 q(R_k) 的映射
        # 如果没有提供 qoe_quality_map，我们可以用码率本身或其对数作为 q(R_k)
        # 或者使用你之前代码中的 quality[level_idx] （如果它代表的是每个分片的质量而非每秒）
        
        # 假设 qoe_quality_map = {0: 800, 1: 1500, 2: 4000, 3: 8000, 4: 16000} (直接用码率kbps)
        # 或者 qoe_quality_map = {0: np.log(800), 1: np.log(1500), ...} (用码率的对数)
        # 或者你传入的 quality 列表就是这个映射 (quality[level_idx])
        
        # 简化：我们使用时间加权的平均质量得分，乘以总的“无卡顿”播放时间
        # 或者更符合公式：计算每个分片的质量得分总和
        total_quality_score_sum = 0.0
        total_played_segments = 0 # 用于调试或另一种计算方式
        
        # 默认的质量映射 (如果未提供)，可以直接使用码率值 (例如，转换为Mbps)
        # 或者使用你代码中似乎存在的 quality 和 psnr 列表
        default_quality_map = {}
        if available_abr_streams and qoe_quality_map is None:
            logger.info("QoE calculation: Using stream bandwidth / 1e6 as q(R_k) since qoe_quality_map not provided.")
            for i, stream in enumerate(available_abr_streams):
                default_quality_map[i] = stream.get('bandwidth', 0) / 1_000_000.0 # 使用 Mbps 作为质量单位
        elif qoe_quality_map:
            default_quality_map = qoe_quality_map
        else: # 如果两者都不可用，则质量项可能为0或引发错误，这里做个保护
            logger.warning("QoE calculation: available_abr_streams or qoe_quality_map not available for quality scoring.")
            # 可以选择将 default_quality_map 留空，后续检查

        # 遍历 self.time_at_each_level 来计算第一项
        # time_at_each_level 的 key 是 level_idx, value 是 duration_ms
        for level_idx, duration_ms_at_level in self.time_at_each_level.items():
            if level_idx in default_quality_map:
                quality_score_for_level = default_quality_map[level_idx]
                # 计算在这个等级播放了多少个分片
                num_segments_at_level = (duration_ms_at_level / 1000.0) / segment_duration_seconds
                total_quality_score_sum += quality_score_for_level * num_segments_at_level
                total_played_segments += num_segments_at_level
            else:
                logger.warning(f"QoE calculation: Quality score for level_idx {level_idx} not found in map. Skipping for sum(q(R_k)).")
        
        logger.info(f"QoE Metric - Total Quality Score Sum (sum q(R_k)): {total_quality_score_sum:.2f} (based on ~{total_played_segments:.1f} segments)")
        final_qoe_score += total_quality_score_sum

        # 2. 计算缓冲时间惩罚: mu * sum(卡顿时长_k)
        # total_stall_duration_ms 已经是总卡顿时长了
        total_stall_duration_s = total_stall_duration_ms / 1000.0
        rebuffering_penalty = qoe_mu * total_stall_duration_s
        logger.info(f"QoE Metric - Rebuffering Penalty (mu * stall_s): {rebuffering_penalty:.2f} (mu={qoe_mu}, stall={total_stall_duration_s:.2f}s)")
        final_qoe_score -= rebuffering_penalty

        # 3. 计算码率切换惩罚: tau * sum(|q(R_k+1) - q(R_k)|)
        # self.quality_switches_log 包含 {'from_level': idx, 'to_level': idx, ...}
        total_switch_penalty_sum = 0.0
        for switch_event in self.quality_switches_log:
            from_level = switch_event.get('from_level')
            to_level = switch_event.get('to_level')
            
            # 忽略初始设置 (from_level == -1)
            if from_level != -1 and from_level != to_level:
                q_from = default_quality_map.get(from_level)
                q_to = default_quality_map.get(to_level)
                
                if q_from is not None and q_to is not None:
                    total_switch_penalty_sum += abs(q_to - q_from)
                else:
                    logger.warning(f"QoE calculation: Quality score for from_level {from_level} or to_level {to_level} not found in map for switch penalty.")
        
        switch_penalty_component = qoe_tau * total_switch_penalty_sum
        logger.info(f"QoE Metric - Switch Penalty (tau * sum|q_diff|): {switch_penalty_component:.2f} (tau={qoe_tau}, sum|q_diff|={total_switch_penalty_sum:.2f})")
        final_qoe_score -= switch_penalty_component
        
        logger.info(f"QoE Metric - Final Calculated QoE Score: {final_qoe_score:.2f}")
        # --- QoE 公式计算结束 ---

        # --- 更新 status 字典 ---
        status = {
            'final_qoe_score': f'{final_qoe_score:.2f}', # 新增最终QoE得分
            # 'quality': 0.0, # 你之前的 quality 和 PSNR 计算方式
            # 'effective_ratio': ..., # 你之前的
            # 'weighted_quality': 0.0, # 这个将被 final_qoe_score 替代
            'PSNR': 0.0, # 你之前的 PSNR 计算方式，可以保留或也用新模型替代
            'switch_count': f'{len(self.quality_switches_log)}', #
            'rebuffering_counts': f'{num_stalls}', #
            'rebuffering_duration': f'{total_stall_duration_ms:.2f} ms', #
            'total_duration': f'{self.total_session_duration_ms:.2f} ms',  #
            'rebuffering_ratio': f'{rebuffering_ratio:.2f}%', #
            'startup_latency': f'{self.startup_latency_ms:.2f} ms' if self.startup_latency_ms is not None else 'Not recorded', #
            'time_at_each_level': {str(i): 0 for i in range(5)}, # 初始化为5个级别
            'switching_events': self.quality_switches_log, #
            'rebuffering_events': self.rebuffering_events, #
            # --- QoE 模型参数记录 ---
            'qoe_param_mu': qoe_mu,
            'qoe_param_tau': qoe_tau,
            'qoe_segment_duration_s': segment_duration_seconds,
            'qoe_total_quality_term': f'{total_quality_score_sum:.2f}',
            'qoe_rebuffering_penalty_term': f'{rebuffering_penalty:.2f}',
            'qoe_switch_penalty_term': f'{switch_penalty_component:.2f}'
        }
        
        # 更新 time_at_each_level 和你之前的 quality/PSNR 计算（如果还想保留的话）
        # 注意: 你之前的 quality 和 psnr 列表需要作为参数传入，或者在这里能访问到
        # 我将假设它们作为参数传入，或者你从 qoe_quality_map 中派生它们
        
        # 你之前的PSNR和平均质量计算逻辑可以保留，或者也用新的q(R_k)思路统一
        # 为了演示，我暂时注释掉你之前的 quality_val 和 psnr_val 计算，因为它们依赖
        # 未在此函数参数中明确定义的 'quality' 和 'psnr' 列表。
        # 你需要决定如何提供这些值，或者统一使用 qoe_quality_map。
        
        # tot = 0.0
        # quality_val_old = 0.0 # 你之前的 'quality'
        # psnr_val_old = 0.0    # 你之前的 'PSNR'

        for level_idx, duration_ms_at_level in self.time_at_each_level.items():
            if str(level_idx) in status['time_at_each_level']: # 确保键存在
                 status['time_at_each_level'][str(level_idx)] = duration_ms_at_level #
            # tot += duration_ms
            # if qoe_quality_map and level_idx in qoe_quality_map: # 假设用qoe_quality_map代表旧的quality列表
            #     quality_val_old += qoe_quality_map[level_idx] * duration_ms_at_level
            # if psnr_map and level_idx in psnr_map: # 假设有一个psnr_map
            #     psnr_val_old += psnr_map[level_idx] * duration_ms_at_level
        
        # if tot > 0:
        #     status['quality'] = f'{(quality_val_old / tot):.2f}' # 使用 .2f 格式化
        #     status['PSNR'] = f'{(psnr_val_old / tot):.2f}'   # 使用 .2f 格式化
        # else:
        #     status['quality'] = 'N/A'
        #     status['PSNR'] = 'N/A'

        # 移除旧的 weighted_quality
        # status['weighted_quality'] = 1.0 * status['quality'] + 1.5 * status['effective_ratio']

        # 写入文件 (确保 self.write_path 已定义)
        if hasattr(self, 'write_path') and self.write_path: #
            try:
                with open(self.write_path, 'w', encoding='utf-8') as f:
                    json.dump(status, f, ensure_ascii=False, indent=4)
                logger.info(f"QoE summary successfully written to: {self.write_path}")
            except IOError as e:
                logger.error(f"Failed to write QoE summary to {self.write_path}: {e}")
            except AttributeError: # 如果 self.write_path 未定义
                logger.warning("QoE summary not written: self.write_path is not defined.")
        else:
            logger.warning("QoE summary not written: self.write_path is not defined or is empty.")
            
        return status # 返回包含新分数的status字典