import time
import logging
import json

logger = logging.getLogger(__name__) # Use module-specific logger

quality = [0, 0.287682, 0.693147, 1.098612, 1.791759]
psnr = [29.58124, 31.784398, 35.715179, 39.906998, 49.492304]

# --- QoE Metrics Manager ---
class QoEMetricsManager:
    def __init__(self):
        self.startup_latency_ms = None
        self.rebuffering_events = [] # Stores {'start_ts': timestamp_ms, 'duration_ms': duration_ms, 'end_ts': timestamp_ms}
        self.quality_switches_log = [] # Stores {'timestamp': ms, 'from_level': idx, 'to_level': idx, 'to_bitrate': bps}
        self.write_path = ''
        
        self.session_active = False
        self.session_start_time_ms = 0
        self.total_session_duration_ms = 0 # Actual time content was playing or supposed to be playing

        self.current_level_index = -1
        self.time_at_each_level = {} # {level_index: duration_ms}
        self.last_event_timestamp_ms = 0 # Used to calculate duration at current level before switch/end
        logger.info("QoEMetricsManager initialized.")

    def set_write_path(self, path):
        self.write_path = path

    def start_session_if_needed(self, event_timestamp_ms):
        if not self.session_active:
            self.session_active = True
            self.session_start_time_ms = event_timestamp_ms # Session starts with the first significant event
            self.last_event_timestamp_ms = event_timestamp_ms
            self.current_level_index = 0
            logger.info(f"QoE: Playback session started around {event_timestamp_ms}.")

    def update_time_at_level(self, event_timestamp_ms):
        if self.session_active and self.current_level_index != -1:
            duration_at_current_level_ms = event_timestamp_ms - self.last_event_timestamp_ms
            if duration_at_current_level_ms > 0:
                print('oooooooo ', self.current_level_index, duration_at_current_level_ms, ' oooooooo')
                self.time_at_each_level[self.current_level_index] = \
                    self.time_at_each_level.get(self.current_level_index, 0) + duration_at_current_level_ms
        self.last_event_timestamp_ms = event_timestamp_ms


    def record_startup_latency(self, latency_ms, timestamp_ms):
        self.start_session_if_needed(timestamp_ms - latency_ms) # Session started when play was initiated
        if self.startup_latency_ms is None:
            self.startup_latency_ms = latency_ms
            logger.info(f"QoE Event: Startup Latency = {latency_ms} ms (at {timestamp_ms})")
            self.last_event_timestamp_ms = timestamp_ms # Update last event time after startup

    def record_rebuffering_start(self, timestamp_ms):
        print('wuwuwuwuwuwuwuwu', timestamp_ms)
        self.start_session_if_needed(timestamp_ms)
        self.update_time_at_level(timestamp_ms)
        self.rebuffering_events.append({'start_ts': timestamp_ms, 'duration_ms': 0, 'end_ts': None}) # duration will be updated
        logger.info(f"QoE Event: Rebuffering Started at {timestamp_ms}")

    def record_rebuffering_end(self, timestamp_ms):
        self.start_session_if_needed(timestamp_ms) # Should already be active
        # Find the last open rebuffering event
        for event in reversed(self.rebuffering_events):
            if event['end_ts'] is None:
                duration_ms = timestamp_ms - event['start_ts']
                event['duration_ms'] = duration_ms
                event['end_ts'] = timestamp_ms
                logger.info(f"QoE Event: Rebuffering Ended. Duration = {duration_ms} ms (at {timestamp_ms})")
                break
        self.last_event_timestamp_ms = timestamp_ms # Update last event time after rebuffering

    def record_quality_switch(self, from_level_index, to_level_index, to_bitrate, timestamp_ms):
        self.start_session_if_needed(timestamp_ms)
        self.update_time_at_level(timestamp_ms)

        # If from_level_index is -1, it's the initial level setting.
        # current_level_index helps track the *actual* previous playing level.
        actual_from_level = self.current_level_index if from_level_index == -1 or self.current_level_index != -1 else from_level_index

        if actual_from_level != to_level_index : # Log only actual switches or initial set
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
        
        self.update_time_at_level(timestamp_ms) # Account for time since last event
        self.total_session_duration_ms = timestamp_ms - self.session_start_time_ms
        self.session_active = False
        logger.info(f"QoE: Playback session ended at {timestamp_ms}. Total duration: {self.total_session_duration_ms:.0f} ms.")
        self.print_summary(available_abr_streams)

    def print_summary(self, available_abr_streams=None):
        logger.info("--- QoE Summary ---")
        if not self.session_start_time_ms: # Check if any activity was logged
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

        # Average Played Bitrate calculation
        if available_abr_streams:
            total_weighted_bitrate_x_time = 0 # Sum of (bitrate_bps * time_seconds_at_level)
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
            # Effective playing time = total_session_duration - total_stall_duration - startup_latency (if startup is part of session)
            # For rebuffering ratio, typically total_stall_duration / (total_stall_duration + actual_playing_time)
            # Or simpler: total_stall_duration / total_session_duration_ms
            rebuffering_ratio = (total_stall_duration / self.total_session_duration_ms) * 100 if self.total_session_duration_ms > 0 else 0
            logger.info(f"  Rebuffering Ratio (approx): {rebuffering_ratio:.2f}%")
        logger.info("--------------------")
        status = {
            'quality': 0.0,
            'effective_ratio': (self.total_session_duration_ms - total_stall_duration - self.startup_latency_ms if self.startup_latency_ms is not None else 0) / self.total_session_duration_ms if self.total_session_duration_ms > 0 else 1,
            'weighted_quality': 0.0,
            'switch_count': f'{len(self.quality_switches_log)}',
            'PSNR': 0.0,
            'rebuffering_counts': f'{num_stalls}',
            'rebuffering_duration': f'{total_stall_duration:.2f} ms',
            'total_duration': f'{self.total_session_duration_ms:.2f} ms', 
            'rebuffering_ratio': f'{rebuffering_ratio:.2f}%',
            'startup_latency': f'{self.startup_latency_ms:.2f} ms' if self.startup_latency_ms is not None else 'Not recorded',
            'time_at_each_level': {
                '0': 0,
                '1': 0, 
                '2': 0,
                '3': 0,
                '4': 0
            },
            'switching_events': self.quality_switches_log,
            'rebuffering_events': self.rebuffering_events
        }
        tot = 0.0
        quality_val = 0.0
        psnr_val = 0.0
        for level_idx, duration_ms in self.time_at_each_level.items():
            status['time_at_each_level'][str(level_idx)] = duration_ms
            tot += duration_ms
            quality_val += quality[level_idx] * duration_ms
            psnr_val += psnr[level_idx] * duration_ms
        if tot > 0:
            status['quality'] = 1.0 * quality_val / tot
            status['PSNR'] = 1.0 * psnr_val / tot
        status['weighted_quality'] = 1.0 * status['quality'] + 1.5 * status['effective_ratio']
        with open(self.write_path, 'a', encoding='utf-8') as f:
            json.dump(status, f, ensure_ascii=False, indent=4)