import time
import threading
import logging
import socket # 用于socket.error，由调用者捕获
import random

logger = logging.getLogger(__name__) # 使用模块自己的logger名称

# --- 模拟的全局变量 ---
g_simulated_bandwidth_bps = None  # None表示无模拟，单位为比特每秒
g_simulation_lock = threading.Lock()

g_bandwidth_update_callback = None  # 网络更新的回调函数，如有需要

def set_simulated_bandwidth(bps):
    """
    设置目标模拟带宽。

    参数:
        bps (int或None): 目标带宽，单位为比特每秒。None表示禁用模拟。
    """
    global g_simulated_bandwidth_bps
    with g_simulation_lock:
        g_simulated_bandwidth_bps = bps
        status_message = {}
        if bps is None:
            logger.info("=> NET_SIM: Throttling disabled (full speed).") 
            status_message = {"status": "Full Speed"}
        else:
            logger.info(f"=> NET_SIM: Bandwidth target set to {bps / 1_000_000:.2f} Mbps.") 
            status_message = {"bandwidth_Mbps": bps / 1_000_000}
            
        if g_bandwidth_update_callback:
            try:
                g_bandwidth_update_callback(status_message) # 调用回调
            except Exception as e:
                logger.error(f"Error in network simulator bandwidth update callback: {e}")

def get_current_simulated_bandwidth():
    """
    获取当前目标模拟带宽。

    返回:
        int或None: 当前带宽，单位为bps，如果禁用则为None。
    """
    with g_simulation_lock:
        return g_simulated_bandwidth_bps
    
def set_bandwidth_update_callback(callback_func):
    """设置带宽更新回调函数"""
    global g_bandwidth_update_callback
    g_bandwidth_update_callback = callback_func

def throttle_data_transfer(data_to_send, target_bps, output_stream, segment_name_for_log="Unknown Segment"):
    """
    将data_to_send发送到output_stream，限制到target_bps。
    此函数将写入output_stream并引入延迟。
    如果output_stream.write()失败，可能引发socket.error。

    参数:
        data_to_send (bytes): 要发送的字节字符串数据。
        target_bps (int): 目标带宽，单位为比特每秒。
        output_stream (类文件对象): 要写入的流（例如，处理器中的self.wfile）。
        segment_name_for_log (str): 用于记录日志的分片名称。

    返回:
        float: 基于target_bps和data_size的预期传输时间（秒）。
    
    异常:
        socket.error: 如果写入output_stream时发生错误。
    """
    data_size_bytes = len(data_to_send)
    if data_size_bytes == 0:
        return 0.0

    simulated_bytes_per_sec = target_bps / 8
    if simulated_bytes_per_sec <= 0: # 避免除零或负速率
        # 如果速率为零或无效，实际上是无限时间，或以最小延迟发送一个块
        # 出于实用目的，我们假设这是一个非常慢的速率但不为零。
        # 这种情况理想情况下应该由调用者处理，确保target_bps > 0。
        # 如果仍然发生，我们可以将其视为最小正速率。
        logger.warning(f"NET_SIM: Invalid target_bps ({target_bps}). Assuming a very slow rate for segment {segment_name_for_log}.")
        simulated_bytes_per_sec = 1 # 每秒1字节（极慢）

    expected_transfer_time_seconds = data_size_bytes / simulated_bytes_per_sec

    # 在开始实际限速发送之前记录日志
    logger.info(
        f"NET_SIM: Simulating download at {target_bps / 1_000_000:.2f} Mbps "
        f"for {segment_name_for_log} ({data_size_bytes / 1024:.1f} KB), "
        f"expected time: {expected_transfer_time_seconds:.2f}s"
    )

    chunk_size = 4 * 1024  # 以4KB块发送
    bytes_sent = 0

    # 调用者（代理处理器）应该从此块中捕获socket.error
    while bytes_sent < data_size_bytes:
        chunk = data_to_send[bytes_sent : bytes_sent + chunk_size]
        if not chunk: # 如果data_size_bytes > 0，不应该发生
            break
        
        output_stream.write(chunk)
        output_stream.flush() # 确保数据通过套接字发送

        bytes_sent += len(chunk)
        
        # 基于模拟带宽计算此块的延迟
        delay_for_chunk = len(chunk) / simulated_bytes_per_sec
        time.sleep(delay_for_chunk) # 引入延迟
    
    return expected_transfer_time_seconds

class NetworkScenarioPlayer:
    """
    在单独线程中管理和播放网络带宽变化序列。
    """
    def __init__(self):
        self.scenario_steps = [] # (下一步前延迟秒数, 带宽bps或None)的列表
        self._thread = None
        self._stop_event = threading.Event() # 用于信号线程停止

    def add_step(self, duration_seconds, bandwidth_bps):
        """
        向模拟场景添加一个步骤。
        每个步骤定义一个在特定持续时间内活跃的带宽。

        参数:
            duration_seconds (float): 此带宽设置应持续多长时间。
            bandwidth_bps (int或None): 此步骤的目标带宽，单位为bps。None表示全速。
        """
        self.scenario_steps.append((duration_seconds, bandwidth_bps))
        return self # 允许链式调用

    def _play_scenario_target(self):
        logger.info("SIM_CTRL: Network simulation scenario player thread started.")
        total_elapsed_time_for_logging = 0

        for i, (duration_seconds, bandwidth_bps) in enumerate(self.scenario_steps):
            if self._stop_event.is_set():
                logger.info("SIM_CTRL: Stop event detected, terminating scenario early.")
                break
            
            step_description = f"{bandwidth_bps / 1_000_000:.2f} Mbps" if bandwidth_bps is not None else "Full Speed"
            logger.info(
                f"SIM_CTRL: Step {i+1}/{len(self.scenario_steps)} - Setting bandwidth to {step_description} "
                f"for {duration_seconds:.1f}s (Total elapsed in scenario: {total_elapsed_time_for_logging:.1f}s)"
            )
            set_simulated_bandwidth(bandwidth_bps)
            
            if duration_seconds > 0:
                # 等待此步骤的持续时间，或直到设置stop_event
                self._stop_event.wait(timeout=duration_seconds)
            
            total_elapsed_time_for_logging += duration_seconds

        if not self._stop_event.is_set(): # 如果场景自然完成
             logger.info("SIM_CTRL: All scenario steps completed.")
        logger.info("SIM_CTRL: Network simulation scenario player thread finished.")

    def start(self):
        """在新线程中开始播放场景。"""
        if self._thread and self._thread.is_alive():
            logger.warning("SIM_CTRL: Scenario player thread is already running.")
            return
        
        if not self.scenario_steps:
            logger.warning("SIM_CTRL: No scenario steps defined. Player will not start.")
            return

        self._stop_event.clear() # 清除之前运行的停止事件
        self._thread = threading.Thread(target=self._play_scenario_target, daemon=True, name="SimScenarioThread")
        self._thread.start()
        logger.info("SIM_CTRL: Scenario player initiated.")

    def stop(self):
        """信号场景播放器线程停止并等待其加入。"""
        if self._thread and self._thread.is_alive():
            logger.info("SIM_CTRL: Signaling scenario player thread to stop...")
            self._stop_event.set() # 信号线程停止
            self._thread.join(timeout=5.0) # 等待线程结束
            if self._thread.is_alive():
                logger.warning("SIM_CTRL: Scenario player thread did not stop cleanly within timeout.")
            else:
                logger.info("SIM_CTRL: Scenario player thread stopped.")
        self._thread = None # 清除线程引用

# --- Example default scenario ---
def create_default_simulation_scenario(mode = -1):
    """根据模式创建网络模拟场景。"""
    player = NetworkScenarioPlayer()
    # 所有模式都有一个短暂的初始全速阶段，用于页面加载和初始请求
    player.add_step(0.1, None) # 初始0.1秒全速，替换之前0.05秒，稍微延长一点点

    # 定义一些辅助函数来创建更真实的带宽模式
    def add_stable_period_with_fluctuations(duration_seconds, base_bps, fluctuation_percent=0.1, fluctuation_interval_seconds=5):
        """
        添加一个带有小幅波动的稳定带宽期。
        Args:
            duration_seconds (float): 此稳定期的总时长。
            base_bps (int): 基础带宽 (bps)。
            fluctuation_percent (float): 波动幅度百分比 (例如0.1表示+/-10%)。
            fluctuation_interval_seconds (float): 带宽小幅改变的间隔。
        """
        elapsed_in_period = 0
        while elapsed_in_period < duration_seconds:
            interval = min(fluctuation_interval_seconds, duration_seconds - elapsed_in_period)
            # 随机正负波动
            fluctuation = base_bps * fluctuation_percent * (random.random() * 2 - 1)
            current_bps = int(base_bps + fluctuation)
            # 确保带宽不为负或过低 (例如，至少为基础的50%或一个绝对最小值)
            current_bps = max(current_bps, int(base_bps * 0.5), 100_000) # 至少100Kbps
            player.add_step(interval, current_bps)
            elapsed_in_period += interval

    def add_gradual_change(start_bps, end_bps, total_duration_seconds, num_steps=5):
        """
        添加一个带宽渐变期。
        Args:
            start_bps (int): 起始带宽。
            end_bps (int): 结束带宽。
            total_duration_seconds (float): 渐变总时长。
            num_steps (int): 渐变步数。
        """
        if num_steps <= 0:
            player.add_step(total_duration_seconds, end_bps) # 直接设置为结束值
            return
            
        step_duration = total_duration_seconds / num_steps
        delta_bps_per_step = (end_bps - start_bps) / num_steps
        current_bps = float(start_bps)
        for i in range(num_steps):
            if i == num_steps - 1: # 最后一步直接使用目标带宽，避免浮点累积误差
                current_bps = float(end_bps)
            else:
                current_bps += delta_bps_per_step
            player.add_step(step_duration, int(current_bps))

    # --- 根据模式定义场景 ---
    # 目标总时长约为 90-120 秒
    # 每个模式的总持续时间将是其内部add_step持续时间的总和。

    if mode == 1:
        # 低带宽（约1-1.5Mbps）但带有较明显波动，总时长约 100 秒
        logger.info("SIM_CTRL: Configuring Scenario Mode 1 - Fluctuating Low-Medium Bandwidth")
        base_bw = 1_200_000 # 1.2 Mbps
        add_stable_period_with_fluctuations(duration_seconds=100, base_bps=base_bw, fluctuation_percent=0.25, fluctuation_interval_seconds=7)

    elif mode == 2:
        # 高带宽（20-28Mbps）但带有波动，总时长约 100 秒
        logger.info("SIM_CTRL: Configuring Scenario Mode 2 - Fluctuating High Bandwidth")
        base_bw = 25_000_000 # 25 Mbps
        add_stable_period_with_fluctuations(duration_seconds=100, base_bps=base_bw, fluctuation_percent=0.15, fluctuation_interval_seconds=6)

    elif mode == 3:
        # 较大幅度快速波动 (5Mbps <-> 25Mbps)，总时长约 100 秒
        logger.info("SIM_CTRL: Configuring Scenario Mode 3 - Rapid Large Fluctuations (5-25Mbps)")
        for i in range(5): # 5个周期，每个周期20秒
            if i % 2 == 0:
                add_gradual_change(25_000_000, 5_000_000, 8, num_steps=4) 
                add_stable_period_with_fluctuations(12, 5_000_000, 0.2, 4)
            else:
                add_gradual_change(5_000_000, 25_000_000, 8, num_steps=4)
                add_stable_period_with_fluctuations(12, 25_000_000, 0.1, 3)
                
    elif mode == 4:
        # 先高带宽稳定，然后突发深跌至1Mbps后缓慢恢复，总时长约 120 秒
        logger.info("SIM_CTRL: Configuring Scenario Mode 4 - High BW, Deep Drop to 1Mbps, Slow Recovery")
        add_stable_period_with_fluctuations(30, 25_000_000, 0.05, 6) 
        add_gradual_change(25_000_000, 1_000_000, 15, num_steps=5)  # 15秒快速降至1Mbps
        add_stable_period_with_fluctuations(30, 1_000_000, 0.25, 5) # 30秒在1Mbps附近 (800Kbps - 1.25Mbps波动)
        add_gradual_change(1_000_000, 15_000_000, 45, num_steps=10) # 45秒缓慢恢复至15Mbps

    elif mode == 5:
        # 逐步平滑提升带宽，每级带宽带有小波动，总时长约 105 秒
        logger.info("SIM_CTRL: Configuring Scenario Mode 5 - Gradual Increase with Fluctuations (1M to 28M)")
        bandwidth_levels = [1_000_000, 3_000_000, 8_000_000, 15_000_000, 28_000_000]
        duration_per_major_step = 21 
        for i in range(len(bandwidth_levels)):
            if i == 0: # 第一级直接稳定
                add_stable_period_with_fluctuations(duration_per_major_step, bandwidth_levels[i], 0.2, 5)
            else: # 从上一级过渡到当前级
                add_gradual_change(bandwidth_levels[i-1], bandwidth_levels[i], duration_per_major_step * 0.6, num_steps=3)
                add_stable_period_with_fluctuations(duration_per_major_step * 0.4, bandwidth_levels[i], 0.1, 4)
                
    elif mode == 6:
        # 逐步平滑降低带宽，每级带宽带有小波动，总时长约 105 秒
        logger.info("SIM_CTRL: Configuring Scenario Mode 6 - Gradual Decrease with Fluctuations (28M to 1M)")
        bandwidth_levels = [28_000_000, 15_000_000, 8_000_000, 3_000_000, 1_000_000]
        duration_per_major_step = 21
        for i in range(len(bandwidth_levels)):
            if i == 0:
                add_stable_period_with_fluctuations(duration_per_major_step, bandwidth_levels[i], 0.1, 6)
            else:
                add_gradual_change(bandwidth_levels[i-1], bandwidth_levels[i], duration_per_major_step * 0.6, num_steps=3)
                add_stable_period_with_fluctuations(duration_per_major_step * 0.4, bandwidth_levels[i], 0.2, 4)

    elif mode == 7:
        # V形反转: 中低 -> 高 -> 中低，总时长约 120 秒
        logger.info("SIM_CTRL: Configuring Scenario Mode 7 - MediumLow-High-MediumLow (V-Shape)")
        add_gradual_change(18_000_000, 2_000_000, 20, num_steps=6)  # 20秒：从18M降到2M
        add_stable_period_with_fluctuations(25, 2_000_000, 0.25, 5) # 25秒：稳定在2M (1M-2.5M波动)
        add_gradual_change(2_000_000, 28_000_000, 30, num_steps=7)  # 30秒：从2M升到28M
        add_stable_period_with_fluctuations(30, 28_000_000, 0.1, 5) # 30秒：稳定在28M
        add_gradual_change(28_000_000, 2_500_000, 15, num_steps=5)  # 15秒：从28M降回2.5M

    elif mode == 8:
        # A形: 高 -> 中低 -> 高，总时长约 120 秒
        logger.info("SIM_CTRL: Configuring Scenario Mode 8 - High-MediumLow-High (A-Shape)")
        add_stable_period_with_fluctuations(25, 22_000_000, 0.1, 6)  # 25秒：稳定在22M
        add_gradual_change(22_000_000, 1_500_000, 30, num_steps=7) # 30秒：从22M降到1.5M
        add_stable_period_with_fluctuations(30, 1_500_000, 0.3, 4)   # 30秒：稳定在1.5M (1.05M-1.95M波动)
        add_gradual_change(1_500_000, 20_000_000, 35, num_steps=8) # 35秒：从1.5M升到20M
        
    else: # 默认模式 (mode 9 或其他未指定模式) - 一个更长、更全面的综合测试场景
        logger.info("SIM_CTRL: Configuring Scenario Mode 9 (Default Long Comprehensive Test)")
        # 阶段1: 稳定高带宽 (25秒)
        add_stable_period_with_fluctuations(25, 20_000_000, 0.1, 5)  # 20M +/- 2M
        # 阶段2: 平滑下降到中带宽 (25秒)
        add_gradual_change(20_000_000, 6_000_000, 25, num_steps=5)
        # 阶段3: 中带宽波动 (30秒)
        add_stable_period_with_fluctuations(30, 6_000_000, 0.15, 6) # 6M +/- 0.9M
        # 阶段4: 快速深跌至较低带宽 (10秒)
        add_gradual_change(6_000_000, 1_200_000, 10, num_steps=3)   # 降至1.2M
        # 阶段5: 较低带宽挣扎并有较大波动 (30秒)
        add_stable_period_with_fluctuations(30, 1_200_000, 0.25, 4) # 1.2M +/- 0.3M (确保不低于800k)
        # 阶段6: 尝试快速恢复至中高带宽 (20秒)
        add_gradual_change(1_200_000, 12_000_000, 20, num_steps=5)
        # 阶段7: 中高带宽稳定期 (20秒)
        add_stable_period_with_fluctuations(20, 12_000_000, 0.1, 5) # 总时长约160秒
        
    player.add_step(10, None) # 所有模式最后10秒全速
    return player

if __name__ == '__main__':
    # 模拟器模块的基本测试
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s')
    logger.info("Testing network_simulator.py standalone...")
    
    test_player = create_default_simulation_scenario()
    test_player.start()
    
    try:
        count = 0
        while test_player._thread and test_player._thread.is_alive() and count < 300: # 最多运行5分钟
            time.sleep(1)
            current_bw = get_current_simulated_bandwidth()
            # logger.info(f"MainTest: Current simulated BW: {current_bw / 1_000_000 if current_bw else 'None'} Mbps")
            count +=1
    except KeyboardInterrupt:
        logger.info("MainTest: Interrupted by user.")
    finally:
        test_player.stop()
        logger.info("MainTest: Finished.")