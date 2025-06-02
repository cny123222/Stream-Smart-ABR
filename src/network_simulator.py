import time
import threading
import logging
import socket # 用于socket.error，由调用者捕获

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
    """Creates a default network simulation scenario."""
    player = NetworkScenarioPlayer()
    # 初始 50 毫秒全速
    player.add_step(0.05, None)
    
    if mode == 1:
        # 低带宽稳定模式
        player.add_step(300, 500_000)  # 500 Kbps 持续 5 分钟
        player.add_step(10, None)  # 最后 10 秒全速

    elif mode == 2:
        # 高带宽稳定模式
        player.add_step(300, 10_000_000)  # 10 Mbps 持续 5 分钟
        player.add_step(10, None)  # 最后 10 秒全速

    elif mode == 3:
        # 快速波动模式
        for _ in range(15):
            player.add_step(10, 2_000_000)  # 2 Mbps 持续 10 秒
            player.add_step(10, 800_000)    # 800 Kbps 持续 10 秒
        player.add_step(10, None)  # 最后 10 秒全速

    elif mode == 4:
        # 突发低带宽模式
        player.add_step(120, 5_000_000)  # 5 Mbps 持续 2 分钟
        player.add_step(30, 200_000)     # 200 Kbps 突发低带宽 30 秒
        player.add_step(150, 5_000_000)  # 5 Mbps 持续 2 分 30 秒
        player.add_step(10, None)  # 最后 10 秒全速

    elif mode == 5:
        # 逐步提升带宽模式
        bandwidths = [500_000, 1_000_000, 2_000_000, 5_000_000, 10_000_000]
        for bps in bandwidths:
            player.add_step(60, bps)  # 每个带宽级别持续 1 分钟
        player.add_step(10, None)  # 最后 10 秒全速

    elif mode == 6:
        # 逐步降低带宽模式
        bandwidths = [10_000_000, 5_000_000, 2_000_000, 1_000_000, 500_000]
        for bps in bandwidths:
            player.add_step(60, bps)  # 每个带宽级别持续 1 分钟
        player.add_step(10, None)  # 最后 10 秒全速

    elif mode == 7:
        # 突然降低的台阶状模式
        player.add_step(180, 5_000_000)
        player.add_step(120, 800_000)

    elif mode == 8:
        # 突然升高的台阶状模式
        player.add_step(180, 800_000)
        player.add_step(120, 5_000_000)

    else:
        # 默认模式
        player.add_step(20, 10_000_000)  # 从20秒全速开始（允许初始缓冲）
        player.add_step(20, 5_000_000)   # 然后，40秒5 Mbps
        player.add_step(20, 800_000)    # 然后，60秒0.8 Mbps
        player.add_step(20, 10_000_000)  # 然后，60秒10 Mbps
        # 总共20秒更快速波动的示例
        player.add_step(5, 500_000)    # 5秒0.5 Mbps
        player.add_step(5, 2_000_000)   # 5秒2 Mbps
        player.add_step(5, 500_000)    # 5秒0.5 Mbps
        player.add_step(5, 2_000_000)   # 5秒2 Mbps
        player.add_step(30, None)       # 最后，再30秒全速
        
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