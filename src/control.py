import subprocess
import time
import os
import signal
import ctypes
import logging # 导入logging模块

# --- 日志记录器设置 ---
# 为此脚本设置一个独立的日志记录器
logger = logging.getLogger("AutomatedTestRunner")
logger.setLevel(logging.INFO) # 默认级别可以设为INFO
if not logger.handlers: # 防止重复添加处理器
    console_handler = logging.StreamHandler()
    # 定义日志格式，可以与其他模块保持一致
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

# 定义不同的参数组合
parameter_combinations = [
    # 例如: [决策逻辑类型, 网络环境场景]
    [6, 9]
]

# 生成参数组合 (决策逻辑 1-7, 网络环境 1-9)
# for i in [6]:
#     for j in range(1, 10):
#         parameter_combinations.append([i, j])

# 用于跟踪当前正在运行的子进程
current_process = None

def cleanup_process_gracefully(proc):
    """
    尝试优雅地终止指定的子进程。
    如果优雅终止失败（超时），则会尝试强制终止。

    参数:
        proc (subprocess.Popen): 需要被终止的子进程对象。

    返回:
        bool: 如果尝试了清理操作（无论成功与否）则返回True，否则返回False。
    """
    if proc and proc.poll() is None:  # 检查进程对象是否存在且仍在运行
        logger.info(f"Interrupt or cleanup signal received. Handling client.py (PID: {proc.pid})...") # 日志英文
        if os.name == 'nt': # Windows 操作系统
            logger.info(f"Sending Terminate signal to Windows process {proc.pid} as a fallback...") # 日志英文
            proc.terminate()
            try:
                proc.wait(timeout=10) 
                logger.info(f"Windows process {proc.pid} terminated or was already finished.") # 日志英文
            except subprocess.TimeoutExpired:
                logger.warning(f"Windows process {proc.pid} did not terminate within 10s after TerminateProcess call.") # 日志英文
        else:  # Linux 和 macOS 操作系统
            logger.info(f"Sending SIGTERM to process {proc.pid}. Expecting client.py's 30s timeout or its signal handler to respond...") # 日志英文
            proc.terminate()  # 发送 SIGTERM 信号
            try:
                proc.wait(timeout=35)
                logger.info(f"Process {proc.pid} terminated gracefully or completed its run.") # 日志英文
            except subprocess.TimeoutExpired:
                logger.warning(f"Process {proc.pid} did not terminate after SIGTERM and expected runtime. Forcing kill...") # 日志英文
                proc.kill()  # 发送 SIGKILL 信号
                try:
                    proc.wait(timeout=5) 
                    logger.info(f"Process {proc.pid} was killed.") # 日志英文
                except subprocess.TimeoutExpired:
                    logger.error(f"Process {proc.pid} did not terminate even after SIGKILL. Manual check may be needed.") # 日志英文
            except Exception as e:
                logger.error(f"Error during Unix process termination for PID {proc.pid}: {e}", exc_info=True) # 日志英文
        return True
    logger.debug(f"Cleanup process: Process does not exist or has already finished (PID: {proc.pid if proc else 'N/A'}).") # 日志英文
    return False

try:
    for params_index, params in enumerate(parameter_combinations):
        current_process = None # 每次循环开始时重置当前进程跟踪变量
        logger.info(f"--- Starting test {params_index + 1}/{len(parameter_combinations)} with parameters: {params} ---") # 日志英文
        
        command = [
            'python',
            './src/client.py', 
            str(params[0]),    # 决策逻辑参数
            str(params[1]),    # 网络环境参数
            f'./test/case_{params[0]}_{params[1]}' # 测试用例/日志输出目录参数
        ]
        
        logger.info(f"Executing command: {' '.join(command)}") # 日志英文
        
        # 启动子进程
        current_process = subprocess.Popen(command)
        logger.info(f"Started client.py (PID: {current_process.pid}) with parameters: {params}.") # 日志英文
        
        try:
            # 主要依赖 client.py 内部的30秒退出机制。
            # Popen.wait() 会等待子进程结束。
            # 此处的超时（例如45秒）是为了测试脚本本身的健壮性，
            # 以防 client.py 内部逻辑卡死导致无法自行退出。
            current_process.wait(timeout=120) 
            logger.info(f"client.py (PID: {current_process.pid}) for parameters {params} has finished its run.") # 日志英文
        
        except subprocess.TimeoutExpired:
            # 如果 client.py 超过了此脚本设置的外部超时（例如45秒）仍未退出
            logger.warning(f"External timeout (120s) reached for client.py (PID: {current_process.pid}). " 
                           f"This is unexpected if client.py has a 30s internal exit. Terminating...") # 日志英文
            cleanup_process_gracefully(current_process)
        
        except KeyboardInterrupt:
            # 如果在 Popen.wait() 期间（即client.py运行时）按下了 Ctrl+C
            logger.info(f"\nTest script received KeyboardInterrupt during client.py (PID: {current_process.pid}) execution.") # 日志英文
            cleanup_process_gracefully(current_process) # 尝试优雅关闭client.py
            logger.info("--- Test case interrupted by user (Ctrl+C) ---") # 日志英文
            raise # 重新抛出 KeyboardInterrupt 以终止外层 for 循环
        
        except Exception as e: # 捕获运行单个client.py实例时可能发生的其他异常
            logger.error(f"An error occurred with client.py (PID: {current_process.pid if current_process else 'N/A'}) for parameters {params}: {e}", exc_info=True) # 日志英文
            cleanup_process_gracefully(current_process) # 尝试清理出错的进程
            logger.info("Continuing to the next parameter combination if any.") # 日志英文
            continue # 跳过当前参数组合，继续下一个
        
        finally:
            # 确保在每次循环迭代后，如果进程由于某种原因仍在运行，其状态能被知晓
            if current_process and current_process.poll() is None:
                logger.warning(f"Warning: client.py (PID: {current_process.pid}) for parameters {params} might still be running unexpectedly after its block.") # 日志英文
                cleanup_process_gracefully(current_process) # 再次尝试清理
            current_process = None # 为下一次循环重置

except KeyboardInterrupt: # 捕获外层 for 循环的 KeyboardInterrupt (例如在参数迭代之间按下Ctrl+C)
    logger.info("\n--- Test script execution interrupted by user (Ctrl+C) ---") # 日志英文
    cleanup_process_gracefully(current_process) 
finally:
    logger.info("Test script finished or was interrupted.") # 日志英文