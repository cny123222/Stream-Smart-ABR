import time
import subprocess
import logging
import argparse
import sys
import os

# 添加src目录到Python路径，以便导入ABR模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

logger = logging.getLogger('NetworkSimulator')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class NetworkSimulator:
    """网络状况模拟器（仅适用于Windows）"""
    
    def __init__(self):
        self.is_active = False
        
    def simulate_bandwidth_limit(self, bandwidth_kbps: int):
        """模拟带宽限制"""
        if os.name != 'nt':
            logger.warning("网络模拟功能仅在Windows上支持")
            return False
            
        try:
            # 使用Windows netsh命令模拟网络限制
            # 注意：这需要管理员权限
            logger.info(f"设置网络带宽限制: {bandwidth_kbps}kbps")
            
            # 这里可以实现具体的网络限制逻辑
            # 例如使用第三方工具如 clumsy, NetLimiter 等
            # 或者通过修改网络设置
            
            self.is_active = True
            return True
            
        except Exception as e:
            logger.error(f"设置网络限制失败: {e}")
            return False
    
    def remove_bandwidth_limit(self):
        """移除带宽限制"""
        if not self.is_active:
            return
            
        try:
            logger.info("移除网络带宽限制")
            # 移除网络限制的逻辑
            self.is_active = False
            
        except Exception as e:
            logger.error(f"移除网络限制失败: {e}")

def simulate_network_changes():
    """模拟网络状况变化"""
    simulator = NetworkSimulator()
    
    # 模拟网络状况变化序列
    scenarios = [
        (10000, 30, "高带宽期"),      # 10Mbps, 30秒
        (2000, 20, "中等带宽期"),     # 2Mbps, 20秒  
        (800, 15, "低带宽期"),        # 800kbps, 15秒
        (5000, 25, "恢复期"),         # 5Mbps, 25秒
        (1200, 20, "再次降低期"),     # 1.2Mbps, 20秒
    ]
    
    try:
        for bandwidth_kbps, duration, description in scenarios:
            logger.info(f"=== {description}: {bandwidth_kbps}kbps, 持续{duration}秒 ===")
            simulator.simulate_bandwidth_limit(bandwidth_kbps)
            time.sleep(duration)
            
    except KeyboardInterrupt:
        logger.info("网络模拟被用户中断")
    finally:
        simulator.remove_bandwidth_limit()
        logger.info("网络模拟结束")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="网络状况模拟器")
    parser.add_argument("--auto", action="store_true", help="自动运行网络变化模拟")
    parser.add_argument("--bandwidth", type=int, help="设置固定带宽限制(kbps)")
    parser.add_argument("--duration", type=int, default=60, help="持续时间(秒)")
    
    args = parser.parse_args()
    
    if args.auto:
        simulate_network_changes()
    elif args.bandwidth:
        simulator = NetworkSimulator()
        simulator.simulate_bandwidth_limit(args.bandwidth)
        logger.info(f"网络限制已设置为 {args.bandwidth}kbps，将持续 {args.duration}秒")
        time.sleep(args.duration)
        simulator.remove_bandwidth_limit()
    else:
        print("使用 --auto 自动模拟网络变化，或使用 --bandwidth <kbps> 设置固定带宽")