import json
import argparse
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import os
import sys
import glob

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def find_latest_qoe_report(pattern: str):
    """查找最新的QoE报告文件"""
    if '*' in pattern:
        # 处理通配符
        files = glob.glob(pattern)
        if not files:
            return None
        # 返回最新的文件
        return max(files, key=os.path.getctime)
    else:
        # 直接检查文件是否存在
        return pattern if os.path.exists(pattern) else None

def analyze_qoe_report(report_path: str):
    """分析QoE报告并生成可视化图表"""
    
    if not os.path.exists(report_path):
        print(f"报告文件不存在: {report_path}")
        return
    
    with open(report_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 处理通配符
    actual_file = find_latest_qoe_report(report_path)
    
    if not actual_file:
        print(f"报告文件不存在: {report_path}")
        
        # 列出可用的QoE报告文件
        logs_dir = os.path.join(os.path.dirname(report_path), "") if os.path.dirname(report_path) else "logs"
        if os.path.exists(logs_dir):
            qoe_files = glob.glob(os.path.join(logs_dir, "qoe_report_*.json"))
            if qoe_files:
                print(f"\n可用的QoE报告文件:")
                for file in sorted(qoe_files, key=os.path.getctime, reverse=True):
                    mtime = datetime.fromtimestamp(os.path.getctime(file))
                    print(f"  {file} (创建时间: {mtime.strftime('%Y-%m-%d %H:%M:%S')})")
                print(f"\n使用示例: python tools/analyze_qoe.py \"{qoe_files[0]}\"")
            else:
                print(f"\n在 {logs_dir} 目录中没有找到QoE报告文件。")
                print("请先运行客户端生成QoE报告:")
                print("  python src/client.py")
        return
    
    print(f"分析QoE报告: {actual_file}")
    
    try:
        with open(actual_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
        return
    except Exception as e:
        print(f"读取文件错误: {e}")
        return

    print("=== QoE分析报告 ===")
    print(f"会话开始时间: {datetime.fromtimestamp(data['session_start_time'])}")
    
    if data['session_end_time']:
        print(f"会话结束时间: {datetime.fromtimestamp(data['session_end_time'])}")
        session_duration = data['session_end_time'] - data['session_start_time']
        print(f"总播放时长: {session_duration:.1f}秒")
    else:
        session_duration = 0
        print("会话未正常结束")
        
    print(f"下载分片总数: {data['total_segments_downloaded']}")
    print(f"总下载量: {data['total_bytes_downloaded'] / (1024*1024):.1f}MB")
    print(f"平均下载速度: {data['average_download_speed_kbps']:.1f}kbps")
    print(f"质量切换次数: {len(data['quality_switches'])}")
    print(f"缓冲事件数: {len(data['buffering_events'])}")
    print(f"总缓冲时间: {data['total_buffering_time']:.1f}秒")
    print(f"启动延迟: {data['startup_delay_seconds']:.1f}秒")
    print(f"平均主观评分(MOS): {data['mean_opinion_score']:.2f}/5.0")
    
    print("\n=== 质量分布 ===")
    for quality, duration in data['quality_time_distribution'].items():
        percentage = (duration / session_duration) * 100 if 'session_duration' in locals() else 0
        print(f"{quality}: {duration:.1f}秒 ({percentage:.1f}%)")
    
    if data['quality_switches']:
        print(f"\n=== 质量切换历史 ===")
        for i, switch in enumerate(data['quality_switches'][:10]):  # 显示前10次切换
            timestamp = datetime.fromtimestamp(switch['timestamp'])
            print(f"{i+1}. {timestamp.strftime('%H:%M:%S')} "
                  f"{switch['from_quality']} -> {switch['to_quality']} "
                  f"(原因: {switch['reason']})")
        
        if len(data['quality_switches']) > 10:
            print(f"... 还有 {len(data['quality_switches']) - 10} 次切换")
    
    # 生成图表（如果有matplotlib）
    try:
        create_qoe_charts(data, report_path)
    except ImportError:
        print("\n注意: 未安装matplotlib，跳过图表生成")
    except Exception as e:
        print(f"\n图表生成失败: {e}")

def create_qoe_charts(data, report_path):
    """创建QoE分析图表"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('QoE Analysis Report', fontsize=16)
    
    # 1. 质量分布饼图
    if data['quality_time_distribution']:
        labels = list(data['quality_time_distribution'].keys())
        sizes = list(data['quality_time_distribution'].values())
        axes[0, 0].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title('Quality Distribution')
    
    # 2. 质量切换时间线
    if data['quality_switches']:
        switch_times = [s['timestamp'] - data['session_start_time'] for s in data['quality_switches']]
        quality_levels = {'480p-1500k': 1, '720p-4000k': 2, '1080p-8000k': 3}
        
        switch_qualities = [quality_levels.get(s['to_quality'], 1) for s in data['quality_switches']]
        
        axes[0, 1].plot(switch_times, switch_qualities, 'o-', markersize=4)
        axes[0, 1].set_xlabel('Time (seconds)')
        axes[0, 1].set_ylabel('Quality Level')
        axes[0, 1].set_title('Quality Switches Over Time')
        axes[0, 1].set_yticks([1, 2, 3])
        axes[0, 1].set_yticklabels(['480p', '720p', '1080p'])
        axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 缓冲事件
    if data['buffering_events']:
        buffering_times = [b['timestamp'] - data['session_start_time'] for b in data['buffering_events']]
        buffering_durations = [b['duration_seconds'] for b in data['buffering_events']]
        
        axes[1, 0].bar(range(len(buffering_events)), buffering_durations)
        axes[1, 0].set_xlabel('Buffering Event #')
        axes[1, 0].set_ylabel('Duration (seconds)')
        axes[1, 0].set_title('Buffering Events')
    
    # 4. QoE指标摘要
    metrics = {
        'MOS Score': data['mean_opinion_score'],
        'Startup Delay': min(data['startup_delay_seconds'], 10),  # 限制显示范围
        'Buffering Ratio': (data['total_buffering_time'] / 
                           (data['session_end_time'] - data['session_start_time']) * 100) 
                           if data['session_end_time'] else 0,
        'Switch Rate': len(data['quality_switches']) / 
                      ((data['session_end_time'] - data['session_start_time']) / 60) 
                      if data['session_end_time'] else 0
    }
    
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    
    bars = axes[1, 1].bar(metric_names, metric_values)
    axes[1, 1].set_title('QoE Metrics Summary')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # 为每个条添加数值标签
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # 保存图表
    chart_path = report_path.replace('.json', '_charts.png')
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存到: {chart_path}")
    
    # 显示图表
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="QoE报告分析工具")
    parser.add_argument("report_path", help="QoE报告JSON文件路径")
    
    args = parser.parse_args()
    analyze_qoe_report(args.report_path)

if __name__ == "__main__":
    main()