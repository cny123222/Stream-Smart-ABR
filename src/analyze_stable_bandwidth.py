import json
import os
import pandas as pd
import matplotlib.pyplot as plt

# 带宽标签映射
BANDWIDTH_LABELS = {
    10: "0.5 Mbps", 11: "1 Mbps", 12: "2 Mbps", 13: "4 Mbps",
    14: "6 Mbps", 15: "8 Mbps", 16: "12 Mbps", 17: "16 Mbps", 
    18: "25 Mbps", 19: "50 Mbps", 20: "100 Mbps"
}

# ABR策略名称映射
ABR_NAMES = {
    1: "SLBW (简单最后带宽)",
    2: "SWMA (简单移动窗口平均)",
    3: "EWMA (指数加权移动平均)",
    4: "BUFFER_ONLY (仅基于缓冲区)",
    5: "BANDWIDTH_BUFFER (带宽+缓冲区)",
    6: "COMPREHENSIVE (综合规则)",
    7: "DQN (深度Q网络)"
}

def analyze_results():
    results = []
    test_dir = "./Test"  
    
    if not os.path.exists(test_dir):
        print(f"目录 {test_dir} 不存在")
        return
    
    # 收集结果 - 只处理稳定带宽测试（模式10-20）
    for filename in os.listdir(test_dir):
        if filename.startswith('case_'):
            try:
                parts = filename.split('_')
                if len(parts) >= 3:
                    abr_strategy = int(parts[1])
                    bandwidth_mode = int(parts[2])
                    
                    # 只分析稳定带宽测试结果（模式10-20）
                    # if bandwidth_mode >= 10 and bandwidth_mode <= 20:
                    if bandwidth_mode >= 10 and bandwidth_mode <= 20:
                        filepath = os.path.join(test_dir, filename)
                        
                        with open(filepath, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        results.append({
                            'ABR策略': abr_strategy,
                            'ABR策略名称': ABR_NAMES.get(abr_strategy, f"Strategy {abr_strategy}"),
                            '带宽模式': bandwidth_mode,
                            '带宽标签': BANDWIDTH_LABELS.get(bandwidth_mode, f"Mode {bandwidth_mode}"),
                            'QoE得分': float(data.get('final_qoe_score', 0)),
                            'PSNR': float(data.get('PSNR', 0)),
                            '卡顿率(%)': float(data.get('rebuffering_ratio', '0%').replace('%', '')),
                            '卡顿次数': int(data.get('rebuffering_counts', 0)),
                            '视频质量': float(data.get('quality', 0)),
                            '切换次数': int(data.get('switch_count', 0)),
                            '平均码率(Kbps)': float(data.get('average_played_bitrate_kbps', '0 Kbps').split()[0])
                        })
            except (ValueError, FileNotFoundError, json.JSONDecodeError) as e:
                print(f"处理文件 {filename} 时出错: {e}")
    
    if not results:
        print("没有找到稳定带宽测试结果（模式10-20）")
        print("请确保已运行稳定带宽测试实验")
        return
    
    df = pd.DataFrame(results)
    
    # 创建输出目录（如果不存在）
    output_dir = "./Test_bandwidth"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")

    # 保存结果
    df.to_csv('./Test_bandwidth/stable_bandwidth_analysis.csv', index=False)
    print(f"找到 {len(results)} 个稳定带宽测试结果")
    print("详细结果已保存到 ./Test_bandwidth/stable_bandwidth_analysis.csv")
    
    # 打印统计信息
    print("\n=== 各ABR策略在不同带宽下的平均表现 ===")
    
    # 按ABR策略分组统计
    strategy_summary = df.groupby('ABR策略').agg({
        'QoE得分': ['mean', 'std'],
        '卡顿率(%)': ['mean', 'std'], 
        'PSNR': ['mean', 'std'],
        '切换次数': 'mean'
    }).round(2)
    
    print("\n各ABR策略总体表现:")
    for strategy in sorted(df['ABR策略'].unique()):
        strategy_data = df[df['ABR策略'] == strategy]
        avg_qoe = strategy_data['QoE得分'].mean()
        avg_rebuf = strategy_data['卡顿率(%)'].mean()
        avg_psnr = strategy_data['PSNR'].mean()
        
        print(f"策略 {strategy} - {ABR_NAMES[strategy]}:")
        print(f"  平均QoE: {avg_qoe:.2f}, 平均卡顿率: {avg_rebuf:.2f}%, 平均PSNR: {avg_psnr:.2f}")
    
    # 找出各带宽下的最佳策略
    print("\n=== 各带宽下的最佳ABR策略 ===")
    for bw_mode in sorted(df['带宽模式'].unique()):
        bw_data = df[df['带宽模式'] == bw_mode]
        if not bw_data.empty:
            best_qoe = bw_data.loc[bw_data['QoE得分'].idxmax()]
            best_rebuf = bw_data.loc[bw_data['卡顿率(%)'].idxmin()]
            
            print(f"\n{BANDWIDTH_LABELS[bw_mode]}:")
            print(f"  最佳QoE: ABR策略{best_qoe['ABR策略']} (得分: {best_qoe['QoE得分']:.2f})")
            print(f"  最低卡顿: ABR策略{best_rebuf['ABR策略']} (卡顿率: {best_rebuf['卡顿率(%)']:.2f}%)")
    
    # 生成简单的对比图表
    generate_plots(df)

def generate_plots(df):
    """生成对比图表"""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 中文字体
    plt.rcParams['axes.unicode_minus'] = False

    # 确保输出目录存在
    output_dir = "./Test_bandwidth"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('不同稳定带宽下的视频播放性能分析', fontsize=16)
    
    # 获取排序后的带宽模式
    bandwidth_modes = sorted(df['带宽模式'].unique())
    bandwidth_labels = [BANDWIDTH_LABELS[mode] for mode in bandwidth_modes]
    
    # 1. 卡顿率对比
    ax1 = axes[0, 0]
    for abr in sorted(df['ABR策略'].unique()):
        abr_data = df[df['ABR策略'] == abr].sort_values('带宽模式')
        if not abr_data.empty:
            ax1.plot(bandwidth_labels, abr_data['卡顿率(%)'], 
                    marker='o', label=f'ABR-{abr}', linewidth=2)
    ax1.set_title('不同带宽下的卡顿率比较')
    ax1.set_xlabel('网络带宽')
    ax1.set_ylabel('卡顿率 (%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. PSNR对比
    ax2 = axes[0, 1]
    for abr in sorted(df['ABR策略'].unique()):
        abr_data = df[df['ABR策略'] == abr].sort_values('带宽模式')
        if not abr_data.empty:
            ax2.plot(bandwidth_labels, abr_data['PSNR'], 
                    marker='s', label=f'ABR-{abr}', linewidth=2)
    ax2.set_title('不同带宽下的PSNR比较')
    ax2.set_xlabel('网络带宽')
    ax2.set_ylabel('PSNR (dB)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. QoE得分对比
    ax3 = axes[1, 0]
    for abr in sorted(df['ABR策略'].unique()):
        abr_data = df[df['ABR策略'] == abr].sort_values('带宽模式')
        if not abr_data.empty:
            ax3.plot(bandwidth_labels, abr_data['QoE得分'], 
                    marker='^', label=f'ABR-{abr}', linewidth=2)
    ax3.set_title('不同带宽下的QoE得分比较')
    ax3.set_xlabel('网络带宽')
    ax3.set_ylabel('QoE得分')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. 切换次数对比
    ax4 = axes[1, 1]
    for abr in sorted(df['ABR策略'].unique()):
        abr_data = df[df['ABR策略'] == abr].sort_values('带宽模式')
        if not abr_data.empty:
            ax4.plot(bandwidth_labels, abr_data['切换次数'], 
                    marker='d', label=f'ABR-{abr}', linewidth=2)
    ax4.set_title('不同带宽下的切换次数比较')
    ax4.set_xlabel('网络带宽')
    ax4.set_ylabel('切换次数')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('./Test_bandwidth/stable_bandwidth_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n分析图表已保存到 ./Test_bandwidth/stable_bandwidth_analysis.png")

if __name__ == "__main__":
    analyze_results()