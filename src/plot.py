import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np # 用于处理可能的NaN值

# --- 配置区域 ---
RESULTS_BASE_DIR = "./test_old"  # 存放 case_{a}_{b} 文件夹的基础目录
# 决策方法 a 的范围 (假设是 1 到 7)
DECISION_METHODS = list(range(1, 7)) # [1, 2, 3, 4, 5, 6, 7]
# 网络环境 b 的范围 (假设是 1 到 9)
NETWORK_ENVIRONMENTS = list(range(1, 9 + 1)) # [1, 2, ..., 9]
# 图表保存目录
OUTPUT_DIR = "./plots" # 可以指定一个新的目录名

# 创建图表输出目录 (如果不存在)
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- 数据读取与处理 ---
all_experiment_data = []

for method_id in DECISION_METHODS:
    for env_id in NETWORK_ENVIRONMENTS:
        # 构建每个实验结果JSON文件的完整路径
        case_name = f"case_{method_id}_{env_id}"
        file_path = os.path.join(RESULTS_BASE_DIR, case_name,)
        
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    
                    # 提取 QoE 公式细节
                    qoe_details = data.get("qoe_formula_details", {})
                    
                    term1 = float(qoe_details.get("term1_total_time_weighted_quality", np.nan))
                    term2 = float(qoe_details.get("term2_rebuffering_penalty", np.nan))
                    term3 = float(qoe_details.get("term3_switch_penalty", np.nan))
                    
                    # 数据清洗和类型转换
                    # 对于可能缺失的字段，使用 np.nan 作为默认值
                    record = {
                        "method": method_id,
                        "environment": env_id,
                        # "final_qoe_score": float(data.get("final_qoe_score", np.nan)),
                        # "term1_quality": float(qoe_details.get("term1_total_time_weighted_quality", np.nan)),
                        # "term2_rebuffering": float(qoe_details.get("term2_rebuffering_penalty", np.nan)),
                        # "term3_switching": float(qoe_details.get("term3_switch_penalty", np.nan)),
                        "term1_quality": term1,
                        "term2_rebuffering": term2,
                        "term3_switching": term3,
                        "final_qoe_score": term1 - (term2 / 2.8 * 3.5) - term3 * 0.8,
                        # 你也可以在这里提取其他你感兴趣的指标，如：
                        # "avg_bitrate_kbps": float(data.get("average_played_bitrate_kbps", "").replace(" Kbps", "") if isinstance(data.get("average_played_bitrate_kbps"), str) else data.get("average_played_bitrate_kbps", np.nan)),
                        # "rebuffering_ratio_percent": float(data.get("rebuffering_ratio", "").replace("%", "") if isinstance(data.get("rebuffering_ratio"), str) else data.get("rebuffering_ratio", np.nan)),
                    }
                    all_experiment_data.append(record)
            except Exception as e:
                print(f"读取或解析文件 {file_path} 时出错: {e}")
        else:
            print(f"文件未找到: {file_path}")

if not all_experiment_data:
    print("未能加载任何实验数据，请检查路径和文件名。脚本退出。")
    exit()

# 将数据转换为Pandas DataFrame，方便处理
df_results = pd.DataFrame(all_experiment_data)

# --- 绘图函数 ---
def plot_line_chart_comparison(dataframe, y_metric_column, y_label, chart_title, output_filename):
    """
    绘制折线对比图。
    X轴是网络环境，每条线代表一种决策方法。
    """
    plt.figure(figsize=(12, 7)) # 设置图表大小

    # 获取所有决策方法并排序，确保图例顺序一致
    methods_in_data = sorted(dataframe['method'].unique())
    
    for method_id in methods_in_data:
        #筛选出当前决策方法的数据
        method_df = dataframe[dataframe['method'] == method_id].sort_values(by='environment')
        if not method_df.empty:
            plt.plot(method_df['environment'], method_df[y_metric_column], marker='o', linestyle='-', label=f"决策方法 {method_id}")
    
    plt.xlabel("网络环境 (Network Environment ID)")
    plt.ylabel(y_label)
    plt.title(chart_title)
    
    # 确保X轴刻度清晰显示所有网络环境
    plt.xticks(NETWORK_ENVIRONMENTS) 
    
    plt.legend(title="决策方法") # 添加图例
    plt.grid(True, linestyle='--', alpha=0.7) # 添加网格线
    plt.tight_layout() # 自动调整布局，防止标签重叠
    
    # 保存图表
    full_output_path = os.path.join(OUTPUT_DIR, output_filename)
    plt.savefig(full_output_path)
    plt.close() # 关闭当前图表，为下一个图表做准备
    print(f"图表已保存: {full_output_path}")

# --- 调用绘图函数生成所需的图表 ---

# 1. 最终QoE得分对比图
plot_line_chart_comparison(
    dataframe=df_results,
    y_metric_column="final_qoe_score",
    y_label="最终QoE得分 (Final QoE Score)",
    chart_title="不同决策方法在各网络环境下的最终QoE得分对比",
    output_filename="final_qoe_score_comparison.png"
)

# 2. QoE项1: 时间加权质量项对比图
plot_line_chart_comparison(
    dataframe=df_results,
    y_metric_column="term1_quality",
    y_label="QoE项1 - 时间加权质量 (Time-Weighted Quality)",
    chart_title="不同决策方法在各网络环境下的QoE项1 (质量)对比",
    output_filename="qoe_term1_quality_comparison.png"
)

# 3. QoE项2: 卡顿惩罚项对比图
# 注意：卡顿惩罚项通常是负值或0，数值越“大”（越接近0）越好。
plot_line_chart_comparison(
    dataframe=df_results,
    y_metric_column="term2_rebuffering",
    y_label="QoE项2 - 卡顿惩罚 (Rebuffering Penalty)",
    chart_title="不同决策方法在各网络环境下的QoE项2 (卡顿惩罚)对比",
    output_filename="qoe_term2_rebuffering_comparison.png"
)

# 4. QoE项3: 切换惩罚项对比图
# 注意：切换惩罚项通常是负值或0，数值越“大”（越接近0）越好。
plot_line_chart_comparison(
    dataframe=df_results,
    y_metric_column="term3_switching",
    y_label="QoE项3 - 切换惩罚 (Switching Penalty)",
    chart_title="不同决策方法在各网络环境下的QoE项3 (切换惩罚)对比",
    output_filename="qoe_term3_switching_comparison.png"
)

print(f"\n所有折线图已成功生成并保存到 '{OUTPUT_DIR}' 目录下。")

# 如果你想查看数据的概览，可以取消下面这行的注释
# print("\n数据概览:")
# print(df_results.head())

# 如果你想看按方法和环境分组的平均值，可以取消下面这行的注释
print("\n按方法和环境分组的平均QoE得分:")
print(df_results.groupby(['method', 'environment'])['final_qoe_score'].mean().unstack())

print("\n每个决策方法的平均QoE得分:")
print(df_results.groupby('method')['final_qoe_score'].mean())