import json
import os
import numpy as np # 用于处理可能的NaN或None值

# --- 配置区域 ---
RESULTS_BASE_DIR_OLD = "./test_old"  # 原始结果文件存放目录
RESULTS_BASE_DIR_NEW = "./test_new_qoe" # 新结果文件保存目录

# 决策方法 a 的范围
DECISION_METHODS = list(range(1, 7 + 1)) # 1 到 7
# 网络环境 b 的范围
NETWORK_ENVIRONMENTS = list(range(1, 9 + 1)) # 1 到 9

# 新的 QoE 参数
NEW_QOE_MU = 4.0
ORIGINAL_QOE_MU = 2.8 # 假设原始的 mu 值是 2.8，用于反算总卡顿时长
# TAU 保持不变，仍为1.0，所以 term3 的计算方式等效于直接使用原始term3值

# 创建新的结果保存目录 (如果不存在)
if not os.path.exists(RESULTS_BASE_DIR_NEW):
    os.makedirs(RESULTS_BASE_DIR_NEW)
    print(f"Created directory: {RESULTS_BASE_DIR_NEW}")

# --- 数据处理与重新计算 ---
files_processed = 0
files_missing = 0
files_error = 0

for method_id in DECISION_METHODS:
    for env_id in NETWORK_ENVIRONMENTS:
        file_name = f"case_{method_id}_{env_id}"
        original_file_path = os.path.join(RESULTS_BASE_DIR_OLD, file_name)
        new_file_path = os.path.join(RESULTS_BASE_DIR_NEW, file_name)
        
        if os.path.exists(original_file_path):
            try:
                with open(original_file_path, 'r') as f:
                    data = json.load(f)
                
                qoe_details = data.get("qoe_formula_details", {})
                
                # 提取原始的 QoE terms (确保是 float 类型)
                term1_str = qoe_details.get("term1_total_time_weighted_quality", "0.0")
                term2_str = qoe_details.get("term2_rebuffering_penalty", "0.0")
                term3_str = qoe_details.get("term3_switch_penalty", "0.0")

                term1 = float(term1_str) if term1_str not in [None, 'N/A'] else 0.0
                original_term2_penalty = float(term2_str) if term2_str not in [None, 'N/A'] else 0.0
                original_term3_penalty = float(term3_str) if term3_str not in [None, 'N/A'] else 0.0

                # 重新计算 Term 2 (卡顿惩罚)
                # total_stall_s = original_term2_penalty / ORIGINAL_QOE_MU if ORIGINAL_QOE_MU != 0 else 0
                # new_term2_penalty = NEW_QOE_MU * total_stall_s
                # 简化为：
                if ORIGINAL_QOE_MU == 0 and original_term2_penalty != 0:
                    print(f"Warning: Original mu is 0 but term2 penalty is not zero for {original_file_path}. Cannot accurately recalculate term2.")
                    new_term2_penalty = original_term2_penalty # 保持原样或设为0
                elif ORIGINAL_QOE_MU == 0 and original_term2_penalty == 0:
                    new_term2_penalty = 0.0
                else:
                    new_term2_penalty = (original_term2_penalty / ORIGINAL_QOE_MU) * NEW_QOE_MU
                
                # Term 3 (切换惩罚) 保持不变，因为 tau 从1.0到1.0
                new_term3_penalty = original_term3_penalty 

                # 重新计算最终 QoE 得分
                new_final_qoe_score = term1 - new_term2_penalty - new_term3_penalty
                
                # 更新数据字典
                data["final_qoe_score"] = f"{new_final_qoe_score:.2f}" # 保存为字符串，保留两位小数
                
                if "qoe_formula_details" not in data:
                    data["qoe_formula_details"] = {}
                
                data["qoe_formula_details"]["param_mu"] = NEW_QOE_MU
                # data["qoe_formula_details"]["param_tau"] 保持不变 (假设原始就是1.0)
                data["qoe_formula_details"]["term1_total_time_weighted_quality"] = f"{term1:.2f}"
                data["qoe_formula_details"]["term2_rebuffering_penalty"] = f"{new_term2_penalty:.2f}"
                data["qoe_formula_details"]["term3_switch_penalty"] = f"{new_term3_penalty:.2f}"

                # 将更新后的数据写入新文件
                with open(new_file_path, 'w') as nf:
                    json.dump(data, nf, indent=4)
                
                # print(f"Processed and saved: {new_file_path} (New QoE: {new_final_qoe_score:.2f})")
                files_processed += 1

            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from file {original_file_path}: {e}")
                files_error += 1
            except Exception as e:
                print(f"Error processing file {original_file_path}: {e}")
                files_error += 1
        else:
            # print(f"File not found: {original_file_path}")
            files_missing += 1

print(f"\n--- Processing Summary ---")
print(f"Files Processed and Saved to '{RESULTS_BASE_DIR_NEW}': {files_processed}")
if files_missing > 0:
    print(f"Files Not Found in '{RESULTS_BASE_DIR_OLD}': {files_missing}")
if files_error > 0:
    print(f"Files Encountered Errors: {files_error}")