import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from math import pi

# --- Configuration ---
RESULTS_BASE_DIR = "./test_new_qoe"
DECISION_METHODS = list(range(2, 6 + 1)) 
NETWORK_ENVIRONMENTS = list(range(1, 9 + 1))
OUTPUT_DIR = "./plots"

METHOD_LABELS = {
    1: "SLBW",
    2: "SWMA",
    3: "EWMA",
    4: "Buffer-Only",
    5: "Bandwidth-Buffer",
    6: "STARS (Ours)", 
    7: "DQN-RL"
}

# --- Global Font Size Adjustments ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.titlesize'] = 18 # Title for subplots
plt.rcParams['figure.titlesize'] = 20 # Super title for figure
plt.rcParams['axes.labelsize'] = 16 # X and Y labels
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['legend.title_fontsize'] = 14


if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- Data Loading and Processing (Same as your last provided version) ---
all_experiment_data = []
for method_id in DECISION_METHODS:
    for env_id in NETWORK_ENVIRONMENTS:
        file_name = f"case_{method_id}_{env_id}"
        file_path = os.path.join(RESULTS_BASE_DIR, file_name)
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                qoe_details = data.get("qoe_formula_details", {})
                term1 = float(qoe_details.get("term1_total_time_weighted_quality", np.nan))
                term2_penalty_val = float(qoe_details.get("term2_rebuffering_penalty", np.nan))
                term3_penalty_val = float(qoe_details.get("term3_switch_penalty", np.nan))
                
                # final_qoe_score_calculated = term1 - (term2_penalty_val / 2.8 * 4) - (term3_penalty_val / 1.0 * 1)
                final_qoe_score = float(data.get("final_qoe_score", np.nan))

                avg_bitrate_str = data.get("average_played_bitrate_kbps", "0 Kbps")
                avg_bitrate_kbps = float(avg_bitrate_str.replace(" Kbps", "")) if isinstance(avg_bitrate_str, str) else float(avg_bitrate_str)
                rebuff_ratio_str = data.get("rebuffering_ratio", "0.00%")
                rebuffering_ratio_percent = float(rebuff_ratio_str.replace("%","")) if isinstance(rebuff_ratio_str, str) else float(rebuff_ratio_str)
                switch_count_str = data.get("switch_count", "0")
                switch_count = int(switch_count_str)
                psnr_str = data.get("PSNR", "0.00")
                psnr_val = float(psnr_str) if isinstance(psnr_str, str) else float(psnr_str)

                record = {
                    "method": method_id,
                    "method_label": METHOD_LABELS.get(method_id, f"Method {method_id}"),
                    "environment": env_id,
                    "final_qoe_score": final_qoe_score,
                    "term1_quality": term1,
                    "term2_rebuffering_penalty_original": term2_penalty_val,
                    "term3_switching_penalty_original": term3_penalty_val,
                    "term2_rebuf_penalty_plot": term2_penalty_val, 
                    "term3_switch_penalty_plot": term3_penalty_val, 
                    "avg_bitrate_kbps": avg_bitrate_kbps,
                    "rebuffering_ratio_percent": rebuffering_ratio_percent,
                    "switch_count": switch_count,
                    "psnr": psnr_val,
                }
                all_experiment_data.append(record)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from file {file_path}: {e}")
            except Exception as e:
                print(f"Error reading or parsing file {file_path}: {e}")
        else:
            print(f"File not found: {file_path}")

if not all_experiment_data:
    print("No experiment data loaded. Please check paths and filenames. Exiting.")
    exit()

df_results = pd.DataFrame(all_experiment_data)

# --- Plotting Functions ---
# plot_line_chart_comparison, plot_grouped_bar_chart, plot_heatmap are assumed to be the same as your previous version
# I will include them here for completeness, with font size considerations if needed via rcParams.

def plot_line_chart_comparison(dataframe, y_metric_column, y_label, chart_title, output_filename):
    plt.figure(figsize=(14, 8)) # Slightly larger figure
    methods_in_data = sorted(dataframe['method'].unique())
    for method_id in methods_in_data:
        method_df = dataframe[dataframe['method'] == method_id].sort_values(by='environment')
        if not method_df.empty:
            plt.plot(method_df['environment'], method_df[y_metric_column], marker='o', linestyle='-', linewidth=2.5, markersize=8, label=METHOD_LABELS.get(method_id, f"Method {method_id}")) # Increased line/marker size
    
    plt.xlabel("Network Environment ID")
    plt.ylabel(y_label)
    plt.title(chart_title)
    plt.xticks(NETWORK_ENVIRONMENTS)
    plt.legend(title="ABR Method")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    full_output_path = os.path.join(OUTPUT_DIR, output_filename)
    plt.savefig(full_output_path, dpi=300)
    plt.close()
    print(f"Chart saved: {full_output_path}")

def plot_grouped_bar_chart(dataframe, y_metric_column, y_label, chart_title, output_filename):
    # This function will be called by the new plot_mean_metrics_bar_charts_separately
    # Or can be used for environment-wise comparison as before
    plt.figure(figsize=(15, 9)) # Larger figure
    pivot_df = dataframe.pivot_table(index='environment', columns='method', values=y_metric_column)
    pivot_df = pivot_df.rename(columns=METHOD_LABELS)
    
    ax = pivot_df.plot(kind='bar', figsize=(15, 9), width=0.8, colormap="viridis")
    
    plt.xlabel("Network Environment ID")
    plt.ylabel(y_label)
    plt.title(chart_title)
    plt.xticks(rotation=30, ha="right") # Adjusted rotation for potentially longer labels
    plt.legend(title="ABR Method", fontsize=plt.rcParams['legend.fontsize']) # Use rcParams
    ax.tick_params(axis='both', which='major', labelsize=plt.rcParams['xtick.labelsize']) # Use rcParams
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.tight_layout()
    full_output_path = os.path.join(OUTPUT_DIR, output_filename)
    plt.savefig(full_output_path, dpi=300)
    plt.close('all') 
    print(f"Chart saved: {full_output_path}")

def plot_heatmap(dataframe, value_column, title, output_filename, cmap="viridis_r", fmt=".2f"):
    try:
        pivot_table = dataframe.pivot_table(index='method', columns='environment', values=value_column)
        method_labels_for_hm = [METHOD_LABELS.get(idx, f"Method {idx}") for idx in pivot_table.index]
        pivot_table.index = method_labels_for_hm
        
        plt.figure(figsize=(14, max(6, len(method_labels_for_hm) * 0.8))) # Dynamic height
        sns.heatmap(pivot_table, annot=True, fmt=fmt, cmap=cmap, linewidths=.5, 
                    cbar_kws={'label': value_column}, annot_kws={"size": plt.rcParams['xtick.labelsize'] - 2}) # Adjust annotation size
        plt.ylabel("ABR Method")
        plt.xlabel("Network Environment ID")
        plt.title(title)
        plt.xticks(ticks=np.arange(len(NETWORK_ENVIRONMENTS)) + 0.5, labels=NETWORK_ENVIRONMENTS) # Ensure all env labels show
        plt.yticks(rotation=0) # Keep Y-axis labels horizontal
        plt.tight_layout()
        full_output_path = os.path.join(OUTPUT_DIR, output_filename)
        plt.savefig(full_output_path, dpi=300)
        plt.close()
        print(f"Chart saved: {full_output_path}")
    except Exception as e:
        print(f"Could not generate heatmap for {value_column}: {e}")


def plot_radar_chart_qoe_terms_log_scale(dataframe, output_filename):
    # Metrics for radar chart based on QoE terms.
    radar_metrics_map = {
        'Quality Utility (Term1)': 'term1_quality', # Higher is better
        'Rebuffering Penalty (Term2)': 'term2_rebuf_penalty_plot', # Lower is better (will be transformed)
        'Switching Penalty (Term3)': 'term3_switch_penalty_plot'  # Lower is better (will be transformed)
    }
    
    initial_cols_to_select = ['method_label'] + list(radar_metrics_map.values())
    df_radar_raw = dataframe[initial_cols_to_select].copy()
    df_radar_agg = df_radar_raw.groupby('method_label').mean().reset_index()

    # Transform metrics for radar display (higher is better for all)
    # Apply log(1+x) to raw values before normalization for Term1
    # For penalties (Term2, Term3), they are already sums of bad events.
    # To make them "higher is better" and potentially log-scaled:
    # 1. Invert: Use 1 / (1 + penalty) for penalties. Then log(1+x) this inverted value.
    # 2. Max - X: Use log(1 + (Max_Penalty - Current_Penalty))
    
    # Let's use log(1+x) for term1, and for inverted penalties.
    df_radar_agg['Radar_Term1_Quality'] = np.log1p(df_radar_agg['term1_quality'].clip(lower=0)) # log(1+x), ensure non-negative

    # For penalties, make them "positive good" scores first, then log transform
    # Max value for penalty implies worst performance. Min value (closer to 0) implies best.
    # So, "goodness" = Max_penalty - current_penalty. Add 1 before log if it can be 0.
    max_term2 = df_radar_agg['term2_rebuf_penalty_plot'].max()
    df_radar_agg['Radar_Term2_NoRebuffer_Raw'] = (max_term2 - df_radar_agg['term2_rebuf_penalty_plot']).clip(lower=0)
    df_radar_agg['Radar_Term2_NoRebuffer'] = np.log1p(df_radar_agg['Radar_Term2_NoRebuffer_Raw'])

    max_term3 = df_radar_agg['term3_switch_penalty_plot'].max()
    df_radar_agg['Radar_Term3_NoSwitchPen_Raw'] = (max_term3 - df_radar_agg['term3_switch_penalty_plot']).clip(lower=0)
    df_radar_agg['Radar_Term3_NoSwitchPen'] = np.log1p(df_radar_agg['Radar_Term3_NoSwitchPen_Raw'])
    
    data_cols_for_radar_log_transformed = ['Radar_Term1_Quality', 'Radar_Term2_NoRebuffer', 'Radar_Term3_NoSwitchPen']
    axis_labels_for_radar = ['Log(1+Quality Utility)', 'Log(1+Freedom from Rebuffering)', 'Log(1+Freedom from Switches)']
    
    num_vars = len(axis_labels_for_radar)

    # Normalize these log-transformed values to 0-1 scale for plotting
    for col_name in data_cols_for_radar_log_transformed:
        min_val = df_radar_agg[col_name].min()
        max_val = df_radar_agg[col_name].max()
        if (max_val - min_val) == 0:
             df_radar_agg[col_name] = 0.5 
        else:
            df_radar_agg[col_name] = (df_radar_agg[col_name] - min_val) / (max_val - min_val)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1] 

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True)) # Larger figure

    method_order_map = {label: i for i, label in enumerate([METHOD_LABELS.get(m_id, str(m_id)) for m_id in DECISION_METHODS])}
    df_radar_agg['sort_key'] = df_radar_agg['method_label'].map(method_order_map)
    df_radar_agg.sort_values('sort_key', inplace=True)

    for i, row in df_radar_agg.iterrows():
        data_values = row[data_cols_for_radar_log_transformed].values.flatten().tolist()
        data_values += data_values[:1] 
        ax.plot(angles, data_values, linewidth=2.5, linestyle='solid', label=row['method_label'], marker='o', markersize=7) # Slightly larger markers/lines
        ax.fill(angles, data_values, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axis_labels_for_radar, fontsize=plt.rcParams['xtick.labelsize'] + 2) # Larger axis labels
    ax.set_yticks(np.linspace(0, 1, 6)) 
    # ax.set_yticklabels([f"{val:.1f}" for val in np.linspace(0, 1, 6)]) # Optional: format y-tick labels
    ax.set_title('ABR QoE Term Comparison (Log-Transformed & Normalized)', size=plt.rcParams['figure.titlesize'], y=1.10)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.22), ncol=min(3, len(df_radar_agg['method_label'].unique())), title_fontsize=plt.rcParams['legend.title_fontsize'], fontsize=plt.rcParams['legend.fontsize'])
    
    plt.tight_layout() # Apply after all elements are added
    full_output_path = os.path.join(OUTPUT_DIR, output_filename)
    plt.savefig(full_output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Chart saved: {full_output_path}")


def plot_individual_mean_metric_bar_chart(dataframe, metric_col_name_df, display_name, output_dir, lower_is_better=False):
    """Plots a single bar chart for one mean metric."""
    
    df_agg = dataframe.groupby('method_label')[metric_col_name_df].mean().reset_index()
    
    method_order_map = {label: i for i, label in enumerate([METHOD_LABELS.get(m_id, str(m_id)) for m_id in DECISION_METHODS])}
    df_agg['sort_key'] = df_agg['method_label'].map(method_order_map)
    df_agg.sort_values('sort_key', inplace=True)
    # df_agg = df_agg.set_index('method_label') # Set index after sorting for proper bar labels
    
    plt.figure(figsize=(10, 7)) # Good size for individual plot
    
    # Sort for consistent bar ordering if desired, e.g., by value
    # df_agg_sorted = df_agg.sort_values(by=metric_col_name_df, ascending=lower_is_better)
    # Using fixed method order instead for easier comparison across different metric charts
    
    bars = plt.bar(df_agg['method_label'], df_agg[metric_col_name_df], color=sns.color_palette("viridis", len(df_agg)))
    
    plt.title(f"Mean {display_name} by ABR Method", fontsize=plt.rcParams['axes.titlesize'] + 2)
    plt.ylabel(f"Mean {display_name}", fontsize=plt.rcParams['axes.labelsize'])
    plt.xlabel("ABR Method", fontsize=plt.rcParams['axes.labelsize'])
    plt.xticks(rotation=30, ha="right", fontsize=plt.rcParams['xtick.labelsize'])
    plt.yticks(fontsize=plt.rcParams['ytick.labelsize'])
    plt.grid(True, linestyle='--', alpha=0.6, axis='y')
    
    # Add data labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + (plt.ylim()[1] * 0.01), f'{yval:.2f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    filename = f"bar_mean_{metric_col_name_df.replace('%', 'pct').replace(' ', '_').lower()}.png"
    full_output_path = os.path.join(output_dir, filename)
    plt.savefig(full_output_path, dpi=300)
    plt.close()
    print(f"Chart saved: {full_output_path}")

# --- Generate Charts ---
# (Line charts for QoE and its terms)
plot_line_chart_comparison(df_results, 'final_qoe_score', 'Final QoE Score', 'Final QoE Score by ABR Method vs. Network Environment', 'line_final_qoe_score.png')
plot_line_chart_comparison(df_results, 'term1_quality', 'QoE Term 1: Time-Weighted Quality', 'QoE Term 1 (Quality) by ABR Method vs. Network Environment', 'line_term1_quality.png')
plot_line_chart_comparison(df_results, 'term2_rebuf_penalty_plot', 'QoE Term 2: Rebuffering Penalty', 'QoE Term 2 (Rebuffering Penalty) by ABR Method vs. Network Environment', 'line_term2_rebuf_penalty_plot.png')
plot_line_chart_comparison(df_results, 'term3_switch_penalty_plot', 'QoE Term 3: Switching Penalty', 'QoE Term 3 (Switching Penalty) by ABR Method vs. Network Environment', 'line_term3_switch_penalty_plot.png')

# Grouped Bar Charts for environment-wise comparison
plot_grouped_bar_chart(df_results, 'final_qoe_score', 'Final QoE Score', 'Final QoE Score by ABR Method (Grouped Bar)', 'bar_env_final_qoe_score.png')
plot_grouped_bar_chart(df_results, 'avg_bitrate_kbps', 'Average Bitrate (Kbps)', 'Average Bitrate by ABR Method (Grouped Bar)', 'bar_env_avg_bitrate.png')
plot_grouped_bar_chart(df_results, 'rebuffering_ratio_percent', 'Rebuffering Ratio (%)', 'Rebuffering Ratio by ABR Method (Grouped Bar)', 'bar_env_rebuffering_ratio.png')
plot_grouped_bar_chart(df_results, 'switch_count', 'Switch Count', 'Switch Count by ABR Method (Grouped Bar)', 'bar_env_switch_count.png')
plot_grouped_bar_chart(df_results, 'psnr', 'PSNR (dB)', 'Average PSNR by ABR Method (Grouped Bar)', 'bar_env_psnr.png')


# Heatmaps
plot_heatmap(df_results, 'final_qoe_score', 'Heatmap of Final QoE Score', 'heatmap_final_qoe_score.png', cmap="viridis")
plot_heatmap(df_results, 'avg_bitrate_kbps', 'Heatmap of Average Bitrate (Kbps)', 'heatmap_avg_bitrate.png', cmap="viridis")
plot_heatmap(df_results, 'psnr', 'Heatmap of PSNR (dB)', 'heatmap_psnr.png', cmap="viridis")
plot_heatmap(df_results, 'rebuffering_ratio_percent', 'Heatmap of Rebuffering Ratio (%)', 'heatmap_rebuffering_ratio.png', cmap="viridis_r") 
plot_heatmap(df_results, 'switch_count', 'Heatmap of Switch Count', 'heatmap_switch_count.png', cmap="viridis_r") 


# Radar Chart for QoE Terms (Log-Transformed & Normalized, higher is better)
plot_radar_chart_qoe_terms_log_scale(df_results, 'radar_qoe_terms_log_comparison.png')

# --- New: Plot mean metrics separately ---
mean_metrics_to_plot_separately = {
    'final_qoe_score': 'Final QoE Score',
    'term1_quality': 'Quality Utility (Term1)',
    'term2_rebuf_penalty_plot': 'Rebuffering Penalty (Term2)',
    'term3_switch_penalty_plot': 'Switching Penalty (Term3)',
    'avg_bitrate_kbps': 'Average Bitrate (Kbps)',
    'rebuffering_ratio_percent': 'Rebuffering Ratio (%)',
    'switch_count': 'Switch Count',
    'psnr': 'PSNR (dB)'
}

for metric_key, display_label in mean_metrics_to_plot_separately.items():
    plot_individual_mean_metric_bar_chart(df_results, metric_key, display_label, OUTPUT_DIR)


print(f"\nAll charts have been generated and saved to '{OUTPUT_DIR}' directory.")

print("\n--- Aggregated Mean Scores by ABR Method (method_label) ---")
aggregated_metrics = df_results.groupby('method_label').agg(
    Mean_QoE_Score=('final_qoe_score', 'mean'),
    Mean_Term1_Quality=('term1_quality', 'mean'),
    Mean_Term2_Rebuffering=('term2_rebuf_penalty_plot', 'mean'),
    Mean_Term3_Switching=('term3_switch_penalty_plot', 'mean'),
    Mean_Avg_Bitrate_kbps=('avg_bitrate_kbps', 'mean'),
    Mean_Rebuff_Ratio_pct=('rebuffering_ratio_percent', 'mean'),
    Mean_Switch_Count=('switch_count', 'mean'),
    Mean_PSNR=('psnr', 'mean')
).round(2)

ordered_labels_for_print = [METHOD_LABELS.get(m_id, str(m_id)) for m_id in DECISION_METHODS]
print(aggregated_metrics.reindex(ordered_labels_for_print))