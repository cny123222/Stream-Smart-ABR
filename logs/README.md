# 日志文件目录

此目录用于存储应用程序生成的各种日志文件：

- `transmission_log.txt` - 服务器传输日志
- `qoe_report_*.json` - QoE(用户体验质量)报告
- `qoe_report_*_charts.png` - QoE分析图表

## QoE报告说明

QoE报告包含以下信息：
- 播放会话统计
- 质量切换历史
- 缓冲事件记录
- 网络性能指标
- 平均主观评分(MOS)

使用 `python tools/analyze_qoe.py logs/qoe_report_*.json` 命令可以分析报告并生成可视化图表。