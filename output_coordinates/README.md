# 轨迹分析文件索引

## 最终结果

- `final/41_coordinates_locked.csv`：最终逐帧坐标。包含羽毛球展开球场坐标、速度、方向变化、异常跳变标记，以及近端/远端运动员坐标、track ID 和坐标来源（`detect`、`flow`、`missing`）。
- `final/41_coordinates_locked_ball_analysis.json`：羽毛球轨迹汇总指标。
- `final/41_locked_tracking_report.md`：身份锁定、光流补偿和羽毛球轨迹分析说明。

## 标定与参考结果

- `reference/court_points_41.json`：当前使用的球场四点和坐标系。
- `reference/player_kinematics_table.csv`：早期标定版本的运动员运动学表，仅作参考。
- `reference/player_kinematics_corrected2.md`：修正标定后的运动学表，仅作参考。
- `reference/41_summary_table.csv`：早期汇总表，仅作参考。
- `reference/trajectory_analysis_report.md`：提示词对应的指标说明和限制。
- `reference/remote_tracking_diagnostic.md`：远端轨迹诊断。

## 中间文件

`intermediate/` 保存旧版坐标、测试运行和早期分析结果，便于追溯；正式分析优先使用 `final/` 文件。

## 重新生成

在项目根目录运行：

```bash
./.venv/bin/python pipeline_code/export_frame_coordinates.py \
  --video 41.MP4 \
  --ball-csv output_tracknet/41_ball.csv \
  --output output_coordinates/final/41_coordinates_locked.csv \
  --device mps
```

导出脚本：`pipeline_code/export_frame_coordinates.py`。YOLO 权重：项目根目录的 `yolov8s-pose.pt`。

注意：球员和羽毛球的米制指标是单目二维投影结果；`flow` 是光流补偿值，`missing` 表示未能可靠补偿，不应当作为静止处理。
