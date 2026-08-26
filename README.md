# 羽毛球运动分析与轨迹可视化系统

本项目是一个面向羽毛球比赛视频的离线分析系统，用于从普通摄像机视频中提取羽毛球轨迹和运动员移动轨迹，并生成带有可视化统计信息的分析视频。

系统将羽毛球检测、运动员检测与跟踪、球场坐标标定、运动统计和视频渲染组合成一条完整流水线，适合用于：

- 羽毛球比赛视频技术复盘
- 运动员移动范围、速度和回位行为分析
- 羽毛球轨迹检测与修复实验
- 计算机视觉、目标跟踪和 AI+VR 研究原型

> 当前版本是离线处理原型，不是实时分析系统。项目中的样例输入、检测结果、模型权重和输出视频均已保存在仓库中；较大的二进制文件通过 Git LFS 管理。

## 功能概览

### 羽毛球轨迹检测

使用 TrackNetV3 对连续视频帧进行羽毛球检测：

- 输入连续 8 帧，而不是单张图片
- 使用 U-Net 风格网络输出羽毛球热力图
- 对热力图进行二值化和连通域分析，得到每帧球心坐标
- 使用 InpaintNet 对被遮挡或短暂丢失的轨迹进行补全
- 使用时序集成减少逐帧检测抖动
- 输出 Frame, X, Y, Visibility 格式的 CSV 文件

### 运动员检测与多目标跟踪

使用 YOLOv8-Pose 和 ByteTrack 对画面中的运动员进行检测和跟踪：

- 检测人员并提取 COCO 17 点人体关键点
- 使用 ByteTrack 在视频帧之间保持稳定的 track_id
- 以边界框底边中心估计运动员脚点
- 通过球场 ROI、球场四边形和顶部静态目标过滤干扰
- 根据球网位置和历史轨迹将运动员分为近端和远端
- 检测缺失时使用光流或上一有效位置进行短时补偿
- 对异常跳变进行断线和轨迹质量控制

### 球场坐标映射与运动统计

通过单应性变换将视频中的梯形球场映射到真实尺寸的矩形球场：

- 标准羽毛球场尺寸：宽 6.1 m，长 13.4 m
- 将像素坐标转换为球场米制坐标
- 计算运动员速度、距离和加速度
- 统计前场、中场、后场活动区域占比
- 识别回合开始和结束
- 计算回位计时等动态指标

### 分析视频渲染

最终视频可以包含：

- 近端和远端运动员脚下轨迹
- 运动员骨架关键点
- 羽毛球轨迹和球场小地图
- 回合状态、距离、速度等统计面板
- 速度趋势、加速度和回位计时组件
- 中文标题和状态文字

### 浏览器轨迹标注工具

tools/annotate_trajectory.html 是一个无需后端的本地浏览器标注工具，可以：

- 加载本地比赛视频并逐帧查看
- 标注远端脚点、近端脚点和羽毛球
- 标注球场四角，顺序为左上、右上、右下、左下
- 复制上一帧标注
- 对空帧进行插值
- 导出 CSV 或 JSON

## 系统架构

~~~text
                         +-------------------------------+
                         | TrackNetV3                    |
                         | 8帧 -> 热力图 -> 球心坐标       |
                         +---------------+---------------+
                                         |
41.MP4 ---------------------------------+----> 41_ball.csv
  |
  +------------------------> YOLOv8-Pose + ByteTrack
                            运动员检测与跨帧跟踪
                                         |
                            +------------v------------------+
                            | Homography + Analytics         |
                            | 坐标映射、统计、视频渲染         |
                            +------------+------------------+
                                         |
                                         v
                              output_final/41_analysis.mp4
~~~

羽毛球和运动员使用不同的检测策略：羽毛球尺寸小、速度快且容易运动模糊，因此使用时序热力图模型；运动员目标较大且具有稳定的空间结构，因此使用目标检测和多目标跟踪。

## 数据流

~~~text
原始比赛视频
    |
    +-- TrackNetV3 推理
    |      +-- 背景估计
    |      +-- 8 帧序列输入
    |      +-- 热力图检测
    |      +-- 坐标提取
    |      +-- InpaintNet 轨迹补全
    |
    +--> 羽毛球坐标 CSV
    |
    +-- overlay_player_analytics.py
           +-- YOLOv8-Pose 检测运动员
           +-- ByteTrack 关联跨帧目标
           +-- 计算运动员脚点
           +-- 过滤球场外目标
           +-- 近端/远端身份绑定
           +-- 光流和历史位置补偿
           +-- 单应性变换到米制球场坐标
           +-- 计算速度、距离和回合统计
           +-- 渲染最终分析视频
~~~

## 目录结构

~~~text
.
├── 41.MP4                              # 原始样例比赛视频
├── docs/
│   └── 系统实现解析.md                   # 算法、历史修复和扩展方向说明
├── external/TrackNetV3/                 # TrackNetV3 源码、数据和模型
│   ├── predict.py                       # 羽毛球轨迹推理入口
│   ├── model.py                         # TrackNet 和 InpaintNet 定义
│   ├── train.py                         # 训练入口
│   ├── test.py                          # 测试和评估入口
│   ├── preprocess.py                    # 数据预处理
│   ├── generate_mask_data.py             # 生成修复模块训练数据
│   ├── ckpts/TrackNet_best.pt            # TrackNet 预训练权重
│   ├── ckpts/InpaintNet_best.pt          # InpaintNet 预训练权重
│   └── corrected_test_label/             # 修正后的测试标注
├── pipeline_code/
│   ├── overlay_player_analytics.py      # 跟踪、统计和视频渲染
│   ├── run_generate_video.sh             # 全精度生成脚本
│   └── run_generate_video_fast.sh        # 快速生成脚本
├── output_tracknet/
│   ├── 41_ball.csv                       # 羽毛球轨迹坐标
│   └── 41.mp4                            # TrackNetV3 处理后的视频
├── output_final/
│   ├── 41_analysis.mp4                  # 最终分析视频
│   └── 41_far_fix_smoke.mp4             # 远端轨迹修复验证视频
└── tools/
    ├── annotate_trajectory.html          # 浏览器轨迹标注工具
    └── README_annotation.md              # 标注工具说明
~~~

## 环境要求

项目当前主要在 Apple Silicon macOS 环境下开发和验证，推荐：

- macOS，Apple Silicon
- Python 3.9 或更高版本
- PyTorch
- OpenCV
- NumPy
- Pillow
- Ultralytics
- pandas
- Git LFS
- 可选：Apple MPS GPU

TrackNetV3 原始依赖位于 external/TrackNetV3/requirements.txt。运动员分析脚本还需要 ultralytics。建议在独立虚拟环境中安装兼容的 Python、PyTorch 和 Ultralytics 版本。

## 安装依赖

~~~bash
cd /Users/zhujinpeng/Desktop/yumaoqiu
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r external/TrackNetV3/requirements.txt
python -m pip install ultralytics opencv-python numpy pandas pillow
~~~

如果使用 Apple Silicon 的 MPS 加速，请按照 PyTorch 官方说明安装适用于当前系统的 PyTorch，并检查：

~~~bash
python -c "import torch; print(torch.backends.mps.is_available())"
~~~

首次克隆后，使用 Git LFS 下载视频和模型权重：

~~~bash
git lfs install
git lfs pull
~~~

如果只需要阅读代码，不需要运行样例视频，可以不下载 LFS 大文件。

## 快速开始

完整分析流程分为两个阶段。

### 阶段一：生成羽毛球轨迹

从原始视频中提取羽毛球坐标：

~~~bash
cd /Users/zhujinpeng/Desktop/yumaoqiu/external/TrackNetV3
python predict.py \
  --video_file ../../41.MP4 \
  --tracknet_file ckpts/TrackNet_best.pt \
  --inpaintnet_file ckpts/InpaintNet_best.pt \
  --save_dir ../../output_tracknet \
  --output_video
~~~

主要输出：

- output_tracknet/41_ball.csv
- output_tracknet/41.mp4

对于较大视频，可以增加 --large_video，并用 --max_sample_num 或 --video_range 限制背景估计范围。

~~~bash
python predict.py \
  --video_file ../../41.MP4 \
  --tracknet_file ckpts/TrackNet_best.pt \
  --inpaintnet_file ckpts/InpaintNet_best.pt \
  --save_dir ../../output_tracknet \
  --large_video \
  --max_sample_num 1800
~~~

### 阶段二：生成运动分析视频

项目提供了已经配置好样例路径和球场坐标的脚本：

~~~bash
cd /Users/zhujinpeng/Desktop/yumaoqiu
./pipeline_code/run_generate_video.sh
~~~

默认配置：

- 输入视频：output_tracknet/41.mp4
- 羽毛球 CSV：output_tracknet/41_ball.csv
- 输出视频：output_final/41_analysis.mp4
- 检测设备：mps
- 姿态模型：外部路径中的 yolov8s-pose.pt
- 每帧检测：detect_interval 1
- 不绘制骨架：no_draw_pose

快速模式会降低姿态检测频率和输入分辨率：

~~~bash
./pipeline_code/run_generate_video_fast.sh
~~~

快速模式适合调试和流程验证，最终精度优先时建议使用全精度脚本。

## 直接运行运动分析脚本

~~~bash
python pipeline_code/overlay_player_analytics.py \
  --video_path output_tracknet/41.mp4 \
  --output_path output_final/custom_analysis.mp4 \
  --ball_csv output_tracknet/41_ball.csv \
  --yolo_model /path/to/yolov8s-pose.pt \
  --tracker_cfg bytetrack.yaml \
  --court_points "352,232,613,232,719,525,244,525" \
  --device mps \
  --detect_interval 1 \
  --no_draw_pose
~~~

court_points 的顺序必须是：

~~~text
左上 x1,y1, 右上 x2,y2, 右下 x3,y3, 左下 x4,y4
~~~

如果没有可靠的球场坐标，可以启用交互式选点：

~~~bash
python pipeline_code/overlay_player_analytics.py \
  --video_path output_tracknet/41.mp4 \
  --output_path output_final/custom_analysis.mp4 \
  --ball_csv output_tracknet/41_ball.csv \
  --yolo_model /path/to/yolov8s-pose.pt \
  --select_court_points
~~~

## 重要参数

### TrackNetV3 predict.py

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| --video_file | 无 | 输入视频路径 |
| --tracknet_file | 无 | TrackNet 权重路径 |
| --inpaintnet_file | 空 | InpaintNet 权重路径 |
| --batch_size | 16 | 推理批大小 |
| --eval_mode | weight | 可选 nonoverlap、average、weight |
| --save_dir | pred_result | 输出目录 |
| --large_video | 关闭 | 使用大视频迭代数据集 |
| --output_video | 关闭 | 输出带轨迹的视频 |
| --traj_len | 8 | 视频中显示的轨迹长度 |

### overlay_player_analytics.py

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| --video_path | 脚本内默认路径 | 输入视频 |
| --output_path | 脚本内默认路径 | 输出分析视频 |
| --ball_csv | 自动推断 | 羽毛球坐标 CSV |
| --yolo_model | yolov8s-pose.pt | YOLO 姿态模型 |
| --tracker_cfg | bytetrack.yaml | ByteTrack 配置 |
| --device | 自动选择 | mps、cuda 或 cpu |
| --imgsz | 960 | YOLO 输入分辨率 |
| --detect_interval | 1 | 每 N 帧检测一次 |
| --conf_thres | 0.18 | 人员检测置信度阈值 |
| --court_points | 空 | 球场四角坐标 |
| --side_band_px | 45 | 网线附近近端/远端分配带宽 |
| --trail_len | 50 | 主画面轨迹长度 |
| --minimap_trail_len | 120 | 小地图轨迹长度 |
| --trail_jump_split_px | 95 | 异常跳变断开阈值 |
| --no_draw_pose | 关闭 | 关闭骨架绘制 |
| --no_draw_bottom_widgets | 关闭 | 关闭底部统计组件 |
| --profile | 关闭 | 打印阶段耗时 |

远端球员容易受到透视、遮挡和网前动作影响。远端精度优先时建议保持 detect_interval 1，并根据视频重新标定 court_points。

## 羽毛球轨迹 CSV 格式

典型输出文件为 output_tracknet/41_ball.csv：

~~~csv
Frame,X,Y,Visibility
0,0,0,0
1,512,236,1
2,518,241,1
~~~

- Frame：视频帧编号
- X、Y：羽毛球在原始视频坐标系中的像素位置
- Visibility：是否检测到羽毛球，1 表示可见，0 表示未检测到或不可见

渲染阶段会使用该 CSV 进行小地图显示、回合切分和球轨迹分析。

## 标注工具使用

直接在浏览器中打开 tools/annotate_trajectory.html，点击“打开视频”选择本地视频即可。

推荐流程：

1. 将帧率设置为源视频真实帧率。
2. 标注球场四角，顺序为左上、右上、右下、左下。
3. 重点检查远端运动员脚点，轨迹漂移处可以每隔 3 至 5 帧补标一次。
4. 需要训练或评估模型时，再补充羽毛球和近端脚点。
5. 需要连续轨迹时，勾选导出时自动插值补齐空帧。

导出的标注字段为：

~~~text
Frame,
Ball_X,Ball_Y,Ball_Visibility,
Far_X,Far_Y,Far_Visibility,
Near_X,Near_Y,Near_Visibility,
Court_Points
~~~

## 模型和外部文件

### TrackNetV3

external/TrackNetV3/ 保留了 TrackNetV3 源码、说明、测试标注和权重：

- [TrackNetV3 原始项目](https://github.com/qaz812345/TrackNetV3)
- [TrackNetV3 论文](https://dl.acm.org/doi/10.1145/3595916.3626370)

使用外部代码和权重时，请遵守 external/TrackNetV3/LICENSE 与 external/TrackNetV3/LICENSE 3。

### YOLOv8-Pose

运动员分析脚本需要 yolov8s-pose.pt。当前 Shell 脚本引用开发机器上的外部路径：

~~~text
/Users/zhujinpeng/Library/Mobile Documents/com~apple~CloudDocs/pipeline_repro_bundle/weights/yolov8s-pose.pt
~~~

换到其他机器运行时，需要准备兼容的 YOLOv8-Pose 权重，并修改 yolo_model 路径。还要确保 bytetrack.yaml 可以被 Ultralytics 找到。

## 已知限制

- 当前流程为离线视频处理，不能保证实时速度。
- 运动员位置是二维像素和二维球场坐标，不是真实三维姿态。
- 单摄像机视角存在遮挡、透视和身份切换问题。
- 球场四角坐标对视角敏感，标定错误会影响速度、距离和区域统计。
- mps 只适用于支持 Apple MPS 的环境；其他机器应使用 cpu 或 cuda。
- YOLO 权重路径依赖本机环境，不能假设其他机器具有相同路径。
- 仓库包含样例视频、模型和生成结果，克隆和拉取 LFS 对象需要较多磁盘空间和网络流量。
- 仓库保留了部分实验文件、重复命名文件和 Python 缓存，这是当前实验资料的一部分。

## 常见问题

### 找不到 YOLO 模型

~~~bash
ls -lh /path/to/yolov8s-pose.pt
~~~

确认文件存在后，使用绝对路径传入 yolo_model。

### 提示没有提供球场四角

传入 8 个数字：

~~~bash
--court_points "x1,y1,x2,y2,x3,y3,x4,y4"
~~~

或者启用 select_court_points 进行交互式选点。

### 远端运动员轨迹颜色或身份不正确

优先检查球场点顺序、球网位置、side_band_px、detect_interval，以及运动员脚点是否落在球场四边形内。

### Git LFS 文件没有下载

~~~bash
git lfs install
git lfs pull
~~~

## 研究与扩展方向

当前项目已经具备向 AI+VR 扩展的部分基础：

- 单应性变换提供真实米制球场坐标
- 运动员脚点可以作为二维战术位置输入
- COCO 17 点人体关键点可以作为动作捕捉起点
- 小地图提供二维俯视战术复盘原型

进一步扩展可以从以下方向开始：

1. 使用单目深度估计或人体先验补充三维位置和深度。
2. 使用多摄像机融合降低遮挡和透视歧义。
3. 将二维关键点映射到虚拟角色骨骼，构建动作回放原型。
4. 使用 CoreML、TensorRT 或模型量化降低推理延迟。
5. 通过人工标注数据评估远端脚点修复和轨迹质量。

## 许可证与引用

本项目包含外部 TrackNetV3 代码和相关文件，请同时遵守其许可证文件。使用 TrackNetV3 时建议引用其原始论文和项目。

如果基于本项目进行研究或二次开发，请在论文、报告或项目说明中注明：

- 使用 TrackNetV3 进行羽毛球检测和轨迹修复
- 使用 YOLOv8-Pose 与 ByteTrack 进行运动员检测和跟踪
- 使用单应性变换进行球场坐标映射

