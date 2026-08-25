# 轨迹标注工具使用说明

用浏览器打开 `tools/annotate_trajectory.html`，然后选择本地比赛视频即可开始标注。

推荐流程：

1. 把帧率设置成源视频的真实帧率。
2. 先标一次场地四角，顺序是左上、右上、右下、左下。
3. 重点修正远端运动员脚点：自动轨迹漂移明显的位置，每隔 3-5 帧点一次即可。
4. 只有需要训练羽毛球或近端运动员模型时，再补充羽毛球和近端脚点。
5. 勾选“导出时自动插值补齐空帧”，然后导出 CSV。

导出的 CSV 字段如下：

```text
Frame,
Ball_X,Ball_Y,Ball_Visibility,
Far_X,Far_Y,Far_Visibility,
Near_X,Near_Y,Near_Visibility,
Court_Points
```

当前项目最有价值的是远端运动员脚点标注。它可以用来训练一个小型“球场坐标空间轨迹纠错模型”，也可以用来评估 `pipeline_code/overlay_player_analytics.py` 的远端轨迹修正效果。
