# 2026-07-16 Go2 NavDP 真机测试总结

## 测试目标

今天主要验证新训练的 NavDP 模型在 Go2 真机 pointgoal 导航中的表现：直线到点能力、停止行为、推理速度，以及是否初步具备全局/绕障能力。对照组是 NavDP 预训练模型。

## 主要结论

1. NavDP 预训练模型作为基线表现稳定：8m 直线 pointgoal 基本走直线，接近目标后 E 端正常触发 `pointgoal_reached` 并停止。
2. 新训练模型在 1m 短距离可以到点，但已经出现轻微转向；距离变长后，轨迹明显偏曲线，不能稳定保持 `y=0` 的直线目标。
3. “x 方向到了但机器人不停”不是 E 端停止逻辑 bug。E 端按世界系到固定目标点的欧氏距离判断停止；如果机器人到达 `x≈8m` 时 `y≈4m`，它距离 `(8,0)` 仍约 4m，远大于默认停止半径，所以不会停。
4. 把 D 端 `--max-wz` 和 E 端 `--max-angular-speed` 放大到 1.0 后，机器人会更忠实地执行模型输出的曲线轨迹；这能测试模型真实意图，但不会把一个偏曲线的模型变成直线模型。
5. 日志可视化显示，新模型输出的 selected local trajectory 本身就带有明显侧向偏移，所以主要问题在模型输出/critic/训练分布，而不是 D/E 控制链路。
6. 今天的全局/障碍测试只能说明新模型会生成绕行或曲线样式的轨迹，还不能证明它具备可靠的全局避障能力。当前表现更像是学习到了“曲线运动偏好/局部绕行倾向”，但没有稳定地服务于 pointgoal。

## 关键量化结果

### 新模型长距离直线测试

日志目录：

`logs/navdp_analysis/20260716_135804_new_model_8m_wz1`

注意：该目录 label 里写的是 `8m`，但 manifest 记录实际命令目标是 `goal_x=5.0, goal_y=0.0`。

- 共记录 39 次 `/eval_navdp` 响应。
- selected endpoint x：min 0.000 m，mean 0.505 m，max 0.942 m。
- selected endpoint y：min -1.195 m，mean -0.143 m，max 1.000 m。
- selected 轨迹最大横向偏移：mean 0.619 m，max 1.195 m。
- 单帧 selected path length：mean 0.660 m。
- top candidate critic value：min -4.444，mean -1.520，max 1.262。
- 推理耗时：首帧约 30.78 s，warm inference 约 3.2 s。
- 有 11/39 帧 selected trajectory 不是 critic top candidate，主要对应 `all_values.max() < stop_threshold` 后的 fallback 行为。

### 新模型全局/障碍测试 run 1

日志目录：

`logs/global_avoidance_20260716_141656/navdp_analysis/20260716_141834_new_model_global_avoidance_5m`

- 共记录 5 次 `/eval_navdp` 响应。
- selected endpoint y：min 0.296 m，mean 0.543 m，max 0.658 m。
- selected 最大横向偏移：mean 0.543 m。
- 5 帧全部偏向正 y 方向，说明模型从一开始就倾向选择侧向曲线。
- 首帧推理约 35.06 s，warm inference 约 3.6 s。

### 新模型全局/障碍测试 run 2

日志目录：

`logs/global_avoidance_20260716_141656/navdp_analysis/20260716_141943_new_model_global_avoidance_5m`

- 共记录 37 次 `/eval_navdp` 响应。
- E 端最终触发 `pointgoal_reached`，说明这次任务层面到达了目标。
- selected endpoint y：min -1.000 m，mean -0.123 m，max 0.683 m。
- selected 最大横向偏移：mean 0.430 m，max 1.000 m。
- top candidate critic value：min -3.117，mean 0.186，max 1.934。
- 推理耗时：首帧约 32.05 s，warm inference 约 3.4 s。
- 有 1/37 帧出现 fallback/非 top selected。

## 代码与工具改动

今天给 Terminal E 增加了 `--analysis-run` 分析模式。开启后会自动创建时间戳目录，并生成：

- `terminal_E_client.log`：E 端终端日志。
- `responses.jsonl`：每次 `/eval_navdp` 完整响应。
- `manifest.json`：命令行、目标点、速度限制、server URL、相机配置等。
- `trajectory_summary.csv`：每帧轨迹、critic、耗时、横向偏移统计。
- `trajectory_overview.png`：整次测试趋势图。
- `trajectory_frames/idx_*.png`：逐帧候选轨迹和 selected trajectory 可视化。

同时修复了 E 端分析模式中的两个问题：

- 修复 `manager` 作用域导致的 `AttributeError: 'NoneType' object has no attribute 'odom'`。
- 修复 Ctrl+C 或退出时日志文件已关闭、ROS node 已销毁后仍继续写入/发布导致的 shutdown race。

## 重要问题定位

NavDP policy 里存在一个对 Go2 真机不太合适的 fallback：

`/home/unitree/NavDP_workspace/NavDP/baselines/navdp/policy_agent.py`

当 `all_values.max() < stop_threshold` 时，代码会把 selected trajectory 强行改成：

- `x = 0`
- `y = sign(mean_y)`

也就是一个近似纯侧向的轨迹。Go2 当前控制链路主要发送 `vx + wz`，并不直接发送侧向速度，所以这个 fallback 容易表现为奇怪转向、绕圈或不能稳定向目标前进。今天日志中多帧 `x=0, y=±1` 的 selected 轨迹与这个逻辑一致。

另一个关键点是 critic 可能没有足够显式地奖励“朝 pointgoal 直线接近”。候选轨迹生成虽然带 goal，但 critic/rerank 如果更偏向“看起来安全或可行”的曲线，就会在直线可达时仍选择弯曲轨迹。

## 下一步建议

1. 数据集里明确加入“直线可达就走直线”的高质量轨迹，尤其是 1m、3m、5m、8m 不同距离的无障碍 pointgoal。
2. 对有障碍数据，不只要求绕开障碍，还要要求绕开后回到 pointgoal 方向，避免模型学成单纯偏航/绕行偏好。
3. 修改或绕过 Go2 真机部署中的 `x=0,y=±1` fallback；诊断时可以先降低 `stop_threshold`，或把 fallback 改成真正的 stop/hold。
4. 在推理 rerank 或训练目标里加入 endpoint-to-goal distance、cross-track error、曲率/角速度惩罚，让直线可达时直线候选获得更高优先级。
5. 后续每次测试都使用 `--analysis-run`，并同时记录 final distance、max |y|、是否触发 `pointgoal_reached`、是否碰撞/贴近障碍、warm inference latency。

## 日志位置

本机主要日志：

- `logs/navdp_analysis`
- `logs/global_avoidance_20260716_141656`

已同步到电脑：

- `/home/robot/NavDP_workspace/go2_internnav_navdp_deployment/logs/navdp_analysis`
- `/home/robot/NavDP_workspace/go2_internnav_navdp_deployment/logs/global_avoidance_20260716_141656`
- `/home/robot/NavDP_workspace/go2_internnav_navdp_deployment/logs/20260716_test_summary.md`
