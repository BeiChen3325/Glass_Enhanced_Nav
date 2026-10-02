# 当前 NavDP / X-NavDP / GlassRecon 真机链路学习地图

本文解释“当前代码实际怎样运行”，用于阅读、调试和后续优化五终端方案。模型
源码以 `NavDP/` 为准，真机适配位于 `go2_internnav_navdp_deployment/`；本次适配
没有修改 `NavDP/`。

## 1. 先记住三种模式的边界

```text
navdp
  checkpoint: NavDP/model/navdp-cross-modal.ckpt
  tasks: nogoal / pointgoal / imagegoal
  policy: NavDP/baselines/navdp

x-navdp
  checkpoint: NavDP/model/x-navdp_posttrain.ckpt
  task: pointgoal only
  policy: NavDP/baselines/x-navdp/eval/src
  embodiment: quadruped (Unitree Go2)
  real-world timing: is_real=True

x-navdp-glass
  checkpoint: x-navdp_posttrain.ckpt
  DPT: NavDP/model/depth_anything_v2_vits.pth
  task: pointgoal only
  policy: X-NavDP + DPT + Local RANSAC + safe fallback
```

当前 X-NavDP 发布的 eval 网络没有 no-goal/image-goal head 或 route。因此不能只
更换 checkpoint 后继续调用跨模态 NavDP 接口。该边界同时由
`model_profiles.py`、服务端和客户端三处校验。

环境边界同样明确：当前 X-NavDP/DPT 源码使用 `str.removeprefix`，终端 C 必须
是 Python 3.9+；旧 `py38` 只能用于已经验证的官方 NavDP 链，不能运行
`x-navdp` 或 `x-navdp-glass`。

## 2. 文件地图

### 当前模型源码（只读来源）

| 文件 | 作用 |
| --- | --- |
| `NavDP/baselines/navdp/policy_agent.py` | 官方跨模态 NavDP 的历史缓存、三种 goal 推理 |
| `NavDP/baselines/navdp/policy_network.py` | 官方跨模态策略网络 |
| `NavDP/baselines/x-navdp/eval/src/policy_agent.py` | X-NavDP 真机 point-goal wrapper、odom guidance、stuck 状态 |
| `NavDP/baselines/x-navdp/eval/src/policy_network_embodiment.py` | 带 embodiment 调制的 X-NavDP diffusion/critic |
| `NavDP/baselines/x-navdp/eval/src/policy_backbone.py` | RGB-D backbone、DPT prior 和增强深度选择 |
| `NavDP/baselines/x-navdp/eval/src/depth_alignment.py` | Local RANSAC、裁剪与逐 batch 安全回退 |
| `NavDP/baselines/x-navdp/eval/src/policy_server.py` | 当前仿真/评测 HTTP 契约的参考实现 |
| `NavDP/docs/x_navdp_go2_simulation_commands.md` | 三类仿真服务的权威启动参数 |
| `NavDP/GlassRecon/alignment_functions.py` | GlassRecon 原始 partition-sampling 思路的参考 |

### 真机部署适配

| 文件 | 作用 |
| --- | --- |
| `go2_internnav_navdp_deployment/scripts/http_internvla_server_w_navdp_isolated.py` | 三种 `model-mode` 的统一服务端 |
| `go2_internnav_navdp_deployment/scripts/http_internvla_client_qos_isolated.py` | ROS 2 传感器、目标、HTTP、MPC 和安全状态机 |
| `go2_internnav_navdp_deployment/scripts/ros2_navdp_input_bridge.py` | 相机与 Go2 odom 话题桥接 |
| `go2_internnav_navdp_deployment/scripts/ros2_cmd_vel_sport_safe_bridge.py` | 最后一级 sport 命令限幅、超时和显式使能 |
| `go2_internnav_navdp_deployment/scripts/run_go2_navdp_stack.py` | 一/两终端进程监督、健康检查、日志和有序退出 |
| `go2_internnav_navdp_deployment/scripts/ros2_navdp_preflight.py` | E 启动前确认 RGB-D、CameraInfo 与 odom 真正到达 |
| `go2_internnav_navdp_deployment/source/go2_internnav/navdp/model_profiles.py` | 模式别名、默认权重、goal 能力与 alignment-depth 契约 |
| `go2_internnav_navdp_deployment/source/go2_internnav/control/controllers.py` | 已纳入部署目录的 MPC/PID，不再依赖外部 InternNav |
| `go2_internnav_navdp_deployment/source/go2_internnav/safety/depth_stop.py` | 与模型独立的近障急停和恢复扫描 |
| `go2_internnav_navdp_deployment/source/go2_internnav/navdp/deployment.py` | 目标坐标变换、trajectory 规范化 |
| `go2_internnav_navdp_deployment/configs/workspace_paths.env` | IP、端口、设备、CameraInfo 模式 |

## 3. 一次 point-goal 推理怎样流动

```text
RealSense rgb8 + aligned 16UC1 depth
        |
        v
Go2Manager._prepare_rgb_depth_frame
  - RGB -> standards-compliant JPEG
  - depth -> metric float32
  - policy depth -> uint16 PNG (0.1 mm HTTP units)
  - Glass mode only -> raw float32 alignment_depth.npy
        |
        v
odom_callback + build_goal_request
  - 首次 local goal 锚定为固定 world goal
  - 每帧 world goal -> current body-frame [forward, left]
  - 同步挑选最接近 RGB 时间的 [world_x, world_y, yaw]
        |
        v
POST /eval_navdp
  json: model_mode, goal_mode, reset, idx, inference_odom, goal
  files: image.jpg, depth.png, optional alignment_depth.npy
        |
        v
terminal C unified server
  - 校验 C/E model mode
  - 第一次请求创建并 reset agent
  - NavDP: step_pointgoal
  - X-NavDP: step_pointgoal_with_guidance
        |
        v
trajectory [1,N,3] -> normalized [N,3]
        |
        v
terminal E
  local trajectory -> world reference -> optional bounded smoothing -> MPC
  -> depth safety -> speed/acceleration clamp -> /cmd_vel_bridge
        |
        v
terminal D safety bridge -> Unitree sport Move/StopMove
```

客户端用固定 world goal 而不是每帧重新把 `(goal_x, goal_y)` 当新目标。这样
`--goal-x 3 --goal-y 0` 表示“启动位置前方 3 m 的固定点”，机器人转弯后局部
goal 会随姿态正确变化。

## 4. 两条深度数据为什么不能合并

普通策略深度沿用 NavDP HTTP 合同：

```text
RealSense raw uint16
  -> depth_scale_m_per_unit
  -> metric metres
  -> round(depth * 10000)
  -> uint16 PNG
  -> terminal C / 10000
  -> policy process_depth: [0.1 m, 5.0 m]
```

该 PNG 最大只能表达 `6.5535 m`，并且不能保留 NaN/Inf 有效性语义。原始
X-NavDP 分支最终只使用 `[0.1, 5.0] m`，所以这个历史合同仍可使用。

GlassRecon 的 alignment depth 不能走这条链：

```text
metric float32 depth
  -> np.save(..., allow_pickle=False)
  -> alignment_depth.npy
  -> process_depth_for_alignment
  -> independent valid mask + RGB-grid resize
  -> Local RANSAC fit against DPT prior
```

这保证 RANSAC 看到未被 PNG 饱和或 NavDP `[0.1,5.0] m` 裁剪破坏的度量深度。

## 5. X-NavDP 真机特有逻辑

统一服务端创建 X-NavDP agent 时固定：

```text
embodiment=2       # quadruped / Unitree Go2
is_real=True       # 真机历史窗口，不使用仿真降频窗口
memory_size=8
predict_size=24
```

终端 E 提交 `[x, y, yaw]` world odom；服务端转换为：

```text
robot_pos  = [[x, y, 0]]
robot_quat = [[0, 0, sin(yaw/2), cos(yaw/2)]]  # SciPy xyzw
```

随后调用 `step_pointgoal_with_guidance`。这使 X-NavDP 能把上一条预测轨迹转换到
当前机器人坐标系，用作连续规划 guidance。若省略 odom，虽然模型可以生成轨迹，
但会丢失当前发布实现中的真机连续 guidance；因此部署服务将缺失 odom 视为错误。

## 6. GlassRecon 增强与安全回退

`x-navdp-glass` 执行以下顺序：

1. X-NavDP RGB encoder 处理当前 RGB 与历史 RGB。
2. Depth Anything V2 DPT head 生成相对 inverse-depth prior。
3. 从 RGB 对齐的原始 metric depth 构建有效 mask。
4. 把图像分成 `4 × 4` partition；每个 partition 每轮抽取 10 点，共 100 轮候选。
5. 在所有有效像素上按平均绝对误差选择 affine scale/shift。
6. inverse-depth 结果安全求逆，再检查有限值、正深度比例和尺度。
7. 成功时裁剪到 X-NavDP 历史 `[0.1,5.0] m` 合同；失败时逐 batch 保留原始策略深度。

因此 RANSAC 失败不会把未标定相对深度直接送进策略。终端 E 每次打印：

```text
[GLASSRECON] aligned=1/1 fallback=0
```

少量 `fallback=1` 是预期安全行为。若所有帧都回退，应优先检查 aligned depth、
尺度、RGB-depth 配准和有效像素，而不是删除安全检查。

## 7. RGB 通道合同

客户端从 ROS `rgb8` 得到真实 RGB 数值，并用 PIL 写标准 RGB JPEG；服务端用
PIL 解码后直接把 RGB 数组交给模型。当前部署链不再依赖“把 RGB 数组传给
`cv2.imencode`，再在服务端交换通道”的历史补偿技巧。

调试图颜色明显反转时，先确认没有在客户端或服务端额外调用 RGB↔BGR 转换。

## 8. reset、模式匹配与状态

- 客户端首次请求带 `reset=true`；切换目标/恢复导航时会重新触发策略 reset。
- 服务端按进程固定一个 model mode；切模型必须停止并重启终端 C。
- 客户端每个请求都携带 model mode；不一致立即 HTTP 400，不允许“似乎能跑”。
- `navdp` agent reset 接收 stop threshold；X-NavDP reset 使用其当前发布签名。
- checkpoint 延迟到第一次 reset/推理加载，因此 `/healthz` 的
  `agent_loaded=false` 不是模型加载验收。
- E 在 `/navdp/navigation_status` 上发布带 TRANSIENT_LOCAL QoS 的导航语义状态；
  D 即使因速度命令停止而进入超时 StopMove，仍能区分 `goal_reached`、
  `obstacle_blocked`、`holding`、`manual_override` 和单纯 `cmd_timeout`。
- `action=StopMove` 是执行器状态，不等于“已到达”。只有 E 计算
  `pointgoal_distance <= pointgoal_stop_radius` 后发布的
  `navigation_state=goal_reached reason=pointgoal_reached` 才是到达判据。

## 9. 控制与安全层级

模型只输出局部轨迹，不直接控制关节：

```text
model trajectory
  -> trajectory shape/finite validation
  -> 去掉最前 3 个近场 waypoint
  -> smooth_continuous 或 legacy reference
  -> unicycle MPC [v,w]
  -> client speed/acceleration clamp
  -> RGB-D depth stop / recovery scan
  -> /cmd_vel_bridge
  -> independent sport bridge clamp + command timeout
  -> Unitree sport API
```

安全层与模型模式无关。即使 GlassRecon 提供增强深度，终端 E 的原始 RealSense
近障急停仍然保留；不能用 learned/重建 depth 取代最后一级硬安全判断。

首次测试推荐 `0.20 m/s`、`0.50 rad/s`。若模型轨迹异常，客户端保持最近有效
reference 的时间有上限；HTTP/轨迹失败会进入 hold，而不是继续使用无限陈旧轨迹。

## 10. 已废弃的 BEV 痕迹

旧方案包含 BEV encoder checkpoint、历史 NPZ/V4 stateful delta、局部栅格、相机
外参投影和特定发布包。它与当前三个 checkpoint/推理接口不一致，不能作为
X-NavDP 或 GlassRecon 的替代实现。

当前行为：

- 五终端文档不再启动 BEV server；
- 默认 checkpoint 不再指向旧 outputs/releases；
- 客户端显式拒绝 `--v4-history`；
- 旧 BEV 启动脚本和过期 checkpoint README 已移除；
- `logs/`、`photos/`、`backups/` 仅保留为历史资料，不参与 import 或运行。

## 11. 验证分层

### 不需要 ROS/GPU

```bash
cd /home/unitree/Glass_Enhanced_X_NavDP/go2_internnav_navdp_deployment

python3 -m py_compile \
  scripts/http_internvla_server_w_navdp_isolated.py \
  scripts/http_internvla_client_qos_isolated.py \
  source/go2_internnav/navdp/model_profiles.py

PYTHONPATH=source /usr/bin/python3 -m unittest discover -s tests -v
```

### 需要终端 C 环境/GPU

1. 分别启动三种 server mode。
2. 用 `/healthz` 核对 mode/checkpoint。
3. 通过 E dry-run 触发真实权重加载。
4. 核对 checkpoint key、显存、首帧/热身后延迟和 trajectory 形状。
5. Glass 模式额外核对 RANSAC success/fallback 与深度有效率。

### 需要 Go2 现场

1. A/B topic 频率与 RGB-depth 对齐。
2. E dry-run 至少 20 次，无 `/cmd_vel_bridge` 发布。
3. D 单独验证超时 StopMove、限幅和急停。
4. 低速短距离直线 point-goal。
   验收时同时核对 E/D 的 `[GOAL REACHED]` 以及 D 的
   `stop_cause=pointgoal_reached`，不要只核对 `action=StopMove`。
5. 左/右偏置 point-goal。
6. NavDP no-goal/image-goal 独立测试。
7. 玻璃/反光场景中对比 `x-navdp` 与 `x-navdp-glass`，记录回退比例、轨迹、
   最小距离和是否到达。

## 12. 下一阶段优化入口

当前部署已用监督器把 Go2 操作面从 A/B/D/E 四个终端压缩为一或两个终端，同时
保留四个独立进程和 D 的安全边界。后续优化建议按风险从低到高推进：

1. 增加终端 C 的显式 warm-up/readiness，而不仅是延迟加载后的日志判断。
2. 将现有消息级 preflight 扩展为持续健康监控，而不只在启动时检查。
3. 为三种 mode 建立相同输入录制与离线 replay，先对比轨迹再上真机。
4. 给 GlassRecon 增加 RANSAC 诊断落盘与阈值告警，但保留逐帧 fallback。
5. 若进一步合并 ROS 节点，仍不能削弱 D 的独立进程、超时和 StopMove 安全边界。
