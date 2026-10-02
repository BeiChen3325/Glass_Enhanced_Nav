# Go2 NavDP 真机部署适配层

本目录是当前 `NavDP` 工程的真机部署层。模型和推理实现以同级
`../NavDP` 为唯一来源；部署代码不会复制或修改该目录中的模型实现。

## 支持范围

| `--model-mode` | checkpoint | 任务 | 额外深度链路 |
| --- | --- | --- | --- |
| `navdp` | `NavDP/model/navdp-cross-modal.ckpt` | no-goal、point-goal、image-goal | 无 |
| `x-navdp` | `NavDP/model/x-navdp_posttrain.ckpt` | 仅 point-goal | 无 |
| `x-navdp-glass` | `NavDP/model/x-navdp_posttrain.ckpt` | 仅 point-goal | Depth Anything V2 DPT + GlassRecon Local RANSAC |

`x-navdp-glass` 还会加载
`NavDP/model/depth_anything_v2_vits.pth`。客户端在普通深度 PNG 之外上传
无损 `float32` metric-depth NPY，服务端使用当前
`NavDP/baselines/x-navdp/eval/src` 中的 DPT、深度对齐和安全回退实现。

X-NavDP 当前发布的真机评测网络没有 no-goal 或 image-goal head。服务端和
客户端都会在启动/请求阶段拒绝这两种组合，避免误把 checkpoint 当作跨模态
NavDP 使用。

## 关键入口

- `scripts/http_internvla_server_w_navdp_isolated.py`：统一模型服务端；三种模式
  使用同一 `/eval_navdp` 协议。
- `scripts/http_internvla_client_qos_isolated.py`：ROS 2 RGB-D/odom 客户端、目标
  管理、深度安全、轨迹跟踪和 `/cmd_vel_bridge` 发布。
- `scripts/ros2_navdp_input_bridge.py`：RealSense 与 Go2 odom 话题桥接。
- `scripts/ros2_cmd_vel_sport_safe_bridge.py`：Go2 sport 指令安全桥。
- `scripts/run_go2_navdp_stack.py`：推荐的 Go2 进程监督器；可用一个终端管理
  A+B+D+E，或用两个终端把 sensors 与 navigation 分组。
- `scripts/ros2_navdp_preflight.py`：监督器启动 E 前的 RGB-D、CameraInfo 与 odom
  消息级预检。
- `source/go2_internnav/control/controllers.py`：部署内自包含的 MPC/PID；不再依赖
  缺失的 `InternNav` 兄弟仓库。
- `source/go2_internnav/navdp/model_profiles.py`：三种模型模式、能力边界和默认权重。
- `configs/workspace_paths.env`：网络、设备和相机配置；仓库路径按本目录位置自动
  推导，因此同一 checkout 可位于 `/home/unitree` 或 `/home/robot`。

## 已废弃链路

旧 BEV checkpoint、BEV encoder、V4 stateful history 和外参栅格构建不属于当前
三种模型。新的客户端会明确拒绝 `--v4-history`；五终端命令中不再包含 BEV
参数或发布包路径。`logs/`、`photos/` 和 `backups/` 仅作为历史实验资料保留，
不参与运行。

## 操作与验证

推荐的一/两终端方案、兼容五终端排障命令、三种模式的 C/E 配对方式和 dry-run
验收项见：

- [`../README_NAVDP_GO2_FIVE_TERMINALS.md`](../README_NAVDP_GO2_FIVE_TERMINALS.md)
- [`../README_NAVDP_DEPLOYMENT_LEARNING_MAP.md`](../README_NAVDP_DEPLOYMENT_LEARNING_MAP.md)

最低限度的本地检查：

```bash
cd /home/unitree/Glass_Enhanced_X_NavDP/go2_internnav_navdp_deployment

python3 -m py_compile \
  scripts/http_internvla_server_w_navdp_isolated.py \
  scripts/http_internvla_client_qos_isolated.py

PYTHONPATH=source /usr/bin/python3 -m unittest discover -s tests -v
```

真机测试必须先运行终端 E 的 `--dry-run`，确认模式、网络、RGB-D、odom、轨迹和
GlassRecon RANSAC 统计正常后，才启动终端 D 并移除 `--dry-run`。异常时先停 E，
再停 D；物理急停始终优先。
