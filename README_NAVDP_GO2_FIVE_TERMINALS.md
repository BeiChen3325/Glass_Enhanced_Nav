# NavDP / X-NavDP 双机精简终端真机部署

> 当前推荐“笔记本终端 C + Go2 一或两个操作终端”。原五终端命令保留在后半部分，
> 仅用于逐进程排障。本文不再包含已经废弃的 BEV/V4 方案。

本文对应当前工作区中的三种模型模式：

| 模式 | 终端 C checkpoint | 终端 E goal | 能力 |
| --- | --- | --- | --- |
| `navdp` | `NavDP/model/navdp-cross-modal.ckpt` | `nogoal` / `pointgoal` / `imagegoal` | 官方跨模态 NavDP |
| `x-navdp` | `NavDP/model/x-navdp_posttrain.ckpt` | 仅 `pointgoal` | X-NavDP Go2 真机基线 |
| `x-navdp-glass` | `x-navdp_posttrain.ckpt` + `depth_anything_v2_vits.pth` | 仅 `pointgoal` | X-NavDP + DPT + GlassRecon Local RANSAC |

终端 C 和 E 的 `--model-mode` 必须完全一致。服务端会拒绝不一致的请求；
X-NavDP 两种模式也会拒绝 no-goal/image-goal。

## 0. 推荐方案：Go2 一终端或两终端

新增的 `scripts/run_go2_navdp_stack.py` 是轻量进程监督器，不会把 A/B/D/E 合并成
一个 ROS 节点：RealSense、输入桥、安全桥和 NavDP 客户端仍是四个独立进程，
只是由一个操作终端统一启动、显示带 `[A]`/`[B]`/`[D]`/`[E]` 前缀的输出、保存
独立日志并处理退出。这保留了 D 的独立超时与 StopMove 安全边界。

监督器会自动完成：

- 加载 RealSense 与 Unitree ROS 2 环境；
- 创建 `RUN_ID` 和日志目录，并阻止重复启动同一组件组；
- 检查终端 C `/healthz` 及 C/E 模型模式；
- 在启动 E 前实际收到并校验 RGB、aligned depth、CameraInfo 和 odom；
- dry-run 不启动 D，因此不创建 `/api/sport/request` 发布器；
- real 模式先启动 D，再启动 E，并要求显式 `--confirm-real-motion`；
- 任一子进程异常退出时联动停止；`Ctrl+C` 时按 E → D → B → A 退出，D 在退出前
  保留超时窗口并发送独立 StopMove。

运行监督器前，先停止所有手工启动的旧 A/B/D/E 进程，避免同名相机节点或重复
sport 发布器。终端 C 仍按第 5 节在笔记本单独运行。

### 0.1 Go2 单终端：最少窗口

以下命令一次管理 A+B+E，且不会让机器人运动。首次部署、代码或权重改变后先运行：

```bash
cd /home/unitree/Glass_Enhanced_X_NavDP/go2_internnav_navdp_deployment

/usr/bin/python3 scripts/run_go2_navdp_stack.py \
  --components all \
  --run-mode dry-run \
  --run-id go2_navdp_01 \
  --gpu-ip 192.168.8.251 \
  --model-mode navdp \
  --goal-mode pointgoal \
  --goal-x 3.0 \
  --goal-y 0.0 \
  --max-linear-speed 0.20 \
  --max-angular-speed 0.50
```

观察至少 20 次 HTTP 200 后按一次 `Ctrl+C`。确认监督器打印
`[STACK] shutdown complete`，再运行真实模式；它会管理 A+B+D+E：

```bash
/usr/bin/python3 scripts/run_go2_navdp_stack.py \
  --components all \
  --run-mode real \
  --confirm-real-motion \
  --run-id go2_navdp_01 \
  --gpu-ip 192.168.8.251 \
  --model-mode navdp \
  --goal-mode pointgoal \
  --goal-x 3.0 \
  --goal-y 0.0 \
  --max-linear-speed 0.20 \
  --max-angular-speed 0.50
```

单终端方案在 dry-run 与 real 切换时会重启相机。若一天内需要反复切换，下面的
两终端方案更快。

### 0.2 Go2 两终端：推荐日常测试

Go2 终端 G1 启动并常驻 A+B；一次实验期间无需重复启动相机：

```bash
cd /home/unitree/Glass_Enhanced_X_NavDP/go2_internnav_navdp_deployment

/usr/bin/python3 scripts/run_go2_navdp_stack.py \
  --components sensors \
  --run-id go2_navdp_01
```

Go2 终端 G2 先运行无运动 E dry-run：

```bash
cd /home/unitree/Glass_Enhanced_X_NavDP/go2_internnav_navdp_deployment

/usr/bin/python3 scripts/run_go2_navdp_stack.py \
  --components navigation \
  --run-mode dry-run \
  --run-id go2_navdp_01 \
  --gpu-ip 192.168.8.251 \
  --model-mode navdp \
  --goal-mode pointgoal \
  --goal-x 3.0 \
  --goal-y 0.0 \
  --max-linear-speed 0.20 \
  --max-angular-speed 0.50
```

dry-run 验收后只停止 G2；G1 保持运行。在 G2 启动 D+E 真实运动：

```bash
cd /home/unitree/Glass_Enhanced_X_NavDP/go2_internnav_navdp_deployment

/usr/bin/python3 scripts/run_go2_navdp_stack.py \
  --components navigation \
  --run-mode real \
  --confirm-real-motion \
  --run-id go2_navdp_01 \
  --gpu-ip 192.168.8.251 \
  --model-mode x-navdp \
  --goal-mode pointgoal \
  --goal-x 7.0 \
  --goal-y 0.0 \
  --max-linear-speed 1.00 \
  --max-angular-speed 1.00
```

结束实验时先在 G2 按一次 `Ctrl+C`，看到 D 已停止及 `shutdown complete` 后，再在
G1 按一次 `Ctrl+C`。物理急停和人工接管始终高于软件退出顺序。

### 0.3 切换模型和任务

终端 C 切换模型后，G2 或单终端命令的 `--model-mode` 必须同步修改：

```text
NavDP:              --model-mode navdp
X-NavDP:            --model-mode x-navdp --goal-mode pointgoal
X-NavDP+GlassRecon: --model-mode x-navdp-glass --goal-mode pointgoal
```

官方 NavDP 的 no-goal 可将目标参数替换为 `--goal-mode nogoal`；image-goal 使用
`--goal-mode imagegoal --image-goal-path /absolute/path/to/goal.jpg`。X-NavDP 两种
模式若选择非 point-goal，监督器会在启动任何进程前拒绝。

每个子进程仍写入熟悉的 `terminal_A_*`、`terminal_B_*`、`terminal_D_*`、
`terminal_E_*` 日志；此外会生成 `sensor_preflight.log`、`stack_*.json` 和
`responses_dry_run.jsonl`/`responses_real.jsonl`。因此精简终端不会降低排障信息。

### 0.4 怎样确认到达、近障或超时

一终端与两终端方案都会把 E 的导航语义通过
`/navdp/navigation_status` 传给 D。`action` 只表示 Unitree 执行器当前执行
`Move` 还是 `StopMove`；必须结合 `navigation_state` 和 `stop_cause` 判断原因。

点目标进入 `--pointgoal-stop-radius` 后，G2/单终端会先醒目输出：

```text
[E] [GOAL REACHED] state=goal_reached reason=pointgoal_reached distance_to_goal_m=0.4xx; navigation stopped safely
[D] [GOAL REACHED] navigation_state=goal_reached stop_reason=pointgoal_reached distance_to_goal_m=0.4xx; StopMove is expected
```

之后 D 的每秒状态仍会保留明确原因，而不会只显示含糊的 `StopMove`：

```text
action=StopMove navigation_state=goal_reached navigation_reason=pointgoal_reached stop_cause=pointgoal_reached
```

常见状态的含义如下：

| 输出 | 含义 |
| --- | --- |
| `navigation_state=goal_reached stop_cause=pointgoal_reached` | 已进入点目标停止半径，正常完成 |
| `navigation_state=obstacle_blocked navigation_reason=...` | RealSense 近障安全层正在阻止前进；可能同时执行恢复扫描 |
| `navigation_state=holding stop_cause=trajectory_stale` | 轨迹过期、HTTP 或轨迹校验异常，客户端安全保持 |
| `stop_cause=cmd_timeout` 且状态不是上述终止态 | D 未继续收到 E 的速度命令，应按通信/进程异常排查 |
| `stop_cause=manual_override` | 无线手柄发生人工输入，安全锁存已接管 |

`goal_reached` 只由 E 根据 odom 计算的目标距离触发，不能由“速度恰好为零”或
D 的命令超时推断。因此障碍物阻塞不会被误报为已经到达。

## 传统五终端方案：仅用于逐进程排障

以下第 1–10 节保留原始 A/B/C/D/E 手工命令。当监督器报告某个组件启动失败时，
可暂时回到这些命令单独观察该进程；日常运行不再要求打开五个终端。

## 1. 部署关系与数据链路

| 终端 | 位置 | 职责 |
| --- | --- | --- |
| A | Go2 机载计算机 `/home/unitree` | RealSense 对齐 RGB-D |
| B | Go2 机载计算机 | 相机/odom 话题适配 |
| C | RTX GPU 计算机 `/home/robot` | 当前 NavDP/X-NavDP 模型推理 |
| D | Go2 机载计算机 | `/cmd_vel_bridge` 到 Unitree sport 的独立安全桥 |
| E | Go2 机载计算机 | 目标管理、HTTP 请求、MPC、深度安全与速度发布 |

```text
RealSense RGB + aligned depth (A)       Go2 odom
                    \                    /
                     ROS 2 input bridge (B)
                              |
                      RGB-D + /odom_bridge
                              |
               Go2 NavDP client / goal / MPC (E)
                              |
             HTTP /eval_navdp over Wi-Fi to GPU (C)
                              |
                    local trajectory response
                              |
                     /cmd_vel_bridge (E)
                              |
                 sport safety bridge (D) -> Go2
```

模型自己的视觉历史保存在终端 C 的 agent 中。GlassRecon 模式由 E 额外上传
当前帧的原始 `float32` metric depth；PNG 深度仍供 X-NavDP 原有深度分支使用。

## 2. 一次性准备与强制边界

两台机器都应有同名工程目录：

```text
Go2: /home/unitree/Glass_Enhanced_X_NavDP
GPU: /home/robot/Glass_Enhanced_X_NavDP
```

GPU 必须包含当前 `NavDP` 目录和三个权重文件；Go2 至少包含部署目录。先检查：

```bash
# GPU
test -f /home/robot/Glass_Enhanced_X_NavDP/NavDP/model/navdp-cross-modal.ckpt
test -f /home/robot/Glass_Enhanced_X_NavDP/NavDP/model/x-navdp_posttrain.ckpt
test -f /home/robot/Glass_Enhanced_X_NavDP/NavDP/model/depth_anything_v2_vits.pth

sha256sum \
  /home/robot/Glass_Enhanced_X_NavDP/NavDP/model/navdp-cross-modal.ckpt \
  /home/robot/Glass_Enhanced_X_NavDP/NavDP/model/x-navdp_posttrain.ckpt \
  /home/robot/Glass_Enhanced_X_NavDP/NavDP/model/depth_anything_v2_vits.pth

python --version

# Go2
test -f /home/unitree/Glass_Enhanced_X_NavDP/go2_internnav_navdp_deployment/configs/realsense_d435i.json
test -f /home/unitree/Glass_Enhanced_X_NavDP/go2_internnav_navdp_deployment/source/go2_internnav/control/controllers.py
```

当前工作区对应的 SHA-256 依次为：

```text
3bb3ad4ab241e857bb57a4021cc6aab76d5263e81fbf80298d579053ef011947
267089a81bbbe7a913debda6603f3f1b66a79520370ce953b2d888d793b89f24
715fade13be8f229f8a70cc02066f656f2423a59effd0579197bbf57860e1378
```

`x-navdp` 和 `x-navdp-glass` 的终端 C 必须使用 Python 3.9 或更高版本；当前
X-NavDP/DPT 源码使用了 Python 3.9 的字符串 API。不要用旧 `py38` 环境启动这
两种模式。官方 `navdp` 仍按已经验证的 `navdp_rtx50` 环境运行。

运行前约定：

- 下文示例 GPU IP 为 `192.168.8.251`；若地址变化，修改监督器的 `--gpu-ip`
  或手工终端 E 的 `GPU_IP`。
- 五个终端使用人工指定的同一个 `RUN_ID`，例如 `go2_xnavdp_glass_01`。
- 同一端口只运行一个终端 C。
- 第一次测试或代码/权重变化后必须先做 E 的 `--dry-run`，此时不要启动 D。
- 不要添加 `--v4-history`；客户端会把它视为已废弃参数并退出。
- 真机运动必须有现场人员、物理急停和足够安全空间。

## 3. 终端 A：RealSense 对齐 RGB-D

```bash
cd /home/unitree/Glass_Enhanced_X_NavDP/go2_internnav_navdp_deployment
export RUN_ID=go2_xnavdp_glass_01
mkdir -p "logs/${RUN_ID}"
printf '%s\n' "${RUN_ID}" | tee logs/.latest_run_id

source /opt/ros/foxy/setup.bash
source /home/unitree/realsense_foxy_ws/install/setup.bash

ros2 launch realsense2_camera rs_launch.py \
  enable_depth:=true \
  enable_color:=true \
  align_depth:=true \
  depth_width:=640 \
  depth_height:=480 \
  depth_fps:=15.0 \
  color_width:=640 \
  color_height:=480 \
  color_fps:=15.0 \
  2>&1 | tee "logs/${RUN_ID}/terminal_A_realsense.log"
```

在临时检查终端确认以下三类 topic 持续发布，且 aligned depth 与 color 分辨率一致：

```bash
source /opt/ros/foxy/setup.bash
ros2 topic hz /camera/color/image_raw
ros2 topic hz /camera/aligned_depth_to_color/image_raw
ros2 topic echo /camera/color/camera_info --once
```

## 4. 终端 B：输入桥

```bash
cd /home/unitree/Glass_Enhanced_X_NavDP/go2_internnav_navdp_deployment
export RUN_ID=$(cat logs/.latest_run_id)

source /opt/ros/foxy/setup.bash
source /home/unitree/unitree_ros2/cyclonedds_ws/install/setup.bash

/usr/bin/python3 scripts/ros2_navdp_input_bridge.py \
  2>&1 | tee "logs/${RUN_ID}/terminal_B_input_bridge.log"
```

检查客户端使用的固定话题：

```bash
ros2 topic hz /camera/camera/color/image_raw
ros2 topic hz /camera/camera/aligned_depth_to_color/image_raw
ros2 topic hz /odom_bridge
```

## 5. 终端 C：GPU 模型服务（三选一）

三条命令只运行一条。这里统一监听 `0.0.0.0:5802`，E 通过 GPU 的真实 IP
连接。

### 5.1 官方跨模态 NavDP

```bash
source /home/robot/ENTER/etc/profile.d/conda.sh
conda activate navdp_rtx50

export WORKSPACE=/home/robot/Glass_Enhanced_X_NavDP
export RUN_ID=go2_navdp_01
cd "${WORKSPACE}/go2_internnav_navdp_deployment"
mkdir -p "logs/${RUN_ID}"

python scripts/http_internvla_server_w_navdp_isolated.py \
  --model-mode navdp \
  --host 0.0.0.0 \
  --port 5802 \
  --navdp-root "${WORKSPACE}/NavDP" \
  --checkpoint "${WORKSPACE}/NavDP/model/navdp-cross-modal.ckpt" \
  2>&1 | tee "logs/${RUN_ID}/terminal_C_navdp.log"
```

### 5.2 X-NavDP

```bash
source /home/robot/ENTER/etc/profile.d/conda.sh
conda activate env_isaaclab

export WORKSPACE=/home/robot/Glass_Enhanced_X_NavDP_Synchronization
export RUN_ID=go2_xnavdp_01

cd "${WORKSPACE}/go2_internnav_navdp_deployment"
mkdir -p "logs/${RUN_ID}"

python scripts/http_internvla_server_w_navdp_isolated.py \
  --model-mode x-navdp \
  --host 0.0.0.0 \
  --port 5802 \
  --navdp-root "${WORKSPACE}/NavDP" \
  --checkpoint "${WORKSPACE}/NavDP/model/x-navdp_posttrain.ckpt" \
  2>&1 | tee "logs/${RUN_ID}/terminal_C_x_navdp.log"
```

### 5.3 X-NavDP + GlassRecon

参数与当前仿真增强链保持一致：DPT 使用共享 X-NavDP RGB 语义、inverse-depth
表示和 `4 × 4` Local RANSAC 分区。

```bash
source /home/robot/ENTER/etc/profile.d/conda.sh
conda activate env_isaaclab

python -c 'import sys; assert sys.version_info >= (3, 9); print(sys.version)'

export WORKSPACE=/home/robot/Glass_Enhanced_X_NavDP_Synchronization
export RUN_ID=go2_xnavdp_glass_01
cd "${WORKSPACE}/go2_internnav_navdp_deployment"
mkdir -p "logs/${RUN_ID}"

python scripts/http_internvla_server_w_navdp_isolated.py \
  --model-mode x-navdp-glass \
  --host 0.0.0.0 \
  --port 5802 \
  --navdp-root "${WORKSPACE}/NavDP" \
  --checkpoint "${WORKSPACE}/NavDP/model/x-navdp_posttrain.ckpt" \
  --dav2-checkpoint "${WORKSPACE}/NavDP/model/depth_anything_v2_vits.pth" \
  --dpt-color-mode shared_xnavdp \
  --prior-representation inverse_depth \
  --alignment-min-depth 0.01 \
  --alignment-max-depth 100.0 \
  --ransac-num-rows 4 \
  --ransac-num-cols 4 \
  --ransac-points-per-partition 10 \
  --ransac-iterations 100 \
  --ransac-seed 7 \
  2>&1 | tee "logs/${RUN_ID}/terminal_C_x_navdp_glass.log"
```

正常导航默认不把 `relative_depth`、`aligned_depth` 和
`clipped_enhanced_depth` 三张 224×224 调试图从 GPU 回传到 CPU/HTTP。需要诊断
这些中间结果时，在上面的服务端命令中显式增加
`--return-glass-debug-depths`；该开关不改变送入 X-NavDP depth encoder 的补全深度。
本轮实时性优化未改变 DPT/RANSAC 算法参数、0.60 秒轨迹有效期或 500 ms 安全桥
watchdog。

服务端为延迟加载。Flask 开始监听后，先查健康状态：

```bash
curl --fail --show-error http://127.0.0.1:5802/healthz
```

`agent_loaded:false` 在第一次推理前是正常的；E 的首个 dry-run 请求会真正加载
checkpoint。终端 C 必须显示正确的 `model_mode`、checkpoint，且首次请求后没有
未获准的 missing/unexpected key、CUDA OOM 或 traceback。Glass 模式会明确报告
新增 DPT 参数为预期 missing、训练期模块为已知 unexpected；这两类经过白名单
检查的日志不是加载失败。

## 6. 终端 E：先做无运动 dry-run

下面以官方 `navdp`、初始机身正前方 `3 m` 为例。切换到 X-NavDP 时把
`MODEL_MODE` 改为 `x-navdp`；切换到 GlassRecon 增强模式时改为
`x-navdp-glass`。后两种模式都只支持 point-goal。

```bash
cd /home/unitree/Glass_Enhanced_X_NavDP/go2_internnav_navdp_deployment
export RUN_ID=$(cat logs/.latest_run_id)
export GPU_IP=192.168.8.251
export MODEL_MODE=navdp

source /opt/ros/foxy/setup.bash
source /home/unitree/unitree_ros2/cyclonedds_ws/install/setup.bash

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export NO_PROXY="${GPU_IP},127.0.0.1,localhost"
export no_proxy="${NO_PROXY}"

/usr/bin/python3 scripts/http_internvla_client_qos_isolated.py \
  --server-url "http://${GPU_IP}:5802/eval_navdp" \
  --model-mode "${MODEL_MODE}" \
  --goal-mode pointgoal \
  --goal-x 3.0 \
  --goal-y 0.0 \
  --pointgoal-stop-radius 0.5 \
  --max-linear-speed 0.80 \
  --max-angular-speed 0.80 \
  --mpc-tracking-profile smooth_continuous \
  --dry-run \
  --record-responses "logs/${RUN_ID}/responses_dry_run.jsonl" \
  2>&1 | tee "logs/${RUN_ID}/terminal_E_dry_run.log"
```

至少持续观察 20 次成功推理。验收项：

- E 收到 HTTP 200，响应 `model_mode` 与命令一致，trajectory 为有限的 `[N,3]`。
- RGB、aligned depth 与 `/odom_bridge` 持续更新；没有 stale、NaN/Inf 或时间戳跳变。
- C 第一次请求后确实加载正确 checkpoint，显存与延迟稳定。
- `x-navdp-glass` 时 E 持续打印 `[GLASSRECON] aligned=... fallback=...`；少量安全
  回退允许，若长期 `aligned=0/1`，必须先排查深度配准/有效值，不能进入实机运动。
- dry-run 明确打印“不发布 cmd_vel”，且此时终端 D 尚未启动。

从 Go2 侧也应核对服务模式：

```bash
curl --fail --show-error "http://${GPU_IP}:5802/healthz"
```

## 7. 终端 D：运动安全桥

只有 dry-run 通过后才执行。本命令会让 Go2 具备接收真实运动指令的能力。

```bash
cd /home/unitree/Glass_Enhanced_X_NavDP/go2_internnav_navdp_deployment
export RUN_ID=$(cat logs/.latest_run_id)

source /opt/ros/foxy/setup.bash
source /home/unitree/unitree_ros2/cyclonedds_ws/install/setup.bash

/usr/bin/python3 scripts/ros2_cmd_vel_sport_safe_bridge.py \
  --no-dry-run \
  --enable-sport-publish \
  --i-understand-this-can-move-the-robot \
  --max-vx 0.80 \
  --max-wz 0.80 \
  --cmd-timeout-sec 0.50 \
  2>&1 | tee "logs/${RUN_ID}/terminal_D_real_safety_bridge.log"
```

## 8. 终端 E：低速真实运动

停止 dry-run 客户端后运行下面的完整命令块。不要只复制最后的 Python 命令：
新 shell 不会继承此前的工作目录或环境变量。这里再次显式设置目录、`RUN_ID`、
`GPU_IP` 和 `MODEL_MODE`，并在任何一项无效时停止，避免产生 `logs//...` 或
`http://:5802/...`。

```bash
set -o pipefail

export DEPLOYMENT_ROOT=/home/unitree/Glass_Enhanced_X_NavDP/go2_internnav_navdp_deployment
cd "${DEPLOYMENT_ROOT}" || exit 1

test -s logs/.latest_run_id || {
  echo "[FAIL] logs/.latest_run_id 不存在或为空；请先按终端 A 步骤创建 RUN_ID"
  exit 1
}
export RUN_ID="$(tr -d '\r\n' < logs/.latest_run_id)"
export GPU_IP=192.168.8.251
export MODEL_MODE=navdp

test -n "${RUN_ID}" || { echo "[FAIL] RUN_ID 为空"; exit 1; }
test -n "${GPU_IP}" || { echo "[FAIL] GPU_IP 为空"; exit 1; }
test -n "${MODEL_MODE}" || { echo "[FAIL] MODEL_MODE 为空"; exit 1; }
test -f scripts/http_internvla_client_qos_isolated.py || {
  echo "[FAIL] 当前目录不是 go2_internnav_navdp_deployment"
  exit 1
}
mkdir -p "logs/${RUN_ID}"

source /opt/ros/foxy/setup.bash
source /home/unitree/unitree_ros2/cyclonedds_ws/install/setup.bash

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export NO_PROXY="${GPU_IP},127.0.0.1,localhost"
export no_proxy="${NO_PROXY}"

curl --fail --show-error "http://${GPU_IP}:5802/healthz" || exit 1

/usr/bin/python3 scripts/http_internvla_client_qos_isolated.py \
  --server-url "http://${GPU_IP}:5802/eval_navdp" \
  --model-mode "${MODEL_MODE}" \
  --goal-mode pointgoal \
  --goal-x 3.0 \
  --goal-y 0.0 \
  --pointgoal-stop-radius 0.5 \
  --max-linear-speed 0.80 \
  --max-angular-speed 0.80 \
  --mpc-tracking-profile smooth_continuous \
  --record-responses "logs/${RUN_ID}/responses_real.jsonl" \
  2>&1 | tee "logs/${RUN_ID}/terminal_E_real.log"
```

上面是官方 NavDP 配置；终端 C 若运行其他模式，必须同步修改这里的
`MODEL_MODE`。真实运动前再次确认 `/healthz` 返回的 `model_mode` 与它完全一致。

客户端深度急停仍独立于模型：阻塞时前进速度被限制为 `0`，默认以
`0.25 rad/s` 原地扫描；4 秒未恢复会反向。若现场只允许静止急停，增加
`--disable-depth-recovery-scan`。

手工五终端与监督器使用相同的 D/E 脚本，因此也会输出第 0.4 节所述的
`navigation_state`、`navigation_reason` 和 `stop_cause`。点目标完成时应同时看到
E 和 D 的 `[GOAL REACHED]`；若只看到 `stop_cause=cmd_timeout`，不能判定已经到达。

## 9. 官方 NavDP 的 no-goal 与 image-goal

以下命令只能配合终端 C 的 `--model-mode navdp`。

No-goal dry-run：

```bash
/usr/bin/python3 scripts/http_internvla_client_qos_isolated.py \
  --server-url "http://${GPU_IP}:5802/eval_navdp" \
  --model-mode navdp \
  --goal-mode nogoal \
  --max-linear-speed 0.20 \
  --max-angular-speed 0.50 \
  --dry-run \
  --record-responses "logs/${RUN_ID}/responses_nogoal_dry_run.jsonl"
```

Image-goal dry-run：

```bash
/usr/bin/python3 scripts/http_internvla_client_qos_isolated.py \
  --server-url "http://${GPU_IP}:5802/eval_navdp" \
  --model-mode navdp \
  --goal-mode imagegoal \
  --image-goal-path /home/unitree/Glass_Enhanced_X_NavDP/image_goal.jpg \
  --max-linear-speed 0.20 \
  --max-angular-speed 0.50 \
  --dry-run \
  --record-responses "logs/${RUN_ID}/responses_imagegoal_dry_run.jsonl"
```

No-goal 没有自动到达条件，必须由操作员终止。Image-goal 的目标图应来自与当前
相机视角/色彩语义一致的 RGB 图像。

## 10. 推荐顺序、停止与排障

启动顺序：C → A → B → E dry-run → 停 E → D → E real。

正常停止顺序：先 E，确认零速后停 D，再停 B/A/C。异常运动时立即人工接管或
物理急停，不等待 HTTP、MPC 或软件超时。

常用只读检查：

```bash
ss -ltnp | grep ':5802'
ros2 topic hz /odom_bridge
ros2 topic echo /cmd_vel_bridge --once
grep -RInE 'Traceback|CUDA out of memory|HTTP 4|HTTP 5|NaN|Inf|model mismatch' \
  "logs/${RUN_ID}"
```

典型错误：

- `server model mismatch`：C/E 的 `--model-mode` 不同，停止 E 后修正。
- X-NavDP 报 only pointgoal：当前发布 checkpoint 没有该任务 head，改用 `navdp`。
- GlassRecon 缺 `alignment_depth.npy`：E 没有使用 `x-navdp-glass` 或版本未同步。
- RANSAC 长期全部 fallback：检查 aligned depth topic、尺度、有效像素和 RGB-depth
  配准；回退本身是安全设计，不应强制用未标定的相对深度。
- C 只显示 Flask 监听：模型是延迟加载，必须以 E dry-run 首次请求验证真实加载。
