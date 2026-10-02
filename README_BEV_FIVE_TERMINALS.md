# NavDP / X-NavDP 双机五终端真机部署

> 文件名因已有工程引用暂时保留；本文不再包含 BEV 方案。旧 BEV checkpoint、
> V4 stateful history、局部栅格与外参投影链路均已废弃。

本文对应当前工作区中的三种模型模式：

| 模式 | 终端 C checkpoint | 终端 E goal | 能力 |
| --- | --- | --- | --- |
| `navdp` | `NavDP/model/navdp-cross-modal.ckpt` | `nogoal` / `pointgoal` / `imagegoal` | 官方跨模态 NavDP |
| `x-navdp` | `NavDP/model/x-navdp_posttrain.ckpt` | 仅 `pointgoal` | X-NavDP Go2 真机基线 |
| `x-navdp-glass` | `x-navdp_posttrain.ckpt` + `depth_anything_v2_vits.pth` | 仅 `pointgoal` | X-NavDP + DPT + GlassRecon Local RANSAC |

终端 C 和 E 的 `--model-mode` 必须完全一致。服务端会拒绝不一致的请求；
X-NavDP 两种模式也会拒绝 no-goal/image-goal。

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

- 下文示例 GPU IP 为 `10.10.154.201`；若地址变化，先修改终端 E 的 `GPU_IP`。
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

python -c 'import sys; assert sys.version_info >= (3, 9); print(sys.version)'

export WORKSPACE=/home/robot/Glass_Enhanced_X_NavDP
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

export WORKSPACE=/home/robot/Glass_Enhanced_X_NavDP
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

下面以 `x-navdp-glass`、初始机身正前方 `3 m` 为例。切换到 X-NavDP 时只把
`MODEL_MODE` 改为 `x-navdp`；切换到官方 NavDP point-goal 时改为 `navdp`。

```bash
cd /home/unitree/Glass_Enhanced_X_NavDP/go2_internnav_navdp_deployment
export RUN_ID=$(cat logs/.latest_run_id)
export GPU_IP=10.10.154.201
export MODEL_MODE=x-navdp-glass

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
  --max-linear-speed 0.20 \
  --max-angular-speed 0.50 \
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
  --max-vx 0.20 \
  --max-wz 0.50 \
  --cmd-timeout-sec 0.50 \
  2>&1 | tee "logs/${RUN_ID}/terminal_D_real_safety_bridge.log"
```

## 8. 终端 E：低速真实运动

停止 dry-run 客户端后复用第 6 节命令，仅移除 `--dry-run`，并改用新的响应日志：

```bash
/usr/bin/python3 scripts/http_internvla_client_qos_isolated.py \
  --server-url "http://${GPU_IP}:5802/eval_navdp" \
  --model-mode "${MODEL_MODE}" \
  --goal-mode pointgoal \
  --goal-x 3.0 \
  --goal-y 0.0 \
  --pointgoal-stop-radius 0.5 \
  --max-linear-speed 0.20 \
  --max-angular-speed 0.50 \
  --mpc-tracking-profile smooth_continuous \
  --record-responses "logs/${RUN_ID}/responses_real.jsonl" \
  2>&1 | tee "logs/${RUN_ID}/terminal_E_real.log"
```

客户端深度急停仍独立于模型：阻塞时前进速度被限制为 `0`，默认以
`0.25 rad/s` 原地扫描；4 秒未恢复会反向。若现场只允许静止急停，增加
`--disable-depth-recovery-scan`。

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
