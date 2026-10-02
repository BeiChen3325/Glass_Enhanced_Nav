五终端真机部署命令
=============================================================
终端A
```bash
cd /home/unitree/NavDP_workspace/go2_internnav_navdp_deployment
export RUN_ID=global_avoidance_$(date +%Y%m%d_%H%M%S)
mkdir -p logs/$RUN_ID
echo $RUN_ID | tee logs/.latest_run_id
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
  2>&1 | tee logs/$RUN_ID/terminal_A_realsense.log
```
=============================================================
终端B
```bash
cd /home/unitree/NavDP_workspace/go2_internnav_navdp_deployment

export RUN_ID=$(cat logs/.latest_run_id)
source /opt/ros/foxy/setup.bash
source /home/unitree/unitree_ros2/cyclonedds_ws/install/setup.bash

/usr/bin/python3 scripts/ros2_navdp_input_bridge.py \
  2>&1 | tee logs/$RUN_ID/terminal_B_input_bridge.log
```
=============================================================
终端C
```bash
cd /home/unitree/NavDP_workspace/go2_internnav_navdp_deployment
export RUN_ID=$(cat logs/.latest_run_id)
/usr/bin/python3 scripts/http_internvla_server_w_navdp_isolated.py \
  --host 127.0.0.1 \
  --port 5801 \
  --checkpoint /home/unitree/NavDP_workspace/NavDP/outputs/go2w_random_circular_v2_train/navdp_v2_latest.ckpt \
  2>&1 | tee logs/$RUN_ID/terminal_C_server_new_model.log
```
=============================================================
终端D
```bash
cd /home/unitree/NavDP_workspace/go2_internnav_navdp_deployment
export RUN_ID=$(cat logs/.latest_run_id)
source /opt/ros/foxy/setup.bash
source /home/unitree/unitree_ros2/cyclonedds_ws/install/setup.bash
/usr/bin/python3 scripts/ros2_cmd_vel_sport_safe_bridge.py \
  --no-dry-run \
  --enable-sport-publish \
  --i-understand-this-can-move-the-robot \
  --max-vx 0.50 \
  --max-wz 1.0 \
  --cmd-timeout-sec 0.50 \
  2>&1 | tee logs/$RUN_ID/terminal_D_real_safety_bridge.log
```
=============================================================
终端E

下面的命令默认使用 `smooth_continuous` 跟踪档，即使不填写
`--mpc-tracking-profile` 也会启用有界轨迹平滑和连续角速度 MPC。
该命令会运动，只能由现场操作人员执行。需要复现旧控制器时，显式增加
`--mpc-tracking-profile legacy`。

```bash
cd /home/unitree/NavDP_workspace/go2_internnav_navdp_deployment
export RUN_ID=$(cat logs/.latest_run_id)
source /opt/ros/foxy/setup.bash
source /home/unitree/unitree_ros2/cyclonedds_ws/install/setup.bash
/usr/bin/python3 scripts/http_internvla_client_qos_isolated.py \
  --server-host 127.0.0.1 \
  --server-port 5801 \
  --goal-mode pointgoal \
  --goal-x 5.0 \
  --goal-y 0.0 \
  --max-linear-speed 0.50 \
  --max-angular-speed 1.0 \
  --analysis-run \
  --analysis-root logs/$RUN_ID/navdp_analysis \
  --analysis-label new_model_global_avoidance_5m
```

### 平滑连续 MPC 对照实验（终端 E，仅现场操作人员执行，会运动）

先保持终端 A–D 的部署与安全检查不变。使用 `smooth_continuous` 时只在客户端增加 tracking profile，并确保分析标签与实际 5 m 目标一致：

```bash
cd /home/unitree/NavDP_workspace/go2_internnav_navdp_deployment
export RUN_ID=$(cat logs/.latest_run_id)
source /opt/ros/foxy/setup.bash
source /home/unitree/unitree_ros2/cyclonedds_ws/install/setup.bash
/usr/bin/python3 scripts/http_internvla_client_qos_isolated.py \
  --server-host 127.0.0.1 \
  --server-port 5801 \
  --goal-mode pointgoal \
  --goal-x 5.0 \
  --goal-y 0.0 \
  --max-linear-speed 0.50 \
  --max-angular-speed 1.0 \
  --mpc-tracking-profile smooth_continuous \
  --analysis-run \
  --analysis-root logs/$RUN_ID/navdp_analysis \
  --analysis-label MODEL_smooth_continuous_5m
```

其中 `MODEL` 分别替换为 `v3_8000` 或 `v3_16450`。终端 C 必须重启后才会加载新的 checkpoint；两组模型路径分别是：

```text
/home/unitree/NavDP_workspace/NavDP/outputs/go2w_easy4_mixed_v3_train/milestones/navdp_v3_step_008000.ckpt
/home/unitree/NavDP_workspace/NavDP/outputs/go2w_easy4_mixed_v3_train/milestones/navdp_v3_step_016450.ckpt
```

`--analysis-run` 额外生成：

- `tracking_references.jsonl`：每次推理的原始与 MPC 有效参考轨迹。
- `tracking_reference_summary.csv` 和 `tracking_reference_overview.png`：折角与最大平滑偏移。
- `tracking_reference_frames/`：每次原始/平滑轨迹叠加图。
- `control_trace.csv` 和 `control_trace.png`：odom、MPC 输出、最终速度和深度封锁状态。

=============================================================
真机诊断实验（分终端、逐项验收）
=============================================================

安全约定：凡是标记为“会运动”的命令，只能由现场操作人员执行。数据记录工具只订阅 ROS 话题，不创建 publisher，不会发送运动命令。每完成一项先分析日志，通过后再进入下一项。

## 实验 1：静止里程计漂移（只记录，不运动）

开始前让机器狗正常站立并保持完全静止，不要启动终端 D 和终端 E。

终端 1：确认本次唯一实验编号：

```bash
cd /home/unitree/NavDP_workspace/go2_internnav_navdp_deployment
export RUN_ID=$(cat logs/.latest_run_id)
echo "$RUN_ID"
```

终端 2：如 `/odom_bridge` 尚不存在，启动仅传感器输入桥。该命令不发送运动指令：

```bash
cd /home/unitree/NavDP_workspace/go2_internnav_navdp_deployment
export RUN_ID=$(cat logs/.latest_run_id)
source /opt/ros/foxy/setup.bash
source /home/unitree/unitree_ros2/cyclonedds_ws/install/setup.bash
/usr/bin/python3 scripts/ros2_navdp_input_bridge.py \
  2>&1 | tee -a logs/$RUN_ID/terminal_B_input_bridge.log
```

终端 3：记录 60 秒静止里程计。该脚本没有 publisher，不会发送运动指令：

```bash
cd /home/unitree/NavDP_workspace/go2_internnav_navdp_deployment
export RUN_ID=$(cat logs/.latest_run_id)
source /opt/ros/foxy/setup.bash
source /home/unitree/unitree_ros2/cyclonedds_ws/install/setup.bash
mkdir -p logs/$RUN_ID/experiment_01_stationary_odom
/usr/bin/python3 scripts/ros2_odom_stationary_recorder.py \
  --duration 60 \
  --output-dir logs/$RUN_ID/experiment_01_stationary_odom \
  2>&1 | tee logs/$RUN_ID/experiment_01_stationary_odom/terminal_recorder.log
```

输出文件：`odom_samples.csv` 保存每帧原始数据，`odom_summary.json` 保存频率、位置漂移、yaw 漂移和单帧跳变。完成后不要继续运动实验，先分析这两个文件。

静止实验通过标准：位置最大漂移不超过 0.05 m，yaw 最大漂移不超过 3 度，单帧位置跳变不超过 0.02 m，单帧 yaw 跳变不超过 1 度，且两个话题都持续有数据。
