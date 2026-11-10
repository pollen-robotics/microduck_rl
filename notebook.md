# Microduck 本地可视化速查

> 无需 GPU、无需训练就能先看一眼。以下命令都在仓库根目录执行：
> `cd C:/Users/qwe12/microduck_rl`
>
> 本文按 **PowerShell** 编写，命令**全部单行**，可直接复制。
> PowerShell 的续行符是反引号 `` ` ``，不是 `\`；能一行就别折行。
> 占位符不要写成 `<run>` —— PowerShell 会把 `<` 当重定向符报错，用时请换成真实值（或写成不带尖括号的 `run`）。

**目录**

- [1. play —— 主界面（看策略 / 看机器人）](#1-play--主界面看策略--看机器人)
- [2. infer_policy.py —— 部署前的键盘遥控](#2-infer_policypy--部署前的键盘遥控)
  - [键盘操作速查](#键盘操作速查)
- [3. 训练曲线 —— TensorBoard / wandb](#3-训练曲线--tensorboard--wandb)
- [4. 两个辅助可视化脚本](#4-两个辅助可视化脚本)
- [5. 训练 episodic 任务（ground-pick / kick）](#5-训练-episodic-任务ground-pick--kick)
  - [5.5 训练日志字段速查（ground_pick 示例）](#55-训练日志字段速查ground_pick-示例)
- [小结 / 注意事项](#小结--注意事项)

---

## 1. play —— 主界面（看策略 / 看机器人）

最简单，不需要任何 checkpoint，先看环境长什么样：

```powershell
uv run play Mjlab-Velocity-Flat-MicroDuck --agent zero --num-envs 1
```

> ⚠️ `--agent zero` 是零输出 dummy 策略，鸭子会**直接瘫倒**（实测 1.5 s 内 tilt 到 80°）。这是正常的，不代表环境坏了——它只是给你看环境，不做平衡。想看鸭子站立 / 行走要加载训练好的 checkpoint（见下）。

加载训练好的策略（把路径换成你实际有的 run 目录）：

```powershell
# 本地 checkpoint
uv run play Mjlab-Velocity-Flat-MicroDuck --checkpoint-file logs/rsl_rl/velocity/2026-09-18_15-35-50_velocity/model_500.pt

# 从 wandb 拉（run_id 记得加引号）
uv run play Mjlab-Velocity-Flat-MicroDuck --wandb-run-path "entity/project/run_id"
```

> 想无头录视频：加 `--video True --video-width 1280 --video-height 720 --video-length 400`，录到该 run 的 `videos/play/` 下。
>
> ⚠️ **不指定分辨率时默认只有 320×240**（mjlab 的 `ViewerConfig` 默认值），所以视频会很糊。`--video-length` 是**帧数**，录制按 50 Hz，`400` ≈ 8 秒。

### 可视化后端（`--viewer`，实测都能启动）

| 后端 | 命令 | 说明 |
|---|---|---|
| `viser`（浏览器） | `--viewer viser` | 启动本地 web 服务，浏览器打开 <http://localhost:8080>。有暂停 / 单步 / 倍速 / 命令调节 / 检查点热切换等面板 |
| `native`（MuJoCo 窗口） | `--viewer native` | 弹出原生 MuJoCo 窗口，鼠标拖拽旋转 / 缩放 |

> ⚠️ **Windows 注意**：`--viewer auto` 会走 `viser`（`auto` 靠 `DISPLAY` 环境变量判断，Windows 没有）。想要原生窗口必须显式写 `--viewer native`。

---

## 2. infer_policy.py —— 部署前的键盘遥控（原生 MuJoCo 窗口）

```powershell
uv run scripts/infer_policy.py --walking logs/rsl_rl/velocity/2026-09-18_15-35-50_velocity/2026-09-18_15-35-50_velocity.onnx --new-cmd-obs
```

非要折行时，PowerShell 用反引号续行（**不是** `\`），且反引号后不能有空格：

```powershell
uv run scripts/infer_policy.py `
  --walking logs/rsl_rl/velocity/2026-09-18_15-35-50_velocity/2026-09-18_15-35-50_velocity.onnx `
  --new-cmd-obs
```

> ⚠️ **`--new-cmd-obs` 不能省**：当前策略用统一的 61D obs（13D 命令块）。
> 不加这个 flag 时脚本按旧的 51D 布局喂输入，会报
> `Got invalid dimensions ... Got: 51 Expected: 61`。
> 只有很老的 51D 策略才反过来——那种要**去掉**这个 flag。
>
> `--sitstand` / `--kick-left` / `--kick-right` / `--roulade` 会强制要求 `--new-cmd-obs`。

### VLA（自由语句）策略

双输入策略（`obs` + `text_emb`）会自动识别；actor 观测从 61D 变成 **48D（无命令块）**，
指令通过文本输入。

```powershell
# 用真实 encoder.onnx（与真机一致：onnxruntime + tokenizer）
uv run scripts/infer_policy.py --walking logs/rsl_rl/vla/<run>/<run>.onnx --new-cmd-obs --text "往前走" --encoder-onnx encoder.onnx

# 或者用 SentenceTransformer（装了 vla extra 即可）
uv run scripts/infer_policy.py --walking <vla.onnx> --new-cmd-obs --text "站好别动"
```

- `--text-file instruction.txt`：轮询文件，改文件即换指令（非阻塞）
- `N`：循环 `instructions.json` 里的句子
- 指令只在变化时编码一次，50 Hz 循环喂缓存向量
- 无 encoder 参数时会自动加载默认 `paraphrase-multilingual-MiniLM-L12-v2`

### 键盘操作速查

> 按键在**终端**里输入，不是 MuJoCo 窗口（窗口已不抓键）。脚本启动时也会打印同一份菜单。

三个输入模式，按键含义随模式变化。判断优先级：**头部模式 > 身体姿态模式 > 速度模式**（同时开着时头部模式会“吃掉”方向键和 `A/E/Space`）。

| 模式 | 开关 | 作用 |
|---|---|---|
| 速度模式 | 默认 | 控制 twist：前进 / 侧移 / 转向 |
| 身体姿态模式 | `B` | 控制 `body_pose` 命令（蹲高 / 俯仰 / 横滚 / 偏航） |
| 头部模式 | `H` | 控制 `head_pose` 命令（脖子 / 头俯仰 / 偏航 / 横滚） |

**全局键（任何模式都生效）**

| 键 | 作用 |
|---|---|
| `Q` | 退出 |
| `T` | 暂停 / 恢复策略推理（暂停时不查 ONNX，电机保持最后一次目标位置） |
| `P` | 随机推力：躯干世界系 xy 速度 = **1.0 m/s 随机方向**（对应训练的 `push_by_setting_velocity`；直接覆盖，不累积） |
| `G` | 触发 ground pick（需 `--ground-pick`） |
| `K` / `L` | 左 / 右脚踢球（需 `--kick-left` / `--kick-right`，会把球传到踢球脚前） |
| `R` | 前滚翻 roulade（需 `--roulade`） |
| `Y` | 有 `--sit`/`--sitstand` 时切坐 / 站；否则切 `--slope` 下坡模式 |
| `H` | 切头部模式 |
| `B` | 切身体姿态模式 |
| `N` | VLA：循环下一条指令（来自 `instructions.json`） |
| `Space` | 重置：按当前模式重置速度 / 身体姿态 / 头部偏移 |

**速度模式（默认）**

| 键 | 作用 | 范围 |
|---|---|---|
| `↑` | lin_vel_x 直接跳到正最大（前进 / 加速） | 普通 +0.3、roller +0.6 m/s |
| `↓` | lin_vel_x 跳到负最大（后退 / 刹车） | 普通 −0.3、roller −0.5 m/s |
| `←` / `→` | 普通机器人：侧移 lin_vel_y（左 +0.2 / 右 −0.2） | ±0.2 m/s |
| `←` / `→` | roller：航向误差 ang_vel_z **增量** ±0.1 | ±1.0 rad |
| `A` / `E` | 转向 ang_vel_z（左 + / 右 −） | 普通 ±1.5、roller ±1.0 rad/s |
| `Space` | 全归零（coast）；若同时装了站立策略会切回站立 | — |

> 方向键是**直接跳到极值**，不是逐步加减。若同时加载 walking + standing 两个策略，会按命令模长自动切换（阈值 `--switch-threshold`，默认 0.05）：命令≈0 → standing，非零 → walking。

**身体姿态模式（按 `B` 开 / 关）**

| 键 | 作用 | 范围 |
|---|---|---|
| `↑` / `↓` | Δz ±10 mm/次 | ±30 mm |
| `←` / `→` | Δpitch（左 + / 右 −）10°/次 | ±30° |
| `A` / `E` | Δroll（左 + / 右 −）10°/次 | ±30° |
| `Z` / `S` | Δyaw ±10°/次（**仅 `--new-cmd-obs` 61D 模式**） | ±30° |
| `Space` | 身体姿态命令清零 | — |

> 旧的 51D 布局只暴露 (z, pitch, roll)；61D 是 6D `[x,y,z,roll,pitch,yaw]`，但键盘只接了 z/pitch/roll/yaw，x/y 固定为 0。

**头部模式（按 `H` 开 / 关）**

| 键 | 作用 |
|---|---|
| `Z` / `S` | neck_pitch ±step |
| `↑` / `↓` | head_pitch ±step |
| `←` / `→` | head_yaw（左 + / 右 −）±step |
| `A` / `E` | head_roll（左 + / 右 −）±step |
| `Space` | 头部偏移清零 |

> 步长 / 上限：61D 模式 `head_max=1.4 rad`、`head_step=0.1`；旧 51D 模式 `head_max=2.5`、`head_step=0.83`。

**行为键的约束**（`g/k/l/r` 是整段换策略，彼此互斥）

- 一次只能跑一个；坐着、下坡、或正在 ground pick 时不能触发。
- 触发后命令强制归零（这些策略是零命令训练的）；踢球 / 滚翻默认分别 3.0 s / 2.0 s 后自动交还 walking/standing（`--kick-duration`、`--roulade-duration` 可调）。
- 初始命令可用 `--lin-vel-x / --lin-vel-y / --ang-vel-z` 指定，不必手动按键。

---

## 3. 训练曲线 —— TensorBoard / wandb

训练本身没有 GUI，但曲线可视化有：

```powershell
# 用 tensorboard 记录
uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 4096 --agent.logger tensorboard

# 另开一个终端启动面板
uv run tensorboard --logdir logs/rsl_rl
# 浏览器打开 http://localhost:6006
```

默认 logger 是 wandb（会传到 wandb 网页），首次需 `uv run wandb login`。

---

## 4. 两个辅助可视化脚本

```powershell
# 带滑块的姿态编辑器（拖关节，关窗时打印姿态）
uv run python scripts/crouch_pose_editor.py

# 查看坡道地形（验证几何）
uv run python scripts/view_slope_terrain.py
```

---

## 5. 训练 episodic 任务（ground-pick / kick）

> 训练入口都是 `uv run train <TASK_ID>`。**任何长训练前必须先跑冒烟测试**（64 envs、5 iters，抓 95% 的配置错误）。
> 步数换算：steps = iter × 24（`num_steps_per_env = 24`）。

### 5.1 ground-pick（弯腰碰地再站起）

任务 ID：`Mjlab-GroundPick-Flat-MicroDuck`（粗糙地形变体：`Mjlab-GroundPick-Rough-MicroDuck`）

```powershell
# ① 冒烟测试
uv run train Mjlab-GroundPick-Flat-MicroDuck --env.scene.num-envs 64 --agent.max_iterations 5

# ② 正式训练
uv run train Mjlab-GroundPick-Flat-MicroDuck --env.scene.num-envs 4096
```

- wandb project `mjlab_microduck`，experiment/run 名 = `ground_pick`
- `max_iterations = 20_000`；这类 episodic 动作实际收敛约 1000–2000 iters，不必跑满

### 5.2 kick（踢球，左/右脚分开训）

任务 ID：`Mjlab-BallKick-Flat-MicroDuck`（**唯一一个 ID，踢球脚由模块顶部 `KICK_FOOT` 常量决定，默认 `"right"`**）

**左/右脚是两次独立训练，不能镜像复用**。训左脚前改一行：

```python
# src/mjlab_microduck/tasks/microduck_ball_kick_env_cfg.py  第 41 行
KICK_FOOT = "left"   # 默认是 "right"；训右脚时改回来
```

```powershell
# ① 冒烟测试
uv run train Mjlab-BallKick-Flat-MicroDuck --env.scene.num-envs 64 --agent.max_iterations 5

# ② 正式训练
uv run train Mjlab-BallKick-Flat-MicroDuck --env.scene.num-envs 4096
```

- experiment/run 名自动跟随 flag：`ball_kick_left` / `ball_kick_right`
- `max_iterations = 10_000`

### 5.3 训练完的收尾（两个任务通用）

```powershell
# 导出 ONNX（bake obs normalizer，必经此路径，别手搓）
uv run scripts/export.py <TASK_ID> --wandb-run-path <entity/project/run_id>

# 或发布到 HF Hub
uv run publish --task <TASK_ID> --wandb-run-path <...> --checkpoint N --repo <user>/microduck-<name> --kind episodic --duration-s 4.0
```

### 5.4 训练时盯日志（AGENTS.md 强制项）

- mean reward 上升，且主任务项真的在涨（不是靠 regularizer 撑起来）
- 每个 `Episode_Reward/<penalty>` 必须 ≤ 0（>0 说明 weight 符号反了，policy 在刷违规项）
- 看视频 + 确认哪个 geom/轴触地，不要只看 sim 指标

### 5.5 训练日志字段速查（ground_pick 示例）

> 下面逐条解释一份 `ground_pick` 训练日志里每个字段的含义。示例这份是**训练非常早期**的
> checkpoint（Total steps 6,094,848 ÷ (24 步/iter × ~3900 env) ≈ **第 ~62–65 个迭代**），
> 因此 `fell_over` 高达 86.8%、`mean episode length` 只有 40 步是预期现象，不代表跑坏了。
>
> `Episode_Reward/*` 是**已加权、且除以 20 s 归一化**后的值（≈ 该项的“每步时间平均贡献”；
> ×20 才是该项的每集总分）。判符号时注意：mjlab 的 penalty 是自否定函数（返回 ≤0），
> 配**正权重**；因此这些项的值必须 ≤0 才算正常。

#### 顶部：运行信息 + 吞吐

| 字段 | 含义 |
|---|---|
| `Run name` | 实验名（此处 `ground_pick`） |
| `Total steps` | 累计环境步数 = iter × 24 × num_envs |
| `Steps per second` | 采样吞吐（步/秒） |
| `Collection time` | 本迭代收集 rollout 的墙钟时间 |
| `Learning time` | 本迭代梯度更新的墙钟时间 |

#### PPO 学习信号

| 字段 | 含义 | 怎么看 |
|---|---|---|
| `Mean value loss` | Critic 对回报的预测误差（clipped MSE），越小 = 价值估计越准 | 正常范围即可 |
| `Mean surrogate loss` | PPO 裁剪代理目标的**负值**（`max(-A·ratio, -A·clip(ratio))`），实际被最大化的目标 = 它的相反数，**越负 = 策略改进越大** | 裁剪后量级小是正常的 |
| `Mean entropy loss` | 动作分布熵（nats，历史命名叫 loss，实际是熵本身，训练里是**加分项**），越大 = 探索越强 | 对应 14 维高斯 std≈0.37 时约 5.8 |
| `Mean action std` | 当前策略探索噪声标准差（从 `init_std=1.0` 衰减） | 下降 = 从探索转向利用 |

#### 全局指标

| 字段 | 含义 |
|---|---|
| `Mean reward` | **每集总回报**的平均值（episode 内所有 reward 项之和） |
| `Mean episode length` | 每集平均步数。上限 = 1000 步（20 s / 0.02 s）；明显偏低说明 episode 提前终止 |

#### Episode_Reward/* 主任务目标（正权重，越大越好）

| 字段 | 权重 | 含义 |
|---|---|---|
| `mouth_ground_proximity` | 3.0 | 嘴巴尖端贴近地面（高斯 std=0.10、target=0），只在下蹲段开门 |
| `mouth_perpendicular_to_ground` | 2.0 | 嘴巴 x 轴垂直朝下（1=正朝下），只在下蹲段 |
| `ground_pick_return_pose_legs` | 6.0 | 起身段腿部 10 个关节回到站立位姿（高斯 std=0.3） |
| `ground_pick_return_pose_neck` | 6.0 | 起身段颈部 4 个关节回到站立位姿（std=0.15，更紧） |
| `return_upright` | 4.0 | 起身段躯干保持竖直，防起身失衡 |
| `feet_grounded` | 3.0 | 双脚保持接触地面（只查接触，不查脚是否翻） |
| `upright` | 0.2 | 躯干竖直（常开，权重故意调低以免和“前倾下蹲”打架） |
| `mouth_payload_force` | 0.0 | **不是奖励**：每步 hook，起身段把随机 10–40 g 物体重量作为外力加到嘴巴上；恒为 0 是正常的 |

#### Episode_Reward/* 惩罚项（负权重，必须 ≤ 0）

| 字段 | 权重 | 含义 |
|---|---|---|
| `action_rate_l2` | 课程中（当前 -0.8 → 最终 -2.0） | 动作变化率 L2 平滑惩罚；慢速精细任务故意调重 |
| `neck_action_rate_l2` | -1.0 | 颈部动作平滑惩罚（头部用得多，单独加） |
| `body_ang_vel` | -0.05 | 躯干角速度惩罚 |
| `feet_flat` | -2.0 | 脚掌放平惩罚（检测脚是否在脚踝轴上翻转） |
| `neck_vel_descent` | -0.1 | 下蹲段颈部关节速度惩罚（防“头猛扎”），起身段不罚 |
| `dof_pos_limits` | 负权重 | 关节位置极限接近惩罚（只在最后 ~7.5% 行程生效） |
| `joint_torques_l2` | -5e-3 | 关节力矩惩罚 |
| `self_collisions` | -1.0 | 自碰撞惩罚（防深蹲时头/颈蹭腿） |
| `head_impact_penalty` | -2.0 | 嘴巴/头部**触地**惩罚（阈值 1.0）——“贴近但不许碰地”的对立面约束 |
| `angular_momentum` | -0.02 | 全身角动量惩罚 |
| `soft_landing` | -1e-5 | 足部落地冲击惩罚 |

#### 其余分组

| 字段 | 含义 |
|---|---|
| `Episode_Metrics/mean_action_acc` | 动作加速度（二阶差分）均值，越小 = 动作越平滑 |
| `Curriculum/action_rate_weight` | 课程当前给 `action_rate_l2` 的权重（0 → -0.8 / iter250 → -1.5 / iter500 → -2.0），可反推训练到第几阶段 |
| `Curriculum/com_range` / `head_com_range` | 躯干/头部质心随机化当前范围（初始 3mm，到 iter500 才升档） |
| `Episode_Termination/time_out` | 自然超时终止占比 |
| `Episode_Termination/fell_over` | 因躯干倾斜 >70°（`bad_orientation`）终止的 episode 占比 |
| `Episode_Termination/out_of_terrain_bounds` | 越出地形边界终止占比（应恒为 0） |
| `Episode_Termination/nan_state` | 物理 NaN 终止占比（应恒为 0，否则 obs 已被污染） |
| `Metrics/angular_momentum_mean` | 全身角动量幅值均值，小 = 姿态平稳 |
| `Metrics/landing_force_mean` | 足部每次落地冲击力均值（N） |

#### 读日志的结论

1. 训练早期（~62 iter）：86.8% 的 episode 在 ~0.8s 内摔倒、主任务项只有小幅正值，是“动作还没成型”的正常形态。
2. **符号全部正确**：所有惩罚项 ≤ 0、所有主任务/稳定项 > 0，没有“双负变奖励”的翻车。
3. 继续观察 `fell_over` 是否下降、`mean episode length` 是否拉长；几百 iter 后仍 80%+ 摔倒才需要动手查（先跑 headless eval 看真实 rollout，再动 reward）。

---

## 小结 / 注意事项

- 最快速看到界面：直接跑第 1 条的 `--agent zero`，无需训练、无需 wandb（鸭子会瘫倒，正常）。
- `play` 用的并发参数是 `--num-envs`（不是训练时的 `--env.scene.num-envs`）。
- 启动日志里那行 `[microduck] pre-Volta GPU detected ...` 是正常的（工作区加的）。
- `viser` / `native` 都会占用终端，按 Ctrl+C 退出；想录视频可加 `play ... --video True --video-width 1280 --video-height 720 --video-length 400`（不指定分辨率默认只有 320×240）。
- **PowerShell 续行用反引号 `` ` ``**，不是 `\`；多行命令失败基本都是这个原因。能一行就一行。
