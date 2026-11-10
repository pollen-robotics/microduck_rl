# VLA / 语言条件策略 实施计划

> 目标：给 Microduck 策略加**自由语句**文本输入（**暂不做视觉**），保持现有
> PPO + MuJoCo（mjlab）+ 真机 ONNX 热插拔架构。
>
> 本文件是施工蓝图，不是已完成的承诺。Phase 1 已完成，其余按顺序推进。

## 已锁定的设计

| 项 | 决定 |
|---|---|
| 文本 | 自由语句，开放词表 |
| 视觉 | 暂不做 |
| 编码器 | **冻结**的预训练句编码器，`paraphrase-multilingual-MiniLM-L12-v2`（384-d） |
| 编码器位置 | **独立 `encoder.onnx`**（不烘进策略图） |
| 文本接入 | 独立 obs group `"text"`，**不碰 61D** |
| 归一化 | **只归一化 61D proprio**；text 向量绝不进 normalizer |
| actor 输入 | 只有 text embedding，**看不到**数值命令 |
| critic / 奖励 | 继续用 13D 数值命令（privileged） |

## ONNX 契约

```
自由语句 ──tokenize──> encoder.onnx ──> text_emb [1, 384] ──┐
                                                         ▼
      obs [1, 61] (normalizer 已烘进图) ──────────> policy.onnx ──> actions [1, 14]
```

| 文件 | 输入 | 输出 |
|---|---|---|
| `encoder.onnx` | `input_ids [1, L]` int64, `attention_mask [1, L]` int64 | `text_emb [1, 384]` float32 |
| `policy.onnx` | `obs [1, 61]` float32, `text_emb [1, 384]` float32 | `actions [1, 14]` float32 |

manifests 必须钉死：`encoder.name / version / sha256 / output_dim`、
tokenizer 版本、`max_length`。运行时启动校验 `encoder.output_dim == policy.text_emb_dim`。

## 为什么文本要"接地"到数值命令

现有奖励（`track_lin_vel_xy`、`head_pose_tracking`）比对的是数值命令。
自由语句本身无法算奖励，所以：

```
句子 ──意图表──> 13D 数值命令 ──> 奖励 + critic
   └──> text_emb ──> actor
```

actor 必须从句子推断意图；数值命令是 critic 的特权信息，不能给 actor，
否则文本输入就没有存在意义。

## 阶段

### Phase 1 — 文本编码器 + 指令池 ✅

- [x] `src/mjlab_microduck/vla/__init__.py`
- [x] `src/mjlab_microduck/vla/text_encoder.py` — `HFTextEncoder`（冻结、可选依赖）
      + `HashTextEncoder`（无依赖、确定性，测试/CI 用）
- [x] `src/mjlab_microduck/vla/instructions.json` — 12 意图 × 10–12 条中英同义句
- [x] `src/mjlab_microduck/vla/instructions.py` — `InstructionPool`、均衡采样、embedding 预计算
- [x] `tests/test_vla_text_encoder.py` — 14 项 CPU 测试
- [x] `pyproject.toml` — 可选 extra：`uv sync --extra vla`

### Phase 2 — 环境 cfg + obs group ✅

- [x] `src/mjlab_microduck/tasks/mdp.py`
  - `InstructionCommandCfg` / `InstructionCommand`：每 episode 采样一个意图，
    产出 13D 数值命令 + 冻结 embedding
  - `CommandSliceCfg` / `CommandSlice`：`twist`(0:3) / `head_pose`(3:7) /
    `body_pose`(7:13) 只读视图，现有奖励/课程无需改动
  - 支持 `encoder="hf"` / `"hash"` / `embedding_cache=.npz`
- [x] 新建 `src/mjlab_microduck/tasks/microduck_vla_env_cfg.py`
  - actor 去掉数字命令，新增独立 obs group `"text"`
  - critic 保留数字命令（privileged）
  - 移除 `standing_envs` / `head_pose_range` / `body_pose_range` 课程
  - `MicroduckVlaRlCfg`：`experiment_name="vla"`、
    `obs_groups={"actor":("actor","text"), "critic":("critic",)}`
- [x] `src/mjlab_microduck/tasks/__init__.py` — 注册 `Mjlab-VLA-Flat-MicroDuck`
- [x] 测试：`tests/test_vla_cfg.py`（13 项，无需 GPU）
- [x] 门禁：64 envs / 5 iters smoke ✅（2026-10-20，wandb run `sw5sq8bm`，
      无 NaN，全部奖励正常计算）

> ✅ Phase 2 遗留的 stock `MLPModel` 问题已在 Phase 3 解决：
> `MicroduckVlaRlCfg.actor.class_name` 已指向 `MicroduckVLAModel`。

### Phase 3 — 模型 ✅

- [x] 新建 `src/mjlab_microduck/vla/model.py`
  - `MicroduckVLAModel(MLPModel)`：只归一化 proprio，原样拼接 text
  - 重写 `_get_latent_dim()` / `get_latent()` / `update_normalization()`
  - `as_onnx()` 返回双输入 wrapper：`obs` + `text_emb`；无 text group 时
    退化为单输入（critic 安全）
  - `as_jit()` 同步支持
- [x] `MicroduckVlaRlCfg.actor.class_name =
      "mjlab_microduck.vla.model:MicroduckVLAModel"`（走 `resolve_callable`，
      不需要 fork rsl_rl）
- [x] `tests/test_vla_model.py` — 10 项，含 **真实 ONNX 导出 + onnxruntime
      双输入 roundtrip** 与 TorchScript 双输入验证
- 验证点：normalizer 形状 = 48（不含 text）、改 text 不改变 normalizer 统计、
  改 text 会改变动作、ONNX 输入为 `obs[1,48]` + `text_emb[1,D]`

### Phase 4 — 导出 / 发布 ✅

- [x] 新建 `src/mjlab_microduck/vla/export_encoder.py`
  - `encoder.onnx`：`input_ids[1,L]` + `attention_mask[1,L]` → `text_emb[1,D]`
  - mean-pool + L2-normalize；导出后与 `SentenceTransformer.encode` 对拍
    （cosine ≥ 0.999，否则报错）
  - `scripts/export_encoder.py` 是命令行包装
- [x] `export.py`：`ExportConfig` 新增 `encoder_onnx` / `encoder_model`；
  导出后从图里探测 VLA（第二输入 `text_emb`），写入 metadata：
  `vla / obs_len / text_input / text_embedding_dim / text_group /
  text_encoder_model / sha256`
- [x] `publish/manifest.py`
  - `OnnxShape` 支持单/双输入；`check_onnx` 校验
    `obs[1,48] + text_emb[1,D] -> actions[1,14]`
  - `inspect/check_encoder_onnx` 校验 encoder 契约
  - `smoke_run_onnx` 双输入冒烟（固定一句 embedding + 抖 obs）
  - `SCHEMA_VERSION=3`（`VLA_SCHEMA_VERSION`）：`text` 块 + `artifacts`
    （policy + encoder），`command.encoding="text"`
  - `validate_manifest` 对 schema 3 额外校验 embedding_dim / encoder.model
  - README 生成 VLA 段落与 encoder 契约
- [x] `publish/cli.py`：`--encoder-onnx` / `--encoder-model`；VLA 必须带
  encoder（否则拒绝），仓库允许 `policy.onnx` + `encoder.onnx` 两个文件，
  manifest 写入 encoder 的 sha256
- [x] 测试：`tests/test_publish_manifest.py` 新增 13 项（双输入识别/拒绝、
  encoder 维度校验、schema 3 manifest/README、CLI dry-run 双文件）

> ⚠️ **外部运行时依赖**：`pollen-robotics/microduck` 的 daemon 目前按
> schema 2 / 单 `.onnx` 加载。本仓库已能生成 schema 3 + 双文件，但真机
> 要能加载还需要运行时侧同步支持（`docs/policy-manifest.md` schema 3）。

### Phase 5 — 部署排练 ✅

- [x] `scripts/infer_policy.py`
  - 自动探测双输入策略（`obs` + `text_emb`），actor 观测自动去掉 13D 命令（48D）
  - `--text "..."` 初始指令；`--text-file PATH` 轮询文件热更新（非阻塞）
  - `--encoder-onnx encoder.onnx`（与真机一致：onnxruntime + tokenizer）
    或 `--encoder-model <id>`（SentenceTransformer）
  - 指令变化时才重新编码并缓存；50 Hz 循环只喂缓存向量
  - `N` 键循环 instruction table 里的句子
  - 无 encoder 时自动加载默认 SentenceTransformer（装了 `vla` extra 即可）
- [x] 无头验证：VLA 策略 obs=48、text_emb=(1,384)，20 步推理动作有限；
  切换指令后 embedding 改变、动作改变
- [x] 编码器对拍：`encoder.onnx` 与 `SentenceTransformer.encode` 三条中文句
  cosine = 1.000000

### Phase 6 — 训练与课程

- 同义句池分批加入；每个意图从 step 0 起有非零采样概率
- 从零 PPO 可能先站着不动 → 用现成的 `distill.py` 做 BC 预热

## 验证状态（2026-10-20）

- ✅ **5-iter smoke**：`uv run train Mjlab-VLA-Flat-MicroDuck --env.scene.num-envs 64
  --agent.max_iterations 5` 跑通（run `sw5sq8bm`），无 NaN。
- ✅ **策略 ONNX**：训练脚本自动导出的图是 `obs[1,48] + text_emb[1,384]
  -> actions[1,14]`，`check_onnx` 识别为 VLA。
- ✅ **encoder.onnx**：真实 `paraphrase-multilingual-MiniLM-L12-v2` 导出成功
  （384D，`[1,seq] -> [1,384]`，L2 norm=1），与 `SentenceTransformer.encode`
  自检 cosine ≥ 0.999。
- ✅ **ONNX 对**：tokenize → encoder.onnx → policy.onnx 全链路跑通；
  不同指令产生不同动作。
- ✅ **测试**：87 passed / 1 skipped（VLA + publish；skip 项是“未安装
  sentence-transformers”的分支，现在已安装所以跳过）。

## 尚未完成

- 运行时仓库 `pollen-robotics/microduck` 需要支持 schema 3 / 双 ONNX
  （`policy.onnx` + `encoder.onnx`）—— 本仓库侧已完成，运行时侧未改。
- 真正的长训练（当前只跑过 5 iter smoke）；指令池可继续扩充同义句。
- 运行时若算力紧张，可把编码器搬到 companion，策略契约不变。

## 不变量 / 风险

1. **61D 热插拔契约**：文本独立输入保住 61D，但 ONNX 变多输入，runtime + manifest 必须同步。
2. **normalizer 只盖 proprio**（最容易写错）。
3. **tokenizer/编码器版本锁死**，训练=部署。
4. 每个意图从 step 0 有采样概率，否则 embedding 死权重。
5. ONNX 必须走 `scripts/export.py`，不手工转。
6. 句子→意图→数值命令的接地表必须唯一可审计。

## 命令

```bash
# 只装核心训练栈（默认）
uv sync

# 需要自由语句编码器时
uv sync --extra vla

# Phase 1 测试（训练进程占用 train.exe 时加 --no-sync）
uv run --no-sync --with pytest pytest tests/test_vla_text_encoder.py -q
```
