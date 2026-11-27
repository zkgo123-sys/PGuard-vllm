# PGuard vLLM 容器化部署前期调研

> 调研日期：2026-11-25
> 目标：在本机（8×RTX 4090 24G）用 Docker 容器把 PGuard 服务拉起，绑定端口 `13553`，GPU 5。
> 结论先行：**PGuard 不是一个大模型，而是"冻结主干 + 90MB 安全头"的插件式护栏**。当前下载的权重只是安全头，主干 Qwen3.5-9B 需要另下。官方推理后端是 **SGLang**，vLLM 官方不支持安全头集成；vLLM 可以直接跑它的主干。

---

## 1. PGuard 是什么

PGuard（Alibaba-AAIG，Apache-2.0）是一个原生、流式、插件式的多模态 LLM 安全护栏：

```
messages -> 冻结 LLM 主干 -> 若干中间层 hidden states -> 可训练安全头 -> 9 个风险维度的分类概率
```

- 主干不改动、不影响生成行为；安全头在生成过程中逐 token 打分，**不需要额外前向**（官方 SGLang 集成）。
- 风险维度共 9 类（config.json `classification_heads`）：Crimes & Illegal Activities、Hate Speech、Physical & Mental Health、Ethics & Morality、Data Privacy、Cybersecurity、Extremism、Inappropriate Suggestions、Risks Involving Minors。每个维度 id=0 都是 `normal`。
- 评测参考（Qwen3.5-9B）：Lasttok 文本 query 0.822 / EMA 0.836 / GRU 0.836~0.860；多模态上 EMA/GRU 明显更好。

## 2. 本地权重清单（pg_weight/，ModelScope 下载中）

命名规则 `PGuard-{Head}-{Backbone}`，共 12 个 checkpoint，**每个仅约 90MB（只是安全头）**：

| Backbone | Lasttok | EMA | GRU | 输入 |
|---|---|---|---|---|
| Qwen3-8B | ✅ | ✅ | ✅ | 文本 |
| Qwen3.5-9B | ✅ | ✅ | ✅ | 图文 |
| Qwen3.5-35B-A3B | ✅ | ✅ | ✅ | 图文 |
| Qwen3.5-122B-A10B | ✅ | ✅ | ✅ | 图文 |

以 `PGuard-Lasttok-Qwen3.5-9B/config.json` 为准的关键信息：

| 字段 | 值 | 含义 |
|---|---|---|
| `architectures` | `Qwen3_5ForConditionalGeneration` | 主干是 Qwen3.5（多模态，带 vision_config） |
| `feature_extractor` | `Qwen/Qwen3.5-9B` | **主干需单独下载**，本目录没有主干权重 |
| `pguard_capture_layers` / `selected_layers` | `20,24,28` | 安全头取主干第 20/24/28 层 hidden states |
| `pguard_version` | `lasttok` | 本 checkpoint 的头变体（lasttok/ema/gru） |
| `task_schema` | 9 组 `id2label/label2id` | 每个风险维度的类别表 |
| `llm_backend` | `sglang` | 官方推理后端 |
| `hidden_size` | 4096 | 头的输入维度 |

⚠️ 注意：Qwen3.5 是"线性注意力 + 每 4 层一个 full attention"的混合架构（`layer_types`），属于较新模型；vLLM 已在 **v0.34.0**（2026-11-16 发布）中包含 `qwen3_5.py` 实现（已验证该 tag 存在此文件），本机 `sag` 环境里的 vllm dev 版也有。

## 3. 官方推理路径（来自仓库 train/ 与 deploy/）

**离线打分**（train/infer.py，输入 JSONL：`{"traceid", "input_text": [messages]}`，输出各类别概率）：
支持 `--llm_backend hf`（transformers 5.2.0）和 sglang backend。依赖：`transformers==5.2.0`、`sglang==0.5.10`（train 侧）等。

**在线服务**（deploy/，唯一官方 serving 路径 = SGLang）：

```bash
export SGLANG_EXTERNAL_MODEL_PACKAGE=pguard.sglang
export PGUARD_OVERWRITE_MODEL=1
export PGUARD_MODELING_VARIANT=seq          # stateless | seq | cfc（必填）
export PGUARD_SEQ_STATE_PATH=/path/to/head  # seq(EMA)；stateless 用 PGUARD_MODEL_PATH；cfc 用 PGUARD_CFC_STATE_PATH
python -m sglang.launch_server --model-path /path/to/Qwen3.5-9B --tp-size 1 \
    --trust-remote-code --incremental-streaming-output
```

- SGLang 版本要求：Qwen3.5/GLM-4.5V → **0.5.11**；DeepSeek-V4-Flash → 0.5.17。
- 风险分从响应的 `meta_info.guard_logits` 读取（逐 token 的类别概率向量，非打分步为 `[0.0]`）。
- 可选 `--interrupt_config policy.json` 按阈值中断请求，如 `"prefill.Crimes & Illegal Activities###Dangerous Weapons@0.9"`。
- 变体↔权重↔环境变量：`stateless`↔Lasttok↔`PGUARD_MODEL_PATH`；`seq`↔EMA↔`PGUARD_SEQ_STATE_PATH`；`cfc`↔GRU↔`PGUARD_CFC_STATE_PATH`。

## 4. vLLM 适配分析（核心问题）

**主干层面：开箱即用。** vLLM v0.34.0 原生支持 `Qwen3_5ForConditionalGeneration` / `Qwen3_5MoeForConditionalGeneration`（文本+图像+视频），无需 trust-remote-code、无需改代码。

**安全头层面：vLLM 没有对应集成，这是全部适配工作量所在。** PGuard 的 guard 头需要截获主干第 20/24/28 层 hidden states 并逐 token 跑分类头，vLLM 的 OpenAI API 和模型注册机制都不暴露这个能力。可选路线：

| 路线 | 做法 | 工作量 | 得到什么 |
|---|---|---|---|
| A. vLLM 只跑主干 | 直接 `vllm serve Qwen3.5-9B`，OpenAI 兼容接口 | ≈0（已写进 compose 默认服务） | 一个可用的 Qwen3.5 服务；**无安全打分** |
| B. SGLang 官方集成（推荐） | 按 deploy/README 用 SGLang 镜像跑，官方代码零改动 | 低（compose 已含 `sglang` profile） | 完整 PGuard 能力：逐 token `guard_logits`、中断策略 |
| C. 移植到 vLLM | 仿照 `pguard-src/deploy/src/pguard/sglang/` 写 vLLM 插件：通过 `vllm.general_plugins` 注册自定义模型，包一层 `Qwen3_5` 模型类，在 decoder layer 20/24/28 后挂 hidden-state 钩子，decode 每步跑 EMA/GRU 头，再经自定义 API/采样器把分数带出 | 高（要处理 vLLM 的 CUDA graph、TP、调度器耦合，官方仓库 12 个 sglang 适配文件可作为翻译蓝本） | 纯 vLLM 栈的护栏；仅当"必须用 vLLM"时值得立项 |

**建议**：先用路线 A 验证权重与容器链路（GPU5、端口 13553），guard 能力直接走路线 B；除非有强约束，不建议一上来就做 C。

## 5. 单卡 24G 资源预算（GPU 5 = RTX 4090）

- Qwen3.5-9B bf16 权重 ≈ 18–19GB，4090 24G 可跑：`--gpu-memory-utilization 0.90 --max-model-len 8192 --max-num-seqs 32`，KV cache 余量偏紧；可试 `--kv-cache-dtype fp8` 换更大上下文（Ada 卡支持）。
- **35B-A3B / 122B-A10B 单卡 4090 放不下**，需要多卡 TP 或量化；本机其它卡目前被占，先按 9B 方案走。
- 安全头本身 ~90MB，忽略不计；路线 B 中它跑在主干进程内，不额外占卡。

## 6. 部署操作（已备好的文件）

```
PGuard/
├── docker-compose.yml   # 默认服务=vllm 主干；--profile sglang=官方 PGuard 服务
├── Dockerfile.sglang    # 基于 lmsysorg/sglang:latest 装 pguard.sglang 包
├── .env.example         # 端口/GPU/变体/路径配置
├── pguard-src/          # 已浅克隆的 PGuard 源码（deploy 包从这里打进镜像）
└── pg_weight/           # 12 个安全头 checkpoint（下载中）
```

```bash
cd /home/zk_data/SAG-main/vllm-code/PGuard
cp .env.example .env          # 改 HOST_BACKBONE_DIR 为实际主干目录

# 1) 下载主干（约 19GB，ModelScope）
modelscope download --model Qwen/Qwen3.5-9B --local_dir ./backbone/Qwen3.5-9B

# 2a) vLLM 路线（先验证链路）
docker compose up -d
curl http://127.0.0.1:13553/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"pguard-backbone","messages":[{"role":"user","content":"hello"}],"max_tokens":16}'

# 2b) SGLang 路线（完整 PGuard 护栏）
docker compose --profile sglang up -d --build
# 从流式响应 chunk 的 meta_info.guard_logits 读取各维度风险概率
```

两个服务都用 `device_ids: ["5"]` 绑定物理 GPU5、`13553:8000` 端口映射；**不要同时启动**（端口和 GPU 冲突）。

## 7. 待确认 / 风险点

1. **主干权重**：`pg_weight` 里只有头。本机已有 **RedHatAI/Qwen3.5-9B-FP8-dynamic**（ModelScope 缓存 `/root/.cache/modelscope/hub/models/RedHatAI/Qwen3.5-9B-FP8-dynamic/`，14GB，完整），架构与 Qwen/Qwen3.5-9B 一致，4090 原生支持 FP8，vLLM/SGLang 可直接加载；FP8 精度对安全头分数可能有轻微偏移（头在 bf16 主干上训练），如打分异常再下载 bf16 原版 `Qwen/Qwen3.5-9B`。
2. **SGLang 镜像 tag**：Docker Hub `lmsysorg/sglang` 近期 tag 多为 commit 号，compose 暂用 `latest`；启动后应核对容器内 sglang 版本 ≥ 0.5.11（Qwen3.5 要求），必要时改 Dockerfile 基础镜像。
3. **变体选择**：默认 `seq`(EMA)——文本/多模态均衡；偏文本可换 `cfc`(GRU)，最简单是 `stateless`(Lasttok)。改 `.env` 的 `PGUARD_VARIANT` 即可。
4. **多模态输入**：9B 是图文模型；vLLM 路线跑 VLM 需按 vLLM multimodal 参数配置（`--limit-mm-per-prompt` 等），当前 compose 按纯文本最小可用配置。
5. **PGuard 判定有误报/漏报**，阈值与中断策略需按业务调（见 deploy/README 的 interrupt policy 示例）。

## 8. 参考

- 仓库：https://github.com/Alibaba-AAIG/PGuard （train/README.md、deploy/README.md）
- 权重：https://huggingface.co/collections/Alibaba-AAIG/pguard （ModelScope 同步）
- 论文：arXiv 2611.02791
- vLLM 支持矩阵：`Qwen3_5ForConditionalGeneration` ✅（docs/models/supported_models.md，v0.34.0 已含实现）
- 本机基线：Docker 29.1.3；conda env `sag` 内 vllm dev 版含 `qwen3_5.py`、transformers 5.19.0
