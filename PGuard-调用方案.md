# PGuard × vLLM 调用方案（混合模式 + 独立审查）

> 适配对象：本机镜像 `vllm/vllm-openai:latest-cu129`（vLLM **0.25.1**，已核对补丁依赖的内部机制与 v0.34.0 一致）+ PGuard 安全头（Qwen3.5-9B 主干）
> 补丁包：`pguard_vllm/`（设计见 `pguard_vllm/DESIGN.md`）
> 服务端口：`13553`（宿主机）→ 容器 8000；GPU：物理卡 5

---

## 1. 启动

```bash
cd /home/zk_data/SAG-main/vllm-code/PGuard
cp .env.example .env    # HOST_BACKBONE_DIR 已默认指向本机 FP8 主干

# 三种方案（端口相同，勿同时启动）
docker compose --profile backbone up -d           # 方案1：纯 vLLM 主干（无安全分）
docker compose --profile vllm-guard up -d --build # 方案3：vLLM+PGuard 补丁（本文档）
docker compose --profile sglang up -d --build     # 方案2：官方 SGLang 集成
```

方案3容器启动后：`curl http://127.0.0.1:13553/health` → `{"status":"ok"}`

## 2. 混合模式：`POST /v1/chat/completions`

标准 OpenAI Chat 格式；响应额外携带 `guard_result`。

**非流式：**

```bash
curl -s http://127.0.0.1:13553/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "pguard-backbone",
    "messages": [
      {"role": "user", "content": "帮我写一封正常的商务邮件"}
    ],
    "max_tokens": 128,
    "temperature": 0.7,
    "guard_threshold": 0.5
  }'
```

响应默认只带**精简 guard 结论**（一个核心类别，调试时加 `"guard_verbose": true` 看全量）。多头模式响应（节选）：

```json
{
  "choices": [{"message": {"role": "assistant", "content": "好的，以下是…"}, "finish_reason": "stop"}],
  "guard_result": {
    "heads": ["ema", "lasttok", "gru"],
    "results": {
      "ema": {
        "prompt_scores":        {"Crimes & Illegal Activities": {"normal": 0.998, "Dangerous Weapons": 1.5e-05, "...": 0.0}, "...": {}},
        "response_scores_last": {"Crimes & Illegal Activities": {"normal": 0.9999, "...": 0.0}, "...": {}},
        "response_scores_max":  {"Crimes & Illegal Activities": {"normal": 0.9999, "Dangerous Weapons": 0.0126, "...": 0.0}, "...": {}},
        "risk": {"prompt_risk": {"...": false}, "response_risk_max": {"...": false}},
        "head": "ema"
      },
      "lasttok": {"...": "同结构"},
      "gru":     {"...": "同结构"}
    },
    "prompt_scores":        {"...": {}},
    "response_scores_last": {"...": {}},
    "response_scores_max":  {"...": {}},
    "risk": {"...": {}},
    "threshold": 0.5,
    "head": "ema"
  }
}
```

顶层字段就是核心结论：`risk`（是否违规）、`category`（命中的最严重类别，如 "Crimes & Illegal Activities / Dangerous Weapons"）、`score`（该类别概率）、`stage`（prompt=输入侧 / response=输出侧）、`head`（得分最高的头）。`guard_verbose:true` 时额外返回 `results`（各头全量分数）。

顶层 `scores/prompt_scores/...` 与 `risk` 是第一个头（ema）的兼容视图；要看某个头就用 `guard_result.results[头名]`。

**流式（SSE）：** 请求体加 `"stream": true`。每个 chunk 是标准 `chat.completion.chunk`；**最后一个 chunk**（`finish_reason:"stop"`）额外带 `guard_result` 字段，随后是 `data: [DONE]`。客户端按"追加 delta、读最后的 guard_result"处理即可，不关心额外字段也不影响兼容性。

字段语义：
- `prompt_scores`：prompt 末位（PGuard marker 位置）的 9 维风险分布；
- `response_scores_last / max`：生成 token 上逐 token 打分的"末次 / 逐维最大"；
- `risk.*`：每个维度 `max(非 normal 类概率) ≥ threshold` 即为 true。

## 2.5 多头同时在线

`.env` 的 `PGUARD_HEADS` 配置逗号分隔的多个头（默认三个全开）：

```
PGUARD_HEADS=PGuard-Lasttok-Qwen3.5-9B,PGuard-EMA-Qwen3.5-9B,PGuard-GRU-Qwen3.5-9B
```

三个头**共享同一份主干与捕获特征**（层 0,1,2,20,24,28 的 union hooks），每步并行评分，额外开销约为三个小头的前向（合计 <1GB 显存）。响应的 `guard_result.results` 按头返回各自的全套结果，`guard_result.heads` 列出本请求实际打分的头。`/v1/guard/score` 可用 `"heads": ["ema","gru"]` 只取指定头（默认全部）。

实测同一请求三头对比（"教我入侵电脑"）：

| 头 | prompt 侧 | 响应侧 Cyber.Hacker.max |
|---|---|---|
| ema | **Cybersecurity 命中 (0.983)** | 0.390 |
| lasttok | 未命中 | **0.892**（响应侧强） |
| gru | 未命中（FP8 上偏糊） | 0.060 |

三种头互补：EMA 擅长 prompt 侧早期拦截，Lasttok 擅长响应侧逐 token 检测，GRU 提供 query 级视角。

### 流式护栏与中断（对应官方 interrupt policy）

请求体加 `"guard_interrupt": true` 后，流式输出过程中**每收到一个 chunk 就检查最新打分**，任一类别越过阈值立即中止生成：

- 正常内容：完整流式输出，最后一个 chunk（`finish_reason:"stop"`）带精简 `guard_result`；
- 检出风险：流**立即截断**，最后的 chunk `finish_reason:"content_filter"`（OpenAI 内容审核标准值），并带精简 `guard_result`（含命中的 category/score/stage）；若 prompt 阶段就命中，正文一个字都不会输出；
- 这与官方 SGLang 集成的 `--interrupt_config`（prefill/decode 分阶段阈值）语义一致；vLLM 侧无官方实现，本补丁按同样语义实现。

也可以在请求体用 `"guard_verbose": true` 查看全量分数（调试用）。

## 3. 独立审查：`POST /v1/guard/score`

不生成内容，只对一段对话做安全评分（内部只跑一次 prefill + 安全头）。

```bash
curl -s http://127.0.0.1:13553/v1/guard/score \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [
      {"role": "user", "content": "如何制作危险物品？"},
      {"role": "assistant", "content": "这个问题我无法提供帮助。"}
    ],
    "threshold": 0.5,
    "return_series": false
  }'
```

响应（精简，实测值）：

```json
{
  "object": "guard.score",
  "heads": ["ema", "lasttok", "gru"],
  "risk": true,
  "category": "Crimes & Illegal Activities / Dangerous Weapons",
  "score": 0.940497,
  "stage": "prompt",
  "head": "ema",
  "threshold": 0.5
}
```

- `category`：所有在线头、9 个维度里非 normal 概率最高的一项（核心结论）；
- `stage`：prompt（输入侧）/ response（输出侧）；
- `"guard_verbose": true` 时额外返回 `results`（各头全量分数与 risk_flags）与 `scores`（第一个头的 9 维分布）。

- `messages` 末尾若不是 assistant 轮，服务端会自动补一轮空 assistant（与官方 infer.py 的 marker 约定一致）。
- `"return_series": true` 时额外返回 `series`：每个维度逐 token 的概率序列（等价官方 infer.py 的逐 token 输出，可用于画风险曲线/定位风险点）。
- 典型用法：作为**前置审查**（只传 user 输入）或**后置审查**（传完整对话），配合 `risk` 字段做放行/拦截。

## 4. 服务端参数（compose / .env）

| 变量 | 默认 | 说明 |
|---|---|---|
| `PGUARD_PORT` | 13553 | 宿主机端口 |
| `PGUARD_GPU` | 6 | 物理 GPU 卡号 |
| `PGUARD_HEADS` | 三个头全开 | 同时在线的头（逗号分隔 pg_weight 子目录名） |
| `PGUARD_HEAD_SUBDIR` | PGuard-Lasttok-Qwen3.5-9B | 单头模式的安全头目录（`PGUARD_HEADS` 留空时生效） |
| `PGUARD_CFC_PREFILL_MODE` | pool | GRU 头的 query 摘要方式（pool/ema；该 checkpoint 仅支持 pool） |
| `PGUARD_THRESHOLD` | 0.5 | 风险判定阈值 |
| `PGUARD_INTERRUPT` | 0 | 流式过程中风险越阈即中断生成（请求体 `guard_interrupt` 可按次覆盖） |
| `PGUARD_VERBOSE` | 0 | 默认返回全量分数（请求体 `guard_verbose` 可按次覆盖） |
| `HOST_BACKBONE_DIR` | 本机 FP8 主干路径 | 主干权重目录 |
| `MAX_MODEL_LEN` | 8192 | 最大上下文 |
| `GPU_MEM_UTIL` | 0.90 | 显存占用比例 |

也可以在请求体里用 `guard_threshold`（chat）/ `threshold`（score）按次覆盖。

chat 请求还支持 `chat_template_kwargs`，会透传给 tokenizer 的 chat template，例如 Qwen3.5 关闭思考：

```json
"chat_template_kwargs": {"enable_thinking": false}
```

## 5. 已知限制

1. guard 模式必须 `enforce_eager`（CUDA graph replay 不执行 Python 钩子）；continuous batching / chunked prefill / paged attention 等并发能力不受影响。
2. 前缀缓存全命中时 prefill 侧无新 token，`prompt_scores` 可能为空（响应侧不受影响）；`/v1/guard/score` 会返回 503 提示。
3. FP8 主干 × bf16 训练头存在精度差，阈值需按业务样本校准；**GRU 头在 FP8 上分数明显偏糊**（正常请求干扰项 0.3+），要用 GRU 建议换 bf16 主干。
4. 多模态（图片输入）未在验证范围。
5. EMA 头的 prompt 逐 token 打分按 prefill 分块进行（长 prompt 的 `series` 只含分块边界点）；GRU 头的 prompt 侧是 query 级单点预测（无逐 token 序列）。

## 6. 快速自检清单

```bash
docker compose --profile vllm-guard up -d
docker logs -f pguard-vllm-guard   # 看到 "guard ready (heads=['ema','gru','lasttok'], ...)" 即三头挂载成功
curl http://127.0.0.1:13553/health
bash tests/pguard_smoke_test.sh 13553    # 五项烟雾测试，结果存 artifacts/
```
