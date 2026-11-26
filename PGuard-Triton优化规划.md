# PGuard × vLLM Triton 化优化规划

> 日期：2026-11-25
> 对象：`pguard_vllm` 补丁包（PGuard 安全头移植到 vLLM 0.25.1，服务 `pguard-vllm-guard`，GPU 6）
> 参照：`PGuard-实测记录.md`（移植版基线）、`pguard-src/deploy`（官方 SGLang 集成，报告吞吐损失 0.8%~2.3%）
> 状态：**P0/P1 已在本容器落地并复测**（见 §4），P2~P4 为 Triton 化与图恢复的后续规划。

---

## 1. 结论摘要

1. **先修正确性，再谈性能**：移植版误用了训练侧参数 `feature_layers` 选捕获层，导致 Lasttok 头一直在吃第 0/1/2 层（训练时是 20/24/28 层）。修复后 Lasttok 对炸弹问题从漏检（normal 0.791）变为命中（Dangerous Weapons 0.902），良性邮件误报 0.556 → 0.089。这是实测记录 §2.4 "Lasttok 漏检+误报" 的根因。
2. **瓶颈主要在宿主机，不在 GPU**：8 并发下三头护栏开销实测 ~35-44ms/步，其中大头是逐标量 GPU→CPU 同步（每步 ~912 次 `.item()`）、每步整文件重读 + JSON 解析、逐请求 Python 小 kernel。这些都不需要 Triton 就能消掉。
3. **第一批修复已见效**：8 并发 3 头全开，21.0s / 97 tok/s → **13.0s / 157.3 tok/s、11.7s / 174.9 tok/s**（两次实测），与纯主干 eager 基线（169~210 tok/s）的差距收窄到约 0~20%。
4. **Triton 化（P2）解决剩余的小 kernel 与融合问题，CUDA graph 恢复（P3）解决 enforce_eager 的 2.4~3 倍主干损失**。vLLM 0.25.1 自带 EAGLE3 式 `aux_hidden_state_layers` 捕获机制，可替代 Python 钩子，使"主干 FULL CUDA graph + guard 图外打分"成为可能——这是对齐官方 0.8%~2.3% 的关键路径。

## 2. 现状与瓶颈定位

### 2.1 实测量级（修复前，来自实测记录）

| 配置 | 8 并发总耗时 | 吞吐 | 说明 |
|---|---|---|---|
| 纯主干（eager） | 9.8~12.1s | 169~210 tok/s | 公平基线（同为 enforce_eager） |
| + 三头全开 | 21.0s | 97 tok/s | 护栏开销 ≈ 35~44ms/decode 步 |
| + Lasttok 单头 | 18.4s | 111 tok/s | 单头也损失 ~25-34ms/步 |
| 纯主干（CUDA graph，参考） | 4.1s | 505 tok/s | 非公平对照，但说明 eager 损失 2.4~3× |

benchmark 形态：8 请求 × max_tokens=256，纯 decode 负载（256 步），prefill 占比可忽略。

### 2.2 代码走读定位的热点（修复前）

按每 decode 步、B=8、三头全开估算：

| # | 热点 | 位置 | 每步代价 |
|---|---|---|---|
| H1 | **逐标量同步**：`round(float(x), 6)` 对每个概率元素做一次 GPU→CPU 同步 | `store.py::append` | 8 请求 × 3 头 × 38 类 ≈ **912 次 `.item()` 同步** |
| H2 | **每步整文件重读**：`_evict_finished_states` 对每个在途请求 `peek()` 整个 jsonl 并逐行 `json.loads` | `model.py` / `store.py` | 2 有状态头 × 8 请求 = 16 次全文件读+解析，文件随步数增长 |
| H3 | **逐请求特征收集**：每请求 `torch.stack` 6 层切片 + `.to(fp32)` | `model.py::_score_batch` | 8×(stack+cast)，几十个小 kernel |
| H4 | **EMA prefill 逐 token Python 循环**：`for i in range(n)` 每步 ~6 个小 kernel | `model.py::_ema_prefill` | 1000 token prompt ≈ 6000 次启动 |
| H5 | **每步重复分配**：`fuse()` 里 `torch.triu(torch.full(...))` 因果 mask、`alphas.to(device)` | `head_seq.py` | 每头每步 2~3 个多余 kernel + 分配 |
| H6 | **每步全量打分**：官方默认每 10 步打 1 次（`PGUARD_COPY_LOGITS_PROB=0.1`），移植版每步都打 | 全局 | ×10 的打分放大 |
| H7 | **enforce_eager**：CUDA graph 下不执行 Python 钩子，被迫全 eager | `server.py` | 主干 2.4~3× 损失（169-210 vs 505 tok/s） |
| H8 | **fp32 头**：EMA 头 ~805MB 权重每步整流一遍 | `_setup_pguard` | ~0.8ms/步带宽 + fp32 慢算 |
| H9 | **三头同时在线**：官方一个变体，移植版三头全跑 | `.env PGUARD_HEADS` | 护栏成本 ×3 |

> 交叉验证：单头 Lasttok（模型很小、纯 GPU 计算应 <2ms/步）却带来 25-34ms/步损失，说明 H1/H2 这类**宿主机同步与 IO** 才是主项，GPU 算力不是瓶颈。这也是为什么"只上 Triton 内核"不能解决问题，必须先斩断同步链。

### 2.3 官方 SGLang 集成为什么快（对照清单）

| 官方机制 | 出处（pguard-src/deploy/src/pguard/） | 作用 |
|---|---|---|
| 抽样打分：默认 `PGUARD_COPY_LOGITS_PROB=0.1` → 每 10 个 decode 步打 1 次，prefill 必打 | `config.py`、`patches/forward_cadence_patch.py`、`patches/guard_variant.py` | 打分成本 ÷10 |
| breakable CUDA graph：把 guard 段单独成段，非打分步 replay 跳过 | `patches/breakable_patch.py`、`patches/bcg_output_patch.py` | 非打分步完全零 guard 开销 |
| Triton EMA 核：块内 bf16 `tl.dot` + 块间 fp32 carry + 只在采样位输出 | `guard/seq_ema_triton.py`、`guard/sampled_ema_kernel.py` | 消灭逐 token 循环与全序列物化 |
| 融合小 kernel：FUSE_MERGE / FUSE_ADAPTER / FUSE_CLASSIFIERS（34-37 → 24-25 个/步） | `guard/plugin_guard.py` | 省 ~10-13 次启动 + Python 派发 |
| Triton GRU 步 + 权重预融合（3 GEMM → 2）+ 图内 unroll | `guard/cfc_guard.py` | GRU 单步 ~10 kernel → 1 GEMM + 2 kernel |
| 异步 D2H：pinned buffer 池 + event，调度器侧延迟 resolve | `patches/async_copy_patch.py` | 打分步不再阻塞引擎流 |
| 槽位式状态池（`req_pool_indices` 直接索引，无 Python dict） | `guard/history_state.py`、`guard/recurrent_state.py` | 免逐请求 Python、CUDA-graph 友好 |
| 中断策略在 CPU numpy 上批量匹配 | `interrupt_policy.py`、`interrupt_runtime.py` | 无逐 token GPU 轮询 |

另一个重要发现：官方 seq 头的 `forward` **只消费 `ema` 和 `current`**（`hist_seq/hist_mask` 是死参数，默认 `PGUARD_HIST_N=0`），移植版无需移植环形窗口。

## 3. vLLM 0.25.1 侧已核实的能力（规划依据）

- `Qwen3NextModel.forward` 原生支持中间层捕获：设置 `aux_hidden_state_layers` 后返回 `(hidden_states, aux_hidden_states)`，`_maybe_add_hidden_state` 追加 `hidden_states + residual`（`interfaces.py:1329`）——**与现钩子逐位同义**（`qwen3_next.py:607-628`）。
- 该机制即 EAGLE3/dflash/extract_hidden_states 的捕获路径（`gpu_model_runner.py:610-637`、`4362-4368`），**在 CUDA graph 捕获/回放下工作**：`ModelCudaGraphManager`（`vllm/v1/worker/gpu/cudagraph_utils.py`）在图内把 `hidden_states` 与每个 aux 拷入**共享持久缓冲**（`self.hidden_states[:num_tokens] = ...`），replay 后 `run_fullgraph` 返回本步新数据——多尺寸图也没有陈旧性问题。
- **vLLM 0.25.1 自带 breakable CUDA graph**（`vllm/compilation/breakable_cudagraph.py`，注释明确"inspired by sgl-project/sglang#19102"）：按 attention 算子切段、支持 eager 段穿插——官方 breakable_patch 的 vLLM 原生对应物，可用于后续"非打分步跳过 guard 段"。
- **runner 分裂**：默认用 legacy runner（`vllm/v1/worker/gpu_model_runner.py`，有 `_model_forward`、aux 解包逻辑），V2 runner（`vllm/v1/worker/gpu/model_runner.py`，含 `ModelCudaGraphManager`）需 `VLLM_USE_V2_MODEL_RUNNER=1`（envs.py:264）。
- **图模式实验结论（2026-11-25）**：legacy runner + aux 元组在 FULL graph 捕获阶段崩溃（`AttributeError: '_moe_model'`，EngineCore 初始化失败）——legacy runner 的捕获路径不兼容本模型/元组输出。后续路线：`VLLM_USE_V2_MODEL_RUNNER=1` + aux 再试；不通则维持 eager（当前 152~175 tok/s 已可用）。
- 当前服务 `enforce_eager=True`、`cudagraph_mode=NONE`（可用 `PGUARD_ENFORCE_EAGER` 切换）。
- 容器内 vLLM 0.25.1、torch 2.13、triton 3.7.1（宿主机 sag 环境同构，可直接做离线实验）。

## 4. 已完成：P0 正确性修复 + P1 宿主机开销治理（本轮已落地）

全部改动在挂载目录 `pguard_vllm/pguard_vllm/`（容器内 `/opt/pguard_vllm`，live-mount），容器已重启生效。

| 项 | 改动 | 文件 |
|---|---|---|
| P0 捕获层修复 | 解析顺序改为官方语义：`PGUARD_CAPTURE_LAYERS` env → `pguard_capture_layers` → `selected_layers` → `feature_layers`；三头统一捕获 20/24/28 层 | `model.py::_setup_pguard` |
| P1.1 批量传输 | 每头每步 1 次 `.cpu()` 拿回全 batch 概率（`_probs_to_cpu`），替代 912 次/步的逐标量同步 | `model.py` + `store.py` |
| P1.2 结束检测 | 新增 `GuardStore.exists()`（`os.path.exists`），替代每步整文件读+JSON 解析 | `store.py` / `model.py::_evict_finished_states` |
| P1.3 整批收集 | 特征改为整批 1 次 stack + 1 次 cast，请求行变视图（原每请求 stack+cast） | `model.py::_score_batch` |
| P1.4 闭式 EMA 折叠 | prefill 折叠改 `ema_n = Σ a(1-a)^(n-1-t)·x_t + (1-a)^n·ema_0`（一次 einsum），与官方 torch 回退同式；与旧循环逐位对照误差 ~1e-7 | `model.py::_ema_fold` |
| P1.5 缓存 | `fuse()` 因果 mask 按 (device,dtype) 缓存；`_ema_alphas` 按头缓存 | `head_seq.py` / `model.py` |
| P1.6 打分节流 | 官方同名环境变量 `PGUARD_COPY_LOGITS_PROB` / `PGUARD_SCORE_INTERVAL`（默认 1=每步都打，行为不变；EMA/GRU 状态每步仍更新，只跳过打分） | `model.py` |
| P1.7 头精度开关 | `PGUARD_HEAD_DTYPE`（默认 fp32 不变；官方用 bf16，收益 ~0.4ms/步带宽，建议 parity 验证后再切） | `model.py` |
| Lasttok 快路径 | decode 时免 `_pad_rows`；softmax 前统一 `.float()` | `model.py` |

**验证结果（容器内实测）**：
- 自检：`python3 -m pguard_vllm.store`、`python3 -m pguard_vllm.head`（三个真实 checkpoint）全过；`_ema_fold` vs 逐 token 循环（含跨 chunk 链式）最大相对误差 8e-8。
- 冒烟：5 项全过，EMA 分数与记录一致（0.940 / 0.983）；**Lasttok 修复生效**（炸弹 0.902 命中、良性邮件 0.089 不误报）。
- 流式中断：DAN 越狱 → `content_filter`，EMA Dangerous Weapons 0.942。
- 性能（8 并发 3 头，2048 token）：**21.0s/97 tok/s → 13.0s/157.3、11.7s/174.9 tok/s**；单流 20.8 → 25.5-25.7 chunks/s；TTFT 0.11-0.14s 不变。

### 4.1 第二批落地（2026-11-25 下午，官方 infra 对齐 + 数据结构改造）

| 项 | 内容 | 官方对应 |
|---|---|---|
| P2·T3 分类头融合 | 9 头共享 dense+gelu 一次 addmm，out_proj 零填充堆叠成一次 grouped matmul | `plugin_guard.py::FUSE_CLASSIFIERS` |
| P2·T5 adapter 融合 | Linear→gelu→Linear→LayerNorm(残差) 4 算子 | `plugin_guard.py::FUSE_ADAPTER` |
| P2·T4 GRU 门融合 | 3 输入 GEMM→1、2 隐层 GEMM→1（权重预拼接缓存） | `cfc_guard.py::_build_fused_weights` |
| P2·T2 gate 快路径 | 捕获层全保留时免 topk+gather（softmax 加权和等价） | `seq_guard.py` gate 融合 |
| P4-lite·fd 缓存 | 活跃 (head, req) 文件只 open 一次，每步仅 write；结束统一关闭 | — |
| P4-lite·decode 判定 | 直接由 SchedulerOutput 的 `num_computed_tokens >= prompt_len` 推导（与 runner 同判据），零文件 IO | — |
| P4-lite·状态回收 | `finished_req_ids` 推送式释放 + 低频 exists() 兜底 | — |
| P3 基建 | aux 捕获 + `_model_forward` 拦截 + 步后打分 + `PGUARD_ENFORCE_EAGER`/`PGUARD_CAPTURE_MODE` 开关 | 官方 cadence/breakable 思路 |
| 数据结构 | `state_pool.py` 槽位池（预分配 GPU 缓冲 + index_select/put，FIFO 驱逐 + 归还清零） | `recurrent_state.py` |

验证：融合/快路径 parity ≤2e-6（快路径 vs topk 对照 0 误差）；eager 重跑 smoke 分数与基线**逐位一致**（0.940497/0.983335），0 次 scoring failed；bench **11.7s / 175.4 tok/s**（8 并发 3 头，已达到纯主干 eager 基线区间 169~210 tok/s）；图模式实验失败已回退（见 §6）。

## 5. P2：Triton 内核移植（下一阶段主体）

原则：照抄官方内核的数学与分块（已验证正确），按移植版的张量布局适配；每个内核配 torch 回退与数值 parity 断言；热路径内禁止 `.item()`/同步（官方内核亦如此，见 `can_use_fused_sampled_ema` 的注释）。

| # | 内核 | 官方蓝本 | 替代的移植版热点 | 预期收益 | 验收 |
|---|---|---|---|---|---|
| T1 | EMA 抽样输出核（chunk 内 dot + 跨 chunk carry） | `seq_ema_triton.py`、`sampled_ema_kernel.py` | `return_series` / `PREFILL_RISK_MODE=all` 时的逐 token EMA；普通 `last` 模式已由 P1.4 闭式折叠覆盖 | 长 prompt 打分 O(n) kernel → O(1)；series 模式可用 | 与 torch 回退 max rel err < 1e-3（bf16）；8 并发 bench 无回退 |
| T2 | gate+topk+加权合并融合核 | `seq_guard.py`（`fuse_kernels`） | `_ema_score_batch` 里 gate_norm→gate→topk→gather→加权求和 ~8 kernel | 省 ~6 kernel/头/步 + Python 派发 | parity < 1e-5（fp32） |
| T3 | 分类头融合（9 头一次 GEMM） | `plugin_guard.py::FUSE_CLASSIFIERS` | 9×(dense+gelu+out_proj) ≈ 27 kernel | → ~4 kernel；省 ~23 次启动/头/步 | 输出逐类一致（fp32 精确） |
| T4 | CfC/GRU 单步融合核（3 GEMM→2 + 门控逐元素融合） | `cfc_guard.py::_gru_step_triton` | `_gru_decode` 的 ~10 小 kernel | GRU 头每步 ~10 → ~4 kernel | 与 eager GRU 逐位一致（`PRESERVE_ROUNDING` 模式） |
| T5 | adapter 融合（addmm+gelu+addmm+layer_norm） | `plugin_guard.py::FUSE_ADAPTER` | Lasttok 3×AdapterBlock、EMA/GRU post_adapter | 每块 5-6 → 4 kernel，且绕开 HF Module 派发 | parity < 1e-6 |
| T6 | 异步 D2H（pinned 池 + event） | `async_copy_patch.py::PendingGuardCopy` | P1.1 之后的每头每步 1 次同步 `.cpu()` | 打分步不再阻塞引擎流水；与 P3 图恢复配套 | 中断延迟不劣化，bench 无回退 |

预估：三头全开时 guard 每步 kernel 数从 ~100+ 降到 ~30，配合 T6 把打分步的同步也移出关键路径。**建议先做 T3/T5（纯融合、低风险、立竿见影），再做 T4、T2，最后 T1/T6。**

## 6. P3：恢复 CUDA graph（最大单项收益）

**方案：aux 捕获 + 图外 guard 尾巴**（vLLM 原生路径，等价于官方 breakable graph 的效果）：

1. `Qwen3_5PGuardForConditionalGeneration` 里改用 `self.language_model.set_aux_hidden_state_layers((20, 24, 28))`（或直接设 `aux_hidden_state_layers`），删除 6 个 Python forward 钩子——语义已核实与现钩子逐位一致（`interfaces.py:1337`：`hidden_states + residual`）。
2. 模型 `forward` 把 `super().forward()` 的元组原样上抛；patch_runner 拦截 `_model_forward` 取出 aux 暂存（runner 侧靠 `use_aux_hidden_state_outputs=True` 在 `__init__` 后打补丁）。
3. `server.py` 的 `PGUARD_ENFORCE_EAGER` 开关（默认 1=eager）；guard 打分统一在 `execute_model` 包装器里步后执行（真实调度步才会走到这里，天然避开 capture/warmup）。
4. 配合 P1.6 的节流：非打分步跳过打分（EMA/GRU 状态更新保留）。

**实验结果（2026-11-25，最终）**：✅ **R1 成功**。过程记录：legacy runner 的 FULL 捕获对 aux 元组不兼容（`_moe_model` AttributeError）；改用 V2 runner（`VLLM_USE_V2_MODEL_RUNNER=1`）后仍失败于 `load_model` 的 `assert speculative_config is not None`（aux 被认为必须伴随 drafter）；最终解法是 **flag 延迟置位**——`__init__` 后保持 `use_aux_hidden_state_outputs=False`（跳过该分支，aux 层已由 `_setup_pguard` 自行设置），包一层 `load_model` 在权重加载后再置 True，图捕获/回放/解包全部正常（V2 上 11 个 PIECEWISE + 7 个 FULL 图捕获成功）。
- **实测（8 并发 3 头）**：**328.0 / 331.8 tok/s**（两轮），6.1~6.2s；单流 55.5~55.9 chunks/s；TTFT 0.06~0.09s；0 次 scoring failed。
- **对照**：优化前 97 tok/s → **3.4×**；纯主干 eager 169~210 → 图模式后护栏吞吐反超 1.6~1.9×；达纯主干 CUDA graph 参考（505 tok/s）的 65%。
- **分数口径**：图模式（compile + padding）带来 ~1% 相对偏移（0.9292 vs 0.9405、0.9847 vs 0.9833），判定结论与 eager 完全一致；DAN 流式中断回归通过（content_filter，0.9368）。
- aux 获取路径：V2 无 `_model_forward`，`_score_after` 从 `runner.execute_model_state.aux_hidden_states` 读取（每步刷新的持久缓冲），trace 已验证 `mode=aux` 下 `scoring rows=N` 正常。

图模式已可用：8 并发护栏吞吐 328~332 tok/s（优化前 97），单流 55+ chunks/s；guard 尾巴每步 <5ms。后续可选：用 vLLM 原生 breakable graph（`vllm/compilation/breakable_cudagraph.py`）做"非打分步跳过 guard 段"（对应官方 `breakable_patch.py` 的 `_conditional_replay`），作为 P3.5 备选。

## 7. P4：打分通道与存储协议

**已落地（2026-11-25 第二批）**：
- **fd 缓存**：引擎侧对活跃 (head, req) 文件只 open 一次、每步仅 `write`（B=8×3 头时每步系统调用 72→24），`finished_req_ids` 触发关闭；API 侧 pop/unlink 语义不变。
- **槽位池化状态缓冲**（`state_pool.py`，官方 `RecurrentStateBuffer` 同款）：取代每请求 tensor 字典——预分配 `(max_slots, C, n_scales, H)` / `(max_slots, mem_dim)` GPU 缓冲 + 槽位分配器（FIFO 驱逐 + 归还清零），每步读写一次 `index_select`/`index_put_`；固定内存、无字典增长，为 guard 进图铺路。`PGUARD_MAX_SLOTS` 默认 64。
- **`PGUARD_PREFILL_RISK_MODE`**（all 默认 | last）：`last` 时 Lasttok prefill 只打每请求末 token（官方默认语义），长 prompt 纯审查场景省掉全序列打分；`all` 保留完整 prompt_series。
- **分数扁平化**：打分记录从"任务→token→类别"三层嵌套改为**任务主序扁平向量 + (start,stop) 类偏移表**（官方 `interrupt_policy.py` HeadRules 同款）——引擎侧零额外 GPU 操作（`torch.cat` 天然任务主序），store/API 按 `flat_to_rows` 切片；JSON 体积更小、解析更快，也为 GPU 聚合与 mmap 存储定了布局。注意：vLLM 0.25.1 的异步请求路径会给请求 id 追加后缀，store 的前缀 glob 兜底必须保留（曾误删导致 503，已恢复）。
- **GPU 聚合前置（R2）**：`StatePool` 增加每槽 `acc_last_p/acc_last_d/acc_max_d` 聚合缓冲，每步打分后一个 kernel 链完成 running max/last 更新；请求结束（`finished_req_ids`）时一次 D2H 写入 **final 记录**，`pop()` 优先读 final（**O(1)**），series 仍可从逐步行重建。已验证：final 与逐行聚合结果一致，图模式下 bench 350.5 tok/s、0 失败。

**下一步（按需）**：
1. **共享内存槽位协议**：/dev/shm 固定布局 `[max_req][tasks][classes]` + 版本号的 mmap 存储，替代 JSONL 的 open/write/parse；API 侧 `_latest_slim` 读版本号。
2. **中断检查节流**：按步读槽位版本号，替代每 chunk 读 3 个文件尾部。

## 7.1 请求形态 × 优化矩阵（实测/规划对应）

| 请求形态 | 关键路径 | 已优化 | 待做 |
|---|---|---|---|
| 非流式 chat（bench 形态） | prefill 折叠 + 逐 decode 步 | 闭式折叠、批量传输、fd 缓存、槽位池、融合算子、聚合前置（final 记录） | mmap 存储（按需） |
| 流式 + 中断 | 同上 + 每 chunk 读尾部 | 批量写降低 syscall | 槽位版本号轮询 |
| `/v1/guard/score` 纯审查 | 纯 prefill | 闭式折叠 | 实测 **~64ms** 热稳 / 77ms 首拍（HTTP 全程，含分词+prefill+3 头打分+聚合；2026-11-25，"如何制作危险物品？+拒答" 用例，lasttok 命中 Dangerous Weapons 0.7174） |
| 长 prompt | Lasttok 全 token 打分 | — | last 模式 / Triton 抽样 EMA 核（T1） |
| 高并发 decode | 每步 guard | 节流开关、融合算子 | CUDA graph（P3）|

## 8. 评测与验收

- **正确性**：`pguard_smoke_test.sh` 5 项 + Lasttok 单头两组用例（炸弹命中、良性邮件不误报）+ `return_series=true` 的序列对齐检查；P2 每个内核与 torch 回退的 parity 断言进模块自测。
- **性能协议**：`bench_guard.py`，并发 8/16/32，报 3 次中位数；单头 EMA 配置用于与官方 0.8%~2.3% 对齐比较；记录 TTFT 与单流速度。
- **目标**：单头 EMA 8 并发护栏开销 ≤5%（对齐官方量级）；三头全开 ≤15%；`PGUARD_COPY_LOGITS_PROB=0.1` 时三头 ≤5%。恢复 CUDA graph 后 8 并发总耗时目标 <7s（>300 tok/s）。
- **实测对照（11-26）**：三头每步打分约 -33%（333~349 tok/s）；三头 + `0.1` 约 -8%（463~468 tok/s），没达到“三头 + 0.1 ≤5%”。与官方对齐的配置（单头 EMA + `0.1`）实测 **497~506 tok/s**，相对主干参考值 505 ≈0%（-1.6%~+0.2%，噪声内）。

## 9. 风险与回退

| 风险 | 缓解 |
|---|---|
| 抽样打分改变 `response_scores_max` 语义（漏采 token） | 默认 interval=1 保持现行为；抽样为显式 opt-in，文档注明与官方 `[0.0]` 占位的差异 |
| bf16 头导致分数漂移 | `PGUARD_HEAD_DTYPE` 默认 fp32；切换前跑 parity 集 |
| CUDA graph 与 FP8/前缀缓存交互 | 分支灰度：先 `PIECEWISE` 实验，再 FULL；保留 `--enforce-eager` 一键回退 |
| 三头同时在线的成本 | 已默认可配；建议生产单头（EMA）或按请求选头 |
| 改动回归 | 三个文件已备份于 /tmp/pguard_backup；本文件 §4 验证项构成回归集 |

## 10. 路线图

| 阶段 | 内容 | 状态 | 预期 |
|---|---|---|---|
| P0 | Lasttok 捕获层修复 | ✅ 已落地 | Lasttok 恢复判别力 |
| P1 | 同步/IO/逐请求循环治理 + 节流/精度开关 | ✅ 已落地 | 97 → 157-175 tok/s（已实测） |
| P2 | 融合算子 T2/T3/T4/T5（分类头/adapter/GRU 门/gate 快路径） | ✅ 已落地（parity ≤2e-6） | 每头每步 ~30 kernel 启动节省；与 P1 同量级噪声内 |
| P3 | aux 捕获 + FULL CUDA graph + 图外 guard（V2 runner + flag 延迟置位） | ✅ 已落地（328~332 tok/s 实测） | 主干 2.4~3× 恢复，97 → 328+ tok/s |
| P4 | fd 缓存 + 槽位池 + prefill 模式 | ✅ 已落地 | syscall 72→24/步；状态内存固定 |
| P4+ | GPU 聚合前置（✅ 已落地）+ mmap 槽位存储（按需） | ✅ 基本落地 | 写盘 O(steps)→O(1)（final 记录） |
| P2 剩余 | Triton 核 T1（抽样 EMA）/T6（异步 D2H） | 待做 | series 模式可用；打分步移出同步 |

## 11. 剩余优化执行清单（2026-11-25 详细规划）

> 现状基线：eager + hooks，8 并发 3 头 166.8~177.8 tok/s（纯主干 eager 169~210，CUDA graph 参考 505）。
> 按"收益/风险比"排序，每项给出改动点、步骤、验收与回退。

### R1（最高优先）CUDA graph 恢复 —— ✅ 已落地（328~332 tok/s 实测）

- **结论回顾**：legacy runner 的 FULL 捕获对 aux 元组不兼容（`_moe_model` AttributeError，EngineCore 初始化失败）；V2 runner（`vllm/v1/worker/gpu/model_runner.py`）有显式支持 aux 的 `ModelCudaGraphManager`，选择开关为 env `VLLM_USE_V2_MODEL_RUNNER=1`（envs.py:264 已核实）。
- **步骤**：
  1. compose 增加透传：`VLLM_USE_V2_MODEL_RUNNER=${VLLM_USE_V2_MODEL_RUNNER:-0}`；
  2. patch_runner：V2 类没有 `_model_forward`（hasattr 守卫已跳过拦截器），aux 需改从 `runner.cudagraph_manager.aux_hidden_states` 持久缓冲读取（图内每步刷新，已核实该缓冲存在）；先在 V2 源码中确认 execute_model 的元组解包点与缓冲字段名；
  3. `.env` 设 `VLLM_USE_V2_MODEL_RUNNER=1 + PGUARD_CAPTURE_MODE=aux + PGUARD_ENFORCE_EAGER=0`，`up -d` 重建；
  4. 验收：启动无捕获错误；smoke 分数与基线**逐位一致**（0.940497/0.983335，防止图回放行映射错位）；bench 8 并发 ≥250 tok/s；30 min 长稳 score 无停更。
- **回退**：任一验收不过 → 三个 env 回退（eager+hooks），保留 V2 补丁代码待 vLLM 升级。

### R2 GPU 聚合前置 + 结束一次写盘 —— ✅ 已落地（pop O(1)，bench 350.5 tok/s）

- **改动点**：`state_pool.py` 增加 `ScoreAccumulator`（`acc_max/acc_last[max_slots, 38]` fp32）；`model.py` 每步打分后在 GPU 上 `torch.maximum` 更新（1 kernel/头/步）；`note_finished` 时一次 D2H 取回并写一条 `{"s":"final","last":[...],"max":[...],"plast":[...]}`；`store.pop()` 优先读 final 记录（O(1)），series 模式仍读逐步行。
- **步骤**：state_pool 增缓冲 → model 三处打分路径接 acc → note_finished 写 final → store.pop 兼容两种记录 → 自测（final 与逐行聚合逐位一致）→ smoke/bench。
- **验收**：pop 结果与现实现逐位一致；bench 无回退；长 prompt（4k）请求结束时 API 侧 CPU 时间下降。
- **回退**：env `PGUARD_STORE_MODE=perstep` 回到逐行。

### R3 Triton T1：抽样 EMA 核 —— ⏸ 关闭（带触发条件）

- **关闭理由**：`last` 模式（默认路径）已被闭式折叠覆盖，T1 仅服务 `return_series` 的 EMA 长 prompt 场景；当前无该需求，且 EMA series 现为"每 prefill 块末 token"的既有口径。
- **触发条件**：出现"EMA + 4k 级长 prompt + 逐 token series"的实测需求时，按 `seq_ema_triton.py`/`sampled_ema_kernel.py` 蓝本实现（验收：与 torch 回退 rel err < 1e-3，4k series 打分 ≥5×）。

### R4 Triton T6：异步 D2H —— ⏸ 关闭（带触发条件）

- **关闭理由**：扁平化后每头每步仅 1 次 D2H（~0.1-0.3ms，占图模式步时 ~2-5%）；真正的异步需要"拷贝与文件写解耦"（延迟一拍写盘 + pinned 池），复杂度与收益不匹配。
- **触发条件**：P3 图模式长稳 profile 显示 D2H 进入关键路径时，按 `async_copy_patch.py` 的 PendingGuardCopy（pinned 池 + event + 延迟写盘）移植。

### R5 Guard 段进 breakable graph —— ✅ 被架构消化（无需改造）

- 官方需要 breakable graph 是因为 guard 前向在模型图内；本移植的 guard 本就运行在 `execute_model` 之后的 Python 尾巴（图外），非打分步由 cadence（`PGUARD_COPY_LOGITS_PROB`/`PGUARD_SCORE_INTERVAL`）跳过打分、状态更新保留——R5 的目标已由现有架构达成，无需再引入 breakable 集成。

### R6 产品形态建议（零代码）

- 生产建议**单头 EMA**（`PGUARD_HEADS=PGuard-EMA-Qwen3.5-9B`）：EMA 判别力最稳；多头共识可作为旁路部署需求保留。注意每步有不随头数减少的固定开销（特征汇总、按请求分段、写记录），单头成本不是严格 ÷3。实测（11-26）：单头 EMA + 每 10 步打分 **497~506 tok/s**，与纯主干参考值 505 持平。显存（图模式，`nvidia-smi` 实读）：纯主干 20.3GB、单头 21.6GB、三头 21.9GB。
- 纯审查/长 prompt 场景开 `PGUARD_PREFILL_RISK_MODE=last`。
- 精度敏感场景再评估 `PGUARD_HEAD_DTYPE=bf16`（EMA 头带宽减半，需 parity 集通过）。

### R7 修复：间隔打分下 Lasttok 的 prompt 侧分数丢失 —— ✅ 已修复（2026-11-26）

- **现象**：`PGUARD_COPY_LOGITS_PROB` < 1 时，新请求的 prompt 与其他请求的 decode 落在同一步、且该步不打分，Lasttok 头连这段 prompt 也跳过。prompt 只在该步出现，于是该请求的 Lasttok prompt 侧分数永久丢失（10 个到达步里 9 个中招）。EMA / GRU 的 prefill 分支不受影响。
- **修复**：`model.py::run_guard_step` 中 Lasttok 只让 decode 行跟随间隔，prefill 行始终打分（与官方 `begin_forward` 语义一致）。
- **验证**：`python3 -m pguard_vllm.model` 自检，对旧逻辑 9/13 场景失败、对新逻辑全过。默认配置（每步打分）行为不变。
- **附带**：`docker-compose.yml` 补 `PGUARD_COPY_LOGITS_PROB` / `PGUARD_SCORE_INTERVAL` 透传（此前 compose 不传这两个变量，设了也不生效）。
- **端到端**（11-26）：三头 + 每 10 步打分在真实引擎上实测 463~468 tok/s（每步打分同场 333~349），晚到请求 16/16 保留 prompt 分数；详见 `PGuard-实测记录.md` §3.5。

### R8 修复：流式中断被输出记录盖住 prompt 命中 —— ✅ 已修复（2026-11-26）

- **现象**：流式 `guard_interrupt` 在 `content_filter` 之前先吐 2~4 个内容 chunk（首测起挂着的待查项）。
- **根因**：中断检查只读每个头最新一条打分记录（文件尾 4KB）。prompt 记录后面只要跟上一条正常的输出记录，prompt 侧的命中就被盖住；长 prompt 的记录超过 4KB，也读不到。prompt 有风险但输出不越线的请求，可能完全不被打断。
- **修复**：引擎每步维护每个请求的实时状态文件（prompt 结论 + 输出侧累计最大值，与最终 `guard_result` 同口径），只在最高非 normal 概率上升时重写；`server.py::_latest_slim` 改读它，`store.latest` 删除。自检 `python3 -m pguard_vllm.server --self-test`。
- **验证**：3 个风险用例 30/30 `content_filter`，中断前内容 chunk 0 个；良性用例 6/6 正常结束。每步打分的吞吐 338~345 tok/s，与修复前同场（333~349）在噪声内。每步都重写的第一版是 328.7 / 328.6 tok/s，已改成只在最高值上升时写。

### 执行顺序与里程碑

| 顺序 | 项 | 工作量 | 预期 |
|---|---|---|---|
| 1 | R1 V2 runner 图模式实验 | 0.5 天 | ✅ 已通：328~332 tok/s |
| 2 | R2 聚合前置 | 0.5 天 | ✅ 已落地：pop O(1)，350.5 tok/s |
| 3 | R6 单头/形态建议落地 | 0 | ✅ 已实测：单头 EMA + 0.1 ≈ 主干水平（497~506 tok/s） |
| 4 | R5 breakable 段跳过 | — | ✅ 被架构消化（guard 在图外 + cadence） |
| 5 | R3/R4 Triton 补齐 | — | ⏸ 关闭（触发条件见上），按需重启 |
| 6 | R7 间隔打分下 Lasttok prompt 修复 | 0 | ✅ 已修复（自检覆盖） |
| 7 | R8 流式中断修复 | 0 | ✅ 已修复（30/30 零泄漏） |
