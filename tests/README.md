# 测试与验收

所有**服务级测试**假设服务已启动（`docker compose --profile vllm-guard up -d`，默认端口 13553；容器内为 8000）。**模块级自测**不需要 HTTP 服务，在容器内直接执行。

## 服务级验收

前提：`curl http://127.0.0.1:13553/health` 返回 `{"status":"ok"}`。

| 脚本 | 测什么 | 命令 |
|---|---|---|
| `pguard_smoke_test.sh` | 5 项判定冒烟：health、恶意/正常独立审查、正常/诱导对话（`guard_result`）。结果 JSON 存 `tests/artifacts/` | `bash tests/pguard_smoke_test.sh 13553` |
| `pguard_interrupt_e2e.py` | 流式中断正确性：风险请求必须在**任何内容输出前**截断（零泄漏），良性请求不得误断 | `python3 tests/pguard_interrupt_e2e.py http://127.0.0.1:13553 10` |
| `pguard_cadence_e2e.py` | 打分节奏：晚到请求的 prompt 分数保留、输出 token 按 cadence 打分不丢 | `python3 tests/pguard_cadence_e2e.py http://127.0.0.1:13553 16` |
| `bench_guard.py` | 性能基准：8 并发吞吐 / TTFT / 单流速度（对照 README 性能表） | `python3 tests/bench_guard.py http://127.0.0.1:13553 8` |

## 模块级自测（容器内，无需 HTTP）

覆盖补丁包各组件：配置解析、注册表、三种安全头、槽位池、分数存储、cadence 分发、流式中断聚合。

```bash
docker exec pguard-vllm-guard python3 -m pguard_vllm.vllm_compat        # vLLM 版本兼容自检
docker exec pguard-vllm-guard python3 -m pguard_vllm.pguard_config       # 环境变量解析
docker exec pguard-vllm-guard python3 -m pguard_vllm.store               # 分数存储/聚合
docker exec pguard-vllm-guard python3 -m pguard_vllm.state_pool          # 槽位池
docker exec pguard-vllm-guard python3 -m pguard_vllm.model               # cadence 分发 + live 状态
docker exec pguard-vllm-guard python3 -m pguard_vllm.server --self-test  # 流式中断聚合
docker exec -e PGUARD_PW_DIR=/pguard_heads pguard-vllm-guard python3 -m pguard_vllm.heads          # 注册表 + 真实 checkpoint
docker exec pguard-vllm-guard python3 -m pguard_vllm.heads.lasttok       # lasttok 头
docker exec pguard-vllm-guard python3 -m pguard_vllm.heads.ema           # ema 头（含闭式折叠）
docker exec pguard-vllm-guard python3 -m pguard_vllm.heads.gru           # gru 头
```

## 多版本矩阵

逐 vLLM 版本跑同一套验收：`tests/versions/`（版本清单在 `versions.py`，入口 `run_matrix.sh`，结果落 `tests/versions/results/`）。

```bash
bash tests/versions/run_matrix.sh              # 清单里全部版本
PULL=1 bash tests/versions/run_matrix.sh 0.26.0   # 先 pull 再测指定版本
```

## 升级 / 换型验收流程

换 vLLM 版本、换主干或改动补丁代码后，按序执行，红一步停一步：

0. 在目标环境的容器里跑 `python3 -m pguard_vllm.vllm_compat`——patch 依赖报告必须全绿（否则先补 `PGUARD_RUNNER_CANDIDATES` / `vllm_compat.py`）
1. 模块级自测（上节，全绿）
2. 起服务并等 healthy
3. `pguard_smoke_test.sh` —— 判定结论必须与基线一致
4. `pguard_interrupt_e2e.py` —— 零泄漏必须保持
5. `pguard_cadence_e2e.py`
6. `bench_guard.py` —— 吞吐与 README 性能表对照（±10% 内视为正常波动）
