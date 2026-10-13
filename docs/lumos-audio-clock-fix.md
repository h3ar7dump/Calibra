# Lumos WAV 采样时钟重叠修复记录

记录日期：2026-10-13。修改直接位于本 Calibra 仓库，基于 `dojo` 分支的 Calibra 0.11.0；尚未提交或发布。接口说明见 [Lumos 文档](lumos.md#audio-sample-clock-estimates)，发布记录见 [CHANGELOG](../CHANGELOG.md)。

## 问题与原因

原始录制 `EGO_AZEC86400G8_20261009_113030` 在读取阶段抛出 `ValueError: Overlapping WAV sample clocks`，导致请求的图像质检无法执行，即使 `decode_audio=False`。

reader 用首个音频块 PTS、WAV 采样数和标称采样率推算整段的采样时钟，并将推算出的分段重叠直接判为读取错误。该时钟不保证与设备时间完全一致；音频 PTS CSV 记录块时间，不能直接还原每个 PCM 采样的设备时间。

核对原始文件得到：

| 项目 | 数值 |
| --- | --- |
| 标称采样率 | 48,000 Hz |
| 第 1 段 WAV 采样数 | 2,946,179，与实际 PCM 数据长度一致 |
| 第 1 段首个 PTS | 1,791,516,633,903,790 微秒 |
| 第 1 段元数据结束时间 | 1,791,516,695,282,258 微秒 |
| 第 2 段首个 PTS | 1,791,516,695,282,279 微秒 |
| 元数据描述的分段间隔 | 21 微秒 |
| 推算的第 1 段末采样时间比元数据结束时间晚 | 240.333 微秒 |
| 推算的采样区间重叠 | 219.333 微秒 |

推算重叠的计算式为：

```text
overlap_us = (previous_n_samples - 1) × 1,000,000 / sample_rate_hz
             - (current_first_pts_us - previous_first_pts_us)
```

代码先进行整数运算，再换算为微秒，避免 epoch 时间转成浮点数后掩盖小幅重叠。上述测量说明推算时钟与设备元数据存在差异，不能仅凭这一推算证明 PCM 采样重复；本次没有推断或修正硬件时钟。

## 行为变化

- `decode_audio=False`：保留原始音频块 PTS 和 WAV 头信息，将推算重叠记录为 `ingestion_warnings`，继续处理其他模态。
- `decode_audio=True`：推算重叠仍抛出 `ValueError`，避免返回非递增的采样时钟。
- 缺失文件、元数据不一致、非递增块 PTS 等结构错误仍会失败。
- 未裁剪、移动或重采样 PCM，未修改原始元数据、时间戳、算子阈值、评分或剪枝规则。

警告保存在 `EpisodeMetadata.extra["ingestion_warnings"]`、`EpisodeBatch.extra["ingestion_warnings"]` 和 `DiagnosticReport.ingestion_warnings`。`calibra integrity` 的文本及 JSON 输出也会显示警告。警告包含 episode ID、前后分段 index、`overlap_us` 和解释，不追加算子。

## 修改文件

| 文件 | 改动 |
| --- | --- |
| [calibra/ingestion/adapters/lumos.py](../calibra/ingestion/adapters/lumos.py) | 按音频解码模式区分警告与错误；在 episode/batch 中保留警告；兼容未提供警告元数据的已有 reader 子类。 |
| [calibra/pipeline.py](../calibra/pipeline.py) | 将 batch 的读取警告传递给原生报告。 |
| [calibra/schema/report.py](../calibra/schema/report.py) | 增加 `ingestion_warnings` 字段并在文本摘要中显示。 |
| [calibra/integrity.py](../calibra/integrity.py) | 在完整性检查的文本及 JSON 输出中显示读取警告。 |
| [tests/test_lumos.py](../tests/test_lumos.py) | 覆盖推算重叠、严格音频解码、块 PTS 错误、警告序列化、CLI 输出及已有子类兼容性。 |
| [docs/lumos.md](lumos.md) | 说明采样时钟推算的限制和读取行为。 |
| [CHANGELOG.md](../CHANGELOG.md) | 记录默认不解码音频时的行为变化。 |

## 验证结果

环境为 Windows、Python 3.14.7；运行完整测试时设置 `PYTHONUTF8=1`，避免系统默认 GBK 解码已有 UTF-8 JSON 数据失败。

| 验证 | 结果 |
| --- | --- |
| 完整核心测试，排除 `tests/regression` | 1,095 项通过、25 项跳过；覆盖率 63.48%，高于要求的 58%。 |
| 最终子类兼容调整后的 reader、pipeline、integrity 测试 | 88 项通过，其中 Lumos 测试 37 项。 |
| 固定基准测试 `tests/regression` | 5 项通过，既有判级和剪枝结果未改变。 |
| dojo-qc 使用修改后的本地库 | 41 项通过；排除 1 项专门断言旧版本抛异常的复现测试。 |
| Ruff、修改文件格式、diff 空白检查、claims 文档一致性 | 通过。 |
| mypy 错误数检查 | 143 个已有类型错误，低于 210 的基线；本地环境不同于 CI，未改写基线。 |

完整测试在最后一项子类兼容调整前执行；调整后执行了相关 88 项测试和 dojo-qc 联动验证。以下 PowerShell 命令可从本仓库根目录复现核心验证：

```powershell
$env:PYTHONUTF8 = '1'
& '.venv\Scripts\python.exe' -m pytest tests/ -q --tb=short --ignore=tests/regression --cov=calibra
& '.venv\Scripts\python.exe' -m pytest tests/test_lumos.py tests/test_pipeline.py tests/test_integrity.py -q
& '.venv\Scripts\python.exe' -m pytest tests/regression -q
& '.venv\Scripts\ruff.exe' check .
& '.venv\Scripts\python.exe' scripts/generate_claims_doc.py --check
& '.venv\Scripts\python.exe' scripts/mypy_ratchet.py
```

## 原始录制回放

[原生 Calibra 回放报告](../results/lumos-audio-clock-fix/replay-report.json) 保存了真实执行结果，由原生 `LumosReader(decode_images=True)` 和 `Pipeline` 生成。记录实际处理了 3,790 个主时钟帧，重复帧率和相机冻结比例均为 0，音频时钟警告为 219.333 微秒。

`blur` 保留跳过说明：单个 episode 未配置 `sharpness_threshold`。该项不能视为已通过。

报告中的 `source_path` 保留回放时实际读取的数据路径。原始数据当时存放在相邻 dojo-qc 仓库的本地诊断目录；没有在本仓库复制整套采集数据。

## 交付状态

代码、测试、接口文档、CHANGELOG、本说明及回放报告均位于本 Calibra 目录。修改尚未提交或推送。dojo-qc 的 Git 依赖仍锁定旧提交；生产环境使用本修复前，需要发布库修改、更新依赖锁文件并重建任务镜像。
