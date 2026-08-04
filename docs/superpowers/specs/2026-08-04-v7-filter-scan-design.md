# V7 过滤参数扫描优化 — 设计文档

日期: 2026-08-04

## 背景

V7 模型全量回测（2024-01-01~2026-08-03）结论：基准阈值 0.555 为 21,111 笔 / 60.0% / +42,375（盈亏平衡 55.56%），方向过滤不采纳，模型有真实稳定优势。用户目标：**平衡型** — 通过过滤参数（时段 / min_atrp / cooldown）提高单笔胜率至 61-63%，接受日均信号从 ~22 笔降至 8-12 笔，总量损失适度。

## 方法

### 1. simulate 最小重构（唯一代码改动）

`v7_backtest.py` 的 `simulate` 增加两个可选参数：

```python
def simulate(F, col_names, finite, model, threshold, df5,
             cooldown=1, min_atrp=0.0005, max_daily_signals=50,
             max_daily_loss=-75.0, stake=25.0, payout=0.80,
             start_ms=None, end_ms=None,
             prob_of: dict[int, float] | None = None,      # 预计算概率, 扫描复用
             session_hours: set[int] | None = None):       # 允许出信号的北京时间小时
```

- `prob_of` 传入则跳过 `model.predict`（271k 行预测只做一次，全部扫描复用）
- `session_hours` 在信号判定处检查（与 atrp 检查并列），`continue` 必须发生在 `last_signal_idx` 更新**之前** → 被过滤信号不占冷却位、不计数、不影响熔断，与实盘信号层过滤语义一致
- 新增 `beijing_hour(ts_ms) -> int` 辅助函数

### 2. 单维扫描网格

| 参数 | 档位 |
|------|------|
| min_atrp | 0.0005(基准) / 0.0006 / 0.0008 / 0.0010 / 0.0015 |
| cooldown | 1(基准) / 2 / 3 |

15 组合，每组合输出笔数 / 胜率 / 日均 / PnL。

### 3. 时段贪心搜索（北京时间）

基准交易的 24 小时胜率降序排列，从最高小时起逐个累加，每个前缀组合跑一次信号层 simulate，输出完整曲线（胜率随量增加而下降）。目标窗口日均 8-12 笔（全期 7,600-11,400 笔）对应的前缀组合为候选。

### 4. 联合验证

单维最优候选叠加（atrp × cooldown × 时段，约 3 个组合）跑分半年×方向 12 格验证。

**决策标准：** 各方向 ≥4/6 个半年期胜率 ≥60%，且无单期 <55% 崩坏；全期胜率落在 61-63% 目标带；日均 8-12 笔。

### 5. 产出物

- `v7_backtest.py`：simulate 重构 + `scan_filters()` 报告函数 + CLI `--scan` 分支
- `tests/test_v7_backtest.py`：新增 session_hours 语义测试 + prob_of 等价性测试
- 报告保存 `server_data/v7_filter_scan_report.txt`（所有扫描表 + 分析结论 + 推荐配置），提交并推送 GitHub

### 性能

一次预测 + ~30 次模拟（每次 271k 行 Python 循环 ~10-15s）≈ 8-10 分钟。

## 不做的事

- 不重训模型；不修改实盘服务器代码（本阶段只读验证，配置建议在回测确认后另行实施）
- 不做方向过滤（已证伪）；不改结算规则（10 分钟事件合约固定）
- 不做特征级挖掘（另立项目）
