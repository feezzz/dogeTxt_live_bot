# V7 过滤参数扫描实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在现有 V7 回测管线上做过滤参数扫描（时段 / min_atrp / cooldown），找出日均 8-12 笔、全期胜率 61-63%、分半年稳定的平衡型配置。

**Architecture:** 复用 `v7_backtest.py` 管线。simulate 增加两个可选参数（`prob_of` 预计算概率复用、`session_hours` 信号层时段过滤），新增 `scan_filters()` 报告函数与 CLI `--scan` 分支。一次模型预测（271k 行）支撑全部 ~30 次模拟扫描。

**Tech Stack:** Python, lightgbm, pandas, numpy, pytest

## Global Constraints

- simulate 仅新增可选参数，默认行为与现状完全一致（prob_of=None、session_hours=None）
- session_hours 过滤的 `continue` 必须发生在 `last_signal_idx` 更新**之前**（信号层语义：被过滤信号不占冷却位、不计数、不影响熔断）
- 特征列数为 100（非 102），用 len(col_names)/col_names.index，禁止硬编码
- 决策标准：候选组合各方向 ≥4/6 个半年期胜率 ≥60%、无单期 <55% 崩坏、全期胜率 61-63%、日均 8-12 笔
- 报告保存 `server_data/v7_filter_scan_report.txt`，提交并推送 GitHub（用户偏好：每次修改都提交推送）
- 提交范围：仅 `v7_backtest.py`、`tests/test_v7_backtest.py`、`server_data/v7_filter_scan_report.txt`、本计划与 spec 文档；禁止 git add -A（工作区有无关脚本）
- 全量数据已缓存于 `server_data/cache/`，扫描运行 8-10 分钟属正常

---

### Task 1: simulate 重构（prob_of / session_hours + 测试）

**Files:**
- Modify: `v7_backtest.py`（`beijing_hour`、`predict_probs` 新函数；`simulate` 签名与两处逻辑）
- Test: `tests/test_v7_backtest.py`（新增 3 个测试）

**Interfaces:**
- Consumes: 现有 `simulate(F, col_names, finite, model, threshold, df5, cooldown=1, min_atrp=0.0005, max_daily_signals=50, max_daily_loss=-75.0, stake=25.0, payout=0.80, start_ms=None, end_ms=None)`；`BEIJING_TZ`、`FIVE_MINUTES_MS` 模块常量
- Produces: `beijing_hour(ts_ms: int) -> int`；`predict_probs(F, finite, model, df5, start_ms=None, end_ms=None) -> dict[int, float]`；`simulate(..., prob_of: dict[int, float] | None = None, session_hours: set[int] | None = None)`——Task 2 的 scan_filters 依赖这两个新参数

- [ ] **Step 1: 写失败测试**（追加到 `tests/test_v7_backtest.py`）

```python
def test_session_hours_signal_layer_semantics():
    # 核心语义：被过滤的信号不占冷却位（错误的后过滤实现会让本测试失败）
    df5, F, names, finite = _sim_inputs()          # t0=1704067200000 = 2024-01-01 00:00 UTC = 08:00 北京
    model = _model_stub({10: 0.9, 11: 0.9})        # idx10 close 08:55 北京, idx11 close 09:00 北京
    trades = bt.simulate(F, names, finite, model, 0.555, df5,
                         cooldown=2, session_hours={9})
    assert len(trades) == 1
    assert trades[0]["signal_idx"] == 11           # idx10 被过滤不更新 last_signal_idx → idx11 不受冷却拦截

def test_session_hours_blocks_all():
    df5, F, names, finite = _sim_inputs()
    model = _model_stub({5: 0.9, 6: 0.1})
    trades = bt.simulate(F, names, finite, model, 0.555, df5, session_hours={9})
    assert trades == []                            # idx5/6 均在 08 点, 不在 {9}

def test_prob_of_equivalence():
    df5, F, names, finite = _sim_inputs()
    model = _model_stub({5: 0.9, 6: 0.1})
    a = bt.simulate(F, names, finite, model, 0.555, df5)
    prob_of = {5: 0.9, 6: 0.1}                     # 与 stub 输出一致
    b = bt.simulate(F, names, finite, model, 0.555, df5, prob_of=prob_of)
    assert a == b
```

注：`_sim_inputs()` 与 `_model_stub()` 为 Task 3 已在测试文件中定义的辅助（n=130、t0=1704067200000、idx5 close=00:30 UTC=08:30 北京、idx6=08:35 北京、idx10=08:55 北京、idx11=09:00 北京；finite 全 True 无 start/end 限制 → prob_of 字典 key 就是索引）。

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_v7_backtest.py -q`
预期: FAIL（TypeError: simulate() got an unexpected keyword argument 'session_hours'）

- [ ] **Step 3: 实现**

`v7_backtest.py` 新增（放在 `beijing_day` 附近）：

```python
def beijing_hour(ts_ms: int) -> int:
    return datetime.fromtimestamp(ts_ms / 1000, tz=BEIJING_TZ).hour


def predict_probs(F: np.ndarray, finite: np.ndarray, model,
                  df5: pd.DataFrame, start_ms: int | None = None,
                  end_ms: int | None = None) -> dict[int, float]:
    """与 simulate 内部相同的 valid_rows 逻辑，供扫描一次性预测复用。"""
    open_ts = df5["open_time"].to_numpy()
    n = len(df5)
    valid_rows = [i for i in range(n) if finite[i]
                  and (start_ms is None or open_ts[i] + FIVE_MINUTES_MS >= start_ms)
                  and (end_ms is None or open_ts[i] + FIVE_MINUTES_MS <= end_ms)]
    if not valid_rows:
        return {}
    probs = model.predict(F[valid_rows], num_iteration=model.num_trees())
    return dict(zip(valid_rows, probs))
```

`simulate` 修改（三处）：

```python
def simulate(F, col_names, finite, model, threshold, df5, cooldown=1,
             min_atrp=0.0005, max_daily_signals=50, max_daily_loss=-75.0,
             stake=25.0, payout=0.80, start_ms=None, end_ms=None,
             prob_of: dict[int, float] | None = None,
             session_hours: set[int] | None = None) -> list[dict]:
    atrp_col = col_names.index("atrp")
    open_ts = df5["open_time"].to_numpy()
    open_arr = df5["open"].to_numpy()
    close_arr = df5["close"].to_numpy()
    n = len(df5)
    if prob_of is None:                                    # 改动 1: 预测可复用
        prob_of = predict_probs(F, finite, model, df5, start_ms, end_ms)
    # ... 主循环不变 ...

    # 改动 2: threshold 判定之后、atrp 检查之前（与 atrp 并列, 必须在 last_signal_idx 更新前）
        if session_hours is not None and beijing_hour(ts) not in session_hours:
            continue
        if F[i, atrp_col] < min_atrp:
            continue
```

即主循环中信号判定段最终顺序为：`prob_of` 存在性 → 冷却检查 → threshold 方向判定 → **session_hours 检查** → atrp 检查 → `last_signal_idx = i`。

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_v7_backtest.py -q`
预期: PASS（10 个测试全绿，旧 7 个不回归）

- [ ] **Step 5: 回归验证段**

Run: `python v7_backtest.py --validate`
预期: 仍 137 vs 139、交集 98.6%、不一致 0（默认参数行为不变）

- [ ] **Step 6: 提交**

```bash
git add v7_backtest.py tests/test_v7_backtest.py
git commit -m "feat: V7回测-simulate支持预计算概率与信号层时段过滤"
git push origin master
```

---

### Task 2: scan_filters 报告 + CLI --scan + 运行输出

**Files:**
- Modify: `v7_backtest.py`（`scan_filters`、`run_scan`、`main()` 加 `--scan`）
- Create: `server_data/v7_filter_scan_report.txt`（运行产物）

**Interfaces:**
- Consumes: Task 1 的 `predict_probs`、`simulate(prob_of=, session_hours=)`；现有 `report_sweep`、`report_by_period`、`_load_frames`、`load_model`、`load_threshold`
- Produces: `scan_filters(F, names, finite, model, threshold, df5) -> str`（报告全文，含推荐配置与分析结论）

- [ ] **Step 1: 实现 scan_filters**

```python
ATRP_GRID = [0.0005, 0.0006, 0.0008, 0.0010, 0.0015]
COOLDOWN_GRID = [1, 2, 3]
TARGET_MIN_DAILY, TARGET_MAX_DAILY = 8, 12


def scan_filters(F, names, finite, model, threshold, df5) -> str:
    from collections import defaultdict
    lines = []
    prob_of = predict_probs(F, finite, model, df5)

    def run(**kw):
        return simulate(F, names, finite, model, threshold, df5, prob_of=prob_of, **kw)

    base = run()
    days = (df5["open_time"].iloc[-1] - df5["open_time"].iloc[0]) / 86400000.0

    lines.append("=== 基准校验 (阈值0.555, 无过滤) ===")
    lines.append(report_sweep([("全部", base)]))
    lines.append(f"日均 {len(base) / days:.1f} 笔 (窗口 {days:.0f} 天)")

    lines.append("\n=== min_atrp × cooldown 网格 ===")
    grid = []
    for atrp in ATRP_GRID:
        for cd in COOLDOWN_GRID:
            sel = run(min_atrp=atrp, cooldown=cd)
            grid.append((atrp, cd, sel))
    grid_lines = [f"{'atrp':<8}{'cooldown':<10}{'笔数':<8}{'胜率':<8}{'日均':<7}{'PnL':<9}"]
    for atrp, cd, sel in grid:
        if not sel:
            continue
        wins = sum(1 for t in sel if t["result"] == "WIN")
        grid_lines.append(
            f"{atrp:<8}{cd:<10}{len(sel):<8}{wins / len(sel) * 100:<8.1f}"
            f"{len(sel) / days:<7.1f}{sum(t['pnl'] for t in sel):<+9.1f}")
    lines.append("\n".join(grid_lines))
    # 网格候选: 日均在 [8,12] 内且胜率最高的组合
    grid_cand = max(
        ((atrp, cd, sel) for atrp, cd, sel in grid
         if TARGET_MIN_DAILY <= len(sel) / days <= TARGET_MAX_DAILY),
        key=lambda x: sum(1 for t in x[2] if t["result"] == "WIN") / len(x[2]),
        default=None)

    lines.append("\n=== 时段贪心 (北京时间, 按小时胜率降序累加) ===")
    by_h = defaultdict(list)
    for t in base:
        by_h[int(t["signal_time"][11:13])].append(t)
    hours_sorted = sorted(by_h, key=lambda h: -sum(1 for t in by_h[h] if t["result"] == "WIN") / len(by_h[h]))
    sess_lines = [f"{'小时集合':<22}{'笔数':<8}{'胜率':<8}{'日均':<7}"]
    sess_cand = None
    for k in range(1, len(hours_sorted) + 1):
        session = set(hours_sorted[:k])
        sel = run(session_hours=session)
        wins = sum(1 for t in sel if t["result"] == "WIN")
        daily = len(sel) / days
        tag = ""
        if TARGET_MIN_DAILY <= daily <= TARGET_MAX_DAILY:
            tag = " ← 目标带"
            if sess_cand is None:
                sess_cand = (session, sel)
        sess_lines.append(f"{sorted(session)}{'':<{22 - len(str(sorted(session)))}}{len(sel):<8}"
                          f"{wins / len(sel) * 100:<8.1f}{daily:<7.1f}{tag}")
    lines.append("\n".join(sess_lines))

    lines.append("\n=== 联合候选分半年×方向验证 ===")
    cands = []
    if grid_cand:
        cands.append(("atrp=%.4f cd=%d" % (grid_cand[0], grid_cand[1]), grid_cand[2]))
    if sess_cand:
        cands.append(("时段%s" % sorted(sess_cand[0]), sess_cand[1]))
    if grid_cand and sess_cand:
        atrp, cd, _ = grid_cand
        comb = run(min_atrp=atrp, cooldown=cd, session_hours=sess_cand[0])
        cands.append(("叠加 atrp=%.4f cd=%d 时段%s" % (atrp, cd, sorted(sess_cand[0])), comb))
    for name, sel in cands:
        wins = sum(1 for t in sel if t["result"] == "WIN")
        lines.append(f"\n--- {name}: {len(sel)}笔 {wins / len(sel) * 100:.1f}% "
                     f"日均{len(sel) / days:.1f} PnL{sum(t['pnl'] for t in sel):+.0f} ---")
        lines.append(report_by_period(sel))

    lines.append("\n=== 分析结论 ===")
    lines.append(_filter_conclusion(cands, days))
    return "\n".join(lines)
```

`_filter_conclusion` 分析函数（按决策标准输出推荐）：

```python
def _filter_conclusion(cands: list[tuple[str, list[dict]]], days: float) -> str:
    out = []
    for name, sel in cands:
        wins = sum(1 for t in sel if t["result"] == "WIN")
        wr = wins / len(sel) * 100
        daily = len(sel) / days
        periods = {}
        for t in sel:
            y, m = int(t["signal_time"][:4]), int(t["signal_time"][5:7])
            key = (f"{y}H{1 if m <= 6 else 2}", t["direction"])
            periods.setdefault(key, []).append(t)
        stable = all(
            sum(1 for t in v if t["result"] == "WIN") / len(v) * 100 >= 60.0
            for v in periods.values())
        low = [k for k, v in periods.items()
               if sum(1 for t in v if t["result"] == "WIN") / len(v) * 100 < 55.0]
        verdict = "✓ 达标" if (TARGET_MIN_DAILY <= daily <= TARGET_MAX_DAILY
                               and 61.0 <= wr <= 63.0 and stable and not low) else "✗ 未达标"
        out.append(f"{name}: {verdict} 胜率{wr:.1f}% 日均{daily:.1f} 半年格数{len(periods)} "
                   f"全部≥60%: {stable} 崩坏格: {low}")
    out.append("\n推荐: 若存在达标组合, 取胜率最高者; 否则报告最接近目标带的组合及其差距。")
    return "\n".join(out)
```

- [ ] **Step 2: 实现 run_scan + CLI**

```python
def run_scan(proxy_url: str | None) -> None:
    model = load_model()
    threshold = load_threshold()
    s = datetime(2024, 1, 1, tzinfo=timezone.utc)
    e = datetime(2026, 8, 3, tzinfo=timezone.utc)
    frames = _load_frames(s, e, proxy_url)
    F, names, finite = build_aligned_features(*frames)
    report = scan_filters(F, names, finite, model, threshold, frames[0])
    out = Path("server_data/v7_filter_scan_report.txt")
    out.write_text(report, encoding="utf-8")
    print(report)
```

`main()` 增加：

```python
    ap.add_argument("--scan", action="store_true", help="过滤参数扫描(时段/min_atrp/cooldown)")
    ...
    if args.validate:
        run_validate(proxy)
    elif args.scan:
        run_scan(proxy)
    else:
        run_full(args.start, args.end, proxy)
```

- [ ] **Step 3: 回归测试**

Run: `python -m pytest tests/test_v7_backtest.py -q` 与 `python v7_backtest.py --validate`
预期: 10 测试全绿；validate 仍 137 vs 139、不一致 0

- [ ] **Step 4: 运行扫描**

Run: `python v7_backtest.py --scan`（约 8-10 分钟；数据全缓存）
预期: 输出基准校验（21,111 笔 / 60.0% / 日均~22，与全量报告一致）、网格表、贪心曲线、联合验证表、分析结论；文件保存至 `server_data/v7_filter_scan_report.txt`

- [ ] **Step 5: 检查报告合理性**

- 基准校验行必须与已提交的全量报告一致（21,111 / 60.0 / +42,375）
- 贪心曲线单调性：小时集合增大 → 胜率下降、笔数上升
- 联合候选的 12 格数字与 `report_by_period` 输出自洽
- 若有数字异常（胜率 <55.56% 的候选、日均 0、空表），排查后重跑

- [ ] **Step 6: 提交并推送**

```bash
git add v7_backtest.py server_data/v7_filter_scan_report.txt
git commit -m "feat: V7回测-过滤参数扫描(时段/min_atrp/cooldown)与报告"
git push origin master
```

---
