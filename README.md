# DogeTxt Live Bot

加密货币事件合约实时信号机器人。监控 ETHUSDT/BTCUSDT 5 分钟 K 线，运行 V3 多指标集成策略，生成 10 分钟事件合约交易信号，通过 PushPlus 推送到微信。

## 项目结构

```
├── main.py              # 主入口，信号管线和事件循环
├── data_stream.py       # Binance WebSocket + REST 数据流
├── indicator_engine.py  # 指标计算引擎（RSI/MFI/CCI/KDJ/BB 等 20+ 指标）
├── strategy_engine.py   # V3 集成打分策略（14 个打分组件）
├── risk_manager.py      # 风控：日限额、冷却、连亏熔断
├── state_tracker.py     # 状态跟踪：CSV 记录 + 事件合约自动结算
├── notifier.py          # PushPlus 微信推送通知
├── daily_stats.py       # 每日信号统计报告
├── test_signal.py       # 历史数据回放测试
├── config.yaml          # 配置文件（含 PushPlus token，不提交 git）
├── run.py               # 启动脚本
├── requirements.txt     # 依赖
└── logs/                # 信号和结算 CSV 日志（自动生成）
```

依赖同目录下的 `event_backtest/` 模块（需单独 clone，与本仓库在同级目录下）。

## 快速开始

### 1. 克隆并安装依赖

```bash
git clone https://github.com/feezzz/dogeTxt_live_bot.git
cd dogeTxt_live_bot
pip install -r requirements.txt
```

### 2. 配置

```bash
cp config.example.yaml config.yaml
# 编辑 config.yaml，填入你的 PushPlus token（可选，不填则仅控制台输出）
# PushPlus 注册: https://www.pushplus.plus
```

### 3. 启动

```bash
python run.py

# 仅控制台模式（不推送微信）
python run.py --console
```

### 4. 查看每日统计

```bash
python daily_stats.py              # 今日
python daily_stats.py 20260721     # 指定日期
```

### 5. 历史数据测试

```bash
python test_signal.py --symbol ETHUSDT --start 2026-06-01 --end 2026-07-01 --threshold 3.0
```

## 策略概述

**V3 集成打分策略** — 14 个评分组件综合打分：

| 组件 | 说明 |
|------|------|
| RSI(7) | 超买超卖检测 |
| Stochastic RSI | 随机 RSI + 金叉死叉 |
| MFI | 资金流量指数 |
| CCI(14) | 商品通道指数 |
| Williams %R | 威廉指标 |
| Parabolic SAR | 抛物线转向 |
| Aroon | 趋势强度 |
| MA 趋势 | MA5/10/20 排列 |
| EMA 9/21 | 快慢均线交叉 |
| KDJ | KDJ 金叉死叉 + J 值 |
| BB 位置 | 布林带位置 |
| Volume Spike | 成交量放量 |
| K 线形态 | 锤子线、吞没形态等 |
| 行情适配 | 趋势/震荡市加分 |

**信号阈值**：score ≥ 3.0 触发正式信号。

**单一周期**：仅 10 分钟事件合约（80% 赔率）。30 分钟合约已移除（胜率低 6%，ROI 差 40%）。

## 风控

- 日交易上限：50 笔
- 信号冷却：2 根 K 线（10 分钟）
- 连亏暂停：3 笔
- 日亏损熔断：$75
- 低波动时段过滤（UTC 22:00-05:00）
- 分币种 ATR 过滤：ETH ≥ 0.05%，BTC ≥ 0.03%

## 配置说明

```yaml
strategy:
  score_threshold: 3.0      # 正式信号阈值
  preview_threshold: 3.0    # 预览信号阈值（与正式阈值同级，关闭预览）
  max_daily_trades: 50      # 日最大信号数
  min_atr_pct: 0.05         # 默认最低波动率过滤
  min_atr_pct_map:          # 分币种 ATR（BTC 价高 ATR% 天然低）
    ETHUSDT: 0.05
    BTCUSDT: 0.03
  timeframes:
    10m:
      settle_bars: 2        # 2根5m K线 = 10分钟
      payout: 0.80          # 赔率 80%

risk:
  daily_loss_limit: 75
  max_consecutive_loss: 3
  low_vol_hours: [22, 23, 0, 1, 2, 3, 4, 5]
```

## 回测结论

基于 2024-01 ~ 2026-07 共 2.5 年、5 个半年周期全量数据回测（th=3.0, 分币种 ATR, 10m 合约）：

| 币种 | 交易笔数 | 胜率 | 总盈亏 | ROI |
|------|---------|------|--------|-----|
| ETHUSDT | 7,341 | **66.7%** | +$36,795 | +20.0% |
| BTCUSDT | 7,981 | **65.9%** | +$37,085 | +18.6% |
| **合计** | **15,322** | **66.3%** | **+$73,880** | **+19.3%** |

- 全部 5 个周期均超 55.6% 盈亏平衡线
- 工作日日均约 24 笔信号（ETH 11 + BTC 13）
- 无未来函数，入场价使用信号触发后下一根 K 线开盘价
- 详细回测报告见 [BACKTEST_REPORT.md](BACKTEST_REPORT.md)

## 免责声明

本工具仅供学习和研究使用。加密货币交易存在极高风险，过往表现不代表未来收益。使用者需自行承担所有交易风险。
