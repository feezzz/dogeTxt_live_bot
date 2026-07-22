# DogeTxt Live Bot

加密货币事件合约实时信号机器人。监控 ETHUSDT/BTCUSDT 5 分钟 K 线，运行 V3 多指标集成策略，生成 10 分钟和 30 分钟事件合约交易信号，通过 PushPlus 推送到微信。

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
├── config.example.yaml  # 配置文件模板
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
python test_signal.py --symbol ETHUSDT --start 2026-06-01 --end 2026-07-01 --threshold 4.0
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

**信号阈值**：score ≥ 4.0 触发正式信号，score ≥ 3.0 触发预览弹窗。

**双周期**：同时开 10 分钟（80% 赔率）和 30 分钟（85% 赔率）事件合约。

## 风控

- 日交易上限：50 笔
- 信号冷却：2 根 K 线（10 分钟）
- 连亏暂停：3 笔
- 日亏损熔断：$75
- 低波动时段过滤（UTC 22:00-05:00）

## 配置说明

```yaml
strategy:
  score_threshold: 4.0      # 正式信号阈值
  preview_threshold: 3.0    # 预览信号阈值
  max_daily_trades: 50      # 日最大信号数
  min_atr_pct: 0.08         # 最低波动率过滤
  timeframes:
    10m:
      settle_bars: 2        # 2根5m K线 = 10分钟
      payout: 0.80          # 赔率 80%
    30m:
      settle_bars: 6        # 6根5m K线 = 30分钟
      payout: 0.85          # 赔率 85%

risk:
  daily_loss_limit: 75
  max_consecutive_loss: 3
  low_vol_hours: [22, 23, 0, 1, 2, 3, 4, 5]
```

## 回测结论

基于 2026 年 1-6 月全量数据回测：

- **th=4.0 10分钟**：2250 笔，胜率 58.3%，夏普 3.31（风险调整后最优）
- **30分钟合约**：胜率略低但 85% 赔率让盈亏平衡线更低（54.1%），edge 更厚
- V3 策略胜率天花板约 59%
- 做空方向胜率略高于做多（58% vs 57%）

## 免责声明

本工具仅供学习和研究使用。加密货币交易存在极高风险，过往表现不代表未来收益。使用者需自行承担所有交易风险。
