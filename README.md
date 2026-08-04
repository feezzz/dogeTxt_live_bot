# DogeTxt Live Bot

加密货币事件合约实时信号机器人。当前主策略为 **V7 因果 ML 信号机器人**（LightGBM 冻结模型，实盘部署中）；仓库根目录保留 V3 多指标集成策略（遗留，不再推荐）。

## 项目结构

```
live_bot/
├── server_data/ETH_V7_Live_Bot_Complete/  # ★ V7 实盘机器人完整运行包（当前主策略）
│   ├── v7_run.py / v7_main.py             # 入口与主循环（信号层过滤/熔断/冷却）
│   ├── v7_strategy_engine.py              # LightGBM 推理 + session_hours 时段过滤
│   ├── v7_feature_engine.py               # 100 个严格因果特征（EMA/RMA/rolling 向量化）
│   ├── v7_live_tracker.py                 # 下一根开盘入场、再下一根收盘结算
│   ├── data_stream.py / notifier.py       # Binance WS 数据流 / 飞书+PushPlus 通知
│   ├── models/v7/                         # 冻结模型 + 参数（threshold/session_hours 等）
│   ├── config.yaml                        # 运行配置（通知 token）
│   ├── self_check.py                      # 离线自检
│   └── tests/                             # 集成测试（8 例）
├── v7_backtest.py                         # V7 全量回测管线 + 过滤参数扫描(--scan)
├── tests/test_v7_backtest.py              # 回测测试（11 例）
├── server_data/v7_backtest_report.txt     # 全量回测报告（阈值/分半年/分时段）
├── server_data/v7_filter_scan_report.txt  # 过滤参数扫描报告（时段/min_atrp/cooldown）
├── server_data/cache/                     # K线缓存（gitignore）
├── main.py / strategy_engine.py ...       # V3 遗留策略（见文末）
└── config.yaml                            # V3 配置（含密钥，不提交 git）
```

## V7 实盘机器人（当前主策略）

### 策略口径

- 冻结 LightGBM 模型：100 特征、阈值 **0.555**、ETHUSDT 10 分钟事件合约
- 5m 收盘出信号 → 下一根 5m 开盘入场 → 再下一根 5m 收盘结算
- shadow 模式：仅信号通知 + 影子结算，不自动下单
- 风控：冷却 1 根、min_atrp 0.0005、每日熔断（-75 PnL 或 50 信号，按北京时间锁存）

### 部署（服务器，本地环境系统 Python 亦可）

```bash
cd /feez/ETH_V7_Live_Bot_Complete        # 解压完整部署包
python3 -m pip install --break-system-packages -r requirements-v7.txt   # 或 ./install.sh 用 venv
python3 self_check.py                     # 离线自检

# screen 前台启动（不带 --console = 推送通知）
screen -S v7
PYTHONUNBUFFERED=1 python3 v7_run.py --config config.yaml
# Ctrl+A D 脱离；screen -r v7 重新进入
```

日志：`logs/v7_bot_YYYYMMDD.log`；信号/结算 CSV：`logs/v7_signals_*.csv`、`logs/v7_settlements_*.csv`。
完整运维说明见 `server_data/ETH_V7_Live_Bot_Complete/README_SERVER.md`。

### 时段过滤（session_hours，当前关闭）

`models/v7/eth_v7_balanced_config.json` 的 `config.session_hours`：非空数组时只允许这些北京时间小时出信号；`[]` = 不过滤（当前配置）。被过滤信号不占冷却位、不计数、不影响熔断（信号层语义，与回测一致）。扫描曾推荐 `[0,1,2,4,5,8,10,17,20]`（61.1%/日均8.8），但总盈利减半，用户决策不采纳。

## V7 回测结论（2024-01-01 ~ 2026-08-03，272,161 根 5m）

- **基准阈值 0.555：21,111 笔 / 60.0% / +42,375 / 日均 22.3**（盈亏平衡 55.56%，总盈利最优）
- **方向过滤不采纳**：UP 六个半年期全部 ≥58%，各阈值 UP 均不劣于 DOWN（0.57: UP 65.7% vs DOWN 63.4%）——"UP 系统性失效"是 6 天下跌周行情假象
- 阈值越高胜率越高但总量收缩：0.57 → 64.3%（5,408 笔）、0.58 → 67.5%（1,218 笔）
- 分半年 12 格 57.7%~61.8% 无衰减；验证段与实盘 139 笔对齐（137/139、交集 98.6%、0 不一致）
- **过滤参数扫描**（时段/min_atrp/cooldown）：胜率无显著杠杆（24 小时胜率挤在 58~63% 窄带），atrp/cooldown 只降量不升率；**总盈利优先 → 维持无过滤**
- 结论：模型有真实稳定且与方向无关的 ~60% 优势；任何过滤都是拿总盈利换单笔感觉

## V7 回测工具

```bash
python v7_backtest.py --start 2024-01-01 --end 2026-08-03   # 全量回测（数据缓存于 server_data/cache/）
python v7_backtest.py --validate                             # 与实盘记录对齐验证
python v7_backtest.py --scan                                 # 过滤参数扫描（约 8-10 分钟）
python -m pytest tests/test_v7_backtest.py                   # 回测测试
```

数据首次需从 Binance 抓取（约 5-10 分钟），之后走缓存。设计文档见 `docs/superpowers/`。

## V3 遗留策略（历史）

V3 多指标集成打分策略（15+ 评分组件 + Optuna 权重优化）曾回测 68.8% 胜率，但实盘胜率远低于回测，已被 V7 取代。代码保留于仓库根目录（`main.py` / `strategy_engine.py` / `indicator_engine.py` / `state_tracker.py` / `daily_stats.py`），仅作历史参考，不再维护推荐。

## 免责声明

本工具仅供学习和研究使用。加密货币交易存在极高风险，过往表现不代表未来收益。使用者需自行承担所有交易风险。
