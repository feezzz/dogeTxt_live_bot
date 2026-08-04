# ETH V7 完整服务器运行包

这是一个**完整、独立可运行**的 ETHUSDT V7 信号机器人包，不需要再合并原仓库，也不会再引用缺失的 `requirements.txt`。

## 目录说明

- `v7_run.py`：V7 启动入口
- `data_stream.py`：Binance 现货 5m/15m/1h 已收盘K线数据流
- `notifier.py`：飞书与 PushPlus 通知
- `v7_feature_engine.py`：100个严格因果特征
- `v7_strategy_engine.py`：LightGBM 模型推理
- `v7_live_tracker.py`：下一根开盘入场、再下一根收盘结算
- `models/v7/`：冻结模型与参数
- `config.yaml`：运行配置
- `install.sh`：安装依赖
- `run_console.sh`：前台测试
- `start.sh` / `stop.sh` / `status.sh`：后台管理

## 上传并解压

推荐上传 `.tar.gz` 到 `/feez`：

```bash
cd /feez
tar -xzf ETH_V7_Live_Bot_Complete.tar.gz
cd ETH_V7_Live_Bot_Complete
```

ZIP 也可以：

```bash
cd /feez
unzip ETH_V7_Live_Bot_Complete.zip
cd ETH_V7_Live_Bot_Complete
```

## 一键安装

```bash
chmod +x *.sh
./install.sh
```

如果提示没有 `venv`：

```bash
apt update && apt install -y python3-venv
./install.sh
```

## 修改配置

```bash
nano config.yaml
```

海外服务器通常保持：

```yaml
proxy:
  enabled: false
```

需要飞书通知时填写：

```yaml
feishu_webhook_url: "你的飞书 Webhook"
```

需要 PushPlus 时填写：

```yaml
pushplus_token: "你的 Token"
```

## 前台测试

```bash
./run_console.sh
```

控制台模式不推送通知，但会测试 Binance 数据、模型和信号计算。按 `Ctrl+C` 停止。

## 后台启动

```bash
./start.sh
```

查看状态：

```bash
./status.sh
```

查看日志：

```bash
tail -f logs/v7_console.log
```

停止：

```bash
./stop.sh
```

## 时段过滤（session_hours）

信号层时段过滤（北京时间小时），配置在 `models/v7/eth_v7_balanced_config.json` 的 `config.session_hours`：

```json
"session_hours": [0, 1, 2, 4, 5, 8, 10, 17, 20]
```

- 只在这些小时允许出信号；其他小时信号不发出。空数组 `[]` = 不过滤（默认行为）。
- 与回测语义一致：被过滤的信号不占冷却位、不计数、不影响每日熔断。
- 2026-08-04 回测扫描推荐值（8,305 笔 / 61.1% / 日均 8.8，替代原 21,111 笔 / 60.0% / 日均 22.3）。

## 离线自检

```bash
.venv/bin/python self_check.py
.venv/bin/python -m unittest tests.test_v7_live_integration -v
```

## 信号与结算口径

1. 当前 5 分钟K线收盘后计算信号。
2. 下一根 5 分钟K线开盘价作为跟踪入场价。
3. 再下一根 5 分钟K线收盘价进行结算。
4. 15分钟和1小时指标只使用在信号时刻已经完整收盘的K线。

日志输出到：

```text
logs/v7_bot_YYYYMMDD.log
logs/v7_signals_YYYYMMDD.csv
logs/v7_settlements_YYYYMMDD.csv
```

## 说明

当前是信号通知与影子结算机器人，不包含自动下单。真实平台报价、网络延迟和下单延迟可能导致实盘结果与 Binance 现货跟踪结果不同。
