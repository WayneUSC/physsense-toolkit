# PhysSense-Toolkit

用于振动特征提取和离线雷达 IQ 分析的小型 Python 工具箱。

[English](README.md)

**当前为实验性 alpha。** 演示使用合成数据；没有真实传感器采集驱动、已验证滑移检测器、训练后的材料分类模型，或经过测试的 LeRobot 策略接入。

## 已包含

- 带时间戳的多通道滑动窗口。
- RMS、谱质心、主频及高频能量比。
- 针对明确数据布局的离线 ADC 解码、距离维 FFT 和复数 IQ 微多普勒分析。
- 为普通观测字典增加触觉特征的辅助函数。
- 可选 Rerun 标量与外部提供点云的记录接口。
- 示例内容分类规则；输出标签与实测特征，不提供分类置信概率。

NumPy 是必需依赖。“1000 Hz”是输入信号的采样率配置，不是已经测得的实时性能。传感器单位、安装方式与实验条件都会影响特征阈值。

## 运行

需要 Python 3.10 或更新版本：

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m physsense.cli --mode tactile --duration 3
```

可选可视化：`python -m pip install -e '.[rerun]'`，再运行 `python -m physsense.cli --mode rerun --duration 3`。

## 使用边界

ADC 输入是明确布局的离线采样字节，不是任意 DCA1000 网络数据包；不包含丢包重组、板卡配置推断或真实雷达采集。提供采样率和 chirp 参数后才能合理解释频谱。

观测辅助函数只增加字典字段，不会自动修改 LeRobot 模型结构或完成策略训练。合成数据测试也不能证明真实接触、滑移或材料分类准确率。

欢迎贡献脱敏数据样例、采集参数、可复现的数值测试与硬件验证报告。采用 [Apache 2.0](LICENSE) 许可证，由 [@WayneUSC](https://github.com/WayneUSC) 维护。
