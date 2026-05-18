# IGWO-LSTM: 改进灰狼优化算法优化LSTM时间序列预测

基于论文《面向电主轴热误差预测建模分析的改进IGWO-LSTM算法》（马能杰, 王洪申, 2024）的算法复现与改进实现。将标准GWO的线性收敛因子改为余弦非线性衰减，利用改进后的IGWO自动搜索LSTM的最优隐藏层节点数，在公开ETT数据集上验证模型性能。

## 算法概述

### IGWO 改进原理

标准灰狼优化算法（GWO）的收敛因子 $a$ 从 2 线性衰减至 0：

$$a = 2 - \frac{2t}{T}$$

改进后的 IGWO 采用余弦非线性衰减策略：

$$a = 2 - 2 \cdot \cos\left(\frac{\pi}{2} \cdot \frac{t}{T}\right)$$

余弦衰减使算法在勘探（Exploration）与开发（Exploitation）之间的切换更加平滑，有效避免陷入局部最优。

### IGWO-LSTM 混合模型

```
IGWO (外层优化器)
    │
    ├── 决策变量: LSTM 隐藏层节点数 [2, 64]
    ├── 适应度函数: LSTM 在验证集上的 MSE
    └── 输出: 最优隐藏层节点数
            │
            ▼
    LSTM (最终预测模型)
    ├── 输入: 多维时间序列滑动窗口
    ├── 结构: 单隐层 LSTM + 全连接输出层
    └── 输出: 下一时刻预测值
```

## 实验结果

在 ETT（电力变压器温度）数据集上的测试结果：

| 模型 | 隐藏层节点数 | MAE (°C) | RMSE (°C) | MAPE (%) |
|------|:-----------:|:--------:|:---------:|:--------:|
| LSTM（经验公式） | 10 | 0.5084 | 0.7080 | 8.99 |
| GWO-LSTM | 62 | 0.5123 | 0.7182 | 9.01 |
| **IGWO-LSTM（本文方法）** | **52** | **0.4910** | **0.6930** | **8.93** |

> IGWO-LSTM 在三项指标上均取得最优。GWO 搜索陷入局部最优（节点数偏大），IGWO 通过余弦收敛策略有效规避了该问题。

## 项目结构

```
workspace/
├── igwo_lstm.py          # 完整代码（含中文注释）
├── data/
│   └── ETTh1.csv         # ETT 数据集（运行自动下载）
├── results.png           # 实验结果可视化（6张子图）
├── report/
│   ├── 第一部分_理论回顾.md
│   └── 第二部分_算法实验.md
└── README.md
```

## 环境依赖

- Python 3.8+
- PyTorch ≥ 2.0
- NumPy, Pandas, Matplotlib, Scikit-learn

```bash
pip install torch numpy pandas matplotlib scikit-learn
```

## 快速运行

```bash
# 克隆仓库
git clone github.com:Centursy/IGWO-LSTM.git
cd IGWO-LSTM

# 运行实验（首次运行会自动下载 ETT 数据集）
python igwo_lstm.py
```

运行结束后将在当前目录生成 `results.png`，包含：
1. IGWO vs GWO 收敛曲线对比
2. 收敛因子衰减策略对比（线性 vs 余弦）
3. IGWO-LSTM 预测值与实际值对比
4. 预测-实际散点图
5. 三模型性能指标柱状图
6. 预测误差分布对比

## 主要参考文献

[1] 马能杰, 王洪申. 面向电主轴热误差预测建模分析的改进IGWO-LSTM算法[J]. 机床与液压, 2024, 52(1): 11-16.

[2] Mirjalili S, Mirjalili S M, Lewis A. Grey Wolf Optimizer[J]. Advances in Engineering Software, 2014, 69: 46-61.

[3] Hochreiter S, Schmidhuber J. Long Short-Term Memory[J]. Neural Computation, 1997, 9(8): 1735-1780.

[4] Zhou H, Zhang S, Peng J, et al. Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting[C]. AAAI, 2021.

[5] Zeng A, Chen M, Zhang L, et al. Are Transformers Effective for Time Series Forecasting?[C]. AAAI, 2023.

[6] 马驰, 等. 基于PSO-BP神经网络的精密镗床主轴热误差建模[J]. 机械工程学报, 2018.

## License

MIT License
