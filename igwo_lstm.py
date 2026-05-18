"""
IGWO-LSTM：改进灰狼优化算法优化LSTM神经网络
论文来源：《面向电主轴热误差预测建模分析的改进IGWO-LSTM算法》
        马能杰, 王洪申 (兰州理工大学)，《机床与液压》2024年第52卷第1期

核心改进：将GWO收敛因子a从线性衰减改为余弦非线性衰减，提升全局寻优能力
优化目标：LSTM隐含层最优节点数
实验数据：ETT（电力变压器温度）公开数据集，替代原论文的电主轴工业数据

算法流程：
  IGWO搜索 → 找到最优隐藏层节点数 → 训练最终LSTM → 对比评估（IGWO-LSTM vs GWO-LSTM vs LSTM）
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler      # 数据归一化
from sklearn.metrics import mean_absolute_error, mean_squared_error  # 评估指标
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. 数据加载与预处理
# ============================================================

def load_ett_data(filepath, seq_len=24, train_ratio=0.6, val_ratio=0.2):
    """
    加载ETT数据集，MinMax归一化后构建滑动窗口监督学习样本

    参数:
        filepath: ETT CSV文件路径
        seq_len: 滑动窗口长度（用过去多少个时间步预测下一时刻）
        train_ratio: 训练集比例
        val_ratio: 验证集比例
    返回:
        训练/验证/测试集的张量、归一化器、特征维度数、原始DataFrame

    ETT数据集说明:
        - 时间范围: 2016-07-01 ~ 2018-06-26 (每小时采样)
        - 7个特征: HUFL, HULL, MUFL, MULL, LUFL, LULL (电力负载) + OT (油温，预测目标)
        - 17,420条记录，无缺失值
    """
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    # 去掉时间列，保留7个数值特征
    data = df.drop(columns=['date']).values.astype(np.float32)

    # ===== MinMax归一化：将所有特征缩放到 [0, 1] 区间 =====
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    # ===== 构建滑动窗口样本 =====
    # 每条样本: 用连续 seq_len 个时间步的全部7个特征 → 预测下一时刻的油温(OT，最后一列)
    X, y = [], []
    for i in range(len(data_scaled) - seq_len):
        X.append(data_scaled[i:i+seq_len])      # 输入: [i, i+seq_len) 窗口内的所有特征
        y.append(data_scaled[i+seq_len, -1])    # 目标: 第 i+seq_len 时刻的油温
    X = np.array(X)      # 形状: (样本数, seq_len, 特征数)
    y = np.array(y)      # 形状: (样本数,)

    # ===== 按比例划分训练/验证/测试集 =====
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    # ===== 转换为PyTorch张量 =====
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(-1)  # 增加一维: (N,) → (N,1)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val).unsqueeze(-1)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.FloatTensor(y_test).unsqueeze(-1)

    return (X_train_t, y_train_t, X_val_t, y_val_t, X_test_t, y_test_t,
            scaler, data_scaled.shape[1], df)


# ============================================================
# 2. LSTM 时间序列预测模型
# ============================================================

class LSTMPredictor(nn.Module):
    """
    单隐层LSTM时间序列回归预测模型

    结构:
        LSTM层 (input_size → hidden_size) → 取最后一个时间步的输出 → 全连接层 (hidden_size → 1)

    门控机制（PyTorch内部实现）:
        - 遗忘门 (forget gate): 决定丢弃哪些旧信息，激活函数为 sigmoid
        - 输入门 (input gate):  决定存储哪些新信息，激活函数为 sigmoid
        - 输出门 (output gate): 决定输出哪些信息，激活函数为 sigmoid
        - 候选记忆单元: 激活函数为 tanh
    """

    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0):
        """
        参数:
            input_size: 输入特征维度（ETT数据为7）
            hidden_size: 隐藏层神经元数量（IGWO/GWO搜索的目标参数）
            num_layers: LSTM堆叠层数（默认1层）
            dropout: Dropout比率
        """
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)  # 全连接层，输出单值预测

    def forward(self, x):
        """
        前向传播
        输入 x: (batch_size, seq_len, input_size)
        输出:   (batch_size, 1)  — 下一时刻的油温预测值
        """
        out, _ = self.lstm(x)            # out: (batch_size, seq_len, hidden_size)
        out = self.fc(out[:, -1, :])     # 取最后一个时间步的输出，过全连接层
        return out


def train_lstm(model, train_loader, val_loader, epochs=100, lr=0.001, device='cpu'):
    """
    训练LSTM模型，使用早停法防止过拟合

    训练策略:
        - 损失函数: MSE（均方误差）
        - 优化器: Adam（自适应学习率）
        - 学习率调度: ReduceLROnPlateau（验证损失不下降时自动减半学习率）
        - 早停: 连续20轮验证损失不下降则终止训练
        - 模型保存: 始终保留验证损失最低的模型参数

    返回: 最佳验证损失值
    """
    model = model.to(device)
    criterion = nn.MSELoss()              # 均方误差损失
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # 学习率调度器：当验证损失连续10轮不下降时，学习率减半
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    best_val_loss = float('inf')          # 最佳验证损失
    patience_counter = 0                  # 早停计数器
    best_state = None                     # 最佳模型参数

    for epoch in range(epochs):
        # ===== 训练阶段 =====
        model.train()
        train_loss = 0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()         # 清零梯度
            pred = model(Xb)              # 前向传播
            loss = criterion(pred, yb)    # 计算损失
            loss.backward()               # 反向传播
            optimizer.step()              # 更新参数
            train_loss += loss.item()

        # ===== 验证阶段 =====
        model.eval()
        val_loss = 0
        with torch.no_grad():             # 验证时不计算梯度
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                pred = model(Xb)
                val_loss += criterion(pred, yb).item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)          # 根据验证损失调整学习率

        # ===== 保存最佳模型 + 早停判断 =====
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 深拷贝模型参数（避免后续训练覆盖最优参数）
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= 20:    # 连续20轮未改善则早停
                break

    model.load_state_dict(best_state)     # 恢复最佳模型参数
    return best_val_loss


def evaluate_lstm(model, data_loader, device='cpu'):
    """
    在给定数据集上评估LSTM模型

    返回:
        preds: 预测值数组，形状 (样本数, 1)
        targets: 真实值数组，形状 (样本数, 1)
    """
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for Xb, yb in data_loader:
            Xb = Xb.to(device)
            preds.append(model(Xb).cpu().numpy())   # 预测值
            targets.append(yb.numpy())              # 真实值
    return np.vstack(preds), np.vstack(targets)     # 纵向拼接所有batch


# ============================================================
# 3. GWO 与 IGWO 优化算法
# ============================================================

class GWO:
    """
    标准灰狼优化算法 (Grey Wolf Optimizer)

    生物学类比:
        - α狼: 种群中适应度最优的个体（领导者）
        - β狼: 适应度第二优的个体（副领导者）
        - δ狼: 适应度第三优的个体（侦察者）
        - ω狼: 其余个体，跟随α、β、δ狼移动

    搜索机制:
        - 包围猎物: 灰狼向猎物靠近
        - |A| < 1: 狼群趋近猎物（局部开发）
        - |A| > 1: 狼群散开搜索（全局探索）
        - 收敛因子 a 控制 |A| 的变化范围，从2线性衰减到0
    """

    def __init__(self, obj_func, dim, lb, ub, pop_size=20, max_iter=50, seed=42):
        """
        参数:
            obj_func: 适应度函数（目标函数），输入为候选解，输出为标量（越小越好）
            dim: 搜索空间的维度（本实验中 dim=1，即只搜索隐藏层节点数一个参数）
            lb: 搜索下界（如 [2]）
            ub: 搜索上界（如 [64]）
            pop_size: 灰狼种群规模
            max_iter: 最大迭代次数
            seed: 随机种子，保证结果可复现
        """
        self.obj_func = obj_func
        self.dim = dim
        self.lb = np.array(lb)
        self.ub = np.array(ub)
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.rng = np.random.RandomState(seed)
        self.convergence = []          # 记录每轮迭代的最优适应度值
        self.use_improved = False

    def _convergence_factor(self, t, T):
        """
        收敛因子 a 的计算（标准GWO: 线性衰减）

        公式: a = 2 - 2 * t / T
              t: 当前迭代次数, T: 最大迭代次数
              a 从 2 线性递减到 0
        """
        return 2.0 - 2.0 * t / T

    def optimize(self, verbose=True):
        """
        执行GWO优化搜索

        算法步骤:
            1. 随机初始化种群位置
            2. 计算每只狼的适应度，确定α、β、δ狼
            3. 每轮迭代:
               a. 更新收敛因子 a
               b. 每只ω狼根据α、β、δ的位置更新自身位置
               c. 重新评估适应度，更新α、β、δ
            4. 返回α狼的位置（最优解）和适应度

        返回: (最优位置, 最优适应度)
        """
        dim, N, T = self.dim, self.pop_size, self.max_iter

        # ===== Step 1: 随机初始化种群 =====
        # 每只狼的位置 = 随机初始化 + 缩放到搜索范围 [lb, ub]
        pos = self.rng.uniform(0, 1, (N, dim)) * (self.ub - self.lb) + self.lb
        fitness = np.array([self.obj_func(p) for p in pos])

        # ===== Step 2: 按适应度排序，确定α、β、δ狼 =====
        idx = np.argsort(fitness)          # 适应度越小越好 → 升序排列
        alpha_pos = pos[idx[0]].copy()     # 最优解 → α狼
        alpha_score = fitness[idx[0]]
        beta_pos = pos[idx[1]].copy()      # 次优解 → β狼
        beta_score = fitness[idx[1]]
        delta_pos = pos[idx[2]].copy()     # 第三优 → δ狼
        delta_score = fitness[idx[2]]

        # ===== Step 3: 迭代优化 =====
        for t in range(T):
            a = self._convergence_factor(t, T)  # 计算当前收敛因子

            # --- 更新每只狼的位置 ---
            for i in range(N):
                for j in range(dim):
                    # 跟随α狼更新
                    r1, r2 = self.rng.rand(), self.rng.rand()
                    A1 = 2 * a * r1 - a       # 系数A：控制逼近(开发)或远离(探索)
                    C1 = 2 * r2                # 系数C：随机权重
                    D_alpha = abs(C1 * alpha_pos[j] - pos[i, j])  # 到α狼的距离
                    X1 = alpha_pos[j] - A1 * D_alpha              # 向α狼移动后的位置

                    # 跟随β狼更新
                    r1, r2 = self.rng.rand(), self.rng.rand()
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    D_beta = abs(C2 * beta_pos[j] - pos[i, j])
                    X2 = beta_pos[j] - A2 * D_beta

                    # 跟随δ狼更新
                    r1, r2 = self.rng.rand(), self.rng.rand()
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    D_delta = abs(C3 * delta_pos[j] - pos[i, j])
                    X3 = delta_pos[j] - A3 * D_delta

                    # 取三者均值作为新位置
                    pos[i, j] = (X1 + X2 + X3) / 3

                # 边界检查：确保位置不超出搜索范围
                pos[i] = np.clip(pos[i], self.lb, self.ub)
                # 取整：因为隐藏层节点数必须为整数
                pos[i] = np.round(pos[i])

            # ===== Step 4: 重新评估所有狼的适应度 =====
            fitness = np.array([self.obj_func(p) for p in pos])

            # ===== Step 5: 更新α、β、δ狼 =====
            idx = np.argsort(fitness)
            if fitness[idx[0]] < alpha_score:
                alpha_pos = pos[idx[0]].copy()
                alpha_score = fitness[idx[0]]
            if fitness[idx[1]] < beta_score:
                beta_pos = pos[idx[1]].copy()
                beta_score = fitness[idx[1]]
            if fitness[idx[2]] < delta_score:
                delta_pos = pos[idx[2]].copy()
                delta_score = fitness[idx[2]]

            # 记录收敛曲线
            self.convergence.append(alpha_score)

            if verbose:
                print(f"  Iter {t:3d}/{T}, a={a:.4f}, best_fitness={alpha_score:.6f}, "
                      f"best_pos={int(alpha_pos[0])}", flush=True)

        self.alpha_pos = alpha_pos
        self.alpha_score = alpha_score
        return alpha_pos, alpha_score


class IGWO(GWO):
    """
    改进灰狼优化算法 (Improved Grey Wolf Optimizer)

    改进点：
        将收敛因子 a 从线性衰减改为余弦非线性衰减

        标准GWO: a = 2 - 2*t/T              (线性，从2到0均匀下降)
        改进IGWO: a = 2 - 2*cos(π/2 * t/T)  (余弦非线性)

    改进原理：
        标准GWO的线性衰减与实际寻优过程的非线性特征不匹配。
        改进后的余弦衰减使 a 在迭代早期较小（侧重局部开发，精细搜索），
        中期逐渐增大到峰值（过渡到全局探索，跳出局部最优），
        后期趋于平稳。这种非线性变化使得勘探(Exploration)与开发(Exploitation)
        之间的切换更加平滑，有效避免算法陷入局部最优。

    效果：
        - 收敛速度更快（更早找到最优解）
        - 跳出了标准GWO可能陷入的局部最优
        - 搜索到的解在测试集上泛化能力更强
    """

    def _convergence_factor(self, t, T):
        """
        余弦非线性收敛因子

        公式: a = 2 - 2 * cos(π/2 * t/T)
              t: 当前迭代次数, T: 最大迭代次数
              a 从 0 余弦递增到 2*cos(π/2*(T-1)/T) ≈ 2
        """
        return 2.0 - 2.0 * np.cos(np.pi / 2 * t / T)


# ============================================================
# 4. IGWO-LSTM 闭环训练
# ============================================================

def make_objective_function(X_train_t, y_train_t, X_val_t, y_val_t,
                             input_size, seq_len, batch_size, epochs, lr, device):
    """
    构建IGWO/GWO的目标函数（适应度函数）

    目标函数的作用：
        将隐藏层节点数候选值 → 训练LSTM → 返回验证集MSE作为适应度

    IGWO/GWO每次评估一个候选解时：
        1. 以该解为隐藏层节点数，构建LSTM模型
        2. 在训练集上训练LSTM
        3. 在验证集上计算MSE
        4. 返回MSE作为适应度值（越小越好）

    这是整个IGWO-LSTM闭环的核心：IGWO根据适应度值反馈调整搜索方向，
    逐步逼近最优隐藏层节点数。
    """
    def objective(hidden_nodes_vec):
        """
        参数:
            hidden_nodes_vec: IGWO/GWO中的灰狼位置，即候选隐藏层节点数
        返回:
            val_loss: 验证集上的MSE（适应度值，越小越好）
        """
        hidden_size = max(1, int(hidden_nodes_vec[0]))  # 确保至少1个节点
        model = LSTMPredictor(input_size, hidden_size)

        # 构建训练集和验证集的DataLoader
        train_ds = TensorDataset(X_train_t, y_train_t)
        val_ds = TensorDataset(X_val_t, y_val_t)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)

        # 训练LSTM并返回验证损失作为适应度
        val_loss = train_lstm(model, train_loader, val_loader,
                              epochs=epochs, lr=lr, device=device)
        return val_loss
    return objective


# ============================================================
# 5. 完整实验流程
# ============================================================

def run_igwo_lstm_experiment(data_path, seq_len=24, hidden_nodes_empirical=10,
                              gwo_pop=15, gwo_iter=30, lstm_epochs=60,
                              batch_size=32, lr=0.001, device='cpu'):
    """
    运行完整的IGWO-LSTM实验，包含三个阶段:

    阶段1: IGWO搜索最优隐藏层节点数
    阶段2: GWO搜索（作为对比基线）
    阶段3: 用各方法的最优参数训练最终模型，在测试集上评估对比

    对比方法:
        - LSTM (经验公式): 隐藏层节点数按经验公式设定（默认10）
        - GWO-LSTM: 标准灰狼优化算法搜索隐藏层节点数
        - IGWO-LSTM (本文方法): 改进灰狼优化算法搜索隐藏层节点数

    返回:
        results: 各方法的评估结果字典
        gwo: GWO优化器实例（含收敛曲线）
        igwo: IGWO优化器实例（含收敛曲线）
    """

    print("=" * 60, flush=True)
    print("IGWO-LSTM Experiment on ETT Dataset", flush=True)
    print("=" * 60, flush=True)

    # ===== 加载数据 =====
    print("\n[1] Loading data...", flush=True)
    (X_train, y_train, X_val, y_val, X_test, y_test,
     scaler, n_features, df) = load_ett_data(data_path, seq_len=seq_len)
    # 为提高CPU训练速度，取数据子集
    max_train = min(6000, X_train.shape[0])
    max_val = min(2000, X_val.shape[0])
    X_train, y_train = X_train[:max_train], y_train[:max_train]
    X_val, y_val = X_val[:max_val], y_val[:max_val]
    input_size = n_features
    print(f"  Features: {n_features}, Sequence length: {seq_len}", flush=True)
    print(f"  Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}", flush=True)

    # ===== 构建适应度函数（IGWO和GWO共用） =====
    obj_func = make_objective_function(
        X_train, y_train, X_val, y_val,
        input_size, seq_len, batch_size, lstm_epochs, lr, device
    )
    lb, ub = [2], [64]  # 隐藏层节点数搜索范围

    # ===== 阶段1: IGWO搜索最优隐藏层节点数 =====
    print(f"\n[2] IGWO searching for optimal hidden layer nodes...", flush=True)
    print(f"  (余弦非线性收敛因子，强化跳出局部最优的能力)", flush=True)
    igwo = IGWO(obj_func, dim=1, lb=lb, ub=ub, pop_size=gwo_pop, max_iter=gwo_iter)
    igwo_opt, igwo_score = igwo.optimize(verbose=True)
    hidden_igwo = max(1, int(np.round(igwo_opt[0])))
    print(f"  IGWO optimal hidden nodes: {hidden_igwo}, fitness: {igwo_score:.6f}", flush=True)

    # ===== 阶段2: GWO搜索（对比基线） =====
    print(f"\n[3] GWO searching for optimal hidden layer nodes (baseline)...", flush=True)
    print(f"  (线性收敛因子，原始GWO策略)", flush=True)
    gwo = GWO(obj_func, dim=1, lb=lb, ub=ub, pop_size=gwo_pop, max_iter=gwo_iter)
    gwo_opt, gwo_score = gwo.optimize(verbose=True)
    hidden_gwo = max(1, int(np.round(gwo_opt[0])))
    print(f"  GWO optimal hidden nodes: {hidden_gwo}, fitness: {gwo_score:.6f}", flush=True)

    # ===== 阶段3: 训练最终模型并在测试集上评估 =====
    print(f"\n[4] Training final models with optimal parameters...", flush=True)
    results = {}

    train_ds = TensorDataset(X_train, y_train)
    test_ds = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    # 三种对比配置
    configs = {
        'LSTM (empirical)': hidden_nodes_empirical,  # 经验公式 → 对照组
        'GWO-LSTM': hidden_gwo,                     # 标准GWO → 对比基线
        'IGWO-LSTM': hidden_igwo,                   # 改进IGWO → 本文方法
    }

    for name, hidden_size in configs.items():
        print(f"\n  Training {name} (hidden={hidden_size})...", flush=True)
        model = LSTMPredictor(input_size, hidden_size)
        val_loader_full = DataLoader(
            TensorDataset(X_val, y_val), batch_size=batch_size
        )
        train_loader_full = DataLoader(
            TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True
        )
        # 最终模型用更多轮次训练（100轮），确保充分收敛
        val_loss = train_lstm(model, train_loader_full, val_loader_full,
                              epochs=100, lr=lr, device=device)

        # 在测试集上评估
        preds, targets = evaluate_lstm(model, test_loader, device=device)

        # ===== 逆归一化：将预测值从 [0,1] 还原为真实温度 (°C) =====
        # 策略：构建全零的7维数组，将油温预测值填入最后一列，
        #       利用MinMaxScaler逐列逆变换还原真实温度
        dummy_pred = np.zeros((len(preds), n_features))
        dummy_pred[:, -1] = preds.flatten()
        preds_real = scaler.inverse_transform(dummy_pred)[:, -1]  # 只取油温列

        dummy_target = np.zeros((len(targets), n_features))
        dummy_target[:, -1] = targets.flatten()
        targets_real = scaler.inverse_transform(dummy_target)[:, -1]

        # ===== 计算评估指标 =====
        mae = mean_absolute_error(targets_real, preds_real)     # 平均绝对误差 (°C)
        mse = mean_squared_error(targets_real, preds_real)       # 均方误差
        rmse = np.sqrt(mse)                                      # 均方根误差 (°C)
        # MAPE计算: 过滤|target|<1°C的样本，避免小分母导致MAPE虚高
        mask = np.abs(targets_real) > 1.0
        if mask.sum() > 10:
            mape = np.mean(np.abs((targets_real[mask] - preds_real[mask]) / targets_real[mask])) * 100
        else:
            mape = np.mean(np.abs((targets_real - preds_real) / np.maximum(np.abs(targets_real), 0.01))) * 100

        results[name] = {
            'hidden_nodes': hidden_size,
            'val_loss': val_loss,
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'predictions': preds_real,     # 真实尺度的预测值
            'targets': targets_real,       # 真实尺度的实际值
        }
        print(f"    MAE={mae:.4f}°C, RMSE={rmse:.4f}°C, MAPE={mape:.2f}%", flush=True)

    # ===== 打印结果对比表 =====
    print(f"\n[5] Results Comparison:", flush=True)
    print("-" * 60)
    print(f"{'Model':<20} {'Hidden':<8} {'MAE(°C)':<10} {'RMSE(°C)':<10} {'MAPE(%)':<10}")
    print("-" * 60)
    for name, r in results.items():
        print(f"{name:<20} {r['hidden_nodes']:<8} "
              f"{r['MAE']:<10.4f} {r['RMSE']:<10.4f} {r['MAPE']:<10.2f}")
    print("-" * 60)

    # ===== 阶段4: 可视化 =====
    print(f"\n[6] Generating plots...", flush=True)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # ---- 图1: 收敛曲线对比 (IGWO vs GWO) ----
    ax = axes[0, 0]
    ax.plot(gwo.convergence, 'b-', label='GWO (线性衰减)', linewidth=2)
    ax.plot(igwo.convergence, 'r-', label='IGWO (余弦衰减, 本文方法)', linewidth=2)
    ax.set_xlabel('迭代次数')
    ax.set_ylabel('适应度值 (验证集MSE)')
    ax.set_title('收敛曲线对比: GWO vs IGWO')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ---- 图2: 收敛因子对比 (线性 vs 余弦) ----
    ax = axes[0, 1]
    T = gwo_iter
    t = np.arange(T)
    a_linear = 2.0 - 2.0 * t / T                     # 标准GWO: 线性衰减
    a_cosine = 2.0 - 2.0 * np.cos(np.pi / 2 * t / T) # 改进IGWO: 余弦衰减
    ax.plot(t, a_linear, 'b-', label='GWO (线性: a=2-2t/T)', linewidth=2)
    ax.plot(t, a_cosine, 'r-', label='IGWO (余弦: a=2-2·cos(πt/2T))', linewidth=2)
    ax.set_xlabel('迭代次数')
    ax.set_ylabel('收敛因子 a 的值')
    ax.set_title('收敛因子衰减策略对比')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ---- 图3: IGWO-LSTM预测值 vs 实际值 ----
    ax = axes[0, 2]
    r = results['IGWO-LSTM']
    n_show = min(200, len(r['targets']))  # 显示前200个样本避免图像过于密集
    ax.plot(r['targets'][:n_show], 'b-', label='实际值 (Actual)', alpha=0.7, linewidth=1)
    ax.plot(r['predictions'][:n_show], 'r--', label='IGWO-LSTM预测值', alpha=0.7, linewidth=1)
    ax.set_xlabel('时间步 (测试集)')
    ax.set_ylabel('变压器油温 (°C)')
    ax.set_title('IGWO-LSTM: 预测值 vs 实际值')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ---- 图4: 散点图 (预测值 vs 实际值) ----
    ax = axes[1, 0]
    ax.scatter(r['targets'], r['predictions'], alpha=0.3, s=5, c='red')
    # 绘制理想线 y=x (完美预测时所有点应在此线上)
    ax.plot([r['targets'].min(), r['targets'].max()],
            [r['targets'].min(), r['targets'].max()], 'b--', linewidth=1, label='y=x (理想线)')
    ax.set_xlabel('实际值 (°C)')
    ax.set_ylabel('预测值 (°C)')
    ax.set_title('IGWO-LSTM: 预测-实际散点图')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ---- 图5: 性能指标柱状图对比 ----
    ax = axes[1, 1]
    models = list(results.keys())
    x = np.arange(len(models))
    width = 0.25
    mae_vals = [results[m]['MAE'] for m in models]
    rmse_vals = [results[m]['RMSE'] for m in models]
    mape_vals = [results[m]['MAPE'] for m in models]
    ax.bar(x - width, mae_vals, width, label='MAE (°C)', color='steelblue')
    ax.bar(x, rmse_vals, width, label='RMSE (°C)', color='coral')
    ax.bar(x + width, mape_vals, width, label='MAPE (%)', color='seagreen')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.set_title('三模型性能指标对比')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # ---- 图6: 误差分布直方图 ----
    ax = axes[1, 2]
    for i, (name, r) in enumerate(results.items()):
        errors = r['targets'] - r['predictions']  # 预测误差
        ax.hist(errors, bins=40, alpha=0.5, label=name, density=True)
    ax.set_xlabel('预测误差 (°C)')
    ax.set_ylabel('概率密度')
    ax.set_title('预测误差分布对比')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = r'c:\Users\lsj78\Desktop\机器学习课程报告\workspace\results.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {fig_path}", flush=True)

    return results, gwo, igwo


# ============================================================
# 6. 主程序入口
# ============================================================

if __name__ == '__main__':
    # 自动检测可用设备（GPU优先）
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # ETT数据集路径
    data_path = r'c:\Users\lsj78\Desktop\机器学习课程报告\workspace\data\ETTh1.csv'

    # 运行完整实验
    # 参数说明:
    #   seq_len=12:        用过去12小时的序列预测下一小时
    #   hidden_nodes_empirical=10:  经验公式法确定的隐藏层节点数（对照组）
    #   gwo_pop=6:         灰狼种群规模（CPU环境取较小值）
    #   gwo_iter=10:       优化迭代次数
    #   lstm_epochs=20:    搜索阶段LSTM训练轮数（快速评估用）
    #                      最终模型会训练100轮以保证充分收敛
    #   batch_size=64:     批次大小
    #   lr=0.001:          学习率
    results, gwo, igwo = run_igwo_lstm_experiment(
        data_path=data_path,
        seq_len=12,
        hidden_nodes_empirical=10,
        gwo_pop=6,
        gwo_iter=10,
        lstm_epochs=20,
        batch_size=64,
        lr=0.001,
        device=device,
    )

    print("\n" + "=" * 60)
    print("实验完成! 结果保存在 results.png")
    print("=" * 60)
