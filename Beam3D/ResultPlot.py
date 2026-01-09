import numpy as np
import matplotlib.pyplot as plt
import Config as cfg

# 绘制训练步骤
range_x=800
steps = np.arange(1, range_x+1)

# 生成三个种子的训练数据（假设为损失值）
FEMPINN_eL2_1=np.loadtxt("Beam3D/Results/mesh4245/int2/errorL2_seed2024.txt")  # 种子1的误差值
FEMPINN_eL2_2=np.loadtxt("Beam3D/Results/mesh4245/int2/errorL2_seed2025.txt")  # 种子2的误差值
FEMPINN_eL2_3=np.loadtxt("Beam3D/Results/mesh4245/int2/errorL2_seed2026.txt")  # 种子3的误差值

PINN_eL2_1=np.loadtxt("DEM3D/Results/mesh40x10x10/errorL2_seed2024.txt")[:range_x,]  # 种子1的误差值
PINN_eL2_2=np.loadtxt("DEM3D/Results/mesh40x10x10/errorL2_seed2025.txt")[:range_x,]  # 种子2的误差值
PINN_eL2_3=np.loadtxt("DEM3D/Results/mesh40x10x10/errorL2_seed2026.txt")[:range_x,]  # 种子3的误差值

# 将三个种子的数据堆叠成一个矩阵
FENPINN_eL2 = np.vstack([FEMPINN_eL2_1, FEMPINN_eL2_2, FEMPINN_eL2_3])
PINN_eL2 = np.vstack([PINN_eL2_1, PINN_eL2_2, PINN_eL2_3])

# 计算平均值和标准差
mean_FEMPINN_eL2 = np.mean(FENPINN_eL2, axis=0)  # 每个步骤的平均损失
std_FEMPINN_eL2 = np.std(FENPINN_eL2, axis=0)   # 每个步骤的标准差

mean_PINN_eL2 = np.mean(PINN_eL2, axis=0)  # 每个步骤的平均损失
std_PINN_eL2 = np.std(PINN_eL2, axis=0)   # 每个步骤的标准差

# 创建图形和轴
plt.figure(figsize=(10, 6))  # 设置图形大小

# 绘制每个种子的训练曲线
# plt.plot(steps, FEMPINN_eL2_1, label='Seed 1', color='blue', linewidth=1.5, alpha=0.5)
# plt.plot(steps, FEMPINN_eL2_2, label='Seed 2', color='red', linewidth=1.5, alpha=0.5)
# plt.plot(steps, FEMPINN_eL2_3, label='Seed 3', color='green', linewidth=1.5, alpha=0.5)

# 绘制平均值曲线
plt.plot(steps, mean_FEMPINN_eL2, label='AD-FEDEM mean error', color='black', linewidth=2)
plt.plot(steps, mean_PINN_eL2, label='DEM mean error', color='red', linewidth=2)

# 绘制置信区间
plt.fill_between(steps, mean_FEMPINN_eL2 - std_FEMPINN_eL2, mean_FEMPINN_eL2 + std_FEMPINN_eL2, 
                 color='gray', alpha=0.3, label='1x Standard Deviation')
plt.fill_between(steps, mean_PINN_eL2 - std_PINN_eL2, mean_PINN_eL2 + std_PINN_eL2,
                 color='lightcoral', alpha=0.3, label='1x Standard Deviation')

# 添加标题和标签
# plt.title('Training Convergence with Confidence Interval', fontsize=16)
plt.xlabel('Iterations', fontsize=16)
plt.ylabel('eL2', fontsize=16)

# 调整坐标刻度的字体大小
plt.tick_params(axis='both', which='major', labelsize=14)

# 添加图例
plt.legend(fontsize=14)

# 添加网格
plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

# 显示图形
plt.show()