#---------------------------------随机种子-----------------------------------------
seed = 2025
#---------------------------------网格设置-----------------------------------------
model_shape='star-2'  # 网格形状
mesh_points=15564  # 网格点数

#--------------------------------高斯积分精度设置-----------------------------------------
n_int3D=2  # 三维高斯积分精度
n_int2D=2  # 二维高斯积分精度

#-------------------------------材料参数设置-------------------------------------
# ----------E：杨氏模量，nu：泊松比---------------
E=5e6  # 单位：Pa
nu=0.498  # 泊松比
#-------------------------------Dirichlet边界条件设置-------------------------------------
# ----------Dir_marker：边界标记，Dir_u：边界指定位移---------------
Dir_marker='OutSurface'
Dir_u=[0.0, 0.0, 0.0]
#-------------------------------压力边界条件设置-------------------------------------
# ----------Pre_marker：边界标记，Pre_value：边界指定力---------------
Pre_marker='InSurface'
Pre_value=7e6  # 单位：Pa
#-------------------------------对称边界条件设置-------------------------------------
# ----------Sym_marker：边界标记---------------
Sym_marker='Symmetry'
#--------------------------------神经网络设置-----------------------------------------
input_size=3  # ResNet的输入大小
hidden_size=200  # ResNet的隐藏层大小
output_size=3  # ResNet的输出大小
depth=4  # ResNet的深度
#--------------------------------训练参数设置-----------------------------------------
epoch_num=100000  # 训练的epoch数
lr=5e-4 #  学习率
lr_scheduler='Exp'
loss_weight=1e6 # 边界损失函数的权重

#---------------------------文件路径-----------------------------------------------
mesh_path=f"Grain3D/mesh/{model_shape}_mesh_{mesh_points}.msh"
model_save_path=f"Grain3D/Results_Adam/{model_shape}_mesh{mesh_points}_Net{depth}x{input_size}-{hidden_size}-{output_size}_{lr_scheduler}{lr:.0e}_weight{loss_weight:.0e}/int{n_int3D}"
Evaluate_save_path=f"Grain3D/Results_Adam/{model_shape}_mesh{mesh_points}_Net{depth}x{input_size}-{hidden_size}-{output_size}_{lr_scheduler}{lr:.0e}_weight{loss_weight:.0e}/int{n_int3D}/{model_shape}_NeoHook"