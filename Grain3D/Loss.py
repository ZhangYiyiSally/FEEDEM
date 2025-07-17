from torch.autograd import grad
import torch
import numpy as np
import torch.nn as nn
import meshio
import time
from scipy.interpolate import griddata
import Config as cfg
from Network import ResNet
from Dataset import Dataset
from GaussIntegral import GaussIntegral


class Loss:
    def __init__(self, model):
        self.dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = model
        pass
        
    def loss_function(self, Tetra_coord: torch.Tensor, Dir_Triangle_coord: torch.Tensor, Pre_Triangle_coord: torch.Tensor, Sym_Triangle_coord: torch.Tensor, Singular_Line_coord: torch.Tensor) -> torch.Tensor:
        self.Tetra_coord=Tetra_coord
        self.Dir_Triangle_coord=Dir_Triangle_coord
        self.Pre_Triangle_coord=Pre_Triangle_coord
        self.Sym_Triangle_coord=Sym_Triangle_coord
        self.Singular_Line_coord=Singular_Line_coord
        integral=GaussIntegral()
        integral_strainenergy=integral.Integral3D(self.StrainEnergy, cfg.n_int3D, Tetra_coord)
        integral_externalwork=integral.Integral2D(self.ExternalWork, cfg.n_int2D, Pre_Triangle_coord)
        # integral_boundaryloss=integral.Integral2D(self.BoundaryLoss, 3, Dir_Triangle_coord)
        integral_boundaryloss=self.BoundaryLoss(Dir_Triangle_coord, Sym_Triangle_coord)

        energy_loss = integral_strainenergy - integral_externalwork
        loss = energy_loss + cfg.loss_weight*integral_boundaryloss
        
        # print("Internal Energy:", integral_strainenergy.item())
        # print("External Work:", integral_externalwork.item())
        # print("Boundary Loss:", integral_boundaryloss.item())
        return loss, energy_loss, cfg.loss_weight*integral_boundaryloss, self.max_grad_norm, self.max_grad_scale_norm

    def GetU(self, xyz_field: torch.Tensor) -> torch.Tensor:
        u = self.model(xyz_field)
        return u
    
    def StrainEnergy(self, xyz_field: torch.Tensor) -> torch.Tensor:
        E=cfg.E
        nu=cfg.nu
        lam = (E * nu) / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))

        xyz_field.requires_grad = True  # 为了计算位移场的梯度，这里需要设置为True
        pred_u = self.GetU(xyz_field)

        # 计算所有点到奇异边线上点的距离，并找到最小距离
        Line_point=torch.cat([self.Singular_Line_coord[:, 0, :], self.Singular_Line_coord[-1:, -1, :]], dim=0)  #[num_point, 3]
        distances = torch.cat([torch.norm(xyz_field - point, dim=-1, keepdim=True) for point in Line_point], dim=-1)  #[N, 4, num_point]
        min_distances, _ = torch.min(distances, dim=-1, keepdim=True)  #[N, 4, 1]

        # 1. 位移梯度正则化参数（可配置）
        GRAD_CLIP_RADIUS = cfg.GRAD_CLIP_RADIUS    # 受影响的半径（特征长度的5-10%）
        MAX_GRAD_NORM = cfg.MAX_GRAD_NORM        # 最大允许梯度范数

        # 2. 计算梯度，并添加梯度范数约束
        duxdxyz = grad(pred_u[:, :, 0], xyz_field, torch.ones_like(pred_u[:, :, 0]), create_graph=True, retain_graph=True)[0] # [N, 4, 3]
        duydxyz = grad(pred_u[:, :, 1], xyz_field, torch.ones_like(pred_u[:, :, 1]), create_graph=True, retain_graph=True)[0]
        duzdxyz = grad(pred_u[:, :, 2], xyz_field, torch.ones_like(pred_u[:, :, 2]), create_graph=True, retain_graph=True)[0]
        grad_u = torch.stack([duxdxyz, duydxyz, duzdxyz], dim=-2)  # [N,4,3,3]
        grad_norm = torch.norm(grad_u, dim=-1)  # [N,4,3]
        
        # 梯度裁剪 (仅在反向传播时影响)
        clipped_grad_norm = torch.where(
            grad_norm > MAX_GRAD_NORM,
            MAX_GRAD_NORM + (grad_norm - MAX_GRAD_NORM).detach(),
            grad_norm
        )
        
        # 创建权重掩码：距离越近权重越小
        reg_mask = torch.exp(-min_distances / GRAD_CLIP_RADIUS)  # [N, 4, 1]
        # 应用正则化：混合原始梯度和裁剪后梯度
        regularized_grad_norm = (1 - reg_mask) * grad_norm + reg_mask * clipped_grad_norm  # [N, 4, 3]

        # 重构梯度张量（保持计算图）
        scale_factor = (regularized_grad_norm / (grad_norm + 1e-8)).unsqueeze(-2)  # [N, 4, 1, 3]
        duxdxyz_scale=duxdxyz*scale_factor[..., 0] # [N, 4, 3]
        duydxyz_scale=duydxyz*scale_factor[..., 1]
        duzdxyz_scale=duzdxyz*scale_factor[..., 2]
        grad_u_scale = torch.stack([duxdxyz_scale, duydxyz_scale, duzdxyz_scale], dim=-2)  # [N, 4, 3, 3]
        self.max_grad_norm=torch.max(grad_norm)
        self.max_grad_scale_norm=torch.max(torch.norm(grad_u_scale, dim=-1))

        # 3. 计算变形梯度张量 F = I + ∇u
        I = torch.eye(3, device=self.dev)  # [3,3]
        F = I + grad_u_scale  # [N,4,3,3]

        # 计算变形梯度的行列式 J = det(F)
        J = torch.det(F).unsqueeze(-1) # 变形梯度行列式[N,4,1]
        # J_safe=nn.functional.softplus(J) # 防止负体积单元
        I1=torch.sum(F**2, dim=[-2, -1]).unsqueeze(-1) # [N,4,1]
        EPS=1e-8

        strainenergy_tmp = 0.5 * lam * (torch.log(J + EPS) * torch.log(J + EPS)) - mu * torch.log(J + EPS) + 0.5 * mu * (I1 - 3)
        strainenergy = strainenergy_tmp[:, :, 0] # [N, 4]
    
        return strainenergy
    
    def ExternalWork(self, pressure_field: torch.Tensor) -> torch.Tensor:
        # 计算压力边界法向量
        edge1 = self.Pre_Triangle_coord[:, 1] - self.Pre_Triangle_coord[:, 0]  # v0 -> v1
        edge2 = self.Pre_Triangle_coord[:, 2] - self.Pre_Triangle_coord[:, 0]  # v0 -> v2
        normals = torch.cross(edge1, edge2, dim=1)
        normals_unity = normals / torch.norm(normals, dim=1, keepdim=True)
        # 确保法向量与半径方向一致
        # radial_yz = self.Pre_Triangle_coord[:, 0, 1:3] # x轴为旋转轴的半径方向
        # radial = torch.cat([torch.zeros_like(radial_yz[:, 0:1]),  radial_yz], dim=1)
        # radial_xy = self.Pre_Triangle_coord[:, 0, 0:2] # z轴为旋转轴的半径方向
        # radial=torch.cat([radial_xy, torch.zeros_like(radial_xy[:, 0:1])], dim=1)  # [N, 3]
        # dot = torch.sum(normals * radial, dim=1)  # 计算点积 [N]
        # normals_unity[dot > 0] *= -1  # 若点积为正，反转法向量方向，因为压力向量前面乘了负号

        #"""计算外力做功"""
        u_pred = self.GetU(pressure_field)
        #计算每个单元的高斯积分点的压力向量
        p= cfg.Pre_value*normals_unity.unsqueeze(1).expand(-1, u_pred.size(1), -1)

        external_work = torch.sum( -u_pred * p, dim=-1)
        return external_work
    
    def BoundaryLoss(self, dirichlet_field: torch.Tensor, symmetry_field: torch.Tensor) -> torch.Tensor:
        #"""计算边界条件损失函数"""
        # 1.计算dirichlet边界条件的损失函数
        u_dir_pred = self.GetU(dirichlet_field)

        u_dir_value=torch.tensor(cfg.Dir_u, dtype=torch.float32).to(self.dev)
        u_dir_true = torch.zeros_like(u_dir_pred)
        u_dir_true[:, :, :]=u_dir_value

        mes_loss = nn.MSELoss(reduction='sum')
        dir_loss = mes_loss(u_dir_pred, u_dir_true)
        
        # 2.计算对称边界条件的损失函数
        u_sym_pred = self.GetU(symmetry_field)
        ## 计算对称边界的法向量
        edge1 = symmetry_field[:, 1] - symmetry_field[:, 0]  # v0 -> v1
        edge2 = symmetry_field[:, 2] - symmetry_field[:, 0]  # v0 -> v2
        normals = torch.cross(edge1, edge2, dim=1)
        normals_unity = normals / torch.norm(normals, dim=1, keepdim=True)
        normals_unity=normals_unity.unsqueeze(1).expand(-1, u_sym_pred.size(1), -1)
        ## 计算对称边界的法向位移
        u_sym_normal = torch.sum(u_sym_pred * normals_unity, dim=-1)
        u_sym_true = torch.zeros_like(u_sym_normal)
        sym_loss = mes_loss(u_sym_normal, u_sym_true)

        boundary_loss = dir_loss + sym_loss
        return boundary_loss

if __name__ =='__main__':
    start_time=time.time()
    torch.manual_seed(2025)
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dem=ResNet(input_size=3, hidden_size=64, output_size=3, depth=4).to(dev)

    mesh = meshio.read("DEFEM3D/Grain3D/cylinder.msh", file_format="gmsh")

    # 提取四面体单元顶点坐标数组：m*4*3（m为四面体单元个数，4为四面体的四个顶点，3为三维空间坐标）
    AllPoint_idx=mesh.cells_dict['tetra']
    Tetra_coord=mesh.points[AllPoint_idx]
    Tetra_coord=torch.tensor(Tetra_coord, dtype=torch.float32).to(dev)

    # 提取狄利克雷边界条件Dirichlet
    DirCell_idx=mesh.cell_sets_dict['OutSurface']['triangle']
    DirPoint_idx=mesh.cells_dict['triangle'][DirCell_idx]
    Dir_coord = mesh.points[DirPoint_idx]
    Dir_coord=torch.tensor(Dir_coord, dtype=torch.float32).to(dev)

    # 提取压力边界条件
    PreCell_idx=mesh.cell_sets_dict['InSurface']['triangle']
    PrePoint_idx=mesh.cells_dict['triangle'][PreCell_idx]
    Pre_coord = mesh.points[PrePoint_idx]
    Pre_coord=torch.tensor(Pre_coord, dtype=torch.float32).to(dev)

    # 提取对称边界条件
    SymCell_idx=mesh.cell_sets_dict['Symmetry']['triangle']
    SymPoint_idx=mesh.cells_dict['triangle'][SymCell_idx]
    Sym_coord = mesh.points[SymPoint_idx]
    Sym_coord=torch.tensor(Sym_coord, dtype=torch.float32).to(dev)

    loss=Loss(dem)
    loss_value=loss.loss_function(Tetra_coord, Dir_coord, Pre_coord, Sym_coord)
    end_time=time.time()
    print('损失函数值为：', loss_value.item())
    print("计算时间:", end_time-start_time, "s")