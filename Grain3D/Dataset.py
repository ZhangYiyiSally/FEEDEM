import numpy as np
import torch
import meshio
import matplotlib.pyplot as plt

class Dataset:
    def __init__(self, model_scale):
        self.dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model_scale=model_scale

    def domain(self, mesh) -> torch.Tensor:
        self.mesh=mesh
        AllPoint_idx=mesh.cells_dict['tetra']
        Tetra_coord=mesh.points[AllPoint_idx]
        Tetra_coord=Tetra_coord*self.model_scale # 缩放模型
        Tetra_coord=torch.tensor(Tetra_coord, dtype=torch.float32).to(self.dev)
        return Tetra_coord
        
    def bc_Dirichlet(self, marker:str) -> torch.Tensor:
        DirCell_idx=self.mesh.cell_sets_dict[marker]['triangle']
        DirPoint_idx=self.mesh.cells_dict['triangle'][DirCell_idx]
        Dir_Triangle_coord = self.mesh.points[DirPoint_idx]
        Dir_Triangle_coord=Dir_Triangle_coord*self.model_scale # 缩放模型
        Dir_Triangle_coord=torch.tensor(Dir_Triangle_coord, dtype=torch.float32).to(self.dev)
        return Dir_Triangle_coord
    
    def bc_Pressure(self, marker:str) -> torch.Tensor:
        PreCell_idx=self.mesh.cell_sets_dict[marker]['triangle']
        PrePoint_idx=self.mesh.cells_dict['triangle'][PreCell_idx]
        Pre_Triangle_coord = self.mesh.points[PrePoint_idx]
        Pre_Triangle_coord=Pre_Triangle_coord*self.model_scale # 缩放模型
        Pre_Triangle_coord=torch.tensor(Pre_Triangle_coord, dtype=torch.float32).to(self.dev)
        return Pre_Triangle_coord
    
    def bc_Symmetry(self, marker:str) -> torch.Tensor:
        SymCell_idx=self.mesh.cell_sets_dict[marker]['triangle']
        SymPoint_idx=self.mesh.cells_dict['triangle'][SymCell_idx]
        Sym_Triangle_coord = self.mesh.points[SymPoint_idx]
        Sym_Triangle_coord=Sym_Triangle_coord*self.model_scale # 缩放模型
        Sym_Triangle_coord=torch.tensor(Sym_Triangle_coord, dtype=torch.float32).to(self.dev)
        return Sym_Triangle_coord
    
    def line_Singular(self, marker:str) -> torch.Tensor:
        SingularCell_idx=self.mesh.cell_sets_dict[marker]['line']
        SingularPoint_idx=self.mesh.cells_dict['line'][SingularCell_idx]
        Singular_Line_coord = self.mesh.points[SingularPoint_idx]
        Singular_Line_coord=Singular_Line_coord*self.model_scale # 缩放模型
        Singular_Line_coord=torch.tensor(Singular_Line_coord, dtype=torch.float32).to(self.dev)
        return Singular_Line_coord
    
if __name__ == '__main__':  # 测试边界条件是否设置正确
    mesh = meshio.read("DEFEM3D/Grain3D/mesh/0.msh", file_format="gmsh")
    data = Dataset()
    dom = data.domain(mesh)
    Dir_coord = data.bc_Dirichlet('OutSurface')
    Pre_coord = data.bc_Pressure('InSurface')
    Sym_coord = data.bc_Symmetry('Symmetry')

    print("全域四面体单元个数*单元顶点个数*坐标方向:", dom.shape)
    print("Dirichlet边界三角形单元个数*单元顶点个数*坐标方向:", Dir_coord.shape)
    print("Pressure 边界三角形单元个数*单元顶点个数*坐标方向:", Pre_coord.shape)
    print("Symmetry边界三角形单元个数*单元顶点个数*坐标方向:", Sym_coord.shape)
    
    