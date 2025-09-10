import torch
import torch.nn as nn
from typing import Optional
import numpy as np

__all__ = ['SimpleSolarPanel']

# 天文常量 (单位: 米)
AU = 149597870700.0  # 1 AU = 149,597,870,700 米
SOLAR_FLUX_EARTH = 1367.0  # 地球附近的太阳通量 (W/m²)


class SimpleSolarPanel(nn.Module):
    """
    PyTorch 实现的太阳能电池板模型

    功能:
    - 计算太阳相对于航天器的位置
    - 计算太阳在航天器本体坐标系中的方向
    - 计算电池板的投影面积
    - 计算太阳距离因子
    - 计算电池板产生的净功率

    参数:
        nHat_B: 面板法向量 (在航天器本体坐标系中)
        panel_area: 面板面积 (m²)
        panel_efficiency: 面板效率 (0-1)
    """

    def __init__(
            self,
            nHat_B: torch.Tensor,
            panel_area: float,
            panel_efficiency: float
    ) -> None:
        """
        初始化太阳能电池板

        参数:
            nHat_B: 面板法向量 (在航天器本体坐标系中), shape [3]
            panel_area: 面板面积 (m²)
            panel_efficiency: 面板效率 (0-1)
        """
        super().__init__()

        # 验证输入参数
        if panel_area <= 0:
            raise ValueError("panel_area must be a positive value")
        if panel_efficiency <= 0 or panel_efficiency > 1:
            raise ValueError("panel_efficiency must be in (0, 1]")
        if torch.norm(nHat_B) < 1e-5:
            raise ValueError("nHat_B must be a non-zero vector")

        # 注册参数
        self.register_buffer('nHat_B', nHat_B / torch.norm(nHat_B))  # 单位化法向量
        self.panel_area = panel_area
        self.panel_efficiency = panel_efficiency

        # 初始化状态变量
        self.shadow_factor: torch.Tensor = torch.tensor(1.0)
        self.projected_area: torch.Tensor = torch.tensor(0.0)
        self.sun_distance_factor: torch.Tensor = torch.tensor(0.0)

    def forward(
            self,
            r_BN_N: torch.Tensor,
            r_SN_N: torch.Tensor,
            sigma_BN: torch.Tensor,
            shadow_factor: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算太阳能电池板的输出功率

        参数:
            r_BN_N: 航天器位置 (在惯性系中), shape [batch_size, 3]
            r_SN_N: 太阳位置 (在惯性系中), shape [batch_size, 3]
            sigma_BN: 航天器姿态 (MRP参数), shape [batch_size, 3]
            shadow_factor: 日食因子 (可选), shape [batch_size]

        返回:
            net_power: 净输出功率, shape [batch_size]
        """
        # 处理日食因子
        if shadow_factor is None:
            shadow_factor = torch.ones(r_BN_N.shape[0], device=r_BN_N.device)

        # 计算太阳相关数据
        self._compute_sun_data(r_BN_N, r_SN_N, sigma_BN)

        # 计算太阳功率因子
        sun_power_factor = SOLAR_FLUX_EARTH * self.sun_distance_factor * shadow_factor

        # 计算净功率
        net_power = sun_power_factor * self.projected_area * self.panel_efficiency

        return net_power

    def _compute_sun_data(
            self,
            r_BN_N: torch.Tensor,
            r_SN_N: torch.Tensor,
            sigma_BN: torch.Tensor
    ) -> None:
        """
        计算太阳相关数据

        参数:
            r_BN_N: 航天器位置 (在惯性系中), shape [batch_size, 3]
            r_SN_N: 太阳位置 (在惯性系中), shape [batch_size, 3]
            sigma_BN: 航天器姿态 (MRP参数), shape [batch_size, 3]
        """
        # 计算太阳相对于航天器的向量
        r_SB_N = r_SN_N - r_BN_N

        # 计算太阳方向的单位向量 (在惯性系中)
        r_SB_norm = torch.norm(r_SB_N, dim=1, keepdim=True)
        sHat_N = r_SB_N / (r_SB_norm + 1e-10)  # 添加小值防止除零

        # 计算惯性系到本体坐标系的旋转矩阵 (从MRP)
        dcm_BN = self._mrp_to_dcm(sigma_BN)

        # 将太阳方向转换到本体坐标系
        sHat_B = torch.bmm(dcm_BN, sHat_N.unsqueeze(-1)).squeeze(-1)

        # 计算投影面积
        cos_theta = torch.sum(sHat_B * self.nHat_B, dim=1)
        projected_area = self.panel_area * cos_theta
        projected_area = torch.clamp(projected_area, min=0.0)  # 负值设为零

        # 计算太阳距离因子 - 修复了括号问题
        sun_distance_factor = (AU ** 2) / (r_SB_norm.squeeze(1) ** 2)

        # 更新状态
        self.projected_area = projected_area
        self.sun_distance_factor = sun_distance_factor

    def _mrp_to_dcm(self, sigma: torch.Tensor) -> torch.Tensor:
        """
        将修正罗德里格斯参数(MRP)转换为方向余弦矩阵(DCM)

        参数:
            sigma: MRP参数, shape [batch_size, 3]

        返回:
            dcm: 方向余弦矩阵, shape [batch_size, 3, 3]
        """
        batch_size = sigma.shape[0]
        sigma_squared = torch.sum(sigma ** 2, dim=1, keepdim=True)

        # 叉乘矩阵
        sigma_tilde = torch.zeros(batch_size, 3, 3, device=sigma.device)
        sigma_tilde[:, 0, 1] = -sigma[:, 2]
        sigma_tilde[:, 0, 2] = sigma[:, 1]
        sigma_tilde[:, 1, 0] = sigma[:, 2]
        sigma_tilde[:, 1, 2] = -sigma[:, 0]
        sigma_tilde[:, 2, 0] = -sigma[:, 1]
        sigma_tilde[:, 2, 1] = sigma[:, 0]
        """
        [ 0     -σ₃    σ₂ ]
        [ σ₃     0     -σ₁]
        [ -σ₂    σ₁     0 ]
        """
        # 计算DCM
        denominator = (1 + sigma_squared) ** 2
        term1 = 8 * torch.bmm(sigma_tilde, sigma_tilde)
        term2 = 4 * (1 - sigma_squared) * sigma_tilde

        # 批量单位矩阵
        eye = torch.eye(3, device=sigma.device).unsqueeze(0).repeat(batch_size, 1, 1)

        dcm = eye + (term1 - term2) / denominator.view(-1, 1, 1)

        return dcm

    def reset(self) -> None:
        """重置状态变量"""
        self.shadow_factor = torch.tensor(1.0)
        self.projected_area = torch.tensor(0.0)
        self.sun_distance_factor = torch.tensor(0.0)


# ====================== 测试代码 ======================
def test_mrp_to_dcm():
    """测试 MRP 到 DCM 转换的正确性"""
    print("\n=== 测试 MRP 到 DCM 转换 ===")
    
    # 创建测试实例
    nHat_B = torch.tensor([0.0, 0.0, 1.0])
    panel = SimpleSolarPanel(nHat_B, panel_area=1.0, panel_efficiency=1.0)
    
    # 测试用例：零旋转 (单位矩阵)
    sigma_zero = torch.tensor([[0.0, 0.0, 0.0]])
    dcm_zero = panel._mrp_to_dcm(sigma_zero)
    print("零旋转 DCM:\n", dcm_zero[0].detach().numpy())
    
    # 测试用例：绕 Z 轴旋转 180 度
    sigma_z180 = torch.tensor([[0.0, 0.0, 1.0]])
    dcm_z180 = panel._mrp_to_dcm(sigma_z180)
    print("\n绕 Z 轴 180 度旋转 DCM:\n", dcm_z180[0].detach().numpy())
    
    # 验证旋转矩阵性质
    identity = torch.eye(3)
    for i, dcm in enumerate([dcm_zero, dcm_z180]):
        # 正交性检查: C^T * C = I
        ortho_check = torch.bmm(dcm.transpose(1, 2), dcm)
        ortho_error = torch.norm(ortho_check - identity)
        print(f"\nDCM {i+1} 正交性误差: {ortho_error.item():.6f}")
        
        # 行列式应为1 (特殊正交群)
        det = torch.det(dcm)
        print(f"DCM {i+1} 行列式: {det.item():.6f}")


def test_solar_panel():
    """测试太阳能电池板功能"""
    print("\n=== 测试太阳能电池板 ===")
    
    # 创建太阳能电池板实例
    nHat_B = torch.tensor([0.0, 0.0, 1.0])  # 面板法向量 (Z轴方向)
    panel = SimpleSolarPanel(nHat_B, panel_area=1.0, panel_efficiency=0.5)
    
    # 测试用例1: 面板正对太阳 (最大功率)
    print("\n测试用例1: 面板正对太阳")
    r_BN_N = torch.tensor([[0.0, 0.0, 0.0]])  # 航天器位置
    r_SN_N = torch.tensor([[0.0, 0.0, AU]])   # 太阳位置 (沿Z轴，距离1AU)
    sigma_BN = torch.tensor([[0.0, 0.0, 0.0]])  # 零旋转姿态
    
    power = panel(r_BN_N, r_SN_N, sigma_BN)
    expected_power = SOLAR_FLUX_EARTH * 1.0 * 0.5
    print(f"计算功率: {power.item():.2f} W, 期望功率: {expected_power:.2f} W")
    assert torch.isclose(power, torch.tensor([expected_power])), "测试用例1失败"
    
    # 测试用例2: 面板背对太阳 (功率应为零)
    print("\n测试用例2: 面板背对太阳")
    r_SN_N = torch.tensor([[0.0, 0.0, -AU]])  # 太阳在负Z轴方向
    power = panel(r_BN_N, r_SN_N, sigma_BN)
    print(f"计算功率: {power.item():.2f} W (应为接近0)")
    assert torch.isclose(power, torch.tensor([0.0])), "测试用例2失败"
    
    # 测试用例3: 面板与太阳成45度角
    print("\n测试用例3: 面板与太阳成45度角")
    r_SN_N = torch.tensor([[AU, 0.0, AU]])  # 太阳在XZ平面45度方向
    power = panel(r_BN_N, r_SN_N, sigma_BN)
    expected_power = SOLAR_FLUX_EARTH * 0.5 * np.cos(np.pi/4) * 0.5
    print(f"计算功率: {power.item():.2f} W, 期望功率: {expected_power:.2f} W")
    assert torch.isclose(power, torch.tensor([expected_power])), "测试用例3失败"
    
    # 测试用例4: 日食情况
    print("\n测试用例4: 日食情况")
    r_SN_N = torch.tensor([[0.0, 0.0, AU]])  # 太阳在正Z轴方向
    shadow_factor = torch.tensor([0.5])  # 50%阴影
    power = panel(r_BN_N, r_SN_N, sigma_BN, shadow_factor)
    expected_power = SOLAR_FLUX_EARTH * 1.0 * 0.5 * 0.5
    print(f"计算功率: {power.item():.2f} W, 期望功率: {expected_power:.2f} W")
    assert torch.isclose(power, torch.tensor([expected_power])), "测试用例4失败"
    
    # 测试用例5: 批处理功能
    print("\n测试用例5: 批处理功能")
    batch_size = 3
    r_BN_N = torch.zeros(batch_size, 3)
    r_SN_N = torch.tensor([
        [0.0, 0.0, AU],         # 正对太阳
        [0.0, 0.0, -AU],        # 背对太阳
        [AU, 0.0, AU]           # 45度角
    ])
    sigma_BN = torch.zeros(batch_size, 3)
    shadow_factor = torch.tensor([1.0, 1.0, 0.8])
    
    powers = panel(r_BN_N, r_SN_N, sigma_BN, shadow_factor)
    expected_powers = torch.tensor([
        SOLAR_FLUX_EARTH * 1.0 * 0.5,  # 正对太阳
        0.0,                           # 背对太阳
        SOLAR_FLUX_EARTH * 0.5 * np.cos(np.pi/4) * 0.5 * 0.8  # 45度角+阴影
    ])
    print("批处理功率输出:", powers.detach().numpy())
    print("期望功率输出:", expected_powers.numpy())
    assert torch.allclose(powers, expected_powers), "测试用例5失败"
    
    print("\n所有太阳能电池板测试通过!")


def run_tests():
    """运行所有测试"""
    print("=" * 60)
    print("开始测试 SimpleSolarPanel 模块")
    print("=" * 60)
    
    # 运行 MRP 到 DCM 转换测试
    test_mrp_to_dcm()
    
    # 运行太阳能电池板功能测试
    test_solar_panel()
    
    print("\n" + "=" * 60)
    print("所有测试通过!")
    print("=" * 60)


if __name__ == "__main__":
    # 当直接运行此文件时执行测试
    run_tests()
