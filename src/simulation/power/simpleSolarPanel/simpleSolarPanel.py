import torch
import torch.nn as nn
from typing import Optional

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