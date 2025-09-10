import torch
from typing import TypedDict, Optional

# 遵循规范，从 satsim.architecture 导入基础组件
from satsim.architecture.module import Module
from satsim.architecture.timer import Timer

# 遵循规范，在文件开头定义 __all__ 属性
__all__ = ['SimpleSolarPanel']

# 从 C++ 代码中移植过来的天文常量
ASTRONOMICAL_UNIT = 149597870.7 * 1000.0  # 转换为米
SOLAR_FLUX_AT_EARTH = 1367.0             # 地球附近的太阳通量 (W/m²)


class SolarPanelStateDict(TypedDict):
    """
    太阳能电池板模组的状态空间 (State Space) [cite: 4]。

    属性:
        projected_area: 面板在太阳光方向上的有效投影面积 (m^2)。
        sun_distance_factor: 因日地距离变化引起的光通量缩放因子。
    """
    projected_area: torch.Tensor
    sun_distance_factor: torch.Tensor


class SimpleSolarPanel(Module[SolarPanelStateDict]):
    """
    一个符合 Satsim 开发规范的、基于 PyTorch 的太阳能电池板模型。

    该模组根据航天器的位置、姿态以及太阳的位置，计算太阳能电池板的瞬时输出功率。
    它被设计为可微分且支持批处理。
    """

    def __init__(
        self,
        *args,
        timer: Timer,
        panel_normal_in_body_frame: torch.Tensor,
        panel_area: float,
        panel_efficiency: float,
        **kwargs,
    ) -> None:
        """
        初始化 SimpleSolarPanel 模组。

        C++ 版本中的 setPanelParameters 和 customReset 中的验证逻辑被合并到此处 。
        
        参数:
            timer: 仿真计时器，必须作为关键字参数传入 [cite: 9]。
            panel_normal_in_body_frame: 面板法向量 (在航天器本体坐标系中)，shape [3]。
            panel_area: 面板面积 (m²)。
            panel_efficiency: 面板效率 (0 到 1 之间)。
        """
        super().__init__(*args, timer=timer, **kwargs)

        # 参数验证，逻辑来自 C++ 的 customReset
        if panel_area <= 0:
            raise ValueError("panel_area 必须是一个正数。")
        if not (0 < panel_efficiency <= 1):
            raise ValueError("panel_efficiency 必须在 (0, 1] 区间内。")
        if torch.norm(panel_normal_in_body_frame) < 1e-5:
            raise ValueError("panel_normal_in_body_frame 必须是一个非零向量。")

        # 存储已验证和处理过的配置参数
        # 归一化法向量，逻辑来自 C++ 的 customReset
        self.panel_normal_in_body_frame = (
            panel_normal_in_body_frame / torch.norm(panel_normal_in_body_frame)
        )
        self.panel_area = panel_area
        self.panel_efficiency = panel_efficiency

    def reset(self) -> SolarPanelStateDict:
        """
        重置并返回模块的初始状态字典 [cite: 5]。
        """
        # 对于批处理，这些值将在 forward 首次调用时被正确地塑造 (shape)
        return {
            'projected_area': torch.tensor(0.0),
            'sun_distance_factor': torch.tensor(0.0),
        }

    def forward(
        self,
        state_dict: SolarPanelStateDict,
        spacecraft_position_inertial: torch.Tensor,
        sun_position_inertial: torch.Tensor,
        spacecraft_attitude_mrp: torch.Tensor,
        shadow_factor: Optional[torch.Tensor] = None,
    ) -> tuple[SolarPanelStateDict, tuple[torch.Tensor]]:
        """
        执行一步功率计算 [cite: 6]。

        此函数整合了 C++ 版本中 customReadMessages, computeSunData, 和 evaluatePowerModel 的功能。
        输入直接作为参数传递，而不是通过消息读取 [cite: 50]。

        参数:
            state_dict: 模组的当前状态字典 [cite: 6]。
            spacecraft_position_inertial: 航天器位置 (在惯性系中), shape [batch_size, 3]。
            sun_position_inertial: 太阳位置 (在惯性系中), shape [batch_size, 3]。
            spacecraft_attitude_mrp: 航天器姿态 (MRP参数), shape [batch_size, 3]。
            shadow_factor: 日食因子 (可选), shape [batch_size]。

        返回:
            一个元组，包含 (更新后的状态字典, (净输出功率,)) [cite: 6]。
        """
        batch_size = spacecraft_position_inertial.shape[0]
        device = spacecraft_position_inertial.device

        # 处理可选的 shadow_factor 输入，逻辑来自 C++ 的 customReadMessages
        if shadow_factor is None:
            shadow_factor = torch.ones(batch_size, device=device)

        # --- 核心计算逻辑，移植自 computeSunData ---
        
        # 计算太阳相对于航天器的矢量 (惯性系)
        sun_vector_inertial = sun_position_inertial - spacecraft_position_inertial
        sun_distance = torch.norm(sun_vector_inertial, dim=1, keepdim=True)
        
        # 归一化得到单位矢量，并防止除零
        sun_direction_inertial = sun_vector_inertial / (sun_distance + 1e-10)

        # 计算从惯性系到本体系的旋转矩阵 (DCM)
        dcm_body_wrt_inertial = self._mrp_to_dcm(spacecraft_attitude_mrp)

        # 将太阳方向矢量转换到本体系
        sun_direction_body = torch.bmm(
            dcm_body_wrt_inertial, sun_direction_inertial.unsqueeze(-1)
        ).squeeze(-1)

        # 计算投影面积，使用 torch.clamp 替换 C++ 的 if 语句，以支持批处理和可微性 
        cos_theta = torch.sum(sun_direction_body * self.panel_normal_in_body_frame, dim=1)
        projected_area = self.panel_area * torch.clamp(cos_theta, min=0.0)

        # 计算太阳距离因子
        sun_distance_factor = (ASTRONOMICAL_UNIT ** 2) / (sun_distance.squeeze(-1) ** 2 + 1e-10)

        # --- 功率评估，移植自 evaluatePowerModel ---
        sun_power_factor = (
            SOLAR_FLUX_AT_EARTH * sun_distance_factor * shadow_factor
        )
        net_power = sun_power_factor * projected_area * self.panel_efficiency
        
        # 更新并返回状态字典和输出
        state_dict['projected_area'] = projected_area
        state_dict['sun_distance_factor'] = sun_distance_factor

        return state_dict, (net_power,)

    @staticmethod
    def _mrp_to_dcm(sigma: torch.Tensor) -> torch.Tensor:
        """
        将修正罗德里格斯参数(MRP)转换为方向余弦矩阵(DCM)。
        这是一个纯工具函数，因此设为静态方法。
        """
        sigma_sq = torch.sum(sigma**2, dim=1, keepdim=True)
        s_tilde = torch.zeros(sigma.shape[0], 3, 3, device=sigma.device)
        s_tilde[:, 0, 1] = -sigma[:, 2]
        s_tilde[:, 0, 2] = sigma[:, 1]
        s_tilde[:, 1, 0] = sigma[:, 2]
        s_tilde[:, 1, 2] = -sigma[:, 0]
        s_tilde[:, 2, 0] = -sigma[:, 1]
        s_tilde[:, 2, 1] = sigma[:, 0]

        identity = torch.eye(3, device=sigma.device).expand(sigma.shape[0], 3, 3)
        numerator = 8 * torch.bmm(s_tilde, s_tilde) - 4 * (1 - sigma_sq.view(-1, 1, 1)) * s_tilde
        denominator = (1 + sigma_sq) ** 2
        
        dcm = identity + numerator / denominator.view(-1, 1, 1)
        return dcm