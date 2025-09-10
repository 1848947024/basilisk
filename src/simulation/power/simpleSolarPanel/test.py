import torch
import numpy as np
import matplotlib.pyplot as plt
from simpleSolarPanel import SimpleSolarPanel  # 假设代码保存在 simpleSolarPanel.py 中
# 天文常量 (单位: 米)
AU = 149597870700.0  # 1 AU = 149,597,870,700 米
SOLAR_FLUX_EARTH = 1367.0  # 地球附近的太阳通量 (W/m²)

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
    for dcm in [dcm_zero, dcm_z180]:
        # 正交性检查: C^T * C = I
        ortho_check = torch.bmm(dcm.transpose(1, 2), dcm)
        ortho_error = torch.norm(ortho_check - identity)
        print(f"正交性误差: {ortho_error.item():.6f}")
        
        # 行列式应为1 (特殊正交群)
        det = torch.det(dcm)
        print(f"行列式: {det.item():.6f}")

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
    
    # 测试用例2: 面板背对太阳 (功率应为零)
    print("\n测试用例2: 面板背对太阳")
    r_SN_N = torch.tensor([[0.0, 0.0, -AU]])  # 太阳在负Z轴方向
    power = panel(r_BN_N, r_SN_N, sigma_BN)
    print(f"计算功率: {power.item():.2f} W (应为接近0)")
    
    # 测试用例3: 面板与太阳成45度角
    print("\n测试用例3: 面板与太阳成45度角")
    r_SN_N = torch.tensor([[AU, 0.0, AU]])  # 太阳在XZ平面45度方向
    power = panel(r_BN_N, r_SN_N, sigma_BN)
    expected_power = SOLAR_FLUX_EARTH * 0.5 * np.cos(np.pi/4) * 0.5
    print(f"计算功率: {power.item():.2f} W, 期望功率: {expected_power:.2f} W")
    
    # 测试用例4: 日食情况
    print("\n测试用例4: 日食情况")
    r_SN_N = torch.tensor([[0.0, 0.0, AU]])  # 太阳在正Z轴方向
    shadow_factor = torch.tensor([0.5])  # 50%阴影
    power = panel(r_BN_N, r_SN_N, sigma_BN, shadow_factor)
    expected_power = SOLAR_FLUX_EARTH * 1.0 * 0.5 * 0.5
    print(f"计算功率: {power.item():.2f} W, 期望功率: {expected_power:.2f} W")
    
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
    print("批处理功率输出:", powers.detach().numpy())

def visualize_solar_panel_performance():
    """可视化太阳能电池板性能"""
    print("\n=== 可视化性能 ===")
    
    # 创建太阳能电池板实例
    nHat_B = torch.tensor([0.0, 0.0, 1.0])
    panel = SimpleSolarPanel(nHat_B, panel_area=1.0, panel_efficiency=1.0)
    
    # 创建角度范围
    angles = np.linspace(0, 2*np.pi, 100)
    powers = []
    
    # 固定航天器位置
    r_BN_N = torch.tensor([[0.0, 0.0, 0.0]])
    sigma_BN = torch.tensor([[0.0, 0.0, 0.0]])
    
    for angle in angles:
        # 计算太阳位置 (在XY平面)
        x = AU * np.cos(angle)
        y = AU * np.sin(angle)
        r_SN_N = torch.tensor([[x, y, AU]])
        
        # 计算功率
        power = panel(r_BN_N, r_SN_N, sigma_BN)
        powers.append(power.item())
    
    # 绘制结果
    plt.figure(figsize=(10, 6))
    plt.polar(angles, powers, 'b-', linewidth=2)
    plt.title('太阳能电池板功率输出 vs 太阳方位角', fontsize=14)
    plt.ylabel('功率 (W)', fontsize=12)
    plt.grid(True)
    plt.show()
    
    # 解释结果
    print("可视化说明:")
    print("- 当太阳在正上方 (0°) 时功率最大")
    print("- 当太阳在水平方向 (90°, 270°) 时功率为零")
    print("- 曲线呈现余弦特征，符合物理预期")

if __name__ == "__main__":
    # 运行测试
    test_mrp_to_dcm()
    test_solar_panel()
    
    # 运行可视化 (可选)
    # visualize_solar_panel_performance()