//
// simpleSolarPanel.cpp
// 太阳能电池板模型实现
// 核心功能：根据太阳位置、航天器状态和面板特性计算输出功率
// 关键算法：太阳矢量投影 + 距离衰减 + 阴影因子
//

#include <math.h>                   // 基础数学函数
#include "simpleSolarPanel.h"       // 类声明
#include "architecture/utilities/rigidBodyKinematics.h"  // 刚体运动学工具
#include "architecture/utilities/astroConstants.h"       // 天文常量 (AU, SOLAR_FLUX_EARTH)
#include "architecture/utilities/avsEigenSupport.h"      // Eigen向量支持
#include "architecture/utilities/avsEigenMRP.h"          // MRP姿态表示支持

// 构造函数：初始化默认参数
SimpleSolarPanel::SimpleSolarPanel() {
    // 初始化输出功率为0
    this->nodePowerOut = 0.0;
    
    // 面板参数初始化为无效值（强制用户设置）
    this->panelArea = -1;           // 面积必须>0
    this->panelEfficiency = -1;     // 效率必须>0
    
    // 法向量初始化为零向量（需用户设置）
    this->nHat_B.setZero();         // Eigen::Vector3d(0,0,0)
    
    // 阴影因子默认无阴影
    this->shadowFactor = 1;         // 1.0=完全光照, 0.0=完全阴影
    
    return;
}

// 析构函数：空实现（无动态资源需释放）
SimpleSolarPanel::~SimpleSolarPanel() {
    return;
}

/*! 
 * 重置函数：初始化或重置模块状态
 * @param CurrentClock 当前仿真时钟（未使用）
 * 
 * 功能：
 * 1. 验证关键参数有效性
 * 2. 归一化法向量
 * 3. 检查输入消息连接
 */
void SimpleSolarPanel::customReset(uint64_t CurrentClock) {
    // 重置阴影因子为默认值（无阴影）
    this->shadowFactor = 1.0;

    // === 参数校验 ===
    // 检查面板面积是否有效（必须正数）
    if (this->panelArea < 0.0) {
        bskLogger.bskLog(BSK_ERROR, "The panelArea must be a positive value");
    }
    
    // 检查面板效率是否有效（0.0~1.0）
    if (this->panelEfficiency < 0.0) {
        bskLogger.bskLog(BSK_ERROR, "The panelEfficiency variable must be a positive value");
    }
    
    // 检查法向量并归一化
    // 注意：使用0.1阈值避免浮点精度问题（非零检查）
    if (this->nHat_B.norm() > 0.1) {
        // 归一化法向量（确保为单位向量）
        this->nHat_B.normalize();
    } else {
        // 法向量过小或为零，报错
        bskLogger.bskLog(BSK_ERROR, "The nHat_B must be set to a non-zero vector");
    }
    
    // === 输入消息校验 ===
    // 检查太阳位置消息是否连接
    if (!this->sunInMsg.isLinked()) {
        bskLogger.bskLog(BSK_ERROR, "simpleSolarPanel.sunInMsg was not linked.");
    }
    
    // 检查航天器状态消息是否连接
    if (!this->stateInMsg.isLinked()) {
        bskLogger.bskLog(BSK_ERROR, "simpleSolarPanel.stateInMsg was not linked.");
    }

    return;
}

/*!
 * 消息读取函数：从输入消息获取数据
 * @return 总是返回true（未实现错误处理）
 * 
 * 功能：
 * 1. 读取太阳位置消息
 * 2. 读取航天器状态消息
 * 3. 读取日食消息（可选）
 */
bool SimpleSolarPanel::customReadMessages() {
    // 初始化消息数据（安全默认值）
    this->sunData = sunInMsg.zeroMsgPayload;    // 太阳位置 (0,0,0)
    this->stateCurrent = stateInMsg.zeroMsgPayload; // 航天器状态 (零状态)

    // 读取太阳位置消息（如果已连接）
    if (this->sunInMsg.isLinked()) {
        this->sunData = this->sunInMsg();  // 获取最新消息
    }
    
    // 读取航天器状态消息（如果已连接）
    if (this->stateInMsg.isLinked()) {
        this->stateCurrent = this->stateInMsg(); // 获取最新消息
    }
    
    // 读取日食消息（可选，影响阴影因子）
    if (this->sunEclipseInMsg.isLinked()) {
        EclipseMsgPayload sunVisibilityFactor; // 阴影因子消息
        sunVisibilityFactor = this->sunEclipseInMsg(); // 获取阴影因子
        this->shadowFactor = sunVisibilityFactor.shadowFactor; // 更新阴影因子
    }
    
    return true; // 总是返回成功（实际应添加错误检查）
}

/*!
 * 设置面板物理参数
 * @param nHat_B     面板法向量（航天器本体坐标系）
 * @param panelArea  面板面积 (m²)
 * @param panelEfficiency 面板效率 (0.0~1.0)
 * 
 * 注意：不会立即归一化法向量（在customReset中处理）
 */
void SimpleSolarPanel::setPanelParameters(Eigen::Vector3d nHat_B, 
                                         double panelArea, 
                                         double panelEfficiency) {
    this->nHat_B = nHat_B;             // 存储法向量
    this->panelArea = panelArea;        // 存储面积
    this->panelEfficiency = panelEfficiency; // 存储效率
    return;
}

/*!
 * 核心计算函数：计算太阳相关数据
 * 
 * 计算内容：
 * 1. 航天器-太阳矢量
 * 2. 太阳方向单位矢量（本体系）
 * 3. 面板投影面积
 * 4. 日距衰减因子
 */
void SimpleSolarPanel::computeSunData() {
    // 变量声明
    Eigen::Vector3d r_SB_N;         // 太阳到航天器的矢量 (惯性系)
    Eigen::Vector3d sHat_N;         // 太阳方向单位矢量 (惯性系)
    Eigen::Matrix3d dcm_BN;         // 惯性系到本体系的旋转矩阵
    
    // 从消息中提取数据（转换为Eigen格式）
    Eigen::Vector3d r_BN_N = cArray2EigenVector3d(this->stateCurrent.r_BN_N);       // 航天器位置
    Eigen::Vector3d r_SN_N = cArray2EigenVector3d(this->sunData.PositionVector);    // 太阳位置
    Eigen::MRPd sigma_BN = cArray2EigenMRPd(this->stateCurrent.sigma_BN);           // 姿态MRP

    // === 计算太阳方向矢量 ===
    // 矢量计算：r_SB_N = r_SN_N - r_BN_N (太阳->航天器)
    r_SB_N = r_SN_N - r_BN_N;
    
    // 归一化得到单位矢量（防止零除错误）
    if (r_SB_N.norm() > 1e-10) {
        sHat_N = r_SB_N.normalized(); // 单位太阳矢量 (惯性系)
    } else {
        sHat_N.setZero(); // 异常情况处理
    }

    // === 姿态转换 ===
    // 从MRP获取旋转矩阵（本体系到惯性系）
    Eigen::Matrix3d dcm_NB = sigma_BN.toRotationMatrix();
    // 转置得到惯性系到本体系的旋转矩阵
    dcm_BN = dcm_NB.transpose();
    
    // 将太阳矢量转换到本体系
    Eigen::Vector3d sHat_B = dcm_BN * sHat_N;

    // === 计算投影面积 ===
    // 点积公式：投影面积 = 实际面积 × cosθ
    // 其中 θ = 太阳矢量与法向量的夹角
    double cosTheta = sHat_B.dot(this->nHat_B);
    
    // 负值处理（太阳在面板背面时面积为0）
    this->projectedArea = (cosTheta > 0) ? 
                          (this->panelArea * cosTheta) : 0.0;

    // === 计算日距因子 ===
    // 平方反比定律：衰减因子 = (AU/r)^2
    double r_SBNorm = r_SB_N.norm();
    // 防止除零错误（当r很小时）
    if (r_SBNorm > 1e-10) {
        // AU单位转换：天文单位 -> 米
        double au_meters = AU * 1000.0;
        this->sunDistanceFactor = (au_meters * au_meters) / (r_SBNorm * r_SBNorm);
    } else {
        this->sunDistanceFactor = 0.0; // 异常情况
    }
}

/*!
 * 功率评估函数：计算面板输出功率
 * @param powerUsageSimMsg 输出消息（存储计算结果）
 * 
 * 功率公式：
 * Power = SOLAR_FLUX_EARTH × (AU²/r²) × ShadowFactor × (Area × cosθ) × Efficiency
 */
void SimpleSolarPanel::evaluatePowerModel(PowerNodeUsageMsgPayload *powerUsageSimMsg) {
    // 步骤1：计算太阳相关数据
    this->computeSunData(); // 更新projectedArea和sunDistanceFactor
    
    // 步骤2：计算太阳功率因子
    // = 地球轨道太阳通量 × 距离衰减 × 阴影因子
    double sunPowerFactor = SOLAR_FLUX_EARTH * this->sunDistanceFactor * this->shadowFactor;
    
    // 步骤3：计算净输出功率
    // = 太阳功率因子 × 有效面积 × 转换效率
    powerUsageSimMsg->netPower = sunPowerFactor * this->projectedArea * this->panelEfficiency;

    return;
}