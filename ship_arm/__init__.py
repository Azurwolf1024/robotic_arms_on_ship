"""
六自由度船载机械臂仿真平台 ( reproduction of Meng et al., 2607.22030 )。

子模块:
    core.lie        SO(3)/SE(3)/四元数
    robot           Newton-Euler 浮动基座动力学
    qp              稠密不等式约束二次规划
    control         TSID 力矩控制器 + 基线
    estimation      ESKF 基座状态估计
    platform        船体运动/IMU/动捕仿真
    sim             闭环仿真引擎、任务、指标
"""

__version__ = "0.1.0"
