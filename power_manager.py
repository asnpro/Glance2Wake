"""
Glance2Wake - Windows 电源管理模块

提供:
  - 用户空闲时间检测 (GetLastInputInfo)
  - 电源状态查询 (AC/DC)
  - 睡眠/屏幕超时查询 (powercfg)
  - 睡眠计时器重置 (SetThreadExecutionState)

仅支持 Windows 平台
"""

import ctypes
import logging
import re
import subprocess

logger = logging.getLogger("Glance2Wake")


# ==================== 空闲时间检测 ====================

class _LASTINPUTINFO(ctypes.Structure):
    """Win32 LASTINPUTINFO 结构体。dwTime 是最后一次输入事件的 GetTickCount 毫秒值。"""
    _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_uint)]


def get_idle_time_seconds():
    """
    获取自上次鼠标/键盘输入以来的空闲时间（秒）。
    原理: GetTickCount() - LastInputInfo.dwTime
    """
    lii = _LASTINPUTINFO()
    lii.cbSize = ctypes.sizeof(_LASTINPUTINFO)
    ctypes.windll.user32.GetLastInputInfo(ctypes.byref(lii))
    millis = ctypes.windll.kernel32.GetTickCount() - lii.dwTime
    return millis / 1000.0


# ==================== 电源状态检测 ====================

class _SYSTEM_POWER_STATUS(ctypes.Structure):
    """Win32 SYSTEM_POWER_STATUS 结构体。ACLineStatus: 0=电池, 1=插电。"""
    _fields_ = [
        ("ACLineStatus", ctypes.c_byte),
        ("BatteryFlag", ctypes.c_byte),
        ("BatteryLifePercent", ctypes.c_byte),
        ("SystemStatusFlag", ctypes.c_byte),
        ("BatteryLifeTime", ctypes.c_ulong),
        ("BatteryFullLifeTime", ctypes.c_ulong),
    ]


def is_on_ac_power():
    """判断当前是否使用交流电源 (插电)"""
    status = _SYSTEM_POWER_STATUS()
    ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status))
    return status.ACLineStatus == 1


# ==================== 电源超时查询 ====================

def _query_power_timeout(sub_group, setting):
    """
    通过 powercfg /query 查询当前电源方案的指定设置。
    返回 (ac_seconds, dc_seconds)，失败返回 (0, 0)。
    值为 0 表示「从不」(已禁用)。

    powercfg 输出中的 0x 十六进制值顺序为:
      [0] Minimum  [1] Maximum  [2] Increment  [3] AC Value  [4] DC Value
    """
    try:
        result = subprocess.run(
            ["powercfg", "/query", "SCHEME_CURRENT", sub_group, setting],
            capture_output=True, text=True, timeout=5,
        )
        # 提取所有十六进制值，取第 4、5 个作为 AC/DC 值
        matches = re.findall(r"0x([0-9a-fA-F]+)", result.stdout)
        if len(matches) >= 5:
            return int(matches[3], 16), int(matches[4], 16)
    except Exception as e:
        logger.warning(f"无法查询电源设置 {sub_group}/{setting}: {e}")
    return 0, 0


def get_sleep_timeout_seconds():
    """
    获取当前电源方案的睡眠超时（秒）。
    自动根据 AC/DC 状态返回对应值。
    返回 0 表示「从不睡眠」。
    查询 powercfg 子组: SUB_SLEEP / STANDBYIDLE
    """
    ac, dc = _query_power_timeout("SUB_SLEEP", "STANDBYIDLE")
    timeout = ac if is_on_ac_power() else dc
    return timeout


def get_display_timeout_seconds():
    """
    获取当前电源方案的关闭显示器超时（秒）。
    自动根据 AC/DC 状态返回对应值。
    返回 0 表示「从不关闭」。
    查询 powercfg 子组: SUB_VIDEO / VIDEOIDLE
    """
    ac, dc = _query_power_timeout("SUB_VIDEO", "VIDEOIDLE")
    timeout = ac if is_on_ac_power() else dc
    return timeout


def get_effective_timeout_seconds():
    """
    获取有效超时时间（秒）= min(睡眠超时, 显示器超时)，忽略为 0 的项。
    返回 0 表示两者都设为 "从不"。
    """
    sleep_t = get_sleep_timeout_seconds()
    display_t = get_display_timeout_seconds()
    candidates = [t for t in (sleep_t, display_t) if t > 0]
    return min(candidates) if candidates else 0


# ==================== 睡眠计时器操作 ====================

# SetThreadExecutionState 标志位
_ES_SYSTEM_REQUIRED = 0x00000001   # 重置系统睡眠计时器
_ES_DISPLAY_REQUIRED = 0x00000002  # 重置显示器关闭计时器


def reset_sleep_timer():
    """
    一次性重置系统睡眠计时器和显示器关闭计时器。
    调用后 Windows 会从零开始重新倒计时。
    不含 ES_CONTINUOUS 标志，因此不会永久阻止睡眠。
    """
    ctypes.windll.kernel32.SetThreadExecutionState(
        _ES_SYSTEM_REQUIRED | _ES_DISPLAY_REQUIRED
    )
    logger.info("已重置系统睡眠/显示器计时器")
