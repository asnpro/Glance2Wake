"""
Glance2Wake — 注视即唤醒
当电脑即将进入睡眠（或关闭显示器）时，自动打开摄像头持续检测用户是否在屏幕前。
检测到存在 → 重置睡眠计时器；未检测到 → 放行，让系统正常睡眠。

检测模式:
    "gaze" — 需要检测到用户正在注视屏幕（虹膜追踪）
    "face" — 只要检测到人脸即可

用法:
    python glance2wake.py          # 启动监控
    Ctrl+C                         # 退出
"""

import logging
import signal
import time
from collections import deque

import cv2

from gaze_detector import GazeDetector
from power_manager import (
    get_idle_time_seconds,
    get_sleep_timeout_seconds,
    get_display_timeout_seconds,
    get_effective_timeout_seconds,
    is_on_ac_power,
    reset_sleep_timer,
)

# ==================== 可调参数 ====================
DETECTION_MODE        = "gaze"  # "gaze" = 注视检测(虹膜追踪), "face" = 仅人脸检测
CHECK_ADVANCE_SECONDS = 60      # 在系统超时前多少秒开始检测
CAMERA_WARMUP         = 1.0     # 摄像头预热秒数，等待自动曝光稳定
POLL_INTERVAL         = 5       # 主循环轮询空闲时间的间隔（秒）
CAMERA_INDEX          = 0       # OpenCV 摄像头索引
ROLLING_WINDOW_SEC    = 4.0     # 滑动窗口长度（秒），取最近这段时间的帧做注视判定
GAZE_CONFIRM_RATIO    = 0.5     # 滑动窗口内注视帧占比 >= 此值即确认用户在注视
SLEEP_MARGIN_SEC      = 8.0     # 安全余量: 距睡眠不足此秒数时停止检测，让系统正常睡眠

# ==================== 日志 ====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Glance2Wake")

# ==================== 优雅退出 ====================
_running = True  # 全局运行标志，Ctrl+C 时置为 False 以通知所有循环退出


def _signal_handler(_sig, _frame):
    global _running
    _running = False
    print()


signal.signal(signal.SIGINT, _signal_handler)


# ==================== 格式化工具 ====================
def _fmt_time(seconds):
    """将秒数格式化为 'Xm Ys' """
    if seconds <= 0:
        return "从不"
    m, s = divmod(int(seconds), 60)
    return f"{m}分{s}秒" if m else f"{s}秒"


# ==================== 主程序 ====================
def main():
    global _running

    logger.info("=" * 50)
    logger.info("  Glance2Wake 正在启动...")
    logger.info("=" * 50)

    # ---------- 读取电源设置 ----------
    sleep_timeout = get_sleep_timeout_seconds()
    display_timeout = get_display_timeout_seconds()
    effective_timeout = get_effective_timeout_seconds()
    power_src = "AC (插电)" if is_on_ac_power() else "DC (电池)"

    logger.info(f"电源: {power_src}")
    logger.info(f"睡眠超时: {_fmt_time(sleep_timeout)}")
    logger.info(f"关屏超时: {_fmt_time(display_timeout)}")

    if effective_timeout == 0:
        logger.warning("系统睡眠和关屏均设为「从不」，Glance2Wake 无需运行")
        logger.info("请在 Windows 设置 > 系统 > 电源 中配置超时后重试")
        return

    if effective_timeout <= CHECK_ADVANCE_SECONDS:
        logger.warning(
            f"有效超时 ({_fmt_time(effective_timeout)})   "
            f"({CHECK_ADVANCE_SECONDS}秒)，已自动调整为超时的一半"
        )
        check_advance = max(effective_timeout // 2, 5)
    else:
        check_advance = CHECK_ADVANCE_SECONDS

    logger.info(f"将在超时前 {_fmt_time(check_advance)} 开始注视检测")

    mode_label = "注视检测 (gaze)" if DETECTION_MODE == "gaze" else "人脸检测 (face)"
    logger.info(f"检测模式: {mode_label}")

    # ---------- 初始化注视检测器 ----------
    logger.info("正在加载检测模型...")
    detector = GazeDetector()

    logger.info("-" * 50)
    logger.info("Glance2Wake 已就绪，按 Ctrl+C 退出")
    logger.info("-" * 50)

    # ---------- 状态追踪 ----------
    # last_activity: 用户最后一次活动的时间戳（增强版，会在注视确认后也更新）
    # detection_done_for_cycle: 防止同一空闲周期内反复开关摄像头
    last_activity = time.time()
    detection_done_for_cycle = False

    try:
        while _running:
            # ---- 获取真实空闲时间 (OS 层面的键鼠输入空闲) ----
            actual_idle = get_idle_time_seconds()

            # 如果检测到真实用户输入，同步虚拟活跃时间并重置周期标志
            if actual_idle < POLL_INTERVAL:
                last_activity = time.time() - actual_idle
                if detection_done_for_cycle:
                    detection_done_for_cycle = False  # 用户有新活动，允许下次再检测
                    logger.info("检测到用户活动，恢复监控")

            # ---- 计算虚拟空闲时间和距超时的剩余时间 ----
            virtual_idle = time.time() - last_activity
            time_to_timeout = effective_timeout - virtual_idle

            # ---- 窗口 1: 安全余量内 → 来不及检测，放行睡眠 ----
            if 0 < time_to_timeout <= SLEEP_MARGIN_SEC:
                logger.info(
                    f"距超时仅剩 {time_to_timeout:.1f}s (< 余量 {SLEEP_MARGIN_SEC}s)，"
                    f"跳过检测，允许系统睡眠"
                )
                time.sleep(POLL_INTERVAL)
                continue

            # ---- 窗口 2: 检测窗口内 → 打开摄像头持续检测 ----
            if SLEEP_MARGIN_SEC < time_to_timeout <= check_advance:
                if detection_done_for_cycle:
                    # 本轮已检测过且未确认注视，不再重复开摄像头，
                    # 等待用户下次活动后 detection_done_for_cycle 会被重置
                    time.sleep(POLL_INTERVAL)
                    continue
                logger.info(
                    f"距超时还有约 {_fmt_time(time_to_timeout)}，正在打开摄像头持续检测..."
                )
                logger.debug(
                    f"  [诊断] actual_idle={actual_idle:.1f}s, "
                    f"virtual_idle={virtual_idle:.1f}s, "
                    f"effective_timeout={effective_timeout}s, "
                    f"检测时长上限={time_to_timeout - SLEEP_MARGIN_SEC:.1f}s"
                )
                confirmed = _continuous_detect(
                    detector, time_to_timeout, effective_timeout
                )
                if confirmed:
                    # 注视已确认，视为用户活动，重置虚拟活跃时间
                    last_activity = time.time()
                else:
                    # 未确认注视，标记本轮检测已完成，避免反复开关摄像头
                    detection_done_for_cycle = True
                    logger.info("本轮检测结束，等待用户下次活动后恢复监控")
            else:
                # ---- 窗口 3: 距超时还早，休眠等待 ----
                time.sleep(POLL_INTERVAL)

    finally:
        detector.close()
        logger.info("Glance2Wake 已退出")


def _continuous_detect(detector, time_to_timeout, effective_timeout):
    """
    打开摄像头持续检测用户是否在注视屏幕。

    检测持续到以下任一条件触发时结束:
      - 滑动窗口内确认注视/人脸 → 重置计时器，返回 True
      - 墙钟截止时间到达 (SLEEP_MARGIN_SEC 前) → 放行睡眠，返回 False
      - 检测到真实用户键鼠输入 → 不再需要检测，返回 True

    参数:
        detector:          GazeDetector 实例
        time_to_timeout:   距系统超时的剩余秒数
        effective_timeout:  系统的有效超时时间（秒）
    """
    # 计算检测最大持续时长（秒），作为硬性墙钟截止
    # 确保在系统真正睡眠前 SLEEP_MARGIN_SEC 秒停止检测
    max_duration = time_to_timeout - SLEEP_MARGIN_SEC
    if max_duration <= 0:
        logger.info("--- 检测时间不足，直接放行睡眠")
        return False

    start_time = time.time()
    deadline = start_time + max_duration
    logger.debug(
        f"  [检测] 最大持续 {max_duration:.1f}s, "
        f"SLEEP_MARGIN={SLEEP_MARGIN_SEC}s"
    )

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        logger.warning("无法打开摄像头")
        return False

    # 减小缓冲区，避免 cap.read() 读到旧帧
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    try:
        # 摄像头预热: 持续 grab 几帧让自动曝光稳定，同时也受 deadline 限制
        warmup_end = min(time.time() + CAMERA_WARMUP, deadline)
        while time.time() < warmup_end and _running:
            cap.grab()
            if time.time() >= deadline:
                logger.info("--- 预热阶段已到截止时间，放行睡眠")
                return False

        # 滑动窗口: 记录 (timestamp, triggered_bool) 用于判定注视确认
        rolling = deque()
        last_status_log = time.time()  # 诊断日志債频控制

        while _running:
            # ---- 墙钟截止（最高优先级，在任何阻塞操作前检查） ----
            now = time.time()
            if now >= deadline:
                elapsed = now - start_time
                logger.info(
                    f"--- 检测窗口结束 (已运行 {elapsed:.1f}s/{max_duration:.1f}s)，"
                    f"未确认注视，允许系统进入睡眠"
                )
                return False

            # ---- 真实用户输入 → 用户已主动操作，无需我们检测 ----
            actual_idle = get_idle_time_seconds()
            if actual_idle < 3:  # 3秒内有输入表示用户正在使用电脑
                logger.info("检测到用户输入，退出检测模式")
                return True

            # 注意: 不在此处用 actual_idle 对比 effective_timeout 做安全网。
            # 因为 reset_sleep_timer() 重置了 OS 计时器，但 actual_idle（基于
            # GetLastInputInfo）仍持续增长，会导致后续检测周期被误判立即退出。
            # 墙钟 deadline 已可靠保证检测在 SLEEP_MARGIN_SEC 前结束。

            # ---- 每 3 秒打印一次诊断状态 ----
            if now - last_status_log >= 3.0:
                elapsed = now - start_time
                remaining = deadline - now
                logger.debug(
                    f"  [检测中] 已运行 {elapsed:.1f}s, "
                    f"剩余 {remaining:.1f}s, "
                    f"actual_idle={actual_idle:.1f}s"
                )
                last_status_log = now

            # ---- 读帧: 使用 grab+retrieve 分离模式减少阻塞 ----
            grabbed = cap.grab()
            # grab 可能阻塞较久，回来后再次检查截止时间
            if time.time() >= deadline:
                logger.info("--- grab() 后已到截止时间，放行睡眠")
                return False

            if not grabbed:
                time.sleep(0.01)
                continue
            ret, frame = cap.retrieve()
            if not ret:
                continue

            # 水平翻转以匹配镜像视角（用户看屏幕时摄像头左右相反）
            frame = cv2.flip(frame, 1)
            face_detected, is_looking, _h, _v = detector.detect_frame(frame)

            # 根据检测模式判断本帧是否触发
            if DETECTION_MODE == "face":
                triggered = face_detected             # 仅需人脸
            else:
                triggered = face_detected and is_looking  # 需要注视

            now = time.time()
            rolling.append((now, triggered))

            # 清除滑动窗口外的旧帧
            cutoff = now - ROLLING_WINDOW_SEC
            while rolling and rolling[0][0] < cutoff:
                rolling.popleft()

            # ---- 滑动窗口判定 ----
            # 当窗口内累积帧足够时 (超过窗口长度的 50%)，开始统计注视占比
            window_span = now - rolling[0][0] if rolling else 0
            if window_span >= ROLLING_WINDOW_SEC * 0.5:
                total = len(rolling)
                positive = sum(1 for _, t in rolling if t)
                ratio = positive / total
                if ratio >= GAZE_CONFIRM_RATIO:
                    mode_txt = "注视" if DETECTION_MODE == "gaze" else "人脸"
                    logger.info(
                        f">>> 确认用户{mode_txt} "
                        f"(最近 {window_span:.1f}s 内 {ratio:.0%})，已重置睡眠计时器"
                    )
                    reset_sleep_timer()
                    return True

            time.sleep(0.03)  # ~30fps

        return False

    finally:
        cap.release()
        logger.debug("摄像头已关闭")


if __name__ == "__main__":
    main()
