"""
Glance2Wake - 注视检测模块
使用 MediaPipe FaceLandmarker (含虹膜追踪) 检测用户是否在注视屏幕
可无头运行（不弹出窗口），供主程序调用
"""

import contextlib
import logging
import os
import time

# 抑制 TensorFlow Lite / MediaPipe 噪音日志（需在 import mediapipe 前设置）
os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GRPC_VERBOSITY"] = "ERROR"

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
logging.getLogger("mediapipe").setLevel(logging.ERROR)

import cv2
import numpy as np
import mediapipe as mp

logger = logging.getLogger("Glance2Wake")


@contextlib.contextmanager
def _suppress_stderr():
    """
    临时屏蔽 C++ 层写入 stderr 的日志。
    Windows 上仅靠环境变量无法抑制 MediaPipe / TFLite 的 C++ 日志，
    因此需要在 OS 层面将 fd=2 重定向到 devnull。
    """
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stderr = os.dup(2)
    os.dup2(devnull, 2)          # stderr → devnull
    try:
        yield
    finally:
        os.dup2(old_stderr, 2)   # 恢复原始 stderr
        os.close(devnull)
        os.close(old_stderr)


# ==================== 关键面部标记索引 (MediaPipe 478 landmarks) ====================
# 左眼：外角 33, 内角 133, 上边 159, 下边 145, 虹膜中心 468
LEFT_EYE_OUTER, LEFT_EYE_INNER = 33, 133
LEFT_EYE_TOP, LEFT_EYE_BOTTOM = 159, 145
LEFT_IRIS_CENTER = 468

# 右眼：外角 362, 内角 263, 上边 386, 下边 374, 虹膜中心 473
RIGHT_EYE_OUTER, RIGHT_EYE_INNER = 362, 263
RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM = 386, 374
RIGHT_IRIS_CENTER = 473


class GazeDetector:
    """
    注视检测器
    封装 MediaPipe FaceLandmarker，提供单帧检测和批量检测接口。
    """

    def __init__(self, model_path=None,
                 h_center_min=0.35, h_center_max=0.65,
                 v_center_min=0.25, v_center_max=0.75):
        """
        参数:
            model_path:       face_landmarker.task 模型文件路径（默认同目录）
            h_center_min/max: 水平虹膜位置阈值 (0=外角, 1=内角)，范围内视为正视
            v_center_min/max: 垂直虹膜位置阈值 (0=上眼睑, 1=下眼睑)，范围内视为正视
        """
        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task"
            )
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"模型文件不存在: {model_path}\n"
                f"请下载: https://storage.googleapis.com/mediapipe-models/"
                f"face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
            )

        self.h_min = h_center_min
        self.h_max = h_center_max
        self.v_min = v_center_min
        self.v_max = v_center_max

        # 以二进制方式读取模型，避免 OneDrive 路径中文/特殊字符问题
        with open(model_path, "rb") as f:
            model_data = f.read()

        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_buffer=model_data),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,  # 视频模式支持帧间追踪
            num_faces=1,                    # 只跟踪一张脸
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # 创建检测器并用空白帧预热（屏蔽初始化时 C++ 层的噪音日志）
        with _suppress_stderr():
            self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
            dummy = np.zeros((100, 100, 3), dtype=np.uint8)
            dummy_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=dummy)
            try:
                self._landmarker.detect_for_video(dummy_img, 0)
            except Exception:
                pass
            time.sleep(0.1)

        self._timestamp_ms = 1
        logger.info("注视检测模型加载完成 (478 landmarks, 含虹膜追踪)")

    # ---------- 内部工具方法 ----------

    @staticmethod
    def _get_coord(landmark, w, h):
        return int(landmark.x * w), int(landmark.y * h)

    def _gaze_ratio_h(self, landmarks, iris_idx, outer_idx, inner_idx, w, h):
        """计算虹膜在眼睛水平方向上的相对位置 (0=外角, 1=内角)"""
        iris = self._get_coord(landmarks[iris_idx], w, h)
        outer = self._get_coord(landmarks[outer_idx], w, h)
        inner = self._get_coord(landmarks[inner_idx], w, h)
        eye_width = inner[0] - outer[0]
        if abs(eye_width) < 1:
            return 0.5
        return float(np.clip((iris[0] - outer[0]) / eye_width, 0.0, 1.0))

    def _gaze_ratio_v(self, landmarks, iris_idx, top_idx, bottom_idx, w, h):
        """计算虹膜在眼睛垂直方向上的相对位置 (0=上眼睑, 1=下眼睑)"""
        iris = self._get_coord(landmarks[iris_idx], w, h)
        top = self._get_coord(landmarks[top_idx], w, h)
        bottom = self._get_coord(landmarks[bottom_idx], w, h)
        eye_height = bottom[1] - top[1]
        if abs(eye_height) < 1:
            return 0.5
        return float(np.clip((iris[1] - top[1]) / eye_height, 0.0, 1.0))

    # ---------- 公共接口 ----------

    def detect_frame(self, frame):
        """
        分析单帧 BGR 图像。
        返回 (face_detected, is_looking, h_ratio, v_ratio)。
        未检测到人脸时返回 (False, False, None, None)。
        """
        img_h, img_w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # 单调递增的虚拟时间戳，VIDEO 模式要求每帧时间戳严格递增
        self._timestamp_ms += 33
        results = self._landmarker.detect_for_video(mp_image, self._timestamp_ms)

        if not results.face_landmarks or len(results.face_landmarks) == 0:
            return False, False, None, None

        lm = results.face_landmarks[0]
        if len(lm) < 478:
            return True, False, None, None  # 有人脸但模型未返回虹膜数据

        # 分别计算左右眼虹膜的水平/垂直位置比，再取平均
        l_h = self._gaze_ratio_h(lm, LEFT_IRIS_CENTER, LEFT_EYE_OUTER, LEFT_EYE_INNER, img_w, img_h)
        l_v = self._gaze_ratio_v(lm, LEFT_IRIS_CENTER, LEFT_EYE_TOP, LEFT_EYE_BOTTOM, img_w, img_h)
        r_h = self._gaze_ratio_h(lm, RIGHT_IRIS_CENTER, RIGHT_EYE_OUTER, RIGHT_EYE_INNER, img_w, img_h)
        r_v = self._gaze_ratio_v(lm, RIGHT_IRIS_CENTER, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM, img_w, img_h)

        avg_h = (l_h + r_h) / 2.0
        avg_v = (l_v + r_v) / 2.0

        # 当双眼虹膜平均位置落在阈值范围内，判定为正在注视屏幕
        looking = (self.h_min <= avg_h <= self.h_max and self.v_min <= avg_v <= self.v_max)
        return True, looking, avg_h, avg_v

    def close(self):
        """释放模型资源"""
        self._landmarker.close()
