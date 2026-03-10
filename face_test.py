"""
注视检测程序 - Gaze Detection
使用 MediaPipe FaceLandmarker (新版 Tasks API, 含虹膜追踪) 检测用户是否在注视屏幕
按 'q' 键退出程序
"""

import contextlib
import logging
import os

# 抑制 TensorFlow Lite 和 MediaPipe 的日志警告 (需要在 import mediapipe 之前设置)
os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GRPC_VERBOSITY"] = "ERROR"

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
logging.getLogger("mediapipe").setLevel(logging.ERROR)


@contextlib.contextmanager
def suppress_stderr():
    """临时屏蔽 C++ 层写入 stderr 的日志 (Windows 上 env var 对 MediaPipe 不生效)"""
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stderr = os.dup(2)
    os.dup2(devnull, 2)
    try:
        yield
    finally:
        os.dup2(old_stderr, 2)
        os.close(devnull)
        os.close(old_stderr)

import cv2
import numpy as np
import mediapipe as mp
from collections import deque

# ==================== 配置参数 ====================
# 水平注视阈值 (值越小越偏向一侧)
H_CENTER_MIN = 0.35  # 小于此值 → 看右边
H_CENTER_MAX = 0.65  # 大于此值 → 看左边

# 垂直注视阈值
V_CENTER_MIN = 0.25  # 小于此值 → 看上方
V_CENTER_MAX = 0.75  # 大于此值 → 看下方

# 平滑帧数 (越大越平滑但延迟越高)
SMOOTHING_FRAMES = 5

# 模型下载地址与本地路径
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")

# ==================== 关键面部标记索引 ====================
# 左眼
LEFT_EYE_OUTER = 33
LEFT_EYE_INNER = 133
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
LEFT_IRIS_CENTER = 468

# 右眼
RIGHT_EYE_OUTER = 362
RIGHT_EYE_INNER = 263
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374
RIGHT_IRIS_CENTER = 473

# 眼睛轮廓 (用于绘制)
LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133,
                    173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249,
                     263, 466, 388, 387, 386, 385, 384, 398]
# 虹膜轮廓
LEFT_IRIS_INDICES = [468, 469, 470, 471, 472]
RIGHT_IRIS_INDICES = [473, 474, 475, 476, 477]





def get_landmark_coord(landmark, img_w, img_h):
    """将归一化的面部标记坐标转换为像素坐标"""
    return int(landmark.x * img_w), int(landmark.y * img_h)


def compute_gaze_ratio_h(landmarks, iris_idx, outer_idx, inner_idx, img_w, img_h):
    """
    计算水平注视比例
    返回值: 0.0(外侧) ~ 1.0(内侧)，0.5 表示正视
    """
    iris = get_landmark_coord(landmarks[iris_idx], img_w, img_h)
    outer = get_landmark_coord(landmarks[outer_idx], img_w, img_h)
    inner = get_landmark_coord(landmarks[inner_idx], img_w, img_h)

    eye_width = inner[0] - outer[0]
    if abs(eye_width) < 1:
        return 0.5
    ratio = (iris[0] - outer[0]) / eye_width
    return float(np.clip(ratio, 0.0, 1.0))


def compute_gaze_ratio_v(landmarks, iris_idx, top_idx, bottom_idx, img_w, img_h):
    """
    计算垂直注视比例
    返回值: 0.0(上方) ~ 1.0(下方)，0.5 表示正视
    """
    iris = get_landmark_coord(landmarks[iris_idx], img_w, img_h)
    top = get_landmark_coord(landmarks[top_idx], img_w, img_h)
    bottom = get_landmark_coord(landmarks[bottom_idx], img_w, img_h)

    eye_height = bottom[1] - top[1]
    if abs(eye_height) < 1:
        return 0.5
    ratio = (iris[1] - top[1]) / eye_height
    return float(np.clip(ratio, 0.0, 1.0))


def get_gaze_direction(h_ratio, v_ratio):
    """根据注视比例判断注视方向"""
    if h_ratio < H_CENTER_MIN:
        h_dir = "Right"
    elif h_ratio > H_CENTER_MAX:
        h_dir = "Left"
    else:
        h_dir = "Center"

    if v_ratio < V_CENTER_MIN:
        v_dir = "Up"
    elif v_ratio > V_CENTER_MAX:
        v_dir = "Down"
    else:
        v_dir = "Center"

    return h_dir, v_dir


def is_looking_at_screen(h_ratio, v_ratio):
    """判断是否在注视屏幕"""
    return (H_CENTER_MIN <= h_ratio <= H_CENTER_MAX and
            V_CENTER_MIN <= v_ratio <= V_CENTER_MAX)


def draw_eye_contour(frame, landmarks, indices, img_w, img_h, color=(0, 255, 0)):
    """绘制眼睛轮廓"""
    points = []
    for idx in indices:
        x, y = get_landmark_coord(landmarks[idx], img_w, img_h)
        points.append([x, y])
    points = np.array(points, dtype=np.int32)
    cv2.polylines(frame, [points], isClosed=True, color=color, thickness=1)


def draw_iris(frame, landmarks, indices, img_w, img_h, color=(255, 0, 255)):
    """绘制虹膜"""
    points = []
    for idx in indices:
        x, y = get_landmark_coord(landmarks[idx], img_w, img_h)
        points.append([x, y])
    center = np.mean(points, axis=0).astype(int)
    radius = int(np.linalg.norm(np.array(points[0]) - np.array(points[2])) / 2)
    cv2.circle(frame, tuple(center), radius, color, 1)
    cv2.circle(frame, tuple(center), 2, color, -1)


def put_text(frame, text, position, font_scale=0.7, color=(0, 255, 0), thickness=2):
    """在帧上显示文字"""
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness, cv2.LINE_AA)


# ==================== 主程序 ====================
def main():


    # --- 用新版 Tasks API 创建 FaceLandmarker ---
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    # 以二进制方式读取模型，避免 OneDrive 路径问题
    with open(MODEL_PATH, "rb") as f:
        model_data = f.read()

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_buffer=model_data),
        running_mode=RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # 创建时临时屏蔽 stderr，避免 C++ 层警告输出
    # 需要包含一次预热检测，因为 MediaPipe 异步线程会在首次推理时输出警告
    with suppress_stderr():
        landmarker = FaceLandmarker.create_from_options(options)
        # 预热：用一张空白图片做一次推理，触发所有异步线程的初始化日志
        dummy = np.zeros((100, 100, 3), dtype=np.uint8)
        dummy_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=dummy)
        try:
            landmarker.detect_for_video(dummy_img, 0)
        except Exception:
            pass
        import time
        time.sleep(0.1)  # 等待异步线程完成日志输出
    print("FaceLandmarker 初始化成功 (478 landmarks, 含虹膜追踪)")

    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头！请检查权限或设备连接。")
        landmarker.close()
        return

    print("=" * 50)
    print("  注视检测程序已启动")
    print("  按 'q' 键退出")
    print("=" * 50)

    # 用于平滑的队列
    h_ratio_buffer = deque(maxlen=SMOOTHING_FRAMES)
    v_ratio_buffer = deque(maxlen=SMOOTHING_FRAMES)

    frame_timestamp_ms = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法接收画面")
            break

        # 水平翻转 (镜像效果，更自然)
        frame = cv2.flip(frame, 1)
        img_h, img_w, _ = frame.shape

        # 转换为 MediaPipe Image (新 API 需要 mp.Image)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # 用 VIDEO 模式检测 (需要递增时间戳)
        frame_timestamp_ms += 33  # ~30fps
        results = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        if results.face_landmarks and len(results.face_landmarks) > 0:
            face_landmarks = results.face_landmarks[0]  # 第一张脸的标记点列表

            # 检查是否包含虹膜标记 (478个点)
            if len(face_landmarks) < 478:
                put_text(frame, "Iris landmarks not available", (20, 45),
                         font_scale=0.7, color=(0, 0, 255), thickness=2)
                cv2.imshow('Gaze Detection - Press Q to Quit', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            # --- 计算左眼注视比例 ---
            l_h = compute_gaze_ratio_h(face_landmarks, LEFT_IRIS_CENTER,
                                       LEFT_EYE_OUTER, LEFT_EYE_INNER, img_w, img_h)
            l_v = compute_gaze_ratio_v(face_landmarks, LEFT_IRIS_CENTER,
                                       LEFT_EYE_TOP, LEFT_EYE_BOTTOM, img_w, img_h)

            # --- 计算右眼注视比例 ---
            r_h = compute_gaze_ratio_h(face_landmarks, RIGHT_IRIS_CENTER,
                                       RIGHT_EYE_OUTER, RIGHT_EYE_INNER, img_w, img_h)
            r_v = compute_gaze_ratio_v(face_landmarks, RIGHT_IRIS_CENTER,
                                       RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM, img_w, img_h)

            # --- 双眼平均 + 平滑 ---
            avg_h = (l_h + r_h) / 2.0
            avg_v = (l_v + r_v) / 2.0
            h_ratio_buffer.append(avg_h)
            v_ratio_buffer.append(avg_v)

            smooth_h = float(np.mean(h_ratio_buffer))
            smooth_v = float(np.mean(v_ratio_buffer))

            # --- 判断注视方向 ---
            h_dir, v_dir = get_gaze_direction(smooth_h, smooth_v)
            looking = is_looking_at_screen(smooth_h, smooth_v)

            # --- 绘制眼睛和虹膜 ---
            draw_eye_contour(frame, face_landmarks, LEFT_EYE_INDICES, img_w, img_h,
                             color=(0, 255, 0))
            draw_eye_contour(frame, face_landmarks, RIGHT_EYE_INDICES, img_w, img_h,
                             color=(0, 255, 0))
            draw_iris(frame, face_landmarks, LEFT_IRIS_INDICES, img_w, img_h,
                      color=(255, 0, 255))
            draw_iris(frame, face_landmarks, RIGHT_IRIS_INDICES, img_w, img_h,
                      color=(255, 0, 255))

            # --- 绘制信息面板背景 ---
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (350, 170), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            # --- 显示注视信息 ---
            if looking:
                status_text = "[O] LOOKING AT SCREEN"
                status_color = (0, 255, 0)
            else:
                status_text = "[X] LOOKING AWAY"
                status_color = (0, 0, 255)

            put_text(frame, status_text, (20, 45),
                     font_scale=0.7, color=status_color, thickness=2)

            h_text = f"Horizontal: {h_dir}  ({smooth_h:.2f})"
            put_text(frame, h_text, (20, 80),
                     font_scale=0.55, color=(255, 255, 255))

            v_text = f"Vertical:   {v_dir}  ({smooth_v:.2f})"
            put_text(frame, v_text, (20, 110),
                     font_scale=0.55, color=(255, 255, 255))

            gaze_summary = f"Gaze: [{h_dir}, {v_dir}]"
            put_text(frame, gaze_summary, (20, 150),
                     font_scale=0.65, color=(0, 255, 255), thickness=2)

            # --- 在屏幕中央绘制注视指示器 ---
            center_x, center_y = img_w // 2, img_h // 2
            indicator_x = int(center_x + (smooth_h - 0.5) * 200)
            indicator_y = int(center_y + (smooth_v - 0.5) * 200)

            cross_size = 20
            cv2.line(frame, (center_x - cross_size, center_y),
                     (center_x + cross_size, center_y), (100, 100, 100), 1)
            cv2.line(frame, (center_x, center_y - cross_size),
                     (center_x, center_y + cross_size), (100, 100, 100), 1)
            cv2.circle(frame, (indicator_x, indicator_y), 8, status_color, -1)
            cv2.circle(frame, (indicator_x, indicator_y), 12, status_color, 2)

        else:
            # 未检测到面部
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (350, 60), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            put_text(frame, "No face detected", (20, 45),
                     font_scale=0.7, color=(0, 0, 255), thickness=2)

        # 显示帧
        cv2.imshow('Gaze Detection - Press Q to Quit', frame)

        # 按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()
    print("程序已退出。")


if __name__ == "__main__":
    main()

