import sys
import os
import queue
import threading
import time
import json
import random
import math
import traceback
import warnings
from datetime import datetime, timedelta
from collections import deque
import statistics
from typing import Optional, Tuple, List
import cv2
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import mediapipe as mp
import dlib
from scipy.ndimage import zoom
from tensorflow.keras.models import load_model
from imutils import face_utils

warnings.filterwarnings('ignore')

try:
    import pyttsx3

    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("注意：pyttsx3 未安装，语音功能将不可用")

# ============================================================================
# 初始化MediaPipe
# ============================================================================
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# ============================================================================
# 注意力识别配置和类
# ============================================================================

class AttentionConfig:
    """注意力识别配置"""

    def __init__(self):
        self.ear_thresh: float = 0.21  # < EAR => 眼睛闭合
        self.ear_consec_frames: int = 1  # 连续帧数判断眨眼/闭合
        self.yaw_thresh_deg: float = 20.0  # |yaw| > => 转头
        self.pitch_thresh_deg: float = 20.0  # |pitch| > => 抬头/低头
        self.roll_thresh_deg: float = 25.0  # 仅用于显示/诊断
        self.gaze_off_center: float = 0.35  # |gaze_x| 或 |gaze_y| > => 视线偏离
        self.min_face_conf: float = 0.5


# MediaPipe FaceMesh 地标索引
R_EYE = [33, 160, 158, 133, 153, 144]
L_EYE = [362, 385, 387, 263, 373, 380]

# 头部姿态2D点
POSE_LANDMARKS = {
    'nose_tip': 1,
    'chin': 152,
    'left_eye_outer': 263,
    'right_eye_outer': 33,
    'left_mouth': 291,
    'right_mouth': 61
}

# 3D模型参考点（毫米）
MODEL_POINTS_3D = np.array([
    [0.0, 0.0, 0.0],  # 鼻尖
    [0.0, -63.6, -12.5],  # 下巴
    [-43.3, 32.7, -26.0],  # 左眼角
    [43.3, 32.7, -26.0],  # 右眼角
    [-28.9, -28.9, -24.1],  # 左嘴角
    [28.9, -28.9, -24.1]  # 右嘴角
], dtype=np.float64)

# 虹膜地标索引范围
RIGHT_IRIS = list(range(468, 473))
LEFT_IRIS = list(range(473, 478))


class AttentionAnalyzer:
    """注意力分析器"""

    def __init__(self):
        self.config = AttentionConfig()
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            refine_landmarks=True,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # 状态变量
        self.closed_counter = 0
        self.blinks = 0
        self.attention_history = []
        self.ear_history = deque(maxlen=30)
        self.gaze_history = deque(maxlen=30)
        self.pose_history = deque(maxlen=30)

        # 当前状态
        self.current_state = {
            "attention_label": "初始化中",
            "ear_left": 0.0,
            "ear_right": 0.0,
            "yaw": 0.0,
            "pitch": 0.0,
            "roll": 0.0,
            "gaze_x": 0.0,
            "gaze_y": 0.0,
            "blink_count": 0
        }

    def landmarks_to_np(self, landmarks, w, h) -> np.ndarray:
        """将地标转换为numpy数组"""
        pts = []
        for lm in landmarks:
            x, y = int(lm.x * w), int(lm.y * h)
            pts.append((x, y))
        return np.array(pts, dtype=np.int32)

    def eye_aspect_ratio(self, eye_pts: np.ndarray) -> float:
        """计算眼睛纵横比"""
        if len(eye_pts) < 6:
            return 0.3

        try:
            p1, p2, p3, p4, p5, p6 = eye_pts[:6]
            A = np.linalg.norm(p2 - p6)
            B = np.linalg.norm(p3 - p5)
            C = np.linalg.norm(p1 - p4)
            ear = (A + B) / (2.0 * C + 1e-6)
            return float(ear)
        except:
            return 0.3

    def head_pose(self, w: int, h: int, pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """计算头部姿态"""
        try:
            idx = POSE_LANDMARKS
            image_points = np.array([
                pts[idx['nose_tip']],
                pts[idx['chin']],
                pts[idx['left_eye_outer']],
                pts[idx['right_eye_outer']],
                pts[idx['left_mouth']],
                pts[idx['right_mouth']]
            ], dtype=np.float64)

            focal_length = w
            center = (w / 2, h / 2)
            camera_matrix = np.array(
                [[focal_length, 0, center[0]],
                 [0, focal_length, center[1]],
                 [0, 0, 1]], dtype=np.float64)

            dist_coeffs = np.zeros((4, 1))

            success, rotation_vec, translation_vec = cv2.solvePnP(
                MODEL_POINTS_3D,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if not success:
                return None, None, None

            rotation_mat, _ = cv2.Rodrigues(rotation_vec)
            pose_mat = np.hstack((rotation_mat, translation_vec))
            _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
            pitch, yaw, roll = euler_angles.flatten()
            return np.array([pitch, yaw, roll]), rotation_vec, translation_vec
        except:
            return None, None, None

    def iris_center(self, all_pts: np.ndarray, iris_idx: List[int]) -> Optional[np.ndarray]:
        """计算虹膜中心"""
        if len(iris_idx) == 0:
            return None
        try:
            iris_pts = all_pts[iris_idx]
            c = iris_pts.mean(axis=0)
            return c
        except:
            return None

    def gaze_vector(self, eye_pts: np.ndarray, iris_c: np.ndarray) -> Tuple[float, float]:
        """计算视线向量"""
        try:
            x_min, y_min = eye_pts.min(axis=0)
            x_max, y_max = eye_pts.max(axis=0)
            cx = (x_min + x_max) / 2.0
            cy = (y_min + y_max) / 2.0

            nx = 0.0 if x_max == x_min else (iris_c[0] - cx) / ((x_max - x_min) / 2.0)
            ny = 0.0 if y_max == y_min else (iris_c[1] - cy) / ((y_max - y_min) / 2.0)

            nx = float(np.clip(nx, -1.5, 1.5))
            ny = float(np.clip(ny, -1.5, 1.5))

            return nx, ny
        except:
            return 0.0, 0.0

    def attention_label(self, ear_l: float, ear_r: float, yaw: float,
                        pitch: float, gaze: Tuple[float, float]) -> str:
        """确定注意力标签"""
        eyes_open = (ear_l > self.config.ear_thresh) and (ear_r > self.config.ear_thresh)
        looking_forward = (abs(yaw) < self.config.yaw_thresh_deg) and (abs(pitch) < self.config.pitch_thresh_deg)
        gaze_centered = (abs(gaze[0]) < self.config.gaze_off_center) and (abs(gaze[1]) < self.config.gaze_off_center)

        if not eyes_open:
            return "眼睛闭合"
        if not looking_forward:
            return "视线偏离"
        if not gaze_centered:
            return "视线偏移"
        return "专注"

    def analyze_frame(self, frame):
        """分析单帧注意力"""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]

            result = self.face_mesh.process(rgb_frame)

            # 默认状态
            label = "未检测到面部"
            ear_l = ear_r = 0.0
            euler = np.array([0.0, 0.0, 0.0])
            gaze_xy = (0.0, 0.0)
            face_detected = False

            if result.multi_face_landmarks:
                face_detected = True
                lms = result.multi_face_landmarks[0].landmark
                pts = self.landmarks_to_np(lms, w, h)

                # 计算EAR
                eye_r = pts[R_EYE]
                eye_l = pts[L_EYE]
                ear_r = self.eye_aspect_ratio(eye_r)
                ear_l = self.eye_aspect_ratio(eye_l)

                # 眨眼/闭眼计数器
                if ear_l < self.config.ear_thresh and ear_r < self.config.ear_thresh:
                    self.closed_counter += 1
                else:
                    if self.closed_counter >= self.config.ear_consec_frames:
                        self.blinks += 1
                    self.closed_counter = 0

                # 头部姿态
                euler_result, _, _ = self.head_pose(w, h, pts)
                if euler_result is not None:
                    euler = euler_result

                # 视线方向
                ir_c_r = self.iris_center(pts, RIGHT_IRIS)
                ir_c_l = self.iris_center(pts, LEFT_IRIS)

                if ir_c_r is not None and ir_c_l is not None:
                    gx_r, gy_r = self.gaze_vector(eye_r, ir_c_r)
                    gx_l, gy_l = self.gaze_vector(eye_l, ir_c_l)
                    gaze_xy = ((gx_r + gx_l) / 2.0, (gy_r + gy_l) / 2.0)

                # 注意力标签
                pitch, yaw, roll = [float(x) for x in euler]
                label = self.attention_label(ear_l, ear_r, yaw, pitch, gaze_xy)

            # 更新当前状态
            self.current_state = {
                "attention_label": label,
                "ear_left": ear_l,
                "ear_right": ear_r,
                "yaw": float(euler[1]) if euler is not None else 0.0,
                "pitch": float(euler[0]) if euler is not None else 0.0,
                "roll": float(euler[2]) if euler is not None else 0.0,
                "gaze_x": gaze_xy[0],
                "gaze_y": gaze_xy[1],
                "blink_count": self.blinks,
                "face_detected": face_detected
            }

            # 更新历史记录
            attention_score = self.calculate_attention_score()
            self.attention_history.append(attention_score)
            self.ear_history.append((ear_l + ear_r) / 2)
            self.gaze_history.append(math.sqrt(gaze_xy[0] ** 2 + gaze_xy[1] ** 2))
            self.pose_history.append(abs(euler[1]) + abs(euler[0]))

            return self.current_state

        except Exception as e:
            print(f"注意力分析错误: {e}")
            return self.current_state

    def calculate_attention_score(self):
        """计算综合注意力分数"""
        state = self.current_state

        if state["attention_label"] == "未检测到面部":
            return 0

        score = 100

        # 眼睛状态扣分
        if state["attention_label"] == "眼睛闭合":
            score -= 40

        # 头部姿态扣分
        if abs(state["yaw"]) > self.config.yaw_thresh_deg:
            yaw_penalty = min(30, abs(state["yaw"]) / self.config.yaw_thresh_deg * 15)
            score -= yaw_penalty

        if abs(state["pitch"]) > self.config.pitch_thresh_deg:
            pitch_penalty = min(25, abs(state["pitch"]) / self.config.pitch_thresh_deg * 12)
            score -= pitch_penalty

        # 视线方向扣分
        gaze_magnitude = math.sqrt(state["gaze_x"] ** 2 + state["gaze_y"] ** 2)
        if gaze_magnitude > self.config.gaze_off_center:
            gaze_penalty = min(35, gaze_magnitude / self.config.gaze_off_center * 20)
            score -= gaze_penalty

        # 眨眼频率（适度眨眼是好的，但过多可能表示疲劳）
        if len(self.attention_history) > 100:
            recent_blinks = min(20, self.blinks / (len(self.attention_history) / 100) * 2)
            if recent_blinks > 15:  # 眨眼过多
                score -= 10

        return max(0, min(100, score))

    def get_attention_stats(self):
        """获取注意力统计"""
        if not self.attention_history:
            return {
                "avg_score": 0,
                "max_score": 0,
                "min_score": 0,
                "trend": "稳定",
                "focus_percentage": 0,
                "blink_rate": 0
            }

        try:
            scores = list(self.attention_history)
            avg_score = statistics.mean(scores) if scores else 0
            max_score = max(scores) if scores else 0
            min_score = min(scores) if scores else 0

            # 计算趋势
            if len(scores) >= 30:
                recent = scores[-15:]
                earlier = scores[-30:-15] if len(scores) >= 30 else recent
                recent_avg = statistics.mean(recent) if recent else 0
                earlier_avg = statistics.mean(earlier) if earlier else 0

                if recent_avg > earlier_avg + 5:
                    trend = "上升"
                elif recent_avg < earlier_avg - 5:
                    trend = "下降"
                else:
                    trend = "稳定"
            else:
                trend = "分析中"

            # 计算专注百分比
            focused_frames = sum(1 for s in scores if s >= 70)
            focus_percentage = (focused_frames / len(scores) * 100) if scores else 0

            # 计算眨眼率（每分钟）
            blink_rate = (self.blinks / (len(scores) / 30)) * 60 if scores else 0

            return {
                "avg_score": round(avg_score, 1),
                "max_score": round(max_score, 1),
                "min_score": round(min_score, 1),
                "trend": trend,
                "focus_percentage": round(focus_percentage, 1),
                "blink_rate": round(blink_rate, 1)
            }
        except:
            return {
                "avg_score": 0,
                "max_score": 0,
                "min_score": 0,
                "trend": "未知",
                "focus_percentage": 0,
                "blink_rate": 0
            }

    def reset(self):
        """重置分析器"""
        self.closed_counter = 0
        self.blinks = 0
        self.attention_history.clear()
        self.ear_history.clear()
        self.gaze_history.clear()
        self.pose_history.clear()


# ============================================================================
# 情绪识别类
# ============================================================================

class EmotionAnalyzer:
    """情绪分析器"""

    def __init__(self):
        # 情绪模型参数
        self.shape_x = 48
        self.shape_y = 48
        self.input_shape = (self.shape_x, self.shape_y, 1)
        self.nClasses = 7

        # 情绪标签
        self.emotion_labels = [
            "生气", "厌恶", "恐惧", "快乐",
            "悲伤", "惊讶", "中性"
        ]

        # 情绪颜色映射
        self.emotion_colors = {
            "生气": (0, 0, 255),  # 红色
            "厌恶": (0, 128, 0),  # 绿色
            "恐惧": (128, 0, 128),  # 紫色
            "快乐": (0, 255, 255),  # 黄色
            "悲伤": (255, 0, 0),  # 蓝色
            "惊讶": (0, 165, 255),  # 橙色
            "中性": (200, 200, 200)  # 灰色
        }

        # 加载模型
        self.model = None
        self.face_detector = None
        self.predictor = None

        try:
            # 尝试加载情绪识别模型
            self.model = load_model('Models/EmotionXCeption/video.h5')
            print("情绪模型加载成功")
        except Exception as e:
            print(f"加载情绪模型失败: {e}")
            print("使用备用情绪检测")

        try:
            # 加载dlib面部检测器和特征点预测器
            self.face_detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor("Models/Landmarks/face_landmarks.dat")
            print("Dlib模型加载成功")
        except Exception as e:
            print(f"加载dlib模型失败: {e}")

        # 状态变量
        self.emotion_history = deque(maxlen=100)
        self.current_emotion = "中性"
        self.emotion_probabilities = [0.0] * 7
        self.emotion_confidence = 0.0

        # 面部特征点索引
        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        (self.nStart, self.nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]
        (self.mStart, self.mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
        (self.jStart, self.jEnd) = face_utils.FACIAL_LANDMARKS_IDXS["jaw"]

    def detect_face_dlib(self, frame):
        """使用dlib检测面部"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = self.face_detector(gray, 1)
        return gray, rects

    def extract_face_features(self, gray, rect):
        """提取面部特征"""
        try:
            shape = self.predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # 获取面部坐标
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            face = gray[y:y + h, x:x + w]

            # 缩放面部图像
            if face.size == 0:
                return None

            face_resized = zoom(face, (self.shape_x / face.shape[0], self.shape_y / face.shape[1]))

            # 转换为浮点数并归一化
            face_resized = face_resized.astype(np.float32)
            if face_resized.max() > 0:
                face_resized /= float(face_resized.max())

            # 重塑为模型输入形状
            face_resized = np.reshape(face_resized, (1, self.shape_x, self.shape_y, 1))

            return face_resized, shape, (x, y, w, h)
        except Exception as e:
            print(f"面部特征提取错误: {e}")
            return None

    def predict_emotion(self, face_image):
        """预测情绪"""
        if self.model is None or face_image is None:
            return "中性", [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.4], 0.5

        try:
            prediction = self.model.predict(face_image, verbose=0)
            emotion_idx = np.argmax(prediction[0])
            emotion = self.emotion_labels[emotion_idx]
            confidence = float(prediction[0][emotion_idx])

            return emotion, prediction[0].tolist(), confidence
        except Exception as e:
            print(f"情绪预测错误: {e}")
            return "中性", [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.4], 0.5

    def analyze_frame(self, frame):
        """分析单帧情绪"""
        try:
            # 使用dlib检测面部
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = self.face_detector(gray, 1)

            emotions = []
            face_shapes = []
            face_boxes = []

            for rect in rects:
                # 提取面部特征
                result = self.extract_face_features(gray, rect)
                if result is None:
                    continue

                face_image, shape, (x, y, w, h) = result

                # 预测情绪
                emotion, probabilities, confidence = self.predict_emotion(face_image)

                emotions.append({
                    "emotion": emotion,
                    "probabilities": probabilities,
                    "confidence": confidence,
                    "box": (x, y, w, h)
                })

                face_shapes.append(shape)
                face_boxes.append((x, y, w, h))

            # 更新当前情绪（使用最大面部或平均）
            if emotions:
                # 选择最大面部的情绪
                max_face_idx = max(range(len(face_boxes)),
                                   key=lambda i: face_boxes[i][2] * face_boxes[i][3])

                self.current_emotion = emotions[max_face_idx]["emotion"]
                self.emotion_probabilities = emotions[max_face_idx]["probabilities"]
                self.emotion_confidence = emotions[max_face_idx]["confidence"]

                # 更新历史记录
                self.emotion_history.append(self.current_emotion)
            else:
                # 没有检测到面部
                self.current_emotion = "未检测到面部"
                self.emotion_probabilities = [0.0] * 7
                self.emotion_confidence = 0.0

            # 确保face_count正确返回
            face_count = len(rects)

            return {
                "emotion": self.current_emotion,
                "probabilities": self.emotion_probabilities,
                "confidence": self.emotion_confidence,
                "face_count": face_count,  # 确保这里返回正确的面部数量
                "face_shapes": face_shapes,
                "face_boxes": face_boxes
            }

        except Exception as e:
            print(f"情绪分析错误: {e}")
            return {
                "emotion": "错误",
                "probabilities": [0.0] * 7,
                "confidence": 0.0,
                "face_count": 0,  # 错误时也返回0
                "face_shapes": [],
                "face_boxes": []
            }

    def get_emotion_stats(self):
        """获取情绪统计"""
        if not self.emotion_history:
            return {
                "dominant_emotion": "未知",
                "emotion_stability": 0,
                "positive_ratio": 0,
                "negative_ratio": 0,
                "emotion_changes": 0
            }

        try:
            history = list(self.emotion_history)

            # 主导情绪
            from collections import Counter
            emotion_counts = Counter(history)
            dominant_emotion = emotion_counts.most_common(1)[0][0] if emotion_counts else "未知"

            # 情绪稳定性（相同情绪连续帧的比例）
            if len(history) > 1:
                changes = sum(1 for i in range(1, len(history)) if history[i] != history[i - 1])
                stability = 1 - (changes / (len(history) - 1))
            else:
                stability = 1.0

            # 积极/消极情绪比例
            positive_emotions = ["快乐", "惊讶", "中性"]
            negative_emotions = ["生气", "厌恶", "恐惧", "悲伤"]

            positive_count = sum(1 for e in history if e in positive_emotions)
            negative_count = sum(1 for e in history if e in negative_emotions)

            positive_ratio = positive_count / len(history) if history else 0
            negative_ratio = negative_count / len(history) if history else 0

            # 情绪变化次数
            emotion_changes = sum(1 for i in range(1, len(history)) if history[i] != history[i - 1])

            return {
                "dominant_emotion": dominant_emotion,
                "emotion_stability": round(stability * 100, 1),
                "positive_ratio": round(positive_ratio * 100, 1),
                "negative_ratio": round(negative_ratio * 100, 1),
                "emotion_changes": emotion_changes
            }
        except:
            return {
                "dominant_emotion": "未知",
                "emotion_stability": 0,
                "positive_ratio": 0,
                "negative_ratio": 0,
                "emotion_changes": 0
            }

    def reset(self):
        """重置分析器"""
        self.emotion_history.clear()
        self.current_emotion = "中性"
        self.emotion_probabilities = [0.0] * 7
        self.emotion_confidence = 0.0


# ============================================================================
# 语音提醒系统
# ============================================================================

class VoiceReminderSystem:
    """增强版语音提醒系统"""

    def __init__(self):
        self.engine = None
        self.is_speaking = False
        self.last_reminder_time = 0
        self.reminder_cooldown = 15

        # 语音队列和线程
        self.voice_queue = queue.Queue()
        self.voice_thread = None
        self.voice_thread_running = False

        # 语音设置
        self.speech_rate = 150
        self.volume = 0.8
        self.pitch = 110

        if TTS_AVAILABLE:
            try:
                self.engine = pyttsx3.init()
                self.setup_voice_engine()
                self.start_voice_thread()
                print("语音系统初始化成功")
            except Exception as e:
                print(f"初始化语音引擎失败: {e}")
                self.engine = None
        else:
            print("语音功能不可用")

    def setup_voice_engine(self):
        """设置语音引擎参数"""
        if not self.engine:
            return

        try:
            voices = self.engine.getProperty('voices')
            chinese_voices = []
            female_voices = []

            for voice in voices:
                voice_info = voice.name.lower()
                if 'chinese' in voice_info or 'zh' in voice_info:
                    chinese_voices.append(voice)
                elif 'female' in voice_info or 'f' in voice_info:
                    female_voices.append(voice)

            if chinese_voices:
                self.engine.setProperty('voice', chinese_voices[0].id)
            elif female_voices:
                self.engine.setProperty('voice', female_voices[0].id)

            self.engine.setProperty('rate', self.speech_rate)
            self.engine.setProperty('volume', self.volume)
            self.engine.setProperty('pitch', self.pitch)

        except Exception as e:
            print(f"语音设置错误: {e}")

    def start_voice_thread(self):
        """启动语音线程"""
        if self.engine and not self.voice_thread_running:
            self.voice_thread_running = True
            self.voice_thread = threading.Thread(
                target=self._voice_worker,
                daemon=True,
                name="语音工作线程"
            )
            self.voice_thread.start()

    def _voice_worker(self):
        """语音工作线程"""
        while self.voice_thread_running:
            try:
                text = self.voice_queue.get(timeout=2)

                if self.engine:
                    try:
                        self.engine.say(text)
                        self.engine.runAndWait()
                        self.is_speaking = False
                    except Exception as e:
                        print(f"语音播放错误: {e}")
                        self.is_speaking = False

                self.voice_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"语音工作线程错误: {e}")
                continue

    def speak(self, text):
        """语音播报"""
        if self.engine is None:
            return False

        current_time = time.time()

        if current_time - self.last_reminder_time < self.reminder_cooldown:
            return False

        try:
            self.voice_queue.put(text)
            self.last_reminder_time = current_time
            self.is_speaking = True
            return True
        except Exception as e:
            print(f"添加语音到队列失败: {e}")
            return False

    def stop(self):
        """停止语音系统"""
        self.voice_thread_running = False
        if self.voice_thread:
            self.voice_thread.join(timeout=2)
        if self.engine:
            try:
                self.engine.stop()
            except:
                pass


# ============================================================================
# 主UI界面
# ============================================================================

class ADHDDetectionSystem(QMainWindow):
    """多动症儿童注意力与情绪检测系统"""

    # 添加信号
    modeling_progress_updated = pyqtSignal(int)
    modeling_finished = pyqtSignal(bool)

    def __init__(self):
        super().__init__()

        # 设置中文字体
        QFontDatabase.addApplicationFont("msyh.ttc")  # 微软雅黑
        font = QFont("Microsoft YaHei", 9)
        QApplication.setFont(font)

        # 初始化分析器
        self.attention_analyzer = AttentionAnalyzer()
        self.emotion_analyzer = EmotionAnalyzer()
        self.voice_system = VoiceReminderSystem()

        # 初始化新功能
        self.facial_modeling = FacialModeling()
        self.attention_scoring = OptimizedAttentionScoringSystem()
        self.calibration_system = CalibrationSystem()
        self.realtime_charts = RealTimeCharts()

        # 摄像头和视频
        self.camera = None
        self.video_path = None
        self.video_capture = None
        self.is_live = False
        self.is_playing = False

        # 记录状态
        self.is_recording = False
        self.video_writer = None
        self.record_data = []
        self.session_start_time = None
        self.frame_count = 0

        # 初始化UI相关的属性
        self.neutral_duration_label = None
        self.attention_stability_label = None
        self.gaze_deviation_label = None
        self.head_stability_label = None
        self.focus_duration_label = None
        self.distraction_count_label = None
        self.refocus_rate_label = None
        self.emotion_change_freq_label = None
        self.extreme_emotion_label = None
        self.face_count_label = None

        # 显示设置
        self.show_attention_overlay = True
        self.show_emotion_overlay = True
        self.show_landmarks = True
        self.show_calibration = False

        # 初始化所有UI相关的属性
        self.min_attention_label = None
        self.focus_percent_label = None
        self.trend_label = None
        self.blink_rate_label = None
        self.dominant_emotion_label = None
        self.emotion_stability_label = None
        self.positive_ratio_label = None
        self.negative_ratio_label = None

        # 详细统计标签
        self.attention_stability_label = None
        self.gaze_deviation_label = None
        self.head_stability_label = None
        self.focus_duration_label = None
        self.distraction_count_label = None
        self.refocus_rate_label = None

        # 情绪统计标签
        self.face_count_label = None
        self.emotion_change_freq_label = None
        self.neutral_duration_label = None
        self.extreme_emotion_label = None

        # 多动症特征标签
        self.inattention_ratio_label = None
        self.hyperactivity_label = None
        self.emotion_volatility_label = None
        self.risk_level_label = None
        self.focus_pattern_label = None
        self.adhd_features_label = None

        # 状态变量
        self.attention_stats_history = deque(maxlen=100)
        self.emotion_stats_history = deque(maxlen=100)
        self.alerts = []
        self.is_calibrating = False
        self.calibration_step = 0

        # 初始化UI
        self.init_ui()

        # 定时器
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # 状态更新定时器
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(1000)

        # 图表更新定时器
        self.chart_timer = QTimer()
        self.chart_timer.timeout.connect(self.update_charts_widgets)
        self.chart_timer.start(200)  # 5 FPS更新图表

        # 语音控制
        self.voice_enabled = True

        # 连接信号和槽
        self.modeling_progress_updated.connect(self.update_modeling_progress)
        self.modeling_finished.connect(self.finish_modeling)

        print("多动症检测系统初始化完成")

    def init_ui(self):
        """初始化UI界面"""
        # 设置中文字体
        self.setFont(QFont("Microsoft YaHei", 9))

        self.setWindowTitle("多动症儿童注意力与情绪检测系统 v5.0")
        self.setGeometry(100, 100, 1200, 800)

        # 设置样式
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f7fa;
            }
            QGroupBox {
                font-size: 14px;
                font-weight: bold;
                border: 2px solid #4a6fa5;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 8px 0 8px;
                color: #2c3e50;
            }
            QPushButton {
                font-size: 13px;
                font-weight: bold;
                padding: 8px 15px;
                border-radius: 6px;
                background-color: #4a6fa5;
                color: white;
                border: 1px solid #385d8a;
            }
            QPushButton:hover {
                background-color: #385d8a;
            }
            QPushButton:pressed {
                background-color: #2c4a6e;
            }
            QLabel {
                font-size: 13px;
                color: #34495e;
            }
            QTextEdit {
                font-size: 12px;
                border: 1px solid #d1d9e6;
                border-radius: 4px;
                background-color: white;
                padding: 5px;
            }
            QProgressBar {
                border: 1px solid #d1d9e6;
                border-radius: 4px;
                text-align: center;
                background-color: white;
            }
            QProgressBar::chunk {
                border-radius: 4px;
                background-color: #3498db;
            }
            QCheckBox {
                font-size: 13px;
                color: #34495e;
            }
            QComboBox {
                padding: 5px;
                border: 1px solid #d1d9e6;
                border-radius: 4px;
                background-color: white;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #d1d9e6;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #3498db;
                border: 1px solid #2980b9;
                width: 18px;
                height: 18px;
                margin: -6px 0;
                border-radius: 9px;
            }
        """)

        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 主布局
        main_layout = QHBoxLayout()
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
        central_widget.setLayout(main_layout)

        # ====================================================================
        # 左侧面板：视频和图表
        # ====================================================================
        left_panel = QVBoxLayout()
        left_panel.setSpacing(10)

        # 视频显示区域
        video_group = QGroupBox("📹 实时视频画面")
        video_layout = QVBoxLayout()

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(1000, 450)  # 减小尺寸
        self.video_label.setMaximumSize(1000, 450)  # 减小尺寸
        self.video_label.setStyleSheet("""
            background-color: #2c3e50;
            border: 3px solid #4a6fa5;
            border-radius: 8px;
            color: white;
            font-size: 16px;
        """)
        self.video_label.setText("等待视频源...")

        # 视频信息标签
        self.video_info_label = QLabel("就绪")
        self.video_info_label.setStyleSheet("color: #7f8c8d; font-size: 12px;")
        self.video_info_label.setAlignment(Qt.AlignCenter)

        video_layout.addWidget(self.video_label)
        video_layout.addWidget(self.video_info_label)

        # 视频控制按钮
        control_layout = QHBoxLayout()

        self.camera_btn = QPushButton("📷 启动摄像头")
        self.camera_btn.setStyleSheet("background-color: #27ae60;")
        self.camera_btn.clicked.connect(self.start_camera)

        self.video_btn = QPushButton("📁 上传视频")
        self.video_btn.setStyleSheet("background-color: #e67e22;")
        self.video_btn.clicked.connect(self.upload_video)

        self.record_btn = QPushButton("● 开始录制")
        self.record_btn.setStyleSheet("background-color: #e74c3c;")
        self.record_btn.clicked.connect(self.toggle_recording)
        self.record_btn.setEnabled(False)

        self.pause_btn = QPushButton("⏸️ 暂停")
        self.pause_btn.setStyleSheet("background-color: #f39c12;")
        self.pause_btn.clicked.connect(self.toggle_pause)
        self.pause_btn.setEnabled(False)

        self.calibrate_btn = QPushButton("🎯 开始校准")
        self.calibrate_btn.setStyleSheet("background-color: #9b59b6;")
        self.calibrate_btn.clicked.connect(self.start_calibration)
        self.calibrate_btn.setEnabled(False)

        control_layout.addWidget(self.camera_btn)
        control_layout.addWidget(self.video_btn)
        control_layout.addWidget(self.record_btn)
        control_layout.addWidget(self.pause_btn)
        control_layout.addWidget(self.calibrate_btn)
        control_layout.addStretch()

        video_layout.addLayout(control_layout)
        video_group.setLayout(video_layout)
        left_panel.addWidget(video_group)

        # 实时图表区域
        charts_group = QGroupBox("📊 实时图表")
        charts_layout = QGridLayout()
        charts_layout.setHorizontalSpacing(5)  # 减少水平间距
        charts_layout.setVerticalSpacing(5)  # 减少垂直间距

        # 创建自定义图表标签
        self.attention_chart_widget = QLabel()
        self.attention_chart_widget.setAlignment(Qt.AlignCenter)
        self.attention_chart_widget.setMinimumSize(320, 160)  # 减小尺寸
        self.attention_chart_widget.setMaximumSize(320, 160)  # 减小尺寸
        self.attention_chart_widget.setStyleSheet("""
            background-color: white;
            border: 1px solid #d1d9e6;
            border-radius: 4px;
        """)
        self.attention_chart_widget.setText("正在初始化...")

        self.gaze_chart_widget = QLabel()
        self.gaze_chart_widget.setAlignment(Qt.AlignCenter)
        self.gaze_chart_widget.setMinimumSize(320, 160)  # 减小尺寸
        self.gaze_chart_widget.setMaximumSize(320, 160)  # 减小尺寸
        self.gaze_chart_widget.setStyleSheet("""
            background-color: white;
            border: 1px solid #d1d9e6;
            border-radius: 4px;
        """)
        self.gaze_chart_widget.setText("正在初始化...")

        self.eye_chart_widget = QLabel()
        self.eye_chart_widget.setAlignment(Qt.AlignCenter)
        self.eye_chart_widget.setMinimumSize(320, 160)  # 减小尺寸
        self.eye_chart_widget.setMaximumSize(320, 160)  # 减小尺寸
        self.eye_chart_widget.setStyleSheet("""
            background-color: white;
            border: 1px solid #d1d9e6;
            border-radius: 4px;
        """)
        self.eye_chart_widget.setText("正在初始化...")

        self.attention_chart_title = QLabel("注意力分数趋势")
        self.attention_chart_title.setAlignment(Qt.AlignCenter)
        self.attention_chart_title.setStyleSheet("font-weight: bold; font-size: 12px;")

        self.gaze_chart_title = QLabel("视线追踪")
        self.gaze_chart_title.setAlignment(Qt.AlignCenter)
        self.gaze_chart_title.setStyleSheet("font-weight: bold; font-size: 12px;")

        self.eye_chart_title = QLabel("眼部与头部特征")
        self.eye_chart_title.setAlignment(Qt.AlignCenter)
        self.eye_chart_title.setStyleSheet("font-weight: bold; font-size: 12px;")

        charts_layout.addWidget(self.attention_chart_title, 0, 0)
        charts_layout.addWidget(self.gaze_chart_title, 0, 1)
        charts_layout.addWidget(self.eye_chart_title, 0, 2)

        charts_layout.addWidget(self.attention_chart_widget, 1, 0)
        charts_layout.addWidget(self.gaze_chart_widget, 1, 1)
        charts_layout.addWidget(self.eye_chart_widget, 1, 2)

        charts_group.setLayout(charts_layout)
        left_panel.addWidget(charts_group)

        # ====================================================================
        # 右侧面板：分析和控制
        # ====================================================================
        right_panel = QVBoxLayout()
        right_panel.setSpacing(10)

        # 创建选项卡
        self.right_tab_widget = QTabWidget()

        # 创建各个选项卡页面
        self.attention_emotion_tab = QWidget()
        self.calibration_tab = QWidget()
        self.control_tab = QWidget()
        self.stats_tab = QWidget()
        self.alert_tab = QWidget()

        # 设置各个选项卡的布局
        self.setup_attention_emotion_tab()
        self.setup_calibration_tab()
        self.setup_control_tab()
        self.setup_stats_tab()
        self.setup_alert_tab()

        # 添加选项卡
        self.right_tab_widget.addTab(self.attention_emotion_tab, "🎯 注意力与情绪")
        self.right_tab_widget.addTab(self.calibration_tab, "🎯 校准")
        self.right_tab_widget.addTab(self.control_tab, "⚙️ 控制")
        self.right_tab_widget.addTab(self.stats_tab, "📈 统计")
        self.right_tab_widget.addTab(self.alert_tab, "⚠️ 警报")

        right_panel.addWidget(self.right_tab_widget)

        # 操作按钮组（放在选项卡下方）
        action_group = QGroupBox("🛠️ 操作")
        action_layout = QHBoxLayout()

        self.export_btn = QPushButton("📊 导出报告")
        self.export_btn.clicked.connect(self.export_report)
        self.export_btn.setStyleSheet("background-color: #9b59b6;")

        self.reset_btn = QPushButton("🔄 重置分析")
        self.reset_btn.clicked.connect(self.reset_analysis)
        self.reset_btn.setStyleSheet("background-color: #95a5a6;")

        self.quit_btn = QPushButton("🚪 退出")
        self.quit_btn.clicked.connect(self.close)
        self.quit_btn.setStyleSheet("background-color: #e74c3c;")

        action_layout.addWidget(self.export_btn)
        action_layout.addWidget(self.reset_btn)
        action_layout.addWidget(self.quit_btn)
        action_group.setLayout(action_layout)
        right_panel.addWidget(action_group)

        # 将左右面板添加到主布局
        main_layout.addLayout(left_panel, 3)
        main_layout.addLayout(right_panel, 2)

    def setup_attention_emotion_tab(self):
        """设置注意力与情绪选项卡"""
        layout = QVBoxLayout()
        layout.setSpacing(10)

        # 注意力分析组
        attention_group = QGroupBox("🎯 注意力分析")
        attention_layout = QVBoxLayout()

        # 注意力分数
        score_layout = QHBoxLayout()
        score_layout.addWidget(QLabel("注意力分数:"))
        self.attention_score_label = QLabel("0")
        self.attention_score_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #2c3e50;")
        score_layout.addWidget(self.attention_score_label)
        score_layout.addStretch()

        # 注意力状态
        state_layout = QHBoxLayout()
        state_layout.addWidget(QLabel("状态:"))
        self.attention_state_label = QLabel("初始化中")
        self.attention_state_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #7f8c8d;")
        state_layout.addWidget(self.attention_state_label)
        state_layout.addStretch()

        # 注意力进度条
        self.attention_progress = QProgressBar()
        self.attention_progress.setRange(0, 100)
        self.attention_progress.setValue(0)
        self.attention_progress.setTextVisible(True)
        self.attention_progress.setFormat("%v/100")

        # 详细指标
        metrics_layout = QGridLayout()
        metrics_layout.addWidget(QLabel("眼睛纵横比:"), 0, 0)
        self.ear_label = QLabel("0.00")
        metrics_layout.addWidget(self.ear_label, 0, 1)

        metrics_layout.addWidget(QLabel("头部偏转:"), 1, 0)
        self.yaw_label = QLabel("0.0°")
        metrics_layout.addWidget(self.yaw_label, 1, 1)

        metrics_layout.addWidget(QLabel("头部俯仰:"), 2, 0)
        self.pitch_label = QLabel("0.0°")
        metrics_layout.addWidget(self.pitch_label, 2, 1)

        metrics_layout.addWidget(QLabel("视线X:"), 0, 2)
        self.gaze_x_label = QLabel("0.00")
        metrics_layout.addWidget(self.gaze_x_label, 0, 3)

        metrics_layout.addWidget(QLabel("视线Y:"), 1, 2)
        self.gaze_y_label = QLabel("0.00")
        metrics_layout.addWidget(self.gaze_y_label, 1, 3)

        metrics_layout.addWidget(QLabel("眨眼次数:"), 2, 2)
        self.blink_label = QLabel("0")
        metrics_layout.addWidget(self.blink_label, 2, 3)

        attention_layout.addLayout(score_layout)
        attention_layout.addLayout(state_layout)
        attention_layout.addWidget(self.attention_progress)
        attention_layout.addLayout(metrics_layout)
        attention_group.setLayout(attention_layout)
        layout.addWidget(attention_group)

        # 情绪分析组
        emotion_group = QGroupBox("😊 情绪分析")
        emotion_layout = QVBoxLayout()

        # 当前情绪
        current_emotion_layout = QHBoxLayout()
        current_emotion_layout.addWidget(QLabel("当前情绪:"))
        self.emotion_label = QLabel("中性")
        self.emotion_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #2c3e50;")
        current_emotion_layout.addWidget(self.emotion_label)
        current_emotion_layout.addStretch()

        # 情绪信心
        confidence_layout = QHBoxLayout()
        confidence_layout.addWidget(QLabel("置信度:"))
        self.confidence_label = QLabel("0%")
        self.confidence_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #7f8c8d;")
        confidence_layout.addWidget(self.confidence_label)
        confidence_layout.addStretch()

        # 情绪概率条
        self.emotion_bars = {}
        emotions = ["生气", "厌恶", "恐惧", "快乐", "悲伤", "惊讶", "中性"]

        for emotion in emotions:
            emotion_bar_layout = QHBoxLayout()
            emotion_bar_layout.addWidget(QLabel(f"{emotion}:"))

            progress_bar = QProgressBar()
            progress_bar.setRange(0, 100)
            progress_bar.setValue(0)
            progress_bar.setTextVisible(True)
            progress_bar.setFormat("%v%")
            progress_bar.setMaximumHeight(20)

            self.emotion_bars[emotion] = progress_bar
            emotion_bar_layout.addWidget(progress_bar)
            emotion_layout.addLayout(emotion_bar_layout)

        emotion_layout.addLayout(current_emotion_layout)
        emotion_layout.addLayout(confidence_layout)
        emotion_group.setLayout(emotion_layout)
        layout.addWidget(emotion_group)

        layout.addStretch()
        self.attention_emotion_tab.setLayout(layout)

    def setup_calibration_tab(self):
        """设置校准选项卡"""
        layout = QVBoxLayout()
        layout.setSpacing(10)

        # 校准状态组
        calibration_group = QGroupBox("🎯 校准状态")
        calibration_layout = QVBoxLayout()

        self.calibration_status_label = QLabel("未校准")
        self.calibration_status_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #7f8c8d;")

        self.calibration_progress = QProgressBar()
        self.calibration_progress.setRange(0, 100)
        self.calibration_progress.setValue(0)
        self.calibration_progress.setTextVisible(True)
        self.calibration_progress.setFormat("校准进度: %p%")

        self.calibration_instruction = QLabel("点击'开始校准'按钮进行校准")
        self.calibration_instruction.setStyleSheet("color: #95a5a6; font-size: 11px;")
        self.calibration_instruction.setWordWrap(True)

        self.calibration_info = QLabel("")
        self.calibration_info.setStyleSheet("color: #34495e; font-size: 10px;")
        self.calibration_info.setWordWrap(True)

        calibration_layout.addWidget(self.calibration_status_label)
        calibration_layout.addWidget(self.calibration_progress)
        calibration_layout.addWidget(self.calibration_instruction)
        calibration_layout.addWidget(self.calibration_info)

        # 校准控制按钮
        calibration_buttons = QHBoxLayout()
        self.calibration_reset_btn = QPushButton("重置校准")
        self.calibration_reset_btn.setStyleSheet("background-color: #95a5a6;")
        self.calibration_reset_btn.clicked.connect(self.reset_calibration)
        self.calibration_reset_btn.setEnabled(False)

        self.calibration_auto_btn = QPushButton("自动面部建模")
        self.calibration_auto_btn.setStyleSheet("background-color: #3498db;")
        self.calibration_auto_btn.clicked.connect(self.auto_facial_modeling)
        self.calibration_auto_btn.setEnabled(False)

        calibration_buttons.addWidget(self.calibration_reset_btn)
        calibration_buttons.addWidget(self.calibration_auto_btn)
        calibration_buttons.addStretch()

        calibration_layout.addLayout(calibration_buttons)
        calibration_group.setLayout(calibration_layout)
        layout.addWidget(calibration_group)

        # 校准结果信息
        result_group = QGroupBox("📋 校准结果")
        result_layout = QVBoxLayout()

        self.calibration_result_label = QLabel("暂无校准结果")
        self.calibration_result_label.setStyleSheet("color: #7f8c8d; font-size: 11px;")
        self.calibration_result_label.setWordWrap(True)

        result_layout.addWidget(self.calibration_result_label)
        result_group.setLayout(result_layout)
        layout.addWidget(result_group)

        layout.addStretch()
        self.calibration_tab.setLayout(layout)

    def setup_control_tab(self):
        """设置控制选项卡"""
        layout = QVBoxLayout()
        layout.setSpacing(10)

        # 显示设置组
        display_group = QGroupBox("👁️ 显示设置")
        display_layout = QVBoxLayout()

        self.show_attention_check = QCheckBox("显示注意力叠加")
        self.show_attention_check.setChecked(True)
        self.show_attention_check.stateChanged.connect(self.toggle_attention_overlay)

        self.show_emotion_check = QCheckBox("显示情绪叠加")
        self.show_emotion_check.setChecked(True)
        self.show_emotion_check.stateChanged.connect(self.toggle_emotion_overlay)

        self.show_landmarks_check = QCheckBox("显示特征点")
        self.show_landmarks_check.setChecked(True)

        display_layout.addWidget(self.show_attention_check)
        display_layout.addWidget(self.show_emotion_check)
        display_layout.addWidget(self.show_landmarks_check)
        display_group.setLayout(display_layout)
        layout.addWidget(display_group)

        # 语音控制组
        voice_group = QGroupBox("🔊 语音控制")
        voice_layout = QVBoxLayout()

        self.voice_check = QCheckBox("启用语音提醒")
        self.voice_check.setChecked(True)
        self.voice_check.stateChanged.connect(self.toggle_voice)

        voice_button_layout = QHBoxLayout()
        self.test_voice_btn = QPushButton("测试语音")
        self.test_voice_btn.clicked.connect(self.test_voice)
        self.test_voice_btn.setMaximumWidth(100)

        voice_button_layout.addWidget(self.test_voice_btn)
        voice_button_layout.addStretch()

        voice_layout.addWidget(self.voice_check)
        voice_layout.addLayout(voice_button_layout)
        voice_group.setLayout(voice_layout)
        layout.addWidget(voice_group)

        # 分析设置组
        analysis_group = QGroupBox("🔍 分析设置")
        analysis_layout = QVBoxLayout()

        self.smooth_check = QCheckBox("平滑分析")
        self.smooth_check.setChecked(True)

        self.auto_reset_check = QCheckBox("自动重置")
        self.auto_reset_check.setChecked(False)

        analysis_layout.addWidget(self.smooth_check)
        analysis_layout.addWidget(self.auto_reset_check)
        analysis_group.setLayout(analysis_layout)
        layout.addWidget(analysis_group)

        layout.addStretch()
        self.control_tab.setLayout(layout)

    def setup_stats_tab(self):
        """设置统计选项卡（完整版）"""
        layout = QVBoxLayout()
        layout.setSpacing(10)

        # 基本统计信息组
        stats_group = QGroupBox("📊 基本统计")
        stats_layout = QGridLayout()

        # 第一行：平均分数和最高分数
        stats_layout.addWidget(QLabel("平均分数:"), 0, 0)
        self.avg_attention_label = QLabel("0")
        stats_layout.addWidget(self.avg_attention_label, 0, 1)

        stats_layout.addWidget(QLabel("最高分数:"), 0, 2)
        self.max_attention_label = QLabel("0")
        stats_layout.addWidget(self.max_attention_label, 0, 3)

        # 第二行：最低分数和专注比例
        stats_layout.addWidget(QLabel("最低分数:"), 1, 0)
        self.min_attention_label = QLabel("0")  # 添加缺失的标签
        stats_layout.addWidget(self.min_attention_label, 1, 1)

        stats_layout.addWidget(QLabel("专注比例:"), 1, 2)
        self.focus_percent_label = QLabel("0%")
        stats_layout.addWidget(self.focus_percent_label, 1, 3)

        # 第三行：趋势和眨眼频率
        stats_layout.addWidget(QLabel("趋势:"), 2, 0)
        self.trend_label = QLabel("稳定")
        stats_layout.addWidget(self.trend_label, 2, 1)

        stats_layout.addWidget(QLabel("眨眼频率:"), 2, 2)
        self.blink_rate_label = QLabel("0/分钟")
        stats_layout.addWidget(self.blink_rate_label, 2, 3)

        # 第四行：主导情绪
        stats_layout.addWidget(QLabel("主导情绪:"), 3, 0)
        self.dominant_emotion_label = QLabel("未知")
        stats_layout.addWidget(self.dominant_emotion_label, 3, 1)

        stats_layout.addWidget(QLabel("情绪稳定性:"), 3, 2)
        self.emotion_stability_label = QLabel("0%")
        stats_layout.addWidget(self.emotion_stability_label, 3, 3)

        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)

        # 详细统计信息组
        detail_stats_group = QGroupBox("📈 详细统计")
        detail_stats_layout = QGridLayout()

        # 第一行
        detail_stats_layout.addWidget(QLabel("注意力稳定性:"), 0, 0)
        self.attention_stability_label = QLabel("0%")
        detail_stats_layout.addWidget(self.attention_stability_label, 0, 1)

        detail_stats_layout.addWidget(QLabel("视线偏移度:"), 0, 2)
        self.gaze_deviation_label = QLabel("0.00")
        detail_stats_layout.addWidget(self.gaze_deviation_label, 0, 3)

        # 第二行
        detail_stats_layout.addWidget(QLabel("头部稳定性:"), 1, 0)
        self.head_stability_label = QLabel("0%")
        detail_stats_layout.addWidget(self.head_stability_label, 1, 1)

        detail_stats_layout.addWidget(QLabel("专注时长:"), 1, 2)
        self.focus_duration_label = QLabel("0秒")
        detail_stats_layout.addWidget(self.focus_duration_label, 1, 3)

        # 第三行
        detail_stats_layout.addWidget(QLabel("分心次数:"), 2, 0)
        self.distraction_count_label = QLabel("0")
        detail_stats_layout.addWidget(self.distraction_count_label, 2, 1)

        detail_stats_layout.addWidget(QLabel("重新专注率:"), 2, 2)
        self.refocus_rate_label = QLabel("0%")
        detail_stats_layout.addWidget(self.refocus_rate_label, 2, 3)

        # 第四行：积极/消极比例
        detail_stats_layout.addWidget(QLabel("积极比例:"), 3, 0)
        self.positive_ratio_label = QLabel("0%")
        detail_stats_layout.addWidget(self.positive_ratio_label, 3, 1)

        detail_stats_layout.addWidget(QLabel("消极比例:"), 3, 2)
        self.negative_ratio_label = QLabel("0%")
        detail_stats_layout.addWidget(self.negative_ratio_label, 3, 3)

        detail_stats_group.setLayout(detail_stats_layout)
        layout.addWidget(detail_stats_group)

        # 情绪统计组
        emotion_stats_group = QGroupBox("😊 情绪统计")
        emotion_stats_layout = QGridLayout()

        # 第一行
        emotion_stats_layout.addWidget(QLabel("面部数量:"), 0, 0)
        self.face_count_label = QLabel("0")
        emotion_stats_layout.addWidget(self.face_count_label, 0, 1)

        emotion_stats_layout.addWidget(QLabel("情绪变化频率:"), 0, 2)
        self.emotion_change_freq_label = QLabel("0次/分钟")
        emotion_stats_layout.addWidget(self.emotion_change_freq_label, 0, 3)

        # 第二行
        emotion_stats_layout.addWidget(QLabel("中性时长:"), 1, 0)
        self.neutral_duration_label = QLabel("0%")
        emotion_stats_layout.addWidget(self.neutral_duration_label, 1, 1)

        emotion_stats_layout.addWidget(QLabel("极端情绪:"), 1, 2)
        self.extreme_emotion_label = QLabel("无")
        self.extreme_emotion_label.setStyleSheet("font-weight: bold; color: #27ae60;")
        emotion_stats_layout.addWidget(self.extreme_emotion_label, 1, 3)

        # 第三行：情绪一致性
        emotion_stats_layout.addWidget(QLabel("情绪一致性:"), 2, 0)
        self.emotion_consistency_label = QLabel("高")
        emotion_stats_layout.addWidget(self.emotion_consistency_label, 2, 1)

        # 添加多动症特征分析组
        adhd_group = QGroupBox("🔍 多动症特征分析")
        adhd_layout = QGridLayout()

        # 第一行
        adhd_layout.addWidget(QLabel("注意力不集中比例:"), 0, 0)
        self.inattention_ratio_label = QLabel("0%")
        adhd_layout.addWidget(self.inattention_ratio_label, 0, 1)

        adhd_layout.addWidget(QLabel("活动过度指数:"), 0, 2)
        self.hyperactivity_label = QLabel("0")
        adhd_layout.addWidget(self.hyperactivity_label, 0, 3)

        # 第二行
        adhd_layout.addWidget(QLabel("情绪波动指数:"), 1, 0)
        self.emotion_volatility_label = QLabel("0")
        adhd_layout.addWidget(self.emotion_volatility_label, 1, 1)

        adhd_layout.addWidget(QLabel("总体风险等级:"), 1, 2)
        self.risk_level_label = QLabel("正常")
        self.risk_level_label.setStyleSheet("font-weight: bold; color: #27ae60;")
        adhd_layout.addWidget(self.risk_level_label, 1, 3)

        # 第三行：专注模式
        adhd_layout.addWidget(QLabel("专注模式:"), 2, 0)
        self.focus_pattern_label = QLabel("分析中")
        adhd_layout.addWidget(self.focus_pattern_label, 2, 1)

        adhd_layout.addWidget(QLabel("ADHD特征:"), 2, 2)
        self.adhd_features_label = QLabel("无")
        adhd_layout.addWidget(self.adhd_features_label, 2, 3)

        adhd_group.setLayout(adhd_layout)
        layout.addWidget(adhd_group)

        layout.addStretch()
        self.stats_tab.setLayout(layout)

    def setup_alert_tab(self):
        """设置警报选项卡"""
        layout = QVBoxLayout()

        # 警报和日志组
        alert_group = QGroupBox("⚠️ 警报与日志")
        alert_layout = QVBoxLayout()

        self.alert_text = QTextEdit()
        self.alert_text.setReadOnly(True)
        self.alert_text.setStyleSheet("font-size: 11px; background-color: #f8f9fa;")

        alert_layout.addWidget(self.alert_text)
        alert_group.setLayout(alert_layout)
        layout.addWidget(alert_group)

        self.alert_tab.setLayout(layout)

    def start_calibration(self):
        """开始校准"""
        if not self.is_playing:
            QMessageBox.warning(self, "警告", "请先启动摄像头或加载视频")
            return

        self.is_calibrating = True
        self.calibration_step = 0
        self.calibration_system.start_calibration()

        # 更新UI
        self.calibrate_btn.setText("🔄 校准中...")
        self.calibrate_btn.setStyleSheet("background-color: #f39c12;")
        self.calibration_status_label.setText("校准中...")
        self.calibration_instruction.setText("请注视屏幕中央的红点")
        self.calibration_progress.setValue(0)
        self.calibration_info.setText("步骤 1/5: 注视中心点")

        self.add_alert("开始校准，请按照提示注视屏幕上的点", "info")

    def reset_calibration(self):
        """重置校准"""
        self.calibration_system.reset_calibration()
        self.facial_modeling.reset_calibration()
        self.attention_scoring.reset()

        self.is_calibrating = False
        self.calibrate_btn.setText("🎯 开始校准")
        self.calibrate_btn.setStyleSheet("background-color: #9b59b6;")
        self.calibration_status_label.setText("未校准")
        self.calibration_instruction.setText("点击'开始校准'按钮进行校准")
        self.calibration_progress.setValue(0)
        self.calibration_info.setText("")

        self.add_alert("校准已重置", "info")

    def auto_facial_modeling(self):
        """自动面部建模"""
        if not self.is_playing:
            QMessageBox.warning(self, "警告", "请先启动摄像头或加载视频")
            return

        self.add_alert("开始自动面部建模，请保持正面注视摄像头", "info")

        # 启用禁用按钮
        self.calibration_auto_btn.setEnabled(False)
        self.calibration_auto_btn.setText("建模中...")

        # 在后台线程中进行面部建模
        modeling_thread = threading.Thread(target=self._perform_facial_modeling)
        modeling_thread.daemon = True
        modeling_thread.start()

    def _perform_facial_modeling(self):
        """执行面部建模（在线程中运行）"""
        frames_collected = 0
        modeling_success = False

        try:
            for i in range(30):  # 收集30帧
                if not self.is_playing:
                    break

                # 模拟获取帧并建模
                time.sleep(0.1)
                frames_collected += 1

                # 使用信号更新进度（线程安全）
                self.modeling_progress_updated.emit(frames_collected)

            if frames_collected >= 20:  # 至少收集20帧
                modeling_success = True

        except Exception as e:
            print(f"面部建模错误: {e}")

        # 使用信号通知完成
        self.modeling_finished.emit(modeling_success)

    def update_modeling_progress(self, frames_collected):
        """更新建模进度（在主线程中执行）"""
        progress = frames_collected * 100 // 30
        self.calibration_progress.setValue(progress)
        self.calibration_info.setText(f"正在采集面部数据: {frames_collected}/30 帧")

    def finish_modeling(self, success):
        """完成建模（在主线程中执行）"""
        if success:
            self.add_alert("自动面部建模成功", "info")
            self.calibration_status_label.setText("面部建模完成")
            self.calibration_instruction.setText("面部建模完成，可以进行校准")
        else:
            self.add_alert("自动面部建模失败", "warning")
            self.calibration_status_label.setText("面部建模失败")
            self.calibration_instruction.setText("面部建模失败，请重试")

        # 恢复按钮状态
        self.calibration_auto_btn.setEnabled(True)
        self.calibration_auto_btn.setText("自动面部建模")

    def start_camera(self):
        """启动摄像头"""
        try:
            if self.camera is not None:
                self.stop_video()

            self.camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

            if not self.camera.isOpened():
                self.camera = cv2.VideoCapture(0)

            if not self.camera.isOpened():
                QMessageBox.critical(self, "错误", "无法打开摄像头")
                return

            # 设置摄像头参数
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, 15)

            self.is_live = True
            self.is_playing = True
            self.video_path = None
            self.video_capture = self.camera

            self.record_btn.setEnabled(True)
            self.pause_btn.setEnabled(True)
            self.calibrate_btn.setEnabled(True)  # 启用校准按钮
            self.calibration_reset_btn.setEnabled(True)
            self.calibration_auto_btn.setEnabled(True)

            self.camera_btn.setText("📷 停止摄像头")
            self.camera_btn.setStyleSheet("background-color: #e74c3c;")

            self.reset_analysis()
            self.session_start_time = datetime.now()
            self.frame_count = 0

            self.timer.start(100)  # 10 FPS

            self.add_alert("摄像头启动成功", "info")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"启动摄像头失败: {str(e)}")

    def upload_video(self):
        """上传视频文件"""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "选择视频文件",
                "", "视频文件 (*.mp4 *.avi *.mov *.mkv)"
            )

            if not file_path:
                return

            self.stop_video()

            self.video_capture = cv2.VideoCapture(file_path)
            if not self.video_capture.isOpened():
                QMessageBox.critical(self, "错误", "无法打开视频文件")
                return

            self.video_path = file_path
            self.is_live = False
            self.is_playing = True

            self.record_btn.setEnabled(True)
            self.pause_btn.setEnabled(True)
            self.calibrate_btn.setEnabled(True)  # 启用校准按钮
            self.calibration_reset_btn.setEnabled(True)
            self.calibration_auto_btn.setEnabled(True)

            self.video_btn.setText("📁 停止视频")
            self.video_btn.setStyleSheet("background-color: #e74c3c;")

            self.reset_analysis()
            self.session_start_time = datetime.now()
            self.frame_count = 0

            self.timer.start(33)  # ~30 FPS for videos

            filename = os.path.basename(file_path)
            self.add_alert(f"视频加载成功: {filename}", "info")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载视频失败: {str(e)}")

    def stop_video(self):
        """停止视频"""
        self.timer.stop()

        if self.camera:
            self.camera.release()
            self.camera = None

        if self.video_capture and not self.is_live:
            self.video_capture.release()
            self.video_capture = None

        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None

        self.is_playing = False
        self.is_recording = False

        self.record_btn.setText("● Start Recording")
        self.record_btn.setStyleSheet("background-color: #e74c3c;")
        self.record_btn.setEnabled(False)

        self.pause_btn.setText("⏸️ Pause")
        self.pause_btn.setEnabled(False)

        self.camera_btn.setText("📷 Start Camera")
        self.camera_btn.setStyleSheet("background-color: #27ae60;")

        self.video_btn.setText("📁 Upload Video")
        self.video_btn.setStyleSheet("background-color: #e67e22;")

        # 显示黑色画面，不显示文字
        black_pixmap = QPixmap(900, 500)
        black_pixmap.fill(Qt.black)
        self.video_label.setPixmap(black_pixmap)

    def toggle_pause(self):
        """切换暂停状态"""
        if not self.is_playing:
            self.is_playing = True
            self.pause_btn.setText("⏸️ 暂停")
            self.timer.start(100 if self.is_live else 33)
            self.add_alert("视频恢复播放", "info")
        else:
            self.is_playing = False
            self.pause_btn.setText("▶️ 恢复")
            self.timer.stop()
            self.add_alert("视频暂停", "info")

    def toggle_recording(self):
        """切换录制状态"""
        if not self.is_recording:
            # 开始录制
            self.is_recording = True
            self.record_btn.setText("⏹️ 停止录制")
            self.record_btn.setStyleSheet("background-color: #2c3e50;")

            # 创建视频写入器
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"多动症分析_{timestamp}.avi"

            # 获取帧尺寸
            ret, frame = self.video_capture.read()
            if ret:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                self.video_writer = cv2.VideoWriter(filename, fourcc, 10.0, (w, h))
                # 将帧写回去
                self.video_capture.set(cv2.CAP_PROP_POS_FRAMES,
                                       self.video_capture.get(cv2.CAP_PROP_POS_FRAMES) - 1)

            self.add_alert("开始录制视频", "info")

        else:
            # 停止录制
            self.is_recording = False
            self.record_btn.setText("● 开始录制")
            self.record_btn.setStyleSheet("background-color: #e74c3c;")

            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None

            self.add_alert("录制已停止", "info")

    def update_frame(self):
        """更新视频帧"""
        if not self.is_playing or self.video_capture is None:
            return

        try:
            ret, frame = self.video_capture.read()

            if not ret:
                if not self.is_live:
                    self.add_alert("视频播放结束", "info")
                    self.stop_video()
                return

            self.frame_count += 1

            # 调整帧尺寸
            frame = cv2.resize(frame, (640, 480))
            display_frame = frame.copy()

            # 分析注意力和情绪
            attention_state = self.attention_analyzer.analyze_frame(frame)
            emotion_state = self.emotion_analyzer.analyze_frame(frame)

            # 计算注意力分数
            attention_score = self.attention_scoring.calculate_attention_score(
                attention_state,
                emotion_state
            )
            attention_state["attention_score"] = attention_score
            attention_state["optimized_score"] = attention_score  # 额外保存，用于区分
            score_analysis = self.attention_scoring.get_score_analysis()
            # 保存当前状态以便在update_status中使用
            self.attention_state = attention_state
            self.score_analysis = score_analysis  # 保存分析结果

            # 计算注意力分数
            if hasattr(self, 'attention_scoring'):
                attention_score = self.attention_scoring.calculate_attention_score(attention_state, emotion_state)
                attention_state["attention_score"] = attention_score
            else:
                attention_score = self.attention_analyzer.calculate_attention_score()
                attention_state["attention_score"] = attention_score

            # 检查校准状态
            if self.is_calibrating:
                # 处理校准帧
                gaze_data = {
                    "gaze_x": attention_state.get("gaze_x", 0),
                    "gaze_y": attention_state.get("gaze_y", 0)
                }

                cal_result = self.calibration_system.process_calibration_frame(frame, gaze_data)

                if cal_result:
                    if cal_result.get("status") == "完成":
                        self.is_calibrating = False
                        self.calibrate_btn.setText("🎯 校准完成")
                        self.calibrate_btn.setStyleSheet("background-color: #27ae60;")
                        self.calibration_status_label.setText("已校准")
                        self.calibration_instruction.setText("校准完成！")
                        self.calibration_progress.setValue(100)

                        # 保存校准结果
                        results = cal_result.get("results", {})
                        tolerance = results.get("tolerance", 0.2)
                        self.calibration_info.setText(f"校准完成！视线容差: {tolerance:.3f}")

                        self.add_alert("校准完成", "info")

                    else:
                        # 更新校准进度
                        current_step = cal_result.get("current_step", "center")
                        progress = cal_result.get("progress", 0) * 100

                        # 更新UI
                        self.calibration_progress.setValue(int(progress))
                        self.calibration_info.setText(f"步骤 {self.calibration_step + 1}/5: {current_step}")

                        # 更新校准步骤显示
                        if "继续" in cal_result.get("status", ""):
                            self.calibration_step += 1

            # 在帧上绘制结果
            if self.show_attention_overlay or self.show_emotion_overlay:
                display_frame = self.draw_analysis_overlay(display_frame, attention_state, emotion_state)

            # 如果正在校准，绘制校准点
            if self.is_calibrating:
                display_frame = self.draw_calibration_point(display_frame)

            # 更新图表数据
            self.realtime_charts.update_data(attention_state, emotion_state)

            # 转换为Qt图像并显示
            rgb_image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)

            scaled_pixmap = pixmap.scaled(
                self.video_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.video_label.setPixmap(scaled_pixmap)

            # 更新UI显示
            self.update_attention_display(attention_state)
            self.update_emotion_display(emotion_state)

            # 更新详细统计
            self.update_detailed_stats(attention_state, emotion_state)

            # 检查警报条件
            self.check_alerts(attention_state, emotion_state)

            # 检查视线是否在容差范围内
            if self.calibration_system.reference_gaze_center != (0, 0):
                gaze_in_tolerance = self.calibration_system.check_gaze_within_tolerance(
                    attention_state.get("gaze_x", 0),
                    attention_state.get("gaze_y", 0)
                )

                if not gaze_in_tolerance and self.frame_count % 20 == 0:
                    self.add_alert("视线偏离正常范围", "warning")

            # 记录数据
            if self.is_recording and self.video_writer:
                try:
                    self.video_writer.write(display_frame)

                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    data_point = {
                        "timestamp": timestamp,
                        "frame": self.frame_count,
                        "attention": attention_state,
                        "emotion": emotion_state
                    }
                    self.record_data.append(data_point)
                except Exception as e:
                    print(f"录制错误: {e}")

            # 语音提醒
            if self.voice_enabled and self.voice_system and self.frame_count % 30 == 0:
                self.check_voice_reminders(attention_state, emotion_state)

        except Exception as e:
            print(f"帧更新错误: {e}")
            traceback.print_exc()

    def draw_calibration_point(self, frame):
        """绘制校准点"""
        try:
            h, w = frame.shape[:2]

            # 根据当前校准步骤确定点位置
            cal_status = self.calibration_system.get_calibration_status()
            current_step = cal_status.get("current_step", "center")

            if current_step == "center":
                point_x, point_y = w // 2, h // 2
            elif current_step == "top_left":
                point_x, point_y = w // 4, h // 4
            elif current_step == "top_right":
                point_x, point_y = 3 * w // 4, h // 4
            elif current_step == "bottom_left":
                point_x, point_y = w // 4, 3 * h // 4
            elif current_step == "bottom_right":
                point_x, point_y = 3 * w // 4, 3 * h // 4
            else:
                point_x, point_y = w // 2, h // 2

            # 绘制外圆
            cv2.circle(frame, (point_x, point_y), 15, (0, 0, 255), 3)

            # 绘制内圆
            cv2.circle(frame, (point_x, point_y), 5, (0, 255, 255), -1)

            # 显示校准步骤
            step_text = f"校准步骤: {current_step}"
            cv2.putText(frame, step_text, (10, h - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            return frame

        except Exception as e:
            print(f"绘制校准点错误: {e}")
            return frame

    def update_detailed_stats(self, attention_state, emotion_state):
        """更新详细统计信息"""
        try:
            # 获取分数分析
            score_analysis = self.attention_scoring.get_score_analysis() if hasattr(self, 'attention_scoring') else {}

            # 更新注意力详细统计
            if hasattr(self, 'attention_stability_label'):
                self.attention_stability_label.setText(f"{score_analysis.get('stability_level', '未知')}")

            if hasattr(self, 'gaze_deviation_label'):
                gaze_x = attention_state.get("gaze_x", 0)
                gaze_y = attention_state.get("gaze_y", 0)
                gaze_magnitude = math.sqrt(gaze_x ** 2 + gaze_y ** 2)
                self.gaze_deviation_label.setText(f"{gaze_magnitude:.3f}")

            if hasattr(self, 'head_stability_label'):
                self.head_stability_label.setText(f"{score_analysis.get('stability_level', '未知')}")

            if hasattr(self, 'focus_duration_label'):
                focus_duration = self.calculate_focus_duration()
                self.focus_duration_label.setText(f"{focus_duration:.1f}秒")

            if hasattr(self, 'distraction_count_label'):
                # 计算分心次数（注意力分数 < 50 的帧数）
                if hasattr(self, 'attention_scoring'):
                    scores = list(self.attention_scoring.score_history)
                    distracted_frames = sum(1 for score in scores if score < 50)
                    self.distraction_count_label.setText(f"{distracted_frames}")
                else:
                    self.distraction_count_label.setText("0")

            if hasattr(self, 'refocus_rate_label'):
                # 计算重新专注率（简化计算）
                self.refocus_rate_label.setText("0%")

            # 更新情绪详细统计
            if hasattr(self, 'emotion_change_freq_label'):
                emotion_stats = self.emotion_analyzer.get_emotion_stats()
                self.emotion_change_freq_label.setText(f"{emotion_stats.get('emotion_changes', 0)}次/分钟")

            if hasattr(self, 'neutral_duration_label'):
                neutral_ratio = self.calculate_neutral_duration()
                self.neutral_duration_label.setText(f"{neutral_ratio:.1f}%")

            if hasattr(self, 'extreme_emotion_label'):
                extreme_emotions = self.check_extreme_emotions()
                if extreme_emotions:
                    self.extreme_emotion_label.setText(f"{', '.join(extreme_emotions)}")
                    self.extreme_emotion_label.setStyleSheet("font-weight: bold; color: #e74c3c;")
                else:
                    self.extreme_emotion_label.setText("无")
                    self.extreme_emotion_label.setStyleSheet("font-weight: bold; color: #27ae60;")

        except Exception as e:
            print(f"更新详细统计错误: {e}")

    def update_charts_widgets(self):
        """更新图表小部件"""
        try:
            # 创建QPixmap来绘制图表
            attention_pixmap = QPixmap(320, 160)
            attention_pixmap.fill(Qt.white)

            gaze_pixmap = QPixmap(320, 160)
            gaze_pixmap.fill(Qt.white)

            eye_pixmap = QPixmap(320, 160)
            eye_pixmap.fill(Qt.white)

            # 创建QPainter来绘制
            attention_painter = QPainter(attention_pixmap)
            gaze_painter = QPainter(gaze_pixmap)
            eye_painter = QPainter(eye_pixmap)

            # 设置抗锯齿
            attention_painter.setRenderHint(QPainter.Antialiasing)
            gaze_painter.setRenderHint(QPainter.Antialiasing)
            eye_painter.setRenderHint(QPainter.Antialiasing)

            # 绘制图表
            self.realtime_charts.draw_attention_chart(attention_painter, 10, 10, 310, 150)
            self.realtime_charts.draw_gaze_chart(gaze_painter, 10, 10, 310, 150)
            self.realtime_charts.draw_eye_chart(eye_painter, 10, 10, 310, 150)

            # 结束绘制
            attention_painter.end()
            gaze_painter.end()
            eye_painter.end()

            # 设置到标签
            self.attention_chart_widget.setPixmap(attention_pixmap)
            self.gaze_chart_widget.setPixmap(gaze_pixmap)
            self.eye_chart_widget.setPixmap(eye_pixmap)

            # 获取统计信息并更新标题
            stats = self.realtime_charts.get_statistics()
            if stats:
                current_score = stats["attention"]["current"]
                self.attention_chart_title.setText(f"注意力分数: {current_score:.1f}")

                gaze_x = stats["gaze"]["x_mean"]
                gaze_y = stats["gaze"]["y_mean"]
                self.gaze_chart_title.setText(f"视线追踪 (X:{gaze_x:.2f}, Y:{gaze_y:.2f})")

                ear_mean = stats["eye"]["ear_mean"]
                self.eye_chart_title.setText(f"眼部特征 (EAR:{ear_mean:.2f})")

        except Exception as e:
            print(f"更新图表错误: {e}")

    def draw_analysis_overlay(self, frame, attention_state, emotion_state):
        """在帧上绘制分析结果（英文显示）"""
        try:
            h, w = frame.shape[:2]

            # 绘制注意力信息
            if self.show_attention_overlay:
                # 注意力标签和分数
                label = attention_state.get("attention_label", "未知")
                score = attention_state.get("attention_score", 0)

                # 根据标签设置颜色和英文文本
                if label == "专注":
                    color = (0, 255, 0)  # 绿色
                    label_en = "Focused"
                elif label == "眼睛闭合":
                    color = (0, 0, 255)  # 红色
                    label_en = "Eyes Closed"
                elif label == "视线偏离":
                    color = (0, 165, 255)  # 橙色
                    label_en = "Head Turned"
                elif label == "视线偏移":
                    color = (0, 255, 255)  # 黄色
                    label_en = "Gaze Offset"
                elif label == "未检测到面部":
                    color = (128, 128, 128)  # 灰色
                    label_en = "No Face Detected"
                else:
                    color = (128, 128, 128)  # 灰色
                    label_en = label

                # 在左上角绘制信息
                cv2.putText(frame, f"Attention: {label_en}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(frame, f"Score: {score:.0f}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # 绘制详细指标 - 使用英文且确保字符编码正确
                cv2.putText(frame,
                            f"EAR: {attention_state.get('ear_left', 0):.2f}/{attention_state.get('ear_right', 0):.2f}",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # 使用英文字符确保没有乱码
                yaw_value = attention_state.get('yaw', 0)
                pitch_value = attention_state.get('pitch', 0)

                cv2.putText(frame, f"Yaw: {yaw_value:+.1f} deg",
                            (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"Pitch: {pitch_value:+.1f} deg",
                            (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"Blinks: {attention_state.get('blink_count', 0)}",
                            (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # 绘制情绪信息（英文显示）
            if self.show_emotion_overlay:
                emotion = emotion_state.get("emotion", "未知")
                confidence = emotion_state.get("confidence", 0)
                face_count = emotion_state.get("face_count", 0)

                # 将中文情绪标签映射为英文
                emotion_map = {
                    "生气": "Angry",
                    "厌恶": "Disgust",
                    "恐惧": "Fear",
                    "快乐": "Happy",
                    "悲伤": "Sad",
                    "惊讶": "Surprise",
                    "中性": "Neutral",
                    "未检测到面部": "No Face",
                    "错误": "Error"
                }

                emotion_en = emotion_map.get(emotion, emotion)

                # 获取情绪颜色
                emotion_colors = self.emotion_analyzer.emotion_colors
                color = emotion_colors.get(emotion, (200, 200, 200))

                # 在右上角绘制情绪信息
                cv2.putText(frame, f"Emotion: {emotion_en}", (w - 200, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(frame, f"Confidence: {confidence * 100:.0f}%", (w - 200, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(frame, f"Faces: {face_count}", (w - 200, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # 绘制面部边界框和特征点
                if self.show_landmarks_check.isChecked():
                    face_boxes = emotion_state.get("face_boxes", [])
                    face_shapes = emotion_state.get("face_shapes", [])

                    for i, (x, y, w_box, h_box) in enumerate(face_boxes):
                        # 绘制边界框
                        cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), color, 2)

                        # 绘制面部编号
                        cv2.putText(frame, f"Face {i + 1}", (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                        # 绘制特征点（如果可用）
                        if i < len(face_shapes):
                            shape = face_shapes[i]
                            for (sx, sy) in shape:
                                cv2.circle(frame, (sx, y), 1, color, -1)

            # 绘制时间戳
            timestamp = datetime.now().strftime("%H:%M:%S")
            cv2.putText(frame, f"Time: {timestamp}", (w - 150, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # 绘制帧计数
            cv2.putText(frame, f"Frame: {self.frame_count}", (10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # 绘制校准状态（如果已校准）
            if hasattr(self, 'calibration_system') and self.calibration_system.get_calibration_status()[
                "is_calibrated"]:
                cv2.putText(frame, "Calibrated", (w - 150, h - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            return frame

        except Exception as e:
            print(f"绘制叠加层错误: {e}")
            return frame

    def update_attention_display(self, attention_state):
        """更新注意力显示（优化版）"""
        try:
            # 更新分数
            score = attention_state.get("attention_score", 0)
            self.attention_score_label.setText(f"{score:.0f}")

            # 从新的分析系统中获取注意力水平
            if hasattr(self, 'score_analysis'):
                attention_level = self.score_analysis.get("attention_level", "未知")
                self.attention_state_label.setText(attention_level)
            else:
                # 回退到原来的逻辑
                label = attention_state.get("attention_label", "未知")
                self.attention_state_label.setText(label)

            # 设置状态颜色
            if hasattr(self, 'score_analysis'):
                attention_level = self.score_analysis.get("attention_level", "一般")
                color_map = {
                    "非常专注": "#27ae60",  # 绿色
                    "专注": "#2ecc71",  # 浅绿
                    "一般": "#f39c12",  # 橙色
                    "轻度分心": "#e67e22",  # 深橙
                    "中度分心": "#e74c3c",  # 红色
                    "严重分心": "#c0392b"  # 深红
                }
                color = color_map.get(attention_level, "#7f8c8d")
                self.attention_state_label.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {color};")
            else:
                # 原来的颜色设置逻辑
                label = attention_state.get("attention_label", "未知")
                if label == "专注":
                    color = "#27ae60"
                elif label == "眼睛闭合":
                    color = "#e74c3c"
                elif label == "视线偏离":
                    color = "#f39c12"
                elif label == "视线偏移":
                    color = "#f1c40f"
                else:
                    color = "#7f8c8d"
                self.attention_state_label.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {color};")

            # 更新进度条
            self.attention_progress.setValue(int(score))

            # 更新详细指标
            ear_left = attention_state.get('ear_left', 0)
            ear_right = attention_state.get('ear_right', 0)
            self.ear_label.setText(f"{ear_left:.2f}/{ear_right:.2f}")

            yaw_value = attention_state.get('yaw', 0)
            pitch_value = attention_state.get('pitch', 0)

            self.yaw_label.setText(f"{yaw_value:+.1f}°")
            self.pitch_label.setText(f"{pitch_value:+.1f}°")
            self.gaze_x_label.setText(f"{attention_state.get('gaze_x', 0):.2f}")
            self.gaze_y_label.setText(f"{attention_state.get('gaze_y', 0):.2f}")
            self.blink_label.setText(f"{attention_state.get('blink_count', 0)}")

            # 如果启用了多动症特征检测，显示相关信息
            if hasattr(self, 'score_analysis') and 'adhd_features' in self.score_analysis:
                adhd_features = self.score_analysis['adhd_features']
                if adhd_features:
                    # 可以在UI中添加新的标签来显示这些信息
                    risk_level = adhd_features.get('risk_level', '未知')
                    self.add_alert(f"注意力风险等级: {risk_level}", "info")

        except Exception as e:
            print(f"更新注意力显示错误: {e}")

    def update_emotion_display(self, emotion_state):
        """更新情绪显示"""
        try:
            # 保存当前情绪状态
            self.current_emotion_state = emotion_state

            # 更新当前情绪
            emotion = emotion_state.get("emotion", "未知")
            self.emotion_label.setText(emotion)

            # 设置情绪颜色
            emotion_colors = {
                "生气": "#e74c3c",
                "厌恶": "#27ae60",
                "恐惧": "#9b59b6",
                "快乐": "#f1c40f",
                "悲伤": "#3498db",
                "惊讶": "#e67e22",
                "中性": "#95a5a6",
                "未检测到面部": "#7f8c8d",
                "错误": "#e74c3c"
            }

            color = emotion_colors.get(emotion, "#7f8c8d")
            self.emotion_label.setStyleSheet(f"font-size: 20px; font-weight: bold; color: {color};")

            # 更新信心
            confidence = emotion_state.get("confidence", 0)
            self.confidence_label.setText(f"{confidence * 100:.0f}%")

            # 更新情绪概率条
            probabilities = emotion_state.get("probabilities", [0.0] * 7)
            emotion_labels = ["生气", "厌恶", "恐惧", "快乐", "悲伤", "惊讶", "中性"]

            for i, emotion_name in enumerate(emotion_labels):
                if emotion_name in self.emotion_bars:
                    prob = probabilities[i] * 100
                    self.emotion_bars[emotion_name].setValue(int(prob))

        except Exception as e:
            print(f"更新情绪显示错误: {e}")
            traceback.print_exc()

    def update_status(self):
        """更新状态信息（修复版本）"""
        try:
            current_time = datetime.now()

            # 1. 获取基础统计信息
            attention_stats = self.attention_analyzer.get_attention_stats()
            emotion_stats = self.emotion_analyzer.get_emotion_stats()

            # 2. 获取优化后的注意力分析结果
            if hasattr(self, 'attention_scoring'):
                score_analysis = self.attention_scoring.get_score_analysis()
                self.score_analysis = score_analysis  # 保存供其他地方使用
            else:
                score_analysis = {}

            # 3. 更新基础注意力统计
            # 平均分数
            if score_analysis and 'statistics' in score_analysis:
                stats = score_analysis['statistics']
                if hasattr(self, 'avg_attention_label'):
                    self.avg_attention_label.setText(f"{stats.get('recent_avg', 0):.1f}")
                if hasattr(self, 'max_attention_label'):
                    self.max_attention_label.setText(f"{stats.get('recent_max', 0):.1f}")
                if hasattr(self, 'min_attention_label'):
                    self.min_attention_label.setText(f"{stats.get('recent_min', 0):.1f}")
            else:
                # 回退到原来的统计
                if hasattr(self, 'avg_attention_label'):
                    self.avg_attention_label.setText(f"{attention_stats.get('avg_score', 0)}")
                if hasattr(self, 'max_attention_label'):
                    self.max_attention_label.setText(f"{attention_stats.get('max_score', 0)}")
                if hasattr(self, 'min_attention_label'):
                    self.min_attention_label.setText(f"{attention_stats.get('min_score', 0)}")

            # 注意力趋势
            if score_analysis and 'statistics' in score_analysis:
                stats = score_analysis['statistics']
                if hasattr(self, 'trend_label'):
                    self.trend_label.setText(f"{stats.get('trend', '稳定')}")
            else:
                if hasattr(self, 'trend_label'):
                    self.trend_label.setText(f"{attention_stats.get('trend', '稳定')}")

            # 专注百分比
            if hasattr(self, 'focus_percent_label'):
                self.focus_percent_label.setText(f"{attention_stats.get('focus_percentage', 0):.1f}%")

            # 眨眼频率
            if hasattr(self, 'blink_rate_label'):
                self.blink_rate_label.setText(f"{attention_stats.get('blink_rate', 0):.1f}/分钟")

            # 4. 更新情绪统计
            if hasattr(self, 'dominant_emotion_label'):
                self.dominant_emotion_label.setText(f"{emotion_stats.get('dominant_emotion', '未知')}")
            if hasattr(self, 'emotion_stability_label'):
                self.emotion_stability_label.setText(f"{emotion_stats.get('emotion_stability', 0)}%")
            if hasattr(self, 'positive_ratio_label'):
                self.positive_ratio_label.setText(f"{emotion_stats.get('positive_ratio', 0)}%")
            if hasattr(self, 'negative_ratio_label'):
                self.negative_ratio_label.setText(f"{emotion_stats.get('negative_ratio', 0)}%")

            # 5. 更新详细注意力统计
            # 注意力稳定性
            if score_analysis and 'statistics' in score_analysis:
                stats = score_analysis['statistics']
                stability_index = stats.get('stability_index', 0)
                if hasattr(self, 'attention_stability_label'):
                    self.attention_stability_label.setText(f"{stability_index:.1f}%")

                # 一致性分数
                consistency_score = stats.get('consistency_score', 0)
                if hasattr(self, 'refocus_rate_label'):
                    self.refocus_rate_label.setText(f"{consistency_score:.1f}%")
            else:
                if hasattr(self, 'attention_stability_label'):
                    self.attention_stability_label.setText(f"{attention_stats.get('trend', '稳定')}")
                if hasattr(self, 'refocus_rate_label'):
                    self.refocus_rate_label.setText("0%")

            # 视线偏移度
            if hasattr(self, 'gaze_deviation_label'):
                if hasattr(self, 'attention_state'):
                    gaze_x = self.attention_state.get("gaze_x", 0)
                    gaze_y = self.attention_state.get("gaze_y", 0)
                    gaze_magnitude = math.sqrt(gaze_x ** 2 + gaze_y ** 2)
                    self.gaze_deviation_label.setText(f"{gaze_magnitude:.3f}")
                else:
                    self.gaze_deviation_label.setText("0.000")

            # 头部稳定性（使用多动症特征中的活动过度指数）
            if score_analysis and 'adhd_features' in score_analysis:
                adhd_features = score_analysis['adhd_features']
                hyperactivity_index = adhd_features.get('hyperactivity_index', 0)
                if hasattr(self, 'head_stability_label'):
                    self.head_stability_label.setText(f"{100 - hyperactivity_index:.1f}%")
            else:
                if hasattr(self, 'head_stability_label'):
                    self.head_stability_label.setText("0%")

            # 6. 更新专注时长统计
            if score_analysis and 'focus_analysis' in score_analysis:
                focus_analysis = score_analysis['focus_analysis']

                # 平均专注时长
                avg_duration = focus_analysis.get('avg_duration', 0)
                if hasattr(self, 'focus_duration_label'):
                    self.focus_duration_label.setText(f"{avg_duration:.1f}秒")

                # 最长专注时长
                longest_duration = focus_analysis.get('longest_duration', 0)
                if longest_duration > 0 and hasattr(self, 'distraction_count_label'):
                    self.distraction_count_label.setText(f"{longest_duration:.1f}秒")

                # 专注中断次数
                interruptions = focus_analysis.get('interruptions', 0)
                if interruptions > 0 and hasattr(self, 'distraction_count_label'):
                    self.distraction_count_label.setText(f"{interruptions}次")

                # 专注模式
                focus_pattern = focus_analysis.get('pattern', '分析中')
                if hasattr(self, 'focus_pattern_label'):
                    self.focus_pattern_label.setText(focus_pattern)
            else:
                # 计算基础专注时长
                focus_duration = self.calculate_focus_duration()
                if hasattr(self, 'focus_duration_label'):
                    self.focus_duration_label.setText(f"{focus_duration:.1f}秒")
                if hasattr(self, 'distraction_count_label'):
                    self.distraction_count_label.setText("0")
                if hasattr(self, 'refocus_rate_label'):
                    self.refocus_rate_label.setText("0%")

            # 7. 更新情绪详细统计
            # 情绪变化频率
            if hasattr(self, 'emotion_change_freq_label'):
                self.emotion_change_freq_label.setText(f"{emotion_stats.get('emotion_changes', 0)}次/分钟")

            # 中性时长比例
            neutral_ratio = self.calculate_neutral_duration()
            if hasattr(self, 'neutral_duration_label'):
                self.neutral_duration_label.setText(f"{neutral_ratio:.1f}%")

            # 极端情绪检测
            extreme_emotions = self.check_extreme_emotions()
            if hasattr(self, 'extreme_emotion_label'):
                if extreme_emotions:
                    self.extreme_emotion_label.setText(f"{', '.join(extreme_emotions)}")
                    self.extreme_emotion_label.setStyleSheet("font-weight: bold; color: #e74c3c;")
                else:
                    self.extreme_emotion_label.setText("无")
                    self.extreme_emotion_label.setStyleSheet("font-weight: bold; color: #27ae60;")

            # 8. 更新面部数量
            face_count = emotion_stats.get('face_count', 0)
            if hasattr(self, 'face_count_label'):
                self.face_count_label.setText(f"{face_count}")

            # 9. 更新多动症特征分析
            if score_analysis and 'adhd_features' in score_analysis:
                adhd_features = score_analysis['adhd_features']

                # 注意力不集中比例
                inattention_ratio = adhd_features.get('inattention_ratio', 0)
                if hasattr(self, 'inattention_ratio_label'):
                    self.inattention_ratio_label.setText(f"{inattention_ratio:.1f}%")

                # 活动过度指数
                hyperactivity_index = adhd_features.get('hyperactivity_index', 0)
                if hasattr(self, 'hyperactivity_label'):
                    self.hyperactivity_label.setText(f"{hyperactivity_index:.1f}")

                # 情绪波动指数
                emotion_volatility = adhd_features.get('emotion_volatility', 0)
                if hasattr(self, 'emotion_volatility_label'):
                    self.emotion_volatility_label.setText(f"{emotion_volatility:.1f}")

                # 总体风险等级
                risk_level = adhd_features.get('risk_level', '正常')
                if hasattr(self, 'risk_level_label'):
                    self.risk_level_label.setText(risk_level)

                    # 根据风险等级设置颜色
                    if risk_level == "高风险":
                        self.risk_level_label.setStyleSheet("font-weight: bold; color: #e74c3c;")
                        if self.frame_count % 60 == 0:  # 每分钟检查一次
                            self.add_alert("检测到高风险注意力问题", "warning")
                    elif risk_level == "中风险":
                        self.risk_level_label.setStyleSheet("font-weight: bold; color: #f39c12;")
                        if self.frame_count % 120 == 0:
                            self.add_alert("检测到中度注意力风险", "info")
                    elif risk_level == "低风险":
                        self.risk_level_label.setStyleSheet("font-weight: bold; color: #f1c40f;")
                    else:
                        self.risk_level_label.setStyleSheet("font-weight: bold; color: #27ae60;")

                # ADHD特征检测
                if hasattr(self, 'adhd_features_label'):
                    features_detected = []
                    if inattention_ratio > 30:
                        features_detected.append("注意力不集中")
                    if hyperactivity_index > 50:
                        features_detected.append("活动过度")
                    if emotion_volatility > 30:
                        features_detected.append("情绪波动")

                    if features_detected:
                        self.adhd_features_label.setText(", ".join(features_detected[:2]))
                        self.adhd_features_label.setStyleSheet("font-weight: bold; color: #e74c3c;")
                    else:
                        self.adhd_features_label.setText("无")
                        self.adhd_features_label.setStyleSheet("font-weight: bold; color: #27ae60;")

                # 检测到的模式
                pattern_detected = adhd_features.get('pattern_detected', False)
                if pattern_detected and self.frame_count % 90 == 0:
                    self.add_alert("检测到注意力分散模式", "info")

            # 10. 更新校准状态
            if hasattr(self, 'calibration_system'):
                cal_status = self.calibration_system.get_calibration_status()

                if cal_status.get('is_calibrated'):
                    if hasattr(self, 'calibration_status_label'):
                        self.calibration_status_label.setText("已校准")

                    if hasattr(self, 'calibration_info'):
                        ref_x, ref_y = cal_status['reference_center']
                        tolerance = cal_status['tolerance']
                        self.calibration_info.setText(
                            f"参考中心: ({ref_x:.3f}, {ref_y:.3f}) | 容差: {tolerance:.3f}"
                        )

                    if hasattr(self, 'calibrate_btn'):
                        self.calibrate_btn.setText("✅ 已校准")
                        self.calibrate_btn.setStyleSheet("background-color: #27ae60;")

                    # 更新校准结果信息
                    if hasattr(self, 'calibration_result_label'):
                        self.calibration_result_label.setText(
                            f"校准状态: 已完成\n"
                            f"参考视线中心: ({ref_x:.3f}, {ref_y:.3f})\n"
                            f"视线容差: {tolerance:.3f}\n"
                            f"校准点: {len(cal_status.get('calibration_results', {}))}"
                        )
                else:
                    if hasattr(self, 'calibration_status_label'):
                        self.calibration_status_label.setText("未校准")

                    if hasattr(self, 'calibration_result_label'):
                        self.calibration_result_label.setText("校准状态: 未完成\n请点击'开始校准'进行校准")

            # 11. 更新会话时间信息
            if self.session_start_time:
                elapsed = current_time - self.session_start_time
                hours, remainder = divmod(elapsed.seconds, 3600)
                minutes, seconds = divmod(remainder, 60)

                # 计算帧率
                if elapsed.total_seconds() > 0:
                    fps = self.frame_count / elapsed.total_seconds()
                else:
                    fps = 0

                # 更新视频信息标签
                time_text = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

                if self.is_live:
                    source_text = "实时摄像头"
                elif self.video_path:
                    filename = os.path.basename(self.video_path)
                    source_text = filename
                else:
                    source_text = "就绪"

                # 添加注意力水平信息
                attention_level = "未知"
                if score_analysis and 'attention_level' in score_analysis:
                    attention_level = score_analysis['attention_level']

                if self.is_playing:
                    status_text = f"{source_text} | {time_text} | 帧: {self.frame_count} | FPS: {fps:.1f} | 注意力: {attention_level}"
                else:
                    status_text = f"就绪 | 上次会话: {time_text} | 总帧数: {self.frame_count}"

                self.video_info_label.setText(status_text)

                # 自动检查长时间运行（超过30分钟建议休息）
                if elapsed.total_seconds() > 1800 and self.frame_count % 300 == 0:  # 30分钟，每5分钟提醒
                    self.add_alert(f"检测已运行{minutes}分钟，建议休息一下", "info")
                    if self.voice_enabled:
                        self.voice_system.speak("已经连续检测30分钟，建议休息一下")

            # 12. 生成实时建议
            if score_analysis and 'recommendations' in score_analysis:
                recommendations = score_analysis['recommendations']
                if recommendations and len(recommendations) > 0 and self.frame_count % 180 == 0:  # 每3分钟
                    random_recommendation = random.choice(recommendations)
                    self.add_alert(f"建议: {random_recommendation}", "info")

            # 13. 更新注意力评分进度条的颜色
            if hasattr(self, 'attention_score_label'):
                current_score_text = self.attention_score_label.text()
                try:
                    current_score = float(current_score_text)
                    self.update_progress_bar_color(current_score)
                except ValueError:
                    pass

            # 14. 更新图表统计信息
            if hasattr(self, 'realtime_charts'):
                chart_stats = self.realtime_charts.get_statistics()
                if chart_stats:
                    # 在这里可以更新图表相关的统计显示
                    pass

            # 15. 检查系统资源使用情况（可选）
            if self.frame_count % 600 == 0:  # 每10秒检查一次
                self.check_system_resources()

        except Exception as e:
            print(f"更新状态错误: {e}")
            traceback.print_exc()

            # 错误时显示基本状态
            try:
                if hasattr(self, 'video_info_label'):
                    self.video_info_label.setText("系统错误 - 尝试重新连接")
                if hasattr(self, 'attention_state_label'):
                    self.attention_state_label.setText("系统错误")
                    self.attention_state_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #e74c3c;")
            except:
                pass

    def update_progress_bar_color(self, score):
        """根据分数更新进度条颜色"""
        try:
            if score >= 80:
                color = "#27ae60"  # 绿色
            elif score >= 60:
                color = "#f1c40f"  # 黄色
            elif score >= 40:
                color = "#e67e22"  # 橙色
            else:
                color = "#e74c3c"  # 红色

            # 设置进度条样式
            style = f"""
                QProgressBar {{
                    border: 1px solid #d1d9e6;
                    border-radius: 4px;
                    text-align: center;
                    background-color: white;
                }}
                QProgressBar::chunk {{
                    border-radius: 4px;
                    background-color: {color};
                }}
            """
            if hasattr(self, 'attention_progress'):
                self.attention_progress.setStyleSheet(style)

        except Exception as e:
            print(f"更新进度条颜色错误: {e}")

    def check_system_resources(self):
        """检查系统资源使用情况"""
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            # 如果资源使用过高，记录警告
            if cpu_percent > 80 or memory_percent > 80:
                self.add_alert(f"系统资源使用偏高 - CPU: {cpu_percent:.1f}%, 内存: {memory_percent:.1f}%", "warning")

        except ImportError:
            # psutil 未安装，跳过资源检查
            pass
        except Exception as e:
            print(f"检查系统资源错误: {e}")

    def calculate_focus_duration(self):
        """计算专注时长（兼容函数）"""
        try:
            if hasattr(self, 'attention_scoring') and hasattr(self.attention_scoring, 'focus_sessions'):
                if self.attention_scoring.focus_sessions:
                    total_duration = sum(session.get('duration', 0)
                                         for session in self.attention_scoring.focus_sessions)
                    return total_duration / len(self.attention_scoring.focus_sessions)

            # 回退计算
            if hasattr(self, 'attention_scoring') and hasattr(self.attention_scoring, 'current_focus_duration'):
                return self.attention_scoring.current_focus_duration

            return 0
        except:
            return 0

    def calculate_neutral_duration(self):
        """计算中性情绪时长比例（兼容函数）"""
        try:
            if hasattr(self, 'emotion_analyzer') and hasattr(self.emotion_analyzer, 'emotion_history'):
                neutral_frames = sum(1 for emotion in self.emotion_analyzer.emotion_history
                                     if emotion == "中性")
                total_frames = len(self.emotion_analyzer.emotion_history)
                if total_frames > 0:
                    return (neutral_frames / total_frames) * 100

            return 0
        except:
            return 0

    def check_extreme_emotions(self):
        """检查极端情绪（兼容函数）"""
        extreme_emotions = []
        try:
            if hasattr(self, 'emotion_analyzer') and hasattr(self.emotion_analyzer, 'emotion_history'):
                recent_emotions = list(self.emotion_analyzer.emotion_history)[-30:] if len(
                    self.emotion_analyzer.emotion_history) >= 30 else list(self.emotion_analyzer.emotion_history)

                for emotion in ["生气", "恐惧"]:
                    count = recent_emotions.count(emotion)
                    if count >= 10:
                        extreme_emotions.append(emotion)
        except:
            pass

        return extreme_emotions

    def check_alerts(self, attention_state, emotion_state):
        """检查警报条件"""
        try:
            attention_label = attention_state.get("attention_label", "未知")
            emotion = emotion_state.get("emotion", "中性")
            face_detected = attention_state.get("face_detected", False)

            current_time = datetime.now().strftime("%H:%M:%S")

            # 注意力相关警报
            if attention_label == "眼睛闭合" and self.frame_count % 30 == 0:
                self.add_alert(f"{current_time} - 检测到眼睛闭合", "warning")

            elif attention_label == "视线偏离" and self.frame_count % 45 == 0:
                self.add_alert(f"{current_time} - 视线偏离屏幕", "warning")

            elif attention_label == "视线偏移" and self.frame_count % 60 == 0:
                self.add_alert(f"{current_time} - 视线偏离中心", "info")

            elif attention_label == "专注" and self.frame_count % 90 == 0:
                self.add_alert(f"{current_time} - 注意力保持良好", "positive")

            # 情绪相关警报
            if emotion in ["生气", "恐惧", "悲伤"] and self.frame_count % 40 == 0:
                self.add_alert(f"{current_time} - 检测到负面情绪: {emotion}", "warning")

            elif emotion == "快乐" and self.frame_count % 50 == 0:
                self.add_alert(f"{current_time} - 正面情绪: 快乐", "positive")

            # 面部检测警报
            if not face_detected and self.frame_count % 60 == 0:
                self.add_alert(f"{current_time} - 未检测到面部", "warning")

        except Exception as e:
            print(f"检查警报错误: {e}")

    def check_voice_reminders(self, attention_state, emotion_state):
        """检查语音提醒（使用新的分析结果）"""
        if not self.voice_enabled or not self.voice_system.engine:
            return

        # 获取新的分析结果
        if hasattr(self, 'score_analysis'):
            attention_level = self.score_analysis.get("attention_level", "一般")
            risk_level = self.score_analysis.get("adhd_features", {}).get("risk_level", "正常")

            # 根据新的分析结果生成提醒
            if attention_level in ["中度分心", "严重分心"]:
                self.voice_system.speak("注意力分散了，请重新集中注意力")

            elif risk_level == "高风险":
                self.voice_system.speak("检测到注意力问题，建议休息一下")

            elif attention_level == "非常专注" and self.frame_count % 100 == 0:
                self.voice_system.speak("太棒了！继续保持专注！")

        else:
            # 原来的逻辑
            attention_label = attention_state.get("attention_label", "未知")
            emotion = emotion_state.get("emotion", "中性")

            if attention_label == "眼睛闭合":
                self.voice_system.speak("请睁开眼睛，看着屏幕")
            elif attention_label == "视线偏离":
                self.voice_system.speak("请看着屏幕")

    def add_alert(self, message, alert_type="info"):
        """添加警报消息"""
        try:
            # 避免重复警报
            if len(self.alerts) > 0 and message in self.alerts[-1]:
                return

            self.alerts.append(message)

            # 限制警报数量
            if len(self.alerts) > 50:
                self.alerts.pop(0)

            # 在文本框中显示
            cursor = self.alert_text.textCursor()
            cursor.movePosition(QTextCursor.End)

            # 设置颜色
            if alert_type == "warning":
                self.alert_text.setTextColor(QColor("#e74c3c"))
            elif alert_type == "positive":
                self.alert_text.setTextColor(QColor("#27ae60"))
            else:
                self.alert_text.setTextColor(QColor("#3498db"))

            self.alert_text.insertPlainText(f"{message}\n")

            # 滚动到底部
            self.alert_text.verticalScrollBar().setValue(
                self.alert_text.verticalScrollBar().maximum()
            )

        except Exception as e:
            print(f"添加警报错误: {e}")

    def toggle_attention_overlay(self, state):
        """切换注意力覆盖层显示"""
        self.show_attention_overlay = (state == Qt.Checked)

    def toggle_emotion_overlay(self, state):
        """切换情绪覆盖层显示"""
        self.show_emotion_overlay = (state == Qt.Checked)

    def toggle_voice(self, state):
        """切换语音功能"""
        self.voice_enabled = (state == Qt.Checked)
        status = "启用" if self.voice_enabled else "禁用"
        self.add_alert(f"语音提醒已{status}", "info")

    def test_voice(self):
        """测试语音功能"""
        if self.voice_system.engine:
            self.voice_system.speak("语音测试成功。多动症检测系统已就绪。")
            self.add_alert("语音测试完成", "info")
        else:
            self.add_alert("语音系统不可用", "warning")

    def reset_analysis(self):
        """重置分析"""
        self.attention_analyzer.reset()
        self.emotion_analyzer.reset()

        self.attention_stats_history.clear()
        self.emotion_stats_history.clear()
        self.record_data.clear()
        self.alerts.clear()
        self.alert_text.clear()

        self.frame_count = 0
        self.session_start_time = datetime.now()

        # 重置UI显示
        self.attention_score_label.setText("0")
        self.attention_state_label.setText("初始化中")
        self.emotion_label.setText("中性")
        self.confidence_label.setText("0%")

        for bar in self.emotion_bars.values():
            bar.setValue(0)

        self.add_alert("分析已重置", "info")

    def export_report(self):
        """导出报告"""
        try:
            if not self.record_data:
                QMessageBox.warning(self, "警告", "没有数据可导出")
                return

            # 选择保存位置
            filename, _ = QFileDialog.getSaveFileName(
                self, "保存分析报告",
                f"多动症分析报告_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                "JSON文件 (*.json)"
            )

            if not filename:
                return

            # 准备报告数据
            attention_stats = self.attention_analyzer.get_attention_stats()
            emotion_stats = self.emotion_analyzer.get_emotion_stats()
            score_analysis = self.attention_scoring.get_score_analysis()
            calibration_status = self.calibration_system.get_calibration_status()
            chart_stats = self.realtime_charts.get_statistics()
            # 获取新的分析结果
            score_analysis = self.attention_scoring.get_score_analysis()

            report = {
                "report_info": {
                "title": "多动症儿童注意力分析报告（优化版）",
                "生成时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "版本": "5.0",
                "分析时长": (
                    datetime.now() - self.session_start_time).total_seconds()
                    if self.session_start_time else 0
            },
                "session_info": {
                    "来源": "实时摄像头" if self.is_live else f"视频: {self.video_path}",
                    "开始时间": self.session_start_time.strftime(
                        "%Y-%m-%d %H:%M:%S") if self.session_start_time else "未知",
                    "结束时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "总帧数": self.frame_count,
                    "录制时长": f"{self.frame_count / 10:.1f} 秒" if self.is_live else f"{self.frame_count / 30:.1f} 秒",
                    "校准状态": calibration_status
                },
                "attention_analysis": {
                "最终分数": score_analysis.get("current_score", 0),
                "注意力水平": score_analysis.get("attention_level", "未知"),
                "统计信息": score_analysis.get("statistics", {}),
                "专注分析": score_analysis.get("focus_analysis", {}),
                "多动症特征": score_analysis.get("adhd_features", {}),
                "建议": score_analysis.get("recommendations", [])
            },
                "emotion_analysis": {
                    "最终情绪": self.emotion_analyzer.current_emotion,
                    "统计信息": emotion_stats,
                    "情绪分布": {
                        emotion: prob for emotion, prob in zip(
                            ["生气", "厌恶", "恐惧", "快乐", "悲伤", "惊讶", "中性"],
                            self.emotion_analyzer.emotion_probabilities
                        )
                    }
                },
                "calibration_results": {
                    "is_calibrated": calibration_status.get("is_calibrated", False),
                    "reference_center": calibration_status.get("reference_center", (0, 0)),
                    "tolerance": calibration_status.get("tolerance", 0.2),
                    "calibration_data": self.calibration_system.calibration_results
                },
                "adhd_indicators": {
                    "注意力缺陷": attention_stats.get("focus_percentage", 0) < 50,
                    "活动过度": self.attention_analyzer.blinks > 20 and attention_stats.get("blink_rate", 0) > 30,
                    "情绪不稳定": emotion_stats.get("emotion_stability", 0) < 70,
                    "视线稳定性问题": chart_stats.get("gaze", {}).get("x_std", 0) > 0.1 if chart_stats else False,
                    "总体风险": self.calculate_overall_risk(attention_stats, emotion_stats, chart_stats)
                },
                "recommendations": self.generate_recommendations(attention_stats, emotion_stats, chart_stats),
                "chart_statistics": chart_stats,
                "sample_data": self.record_data[:100] if len(self.record_data) > 100 else self.record_data,
                "alerts": self.alerts[-20:]
            }

            # 保存报告
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)

            QMessageBox.information(self, "成功",
                                    f"报告保存成功！\n\n"
                                    f"文件: {filename}\n"
                                    f"分析帧数: {self.frame_count}\n"
                                    f"注意力分数: {score_analysis.get('current_score', 0):.1f}\n"
                                    f"主导情绪: {self.emotion_analyzer.current_emotion}\n"
                                    f"校准状态: {'已校准' if calibration_status.get('is_calibrated') else '未校准'}")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出报告失败: {str(e)}")

    def generate_recommendations(self, attention_stats, emotion_stats):
        """生成建议"""
        recommendations = []

        # 注意力相关建议
        focus_percentage = attention_stats.get("focus_percentage", 0)
        if focus_percentage < 40:
            recommendations.append("检测到严重注意力缺陷。建议每15-20分钟安排结构性休息。")
        elif focus_percentage < 60:
            recommendations.append("检测到中度注意力问题。尝试减少环境干扰。")
        else:
            recommendations.append("注意力水平保持良好。继续当前策略。")

        # 情绪相关建议
        positive_ratio = emotion_stats.get("positive_ratio", 0)
        if positive_ratio < 40:
            recommendations.append("积极情绪比例较低。建议增加更多有趣和奖励性的活动。")

        emotion_stability = emotion_stats.get("emotion_stability", 0)
        if emotion_stability < 60:
            recommendations.append("观察到情绪不稳定。建议教授情绪调节技巧。")

        # 综合建议
        blink_rate = attention_stats.get("blink_rate", 0)
        if blink_rate > 25:
            recommendations.append("眨眼频率较高，可能表示疲劳或压力。确保充分休息。")

        if self.emotion_analyzer.current_emotion in ["生气", "恐惧", "悲伤"]:
            recommendations.append("检测到负面情绪。建议提供情感支持和应对策略。")

        # 多动症特定建议
        recommendations.append("针对多动症儿童：使用视觉时间表、计时器和频繁的积极强化。")
        recommendations.append("将任务分解为小步骤，并提供即时反馈。")
        recommendations.append("允许活动休息，如果需要可提供适当的玩具。")

        return recommendations

    def closeEvent(self, event):
        """关闭窗口事件"""
        reply = QMessageBox.question(
            self, "确认退出",
            "确定要退出吗？所有未保存的数据将会丢失。",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self.stop_video()

            if self.voice_system:
                self.voice_system.stop()

            event.accept()
        else:
            event.ignore()


# ============================================================================
# 面部建模
# ============================================================================

class FacialModeling:
    """面部建模功能 - 用于校准和个性化设置"""

    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # 存储用户的面部特征
        self.user_profile = {
            "face_landmarks": None,
            "reference_points": {
                "neutral_gaze": (0.0, 0.0),  # 中性视线位置
                "eye_size": (0.0, 0.0),  # 眼睛尺寸
                "pupil_distance": 0.0,  # 瞳孔距离
                "head_pose_neutral": (0.0, 0.0, 0.0)  # 中性头部姿态
            },
            "calibration_data": [],
            "is_calibrated": False
        }

    def extract_face_features(self, frame):
        """提取面部特征"""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]

            results = self.face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark

                # 转换为numpy数组
                points = []
                for lm in landmarks:
                    x, y = int(lm.x * w), int(lm.y * h)
                    points.append((x, y))

                points = np.array(points)

                # 提取关键特征点
                features = {
                    "face_points": points,
                    "left_eye": points[L_EYE],
                    "right_eye": points[R_EYE],
                    "left_iris": points[LEFT_IRIS],
                    "right_iris": points[RIGHT_IRIS],
                    "nose": points[1],  # 鼻尖
                    "mouth_left": points[61],
                    "mouth_right": points[291]
                }

                # 计算眼部特征
                left_eye_h = np.linalg.norm(features["left_eye"][1] - features["left_eye"][4])
                right_eye_h = np.linalg.norm(features["right_eye"][1] - features["right_eye"][4])

                # 计算瞳孔距离
                left_pupil = features["left_iris"].mean(axis=0)
                right_pupil = features["right_iris"].mean(axis=0)
                pupil_distance = np.linalg.norm(left_pupil - right_pupil)

                # 计算视线方向
                gaze_vector = self.calculate_gaze_vector(features)

                return {
                    "features": features,
                    "eye_size": (left_eye_h, right_eye_h),
                    "pupil_distance": pupil_distance,
                    "gaze_direction": gaze_vector,
                    "valid": True
                }

            return {"valid": False}

        except Exception as e:
            print(f"面部特征提取错误: {e}")
            return {"valid": False}

    def calculate_gaze_vector(self, features):
        """计算视线方向向量"""
        try:
            # 计算左眼视线
            eye_bbox_l = cv2.boundingRect(features["left_eye"])
            iris_center_l = features["left_iris"].mean(axis=0)

            # 计算右眼视线
            eye_bbox_r = cv2.boundingRect(features["right_eye"])
            iris_center_r = features["right_iris"].mean(axis=0)

            # 归一化视线向量
            def normalize_gaze(bbox, iris_center):
                center_x = bbox[0] + bbox[2] / 2
                center_y = bbox[1] + bbox[3] / 2

                dx = (iris_center[0] - center_x) / (bbox[2] / 2)
                dy = (iris_center[1] - center_y) / (bbox[3] / 2)

                return (dx, dy)

            gaze_l = normalize_gaze(eye_bbox_l, iris_center_l)
            gaze_r = normalize_gaze(eye_bbox_r, iris_center_r)

            # 平均双眼视线
            gaze_x = (gaze_l[0] + gaze_r[0]) / 2
            gaze_y = (gaze_l[1] + gaze_r[1]) / 2

            return (gaze_x, gaze_y)

        except Exception as e:
            print(f"计算视线向量错误: {e}")
            return (0.0, 0.0)

    def calibrate(self, frame, calibration_type="neutral"):
        """校准面部特征"""
        result = self.extract_face_features(frame)

        if result["valid"]:
            if calibration_type == "neutral":
                # 记录中性视线
                self.user_profile["reference_points"]["neutral_gaze"] = result["gaze_direction"]
                self.user_profile["reference_points"]["eye_size"] = result["eye_size"]
                self.user_profile["reference_points"]["pupil_distance"] = result["pupil_distance"]

                # 添加到校准数据
                self.user_profile["calibration_data"].append({
                    "type": "neutral",
                    "gaze": result["gaze_direction"],
                    "eye_size": result["eye_size"],
                    "timestamp": time.time()
                })

                # 计算平均值
                if len(self.user_profile["calibration_data"]) >= 5:
                    self.calculate_calibration_average()
                    self.user_profile["is_calibrated"] = True

                return True

        return False

    def calculate_calibration_average(self):
        """计算校准数据的平均值"""
        if not self.user_profile["calibration_data"]:
            return

        neutral_frames = [d for d in self.user_profile["calibration_data"] if d["type"] == "neutral"]

        if neutral_frames:
            # 计算视线平均值
            gaze_x_list = [d["gaze"][0] for d in neutral_frames]
            gaze_y_list = [d["gaze"][1] for d in neutral_frames]

            avg_gaze_x = np.mean(gaze_x_list[-5:])  # 取最后5帧
            avg_gaze_y = np.mean(gaze_y_list[-5:])

            self.user_profile["reference_points"]["neutral_gaze"] = (avg_gaze_x, avg_gaze_y)

            # 计算眼部尺寸平均值
            left_eye_list = [d["eye_size"][0] for d in neutral_frames]
            right_eye_list = [d["eye_size"][1] for d in neutral_frames]

            avg_left_eye = np.mean(left_eye_list[-5:])
            avg_right_eye = np.mean(right_eye_list[-5:])

            self.user_profile["reference_points"]["eye_size"] = (avg_left_eye, avg_right_eye)

    def get_calibration_status(self):
        """获取校准状态"""
        status = {
            "is_calibrated": self.user_profile["is_calibrated"],
            "calibration_frames": len(self.user_profile["calibration_data"]),
            "neutral_gaze": self.user_profile["reference_points"]["neutral_gaze"],
            "remaining_frames": max(0, 5 - len(self.user_profile["calibration_data"]))
        }
        return status

    def reset_calibration(self):
        """重置校准数据"""
        self.user_profile = {
            "face_landmarks": None,
            "reference_points": {
                "neutral_gaze": (0.0, 0.0),
                "eye_size": (0.0, 0.0),
                "pupil_distance": 0.0,
                "head_pose_neutral": (0.0, 0.0, 0.0)
            },
            "calibration_data": [],
            "is_calibrated": False
        }


# ============================================================================
#  注意力得分计算机制
# ============================================================================

# ============================================================================
# 优化的注意力得分计算机制
# ============================================================================

class OptimizedAttentionScoringSystem:
    """优化的注意力得分计算系统（针对多动症儿童）"""

    def __init__(self):
        # 多动症儿童特有的注意力特征权重
        self.weights = {
            "eye_openness": 50,  # 眼睛睁开程度
            "gaze_stability": 20,  # 视线稳定性
            "head_stability": 10,  # 头部稳定性
            "focus_duration": 10,  # 持续专注时间（多动症关键指标）
            "blink_pattern": 5,  # 眨眼模式
            "motor_restlessness": 5  # 动作不安（新增指标）
        }

        # 针对多动症儿童的优化参数
        self.scoring_params = {
            # 眼部参数（多动症儿童可能眨眼更频繁）
            "ear_optimal": 0.22,
            "ear_good_threshold": 0.20,
            "ear_fair_threshold": 0.18,
            "ear_bad_threshold": 0.16,
            "ear_asymmetry_threshold": 0.05,  # 左右眼EAR差异阈值

            # 视线参数（多动症儿童视线更不稳定）
            "gaze_optimal": 0.15,
            "gaze_good_threshold": 0.25,
            "gaze_fair_threshold": 0.35,
            "gaze_bad_threshold": 0.50,
            "gaze_speed_threshold": 0.8,  # 视线移动速度阈值

            # 头部姿态参数（多动症儿童头部移动更频繁）
            "head_optimal": 8.0,
            "head_good_threshold": 15.0,
            "head_fair_threshold": 25.0,
            "head_bad_threshold": 35.0,
            "head_speed_threshold": 10.0,  # 头部移动速度阈值

            # 眨眼参数（多动症儿童眨眼模式异常）
            "blink_optimal_min": 10,
            "blink_optimal_max": 30,
            "blink_too_fast": 40,
            "blink_too_slow": 5,
            "blink_cluster_threshold": 5,  # 连续眨眼阈值

            # 专注时长参数
            "short_focus_threshold": 2.0,  # 短时专注阈值（秒）
            "medium_focus_threshold": 5.0,  # 中等专注阈值
            "long_focus_threshold": 10.0,  # 长时专注阈值

            # 动作不安参数
            "motor_threshold": 0.3,  # 动作不安阈值
            "micro_movement_freq": 3.0  # 微小动作频率阈值（次/秒）
        }

        # 历史数据记录（增加时间戳）
        self.gaze_history = deque(maxlen=300)
        self.head_pose_history = deque(maxlen=300)
        self.ear_history = deque(maxlen=300)
        self.attention_history = deque(maxlen=600)
        self.score_history = deque(maxlen=600)
        self.timestamps = deque(maxlen=600)  # 时间戳记录

        # 专注状态记录（增强版）
        self.focus_start_time = None
        self.current_focus_duration = 0
        self.longest_focus_duration = 0
        self.focus_interruptions = 0
        self.focus_sessions = []  # 记录每次专注会话
        self.focus_quality_history = deque(maxlen=100)  # 专注质量历史

        # 动作不安记录
        self.motor_movements = deque(maxlen=100)  # 动作记录
        self.micro_movement_count = 0  # 微小动作计数
        self.last_head_position = None
        self.head_movement_speed_history = deque(maxlen=50)

        # 眨眼模式分析
        self.blink_timestamps = deque(maxlen=100)  # 眨眼时间戳
        self.blink_clusters = []  # 眨眼簇记录
        self.current_blink_cluster = 0

        # 自适应阈值（根据用户表现动态调整）
        self.adaptive_params = {
            "user_ear_baseline": 0.22,
            "user_gaze_stability": 0.2,
            "user_head_stability": 10.0,
            "learning_rate": 0.01  # 学习率
        }

        # 多动症特征检测
        self.adhd_features = {
            "inattention_count": 0,
            "hyperactivity_count": 0,
            "impulsivity_events": [],
            "pattern_recognition": []
        }

    def calculate_attention_score(self, attention_state, emotion_state=None):
        """计算综合注意力分数（针对多动症儿童优化）"""
        try:
            current_time = time.time()
            self.timestamps.append(current_time)

            # 获取当前状态
            gaze_x = attention_state.get("gaze_x", 0)
            gaze_y = attention_state.get("gaze_y", 0)
            yaw = attention_state.get("yaw", 0)
            pitch = attention_state.get("pitch", 0)
            ear_left = attention_state.get("ear_left", 0)
            ear_right = attention_state.get("ear_right", 0)
            attention_label = attention_state.get("attention_label", "未知")
            face_detected = attention_state.get("face_detected", False)

            # 1. 眼睛特征评分 (0-25分)
            eye_score = self.calculate_eye_score_optimized(
                ear_left, ear_right, attention_label
            )

            # 2. 视线稳定性评分 (0-20分)
            gaze_score = self.calculate_gaze_score_optimized(
                gaze_x, gaze_y, attention_label
            )

            # 3. 头部稳定性评分 (0-15分)
            head_score = self.calculate_head_score_optimized(
                yaw, pitch, attention_label
            )

            # 4. 持续专注时间评分 (0-20分) - 多动症关键指标
            duration_score = self.calculate_duration_score_optimized(
                attention_label, current_time
            )

            # 5. 眨眼模式评分 (0-10分)
            blink_score = self.calculate_blink_score_optimized(
                ear_left, ear_right, current_time
            )

            # 6. 动作不安评分 (0-10分)
            motor_score = self.calculate_motor_score(
                yaw, pitch, current_time
            )

            # 计算基础总分
            base_score = (
                    eye_score + gaze_score + head_score +
                    duration_score + blink_score + motor_score
            )

            # 7. 情绪影响调整（考虑多动症儿童情绪敏感性）
            emotion_adjustment = self.calculate_emotion_adjustment_optimized(
                emotion_state, attention_label
            )

            # 8. 多动症特征检测与调整
            adhd_adjustment = self.detect_adhd_features(
                attention_state, emotion_state, current_time
            )

            # 9. 面部检测状态调整
            if not face_detected:
                base_score = max(0, base_score * 0.6)  # 未检测到面部，惩罚更大

            # 计算最终分数
            total_score = base_score + emotion_adjustment + adhd_adjustment

            # 应用非线性变换，突出临界区域
            total_score = self.apply_nonlinear_scaling(total_score)

            # 限制在0-100范围内
            total_score = max(0, min(100, total_score))

            # 更新自适应参数
            self.update_adaptive_params(
                ear_left, ear_right, gaze_x, gaze_y, yaw, pitch
            )

            # 更新专注状态
            self.update_focus_state_optimized(
                attention_label, total_score, current_time
            )

            # 更新历史记录
            self.update_history_optimized(
                attention_state, total_score, current_time
            )

            # 记录专注质量
            focus_quality = self.calculate_focus_quality(
                eye_score, gaze_score, head_score, duration_score
            )
            self.focus_quality_history.append(focus_quality)

            return total_score

        except Exception as e:
            print(f"计算注意力分数错误: {e}")
            traceback.print_exc()
            return 50  # 返回安全中间分数

    def calculate_eye_score_optimized(self, ear_left, ear_right, attention_label):
        """优化的眼睛特征评分"""
        if attention_label == "眼睛闭合":
            return 0

        ear_avg = (ear_left + ear_right) / 2
        ear_asymmetry = abs(ear_left - ear_right)

        # 基础评分（考虑EAR值）
        if ear_avg >= self.scoring_params["ear_optimal"]:
            base_score = 20
        elif ear_avg >= self.scoring_params["ear_good_threshold"]:
            base_score = 16
        elif ear_avg >= self.scoring_params["ear_fair_threshold"]:
            base_score = 12
        elif ear_avg >= self.scoring_params["ear_bad_threshold"]:
            base_score = 6
        else:
            base_score = 0

        # 惩罚左右眼不对称（可能表示斜视或疲劳）
        if ear_asymmetry > self.scoring_params["ear_asymmetry_threshold"]:
            asymmetry_penalty = min(5, ear_asymmetry * 20)
            base_score -= asymmetry_penalty

        # 奖励眼睛稳定性（EAR值波动小）
        if len(self.ear_history) >= 30:
            recent_ears = list(self.ear_history)[-30:]
            ear_std = np.std(recent_ears)
            if ear_std < 0.02:  # 非常稳定
                base_score += 3
            elif ear_std < 0.04:  # 稳定
                base_score += 2

        return max(0, base_score)

    def calculate_gaze_score_optimized(self, gaze_x, gaze_y, attention_label):
        """优化的视线稳定性评分"""
        if attention_label in ["视线偏离", "视线偏移"]:
            return 0

        gaze_magnitude = math.sqrt(gaze_x ** 2 + gaze_y ** 2)
        current_gaze = (gaze_x, gaze_y)

        # 基础评分（考虑视线偏移）
        if gaze_magnitude <= self.scoring_params["gaze_optimal"]:
            base_score = 15
        elif gaze_magnitude <= self.scoring_params["gaze_good_threshold"]:
            base_score = 12
        elif gaze_magnitude <= self.scoring_params["gaze_fair_threshold"]:
            base_score = 9
        elif gaze_magnitude <= self.scoring_params["gaze_bad_threshold"]:
            base_score = 5
        else:
            base_score = 0

        # 惩罚视线移动速度（快速扫视可能是注意力不集中）
        if len(self.gaze_history) >= 2:
            recent_gaze = list(self.gaze_history)[-2:]
            if len(recent_gaze) == 2:
                prev_gaze_mag = recent_gaze[0]
                gaze_speed = abs(gaze_magnitude - prev_gaze_mag)

                if gaze_speed > self.scoring_params["gaze_speed_threshold"]:
                    speed_penalty = min(5, gaze_speed * 3)
                    base_score -= speed_penalty

        # 奖励视线稳定性（长时间保持稳定）
        if len(self.gaze_history) >= 60:  # 2秒历史
            recent_gazes = list(self.gaze_history)[-60:]
            gaze_std = np.std(recent_gazes)
            if gaze_std < 0.1:  # 非常稳定
                base_score += 3
            elif gaze_std < 0.2:  # 稳定
                base_score += 2

        return max(0, base_score)

    def calculate_head_score_optimized(self, yaw, pitch, attention_label):
        """优化的头部稳定性评分"""
        if attention_label == "视线偏离":
            return 0

        # 计算头部偏移的合成值
        head_offset = math.sqrt(yaw ** 2 + pitch ** 2)
        current_head_pos = (yaw, pitch)

        # 基础评分
        if head_offset <= self.scoring_params["head_optimal"]:
            base_score = 12
        elif head_offset <= self.scoring_params["head_good_threshold"]:
            base_score = 10
        elif head_offset <= self.scoring_params["head_fair_threshold"]:
            base_score = 8
        elif head_offset <= self.scoring_params["head_bad_threshold"]:
            base_score = 4
        else:
            base_score = 0

        # 惩罚头部移动速度（多动症特征）
        if self.last_head_position is not None:
            prev_yaw, prev_pitch = self.last_head_position
            head_movement = math.sqrt(
                (yaw - prev_yaw) ** 2 + (pitch - prev_pitch) ** 2
            )

            if head_movement > self.scoring_params["head_speed_threshold"]:
                movement_penalty = min(4, head_movement * 2)
                base_score -= movement_penalty

            # 记录移动速度
            if len(self.timestamps) >= 2:
                time_diff = self.timestamps[-1] - self.timestamps[-2]
                if time_diff > 0:
                    head_speed = head_movement / time_diff
                    self.head_movement_speed_history.append(head_speed)

        # 更新头部位置
        self.last_head_position = (yaw, pitch)

        # 奖励头部稳定性
        if len(self.head_pose_history) >= 60:
            recent_heads = list(self.head_pose_history)[-60:]
            head_std = np.std(recent_heads)
            if head_std < 5.0:  # 非常稳定
                base_score += 2
            elif head_std < 10.0:  # 稳定
                base_score += 1

        return max(0, base_score)

    def calculate_duration_score_optimized(self, attention_label, current_time):
        """优化的持续专注时间评分（多动症关键指标）"""
        if self.focus_start_time is None:
            return 5  # 基础分

        focus_duration = current_time - self.focus_start_time

        # 多动症儿童通常专注时间较短，适当调整评分标准
        if focus_duration >= self.scoring_params["long_focus_threshold"]:
            return 18  # 长时专注（优秀）
        elif focus_duration >= self.scoring_params["medium_focus_threshold"]:
            return 14  # 中等专注（良好）
        elif focus_duration >= self.scoring_params["short_focus_threshold"]:
            return 9  # 短时专注（一般）
        else:
            return 4  # 短暂专注

    def calculate_blink_score_optimized(self, ear_left, ear_right, current_time):
        """优化的眨眼模式评分"""
        if len(self.ear_history) < 30:
            return 5

        # 检测眨眼（EAR值低于阈值）
        if ear_left < self.scoring_params["ear_bad_threshold"] and \
                ear_right < self.scoring_params["ear_bad_threshold"]:
            self.blink_timestamps.append(current_time)
            self.current_blink_cluster += 1
        else:
            # 如果超过一定时间没有眨眼，结束当前眨眼簇
            if self.current_blink_cluster > 0:
                if len(self.blink_timestamps) > 0:
                    last_blink = self.blink_timestamps[-1]
                    if current_time - last_blink > 0.5:  # 0.5秒内没有新眨眼
                        if self.current_blink_cluster >= self.scoring_params["blink_cluster_threshold"]:
                            self.blink_clusters.append(self.current_blink_cluster)
                        self.current_blink_cluster = 0

        # 计算最近10秒的眨眼频率
        recent_timestamps = [
            ts for ts in self.blink_timestamps
            if current_time - ts <= 10
        ]
        blink_rate = len(recent_timestamps) / 10.0 * 60  # 转换为每分钟

        # 评分（多动症儿童眨眼频率可能偏高）
        if (blink_rate >= self.scoring_params["blink_optimal_min"] and
                blink_rate <= self.scoring_params["blink_optimal_max"]):
            base_score = 8
        elif blink_rate > self.scoring_params["blink_too_fast"]:
            # 眨眼过快（可能是疲劳或焦虑）
            base_score = 3
        elif blink_rate < self.scoring_params["blink_too_slow"]:
            # 眨眼过少（可能是过度专注或疲劳）
            base_score = 4
        else:
            base_score = 6

        # 惩罚眨眼簇（连续快速眨眼）
        if len(self.blink_clusters) > 0 and self.blink_clusters[-1] >= 3:
            base_score -= 2

        return max(0, base_score)

    def calculate_motor_score(self, yaw, pitch, current_time):
        """计算动作不安评分（多动症特征）"""
        if self.last_head_position is None:
            self.last_head_position = (yaw, pitch)
            return 5

        # 计算头部微小移动
        prev_yaw, prev_pitch = self.last_head_position
        movement = math.sqrt((yaw - prev_yaw) ** 2 + (pitch - prev_pitch) ** 2)

        # 记录微小动作
        if movement > 0.5 and movement < 5.0:  # 微小移动范围
            self.micro_movement_count += 1
            self.motor_movements.append({
                "timestamp": current_time,
                "movement": movement
            })

        # 计算最近5秒的微小动作频率
        recent_movements = [
            m for m in self.motor_movements
            if current_time - m["timestamp"] <= 5
        ]
        movement_freq = len(recent_movements) / 5.0

        # 评分（动作不安越多，分数越低）
        base_score = 8
        if movement_freq > self.scoring_params["micro_movement_freq"]:
            # 动作不安明显
            base_score -= 4
            self.adhd_features["hyperactivity_count"] += 1
        elif movement_freq > self.scoring_params["micro_movement_freq"] / 2:
            # 中度动作不安
            base_score -= 2

        # 更新头部位置
        self.last_head_position = (yaw, pitch)

        return max(0, base_score)

    def calculate_emotion_adjustment_optimized(self, emotion_state, attention_label):
        """优化的情绪影响调整（考虑多动症儿童情绪调节困难）"""
        if not emotion_state:
            return 0

        emotion = emotion_state.get("emotion", "中性")
        confidence = emotion_state.get("confidence", 0)

        # 多动症儿童的情绪敏感性调整
        emotion_effects = {
            "生气": -10,  # 多动症儿童生气时注意力更差
            "恐惧": -8,  # 恐惧导致注意力分散
            "悲伤": -6,  # 悲伤影响注意力维持
            "厌恶": -4,  # 厌恶有负面影响
            "中性": 0,  # 中性情绪最利于注意力
            "惊讶": +3,  # 惊讶可能短暂提高注意力
            "快乐": +6  # 快乐情绪有助于注意力，但可能过度兴奋
        }

        base_adjustment = emotion_effects.get(emotion, 0)

        # 考虑情绪强度（置信度）
        adjusted = base_adjustment * confidence

        # 如果是快乐情绪但注意力标签为"专注"，额外奖励
        if emotion == "快乐" and attention_label == "专注":
            adjusted += 2

        return adjusted

    def detect_adhd_features(self, attention_state, emotion_state, current_time):
        """检测多动症特征并调整分数"""
        adjustment = 0

        # 1. 注意力不集中特征
        attention_label = attention_state.get("attention_label", "未知")
        if attention_label in ["视线偏离", "视线偏移", "眼睛闭合"]:
            self.adhd_features["inattention_count"] += 1

            # 连续分心惩罚
            if len(self.attention_history) >= 3:
                recent_labels = list(self.attention_history)[-3:]
                if all(label != "专注" for label in recent_labels):
                    adjustment -= 5
                elif sum(1 for label in recent_labels if label != "专注") >= 2:
                    adjustment -= 3

        # 2. 情绪不稳定特征
        if emotion_state:
            emotion = emotion_state.get("emotion", "中性")
            if emotion in ["生气", "恐惧"]:
                # 记录情绪波动事件
                self.adhd_features["impulsivity_events"].append({
                    "timestamp": current_time,
                    "emotion": emotion
                })

                # 频繁情绪波动惩罚
                recent_events = [
                    e for e in self.adhd_features["impulsivity_events"]
                    if current_time - e["timestamp"] <= 30  # 30秒内
                ]
                if len(recent_events) >= 3:
                    adjustment -= 4

        # 3. 模式识别（分心-重新专注的循环模式）
        if len(self.attention_history) >= 20:
            recent_pattern = list(self.attention_history)[-20:]
            focus_transitions = sum(
                1 for i in range(1, len(recent_pattern))
                if recent_pattern[i] == "专注" and recent_pattern[i - 1] != "专注"
            )

            # 频繁的注意力转移（可能是注意力分散）
            if focus_transitions >= 5:
                adjustment -= 3
                self.adhd_features["pattern_recognition"].append({
                    "timestamp": current_time,
                    "pattern": "frequent_transitions"
                })

        return adjustment

    def apply_nonlinear_scaling(self, score):
        """应用非线性缩放，突出临界区域"""
        if score >= 80:
            # 高分段：轻微压缩
            return 80 + (score - 80) * 0.8
        elif score >= 60:
            # 中等分段：保持线性
            return score
        elif score >= 40:
            # 低分段：适当放大差异
            return 40 + (score - 40) * 1.2
        else:
            # 很低分段：进一步放大差异
            return score * 1.5

    def update_adaptive_params(self, ear_left, ear_right, gaze_x, gaze_y, yaw, pitch):
        """根据用户表现自适应调整参数"""
        # 学习率
        alpha = self.adaptive_params["learning_rate"]

        # 更新EAR基线
        ear_avg = (ear_left + ear_right) / 2
        self.adaptive_params["user_ear_baseline"] = (
                (1 - alpha) * self.adaptive_params["user_ear_baseline"] +
                alpha * ear_avg
        )

        # 更新视线稳定性基线
        gaze_magnitude = math.sqrt(gaze_x ** 2 + gaze_y ** 2)
        self.adaptive_params["user_gaze_stability"] = (
                (1 - alpha) * self.adaptive_params["user_gaze_stability"] +
                alpha * gaze_magnitude
        )

        # 更新头部稳定性基线
        head_offset = math.sqrt(yaw ** 2 + pitch ** 2)
        self.adaptive_params["user_head_stability"] = (
                (1 - alpha) * self.adaptive_params["user_head_stability"] +
                alpha * head_offset
        )

    def update_focus_state_optimized(self, attention_label, current_score, current_time):
        """优化的专注状态更新"""
        if attention_label == "专注" and current_score >= 65:  # 降低专注阈值
            # 进入或保持专注状态
            if self.focus_start_time is None:
                self.focus_start_time = current_time
                self.current_focus_duration = 0

            self.current_focus_duration = current_time - self.focus_start_time

            # 更新最长专注时长
            if self.current_focus_duration > self.longest_focus_duration:
                self.longest_focus_duration = self.current_focus_duration

            # 记录高质量专注
            if current_score >= 80 and self.current_focus_duration >= 3.0:
                self.focus_sessions.append({
                    "start": self.focus_start_time,
                    "duration": self.current_focus_duration,
                    "quality": current_score
                })
        else:
            # 专注中断
            if self.focus_start_time is not None:
                # 记录中断前的专注会话
                if self.current_focus_duration >= 1.0:  # 至少专注1秒
                    self.focus_sessions.append({
                        "start": self.focus_start_time,
                        "duration": self.current_focus_duration,
                        "end": current_time,
                        "interrupted": True
                    })

                self.focus_interruptions += 1
                self.focus_start_time = None
                self.current_focus_duration = 0

    def update_history_optimized(self, attention_state, score, timestamp):
        """优化的历史记录更新"""
        # 记录当前分数
        self.score_history.append(score)

        # 记录注意力标签
        attention_label = attention_state.get("attention_label", "未知")
        self.attention_history.append(attention_label)

        # 记录其他数据
        gaze_x = attention_state.get("gaze_x", 0)
        gaze_y = attention_state.get("gaze_y", 0)
        gaze_magnitude = math.sqrt(gaze_x ** 2 + gaze_y ** 2)
        self.gaze_history.append(gaze_magnitude)

        yaw = attention_state.get("yaw", 0)
        pitch = attention_state.get("pitch", 0)
        self.head_pose_history.append((yaw + pitch) / 2)

        ear_avg = (attention_state.get("ear_left", 0) + attention_state.get("ear_right", 0)) / 2
        self.ear_history.append(ear_avg)

    def calculate_focus_quality(self, eye_score, gaze_score, head_score, duration_score):
        """计算专注质量指数"""
        # 归一化各个分数到0-1范围
        eye_norm = eye_score / 25.0
        gaze_norm = gaze_score / 20.0
        head_norm = head_score / 15.0
        duration_norm = duration_score / 20.0

        # 加权平均
        weights = [0.25, 0.20, 0.15, 0.40]  # 持续专注时间权重最高
        quality = (
                          eye_norm * weights[0] +
                          gaze_norm * weights[1] +
                          head_norm * weights[2] +
                          duration_norm * weights[3]
                  ) * 100

        return quality

    def get_score_analysis(self):
        """获取分数详细分析（增强版）"""
        if not self.score_history:
            return self.get_empty_analysis()

        try:
            scores = list(self.score_history)
            current_score = scores[-1] if scores else 0

            # 计算统计信息
            stats = self.calculate_statistics(scores)

            # 计算专注质量分析
            focus_analysis = self.analyze_focus_patterns()

            # 计算多动症特征分析
            adhd_analysis = self.analyze_adhd_features()

            # 组合分析结果
            analysis = {
                "current_score": round(current_score, 1),
                "statistics": stats,
                "focus_analysis": focus_analysis,
                "adhd_features": adhd_analysis,
                "attention_level": self.get_attention_level(current_score),
                "recommendations": self.generate_recommendations(
                    current_score, focus_analysis, adhd_analysis
                )
            }

            return analysis

        except Exception as e:
            print(f"获取分数分析错误: {e}")
            return self.get_empty_analysis()

    def calculate_statistics(self, scores):
        """计算统计信息"""
        if len(scores) < 10:
            return {"error": "数据不足"}

        try:
            recent_scores = scores[-30:] if len(scores) >= 30 else scores
            long_term_scores = scores[-300:] if len(scores) >= 300 else scores

            return {
                "recent_avg": round(np.mean(recent_scores), 1),
                "long_term_avg": round(np.mean(long_term_scores), 1),
                "recent_max": round(np.max(recent_scores), 1),
                "recent_min": round(np.min(recent_scores), 1),
                "recent_std": round(np.std(recent_scores), 1),
                "trend": self.calculate_trend(scores),
                "stability_index": self.calculate_stability_index(scores),
                "consistency_score": self.calculate_consistency_score(scores)
            }
        except Exception as e:
            print(f"计算统计信息错误: {e}")
            return {}

    def calculate_trend(self, scores):
        """计算分数趋势"""
        if len(scores) < 20:
            return "分析中"

        try:
            recent = scores[-10:]
            earlier = scores[-20:-10] if len(scores) >= 20 else recent

            recent_avg = np.mean(recent)
            earlier_avg = np.mean(earlier)

            if recent_avg > earlier_avg + 5:
                return "上升"
            elif recent_avg < earlier_avg - 5:
                return "下降"
            else:
                return "稳定"
        except:
            return "未知"

    def calculate_stability_index(self, scores):
        """计算稳定性指数"""
        if len(scores) < 30:
            return 0

        try:
            scores_array = np.array(scores[-30:])
            # 计算变化率的稳定性
            changes = np.diff(scores_array)
            stability = 1.0 - (np.std(changes) / 50.0)  # 归一化
            return max(0, min(100, stability * 100))
        except:
            return 0

    def calculate_consistency_score(self, scores):
        """计算一致性分数"""
        if len(scores) < 50:
            return 0

        try:
            scores_array = np.array(scores[-50:])
            # 计算在平均分±10分范围内的比例
            mean_score = np.mean(scores_array)
            within_range = np.sum(np.abs(scores_array - mean_score) <= 10)
            consistency = within_range / len(scores_array)
            return round(consistency * 100, 1)
        except:
            return 0

    def analyze_focus_patterns(self):
        """分析专注模式"""
        if not self.focus_sessions:
            return {"total_sessions": 0, "avg_duration": 0, "pattern": "无专注记录"}

        try:
            durations = [s.get("duration", 0) for s in self.focus_sessions]
            qualities = [s.get("quality", 0) for s in self.focus_sessions if "quality" in s]

            avg_duration = np.mean(durations) if durations else 0
            avg_quality = np.mean(qualities) if qualities else 0

            # 分析专注模式
            if len(durations) >= 5:
                # 计算专注时长分布
                short_focus = sum(1 for d in durations if d < 3.0)
                medium_focus = sum(1 for d in durations if 3.0 <= d < 10.0)
                long_focus = sum(1 for d in durations if d >= 10.0)

                total = len(durations)
                pattern = f"短时专注:{short_focus / total * 100:.0f}%, "
                pattern += f"中时专注:{medium_focus / total * 100:.0f}%, "
                pattern += f"长时专注:{long_focus / total * 100:.0f}%"
            else:
                pattern = "数据不足"

            return {
                "total_sessions": len(self.focus_sessions),
                "avg_duration": round(avg_duration, 1),
                "avg_quality": round(avg_quality, 1),
                "longest_duration": round(self.longest_focus_duration, 1),
                "interruptions": self.focus_interruptions,
                "pattern": pattern
            }
        except Exception as e:
            print(f"分析专注模式错误: {e}")
            return {"error": str(e)}

    def analyze_adhd_features(self):
        """分析多动症特征"""
        try:
            # 计算注意力不集中比例
            total_frames = len(self.attention_history)
            if total_frames == 0:
                return {}

            inattention_frames = sum(
                1 for label in self.attention_history
                if label != "专注" and label != "初始化中"
            )
            inattention_ratio = inattention_frames / total_frames

            # 计算动作不安指数
            motor_index = 0
            if self.head_movement_speed_history:
                avg_speed = np.mean(list(self.head_movement_speed_history))
                motor_index = min(100, avg_speed * 10)

            # 计算情绪波动指数
            emotion_volatility = 0
            if self.adhd_features["impulsivity_events"]:
                recent_events = [
                    e for e in self.adhd_features["impulsivity_events"]
                    if time.time() - e["timestamp"] <= 300  # 5分钟内
                ]
                emotion_volatility = len(recent_events) / 5.0 * 100  # 每分钟事件数×100

            return {
                "inattention_ratio": round(inattention_ratio * 100, 1),
                "hyperactivity_index": round(motor_index, 1),
                "emotion_volatility": round(emotion_volatility, 1),
                "pattern_detected": len(self.adhd_features["pattern_recognition"]) > 0,
                "risk_level": self.calculate_adhd_risk_level(
                    inattention_ratio, motor_index, emotion_volatility
                )
            }
        except Exception as e:
            print(f"分析多动症特征错误: {e}")
            return {}

    def calculate_adhd_risk_level(self, inattention_ratio, motor_index, emotion_volatility):
        """计算多动症风险等级"""
        score = (
                        inattention_ratio * 0.4 +
                        (motor_index / 100) * 0.3 +
                        (emotion_volatility / 100) * 0.3
                ) * 100

        if score >= 70:
            return "高风险"
        elif score >= 50:
            return "中风险"
        elif score >= 30:
            return "低风险"
        else:
            return "正常"

    def get_attention_level(self, score):
        """获取注意力水平描述"""
        if score >= 85:
            return "非常专注"
        elif score >= 70:
            return "专注"
        elif score >= 55:
            return "一般"
        elif score >= 40:
            return "轻度分心"
        elif score >= 25:
            return "中度分心"
        else:
            return "严重分心"

    def generate_recommendations(self, current_score, focus_analysis, adhd_analysis):
        """生成个性化建议"""
        recommendations = []

        # 基于当前分数
        if current_score < 40:
            recommendations.append("注意力水平较低，建议休息后重新开始")
        elif current_score < 60:
            recommendations.append("注意力一般，尝试减少环境干扰")

        # 基于专注模式
        if "avg_duration" in focus_analysis:
            avg_duration = focus_analysis["avg_duration"]
            if avg_duration < 3.0:
                recommendations.append("专注持续时间较短，建议使用番茄工作法（25分钟工作，5分钟休息）")
            elif avg_duration < 10.0:
                recommendations.append("专注时间中等，继续保持")
            else:
                recommendations.append("专注时间良好，注意适时休息")

        # 基于多动症特征
        if "risk_level" in adhd_analysis:
            risk_level = adhd_analysis["risk_level"]
            if risk_level == "高风险":
                recommendations.append("检测到明显的多动症特征，建议咨询专业医生")
            elif risk_level == "中风险":
                recommendations.append("检测到部分多动症特征，建议进行注意力训练")
            elif risk_level == "低风险":
                recommendations.append("有轻微的多动症倾向，保持观察")

        # 通用建议
        recommendations.append("确保充足睡眠和规律作息")
        recommendations.append("进行适当的体育锻炼")
        recommendations.append("使用计时器帮助管理时间")

        return recommendations

    def get_empty_analysis(self):
        """获取空分析结果"""
        return {
            "current_score": 0,
            "statistics": {
                "recent_avg": 0,
                "long_term_avg": 0,
                "recent_max": 0,
                "recent_min": 0,
                "recent_std": 0,
                "trend": "分析中",
                "stability_index": 0,
                "consistency_score": 0
            },
            "focus_analysis": {
                "total_sessions": 0,
                "avg_duration": 0,
                "avg_quality": 0,
                "longest_duration": 0,
                "interruptions": 0,
                "pattern": "无数据"
            },
            "adhd_features": {},
            "attention_level": "未知",
            "recommendations": ["等待更多数据..."]
        }

    def reset(self):
        """重置分数系统"""
        self.gaze_history.clear()
        self.head_pose_history.clear()
        self.ear_history.clear()
        self.attention_history.clear()
        self.score_history.clear()
        self.timestamps.clear()
        self.focus_quality_history.clear()
        self.motor_movements.clear()
        self.head_movement_speed_history.clear()
        self.blink_timestamps.clear()
        self.blink_clusters.clear()

        # 重置专注状态
        self.focus_start_time = None
        self.current_focus_duration = 0
        self.longest_focus_duration = 0
        self.focus_interruptions = 0
        self.focus_sessions.clear()

        # 重置动作记录
        self.micro_movement_count = 0
        self.last_head_position = None
        self.current_blink_cluster = 0

        # 重置多动症特征
        self.adhd_features = {
            "inattention_count": 0,
            "hyperactivity_count": 0,
            "impulsivity_events": [],
            "pattern_recognition": []
        }

# ============================================================================
#  校准
# ============================================================================

class CalibrationSystem:
    """校准系统 - 确保正常专注状态"""

    def __init__(self):
        self.calibration_steps = [
            "center",  # 中心位置
            "top_left",  # 左上
            "top_right",  # 右上
            "bottom_left",  # 左下
            "bottom_right"  # 右下
        ]

        self.current_step = 0
        self.is_calibrating = False
        self.calibration_data = {step: [] for step in self.calibration_steps}
        self.calibration_results = {}
        self.reference_gaze_center = (0.0, 0.0)
        self.gaze_tolerance = 0.2

        # 校准文件路径
        self.calibration_file = "calibration_data.json"

        # 尝试加载已有的校准数据
        self.load_calibration()

    def save_calibration(self):
        """保存校准数据到文件"""
        try:
            data = {
                "reference_gaze_center": self.reference_gaze_center,
                "gaze_tolerance": self.gaze_tolerance,
                "calibration_results": self.calibration_results,
                "save_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            with open(self.calibration_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"校准数据已保存到: {self.calibration_file}")
            return True
        except Exception as e:
            print(f"保存校准数据失败: {e}")
            return False

    def load_calibration(self):
        """从文件加载校准数据"""
        try:
            if os.path.exists(self.calibration_file):
                with open(self.calibration_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.reference_gaze_center = tuple(data.get("reference_gaze_center", (0.0, 0.0)))
                self.gaze_tolerance = data.get("gaze_tolerance", 0.2)
                self.calibration_results = data.get("calibration_results", {})
                print(f"已加载校准数据 (保存时间: {data.get('save_time', '未知')})")
                return True
            else:
                print("未找到校准文件，需要重新校准")
                return False
        except Exception as e:
            print(f"加载校准数据失败: {e}")
            return False

    def start_calibration(self):
        """开始校准"""
        self.is_calibrating = True
        self.current_step = 0
        self.calibration_data = {step: [] for step in self.calibration_steps}
        self.calibration_results = {}

        return {
            "status": "开始",
            "current_step": self.calibration_steps[self.current_step],
            "instruction": "请注视屏幕中央的红点"
        }

    def process_calibration_frame(self, frame, gaze_data):
        """处理校准帧"""
        if not self.is_calibrating:
            return None

        step_name = self.calibration_steps[self.current_step]

        # 记录当前步骤的数据
        self.calibration_data[step_name].append({
            "gaze_x": gaze_data.get("gaze_x", 0),
            "gaze_y": gaze_data.get("gaze_y", 0),
            "timestamp": time.time()
        })

        # 检查是否收集了足够数据（约2秒，60帧）
        if len(self.calibration_data[step_name]) >= 60:
            self.complete_current_step()

            if self.current_step < len(self.calibration_steps) - 1:
                # 进入下一步
                self.current_step += 1
                next_step = self.calibration_steps[self.current_step]

                return {
                    "status": "继续",
                    "current_step": next_step,
                    "progress": (self.current_step + 1) / len(self.calibration_steps),
                    "instruction": self.get_instruction(next_step)
                }
            else:
                # 所有步骤完成
                result = self.finalize_calibration()
                if result.get("success"):
                    # 保存校准结果
                    self.save_calibration()
                return {
                    "status": "完成",
                    "progress": 1.0,
                    "results": self.calibration_results,
                    "success": result.get("success", False)
                }

        # 仍在当前步骤
        return {
            "status": "进行中",
            "current_step": step_name,
            "progress": len(self.calibration_data[step_name]) / 60,
            "samples": len(self.calibration_data[step_name]),
            "instruction": f"保持注视 {step_name}"
        }

    def finalize_calibration(self):
        """完成校准"""
        try:
            # 计算中心位置的视线范围
            center_data = self.calibration_results.get("center", {})
            if center_data:
                gaze_x, gaze_y = center_data["average_gaze"]
                std_x, std_y = center_data["stability"]

                # 设置参考中心点和容差
                self.reference_gaze_center = (gaze_x, gaze_y)

                # 计算容差范围（平均值 ± 2倍标准差）
                self.gaze_tolerance = max(0.15, 2 * max(std_x, std_y))

                # 计算各个方向的视线偏移
                offsets = {}
                for step in self.calibration_steps:
                    if step != "center" and step in self.calibration_results:
                        offset_x = abs(self.calibration_results[step]["average_gaze"][0] - gaze_x)
                        offset_y = abs(self.calibration_results[step]["average_gaze"][1] - gaze_y)
                        offsets[step] = (offset_x, offset_y)

                self.is_calibrating = False

                return {
                    "success": True,
                    "reference_center": self.reference_gaze_center,
                    "tolerance": self.gaze_tolerance,
                    "offsets": offsets
                }

            return {"success": False, "error": "中心校准数据缺失"}

        except Exception as e:
            print(f"完成校准错误: {e}")
            return {"success": False, "error": str(e)}

    def get_calibration_status(self):
        """获取校准状态"""
        status = {
            "is_calibrating": self.is_calibrating,
            "current_step": self.calibration_steps[self.current_step] if self.is_calibrating else "无",
            "reference_center": self.reference_gaze_center,
            "tolerance": self.gaze_tolerance,
            "is_calibrated": self.reference_gaze_center != (0, 0)
        }

        if self.is_calibrating:
            step_name = self.calibration_steps[self.current_step]
            status["progress"] = len(self.calibration_data[step_name]) / 60
            status["samples"] = len(self.calibration_data[step_name])

        return status

    def get_instruction(self, step_name):
        """获取校准指令"""
        instructions = {
            "center": "请注视屏幕中央的红点",
            "top_left": "请注视屏幕左上角的红点",
            "top_right": "请注视屏幕右上角的红点",
            "bottom_left": "请注视屏幕左下角的红点",
            "bottom_right": "请注视屏幕右下角的红点"
        }
        return instructions.get(step_name, "请注视红点")

    def complete_current_step(self):
        """完成当前校准步骤"""
        step_name = self.calibration_steps[self.current_step]

        if self.calibration_data[step_name]:
            # 计算平均值
            gaze_x_list = [d["gaze_x"] for d in self.calibration_data[step_name][-30:]]  # 取后30帧
            gaze_y_list = [d["gaze_y"] for d in self.calibration_data[step_name][-30:]]

            avg_gaze_x = np.mean(gaze_x_list)
            avg_gaze_y = np.mean(gaze_y_list)

            # 计算标准差（评估稳定性）
            std_gaze_x = np.std(gaze_x_list)
            std_gaze_y = np.std(gaze_y_list)

            self.calibration_results[step_name] = {
                "average_gaze": (avg_gaze_x, avg_gaze_y),
                "stability": (std_gaze_x, std_gaze_y),
                "samples": len(self.calibration_data[step_name])
            }

    def check_gaze_within_tolerance(self, gaze_x, gaze_y):
        """检查视线是否在容差范围内"""
        if not hasattr(self, 'reference_gaze_center') or self.reference_gaze_center == (0, 0):
            return True  # 未校准，返回默认值

        ref_x, ref_y = self.reference_gaze_center

        # 计算欧氏距离
        distance = math.sqrt((gaze_x - ref_x) ** 2 + (gaze_y - ref_y) ** 2)

        return distance <= self.gaze_tolerance

    def reset_calibration(self):
        """重置校准"""
        self.is_calibrating = False
        self.current_step = 0
        self.calibration_data = {step: [] for step in self.calibration_steps}
        self.calibration_results = {}
        self.reference_gaze_center = (0.0, 0.0)
        self.gaze_tolerance = 0.2


# ============================================================================
#  实时图表
# ============================================================================

class RealTimeCharts:
    """实时图表绘制系统"""

    def __init__(self):
        self.history_length = 100  # 存储100个数据点
        self.chart_width = 380
        self.chart_height = 180

        # 数据历史
        self.attention_scores = deque(maxlen=self.history_length)
        self.gaze_x_values = deque(maxlen=self.history_length)
        self.gaze_y_values = deque(maxlen=self.history_length)
        self.ear_values = deque(maxlen=self.history_length)
        self.head_yaw_values = deque(maxlen=self.history_length)
        self.head_pitch_values = deque(maxlen=self.history_length)

        # 图表颜色
        self.colors = {
            "attention": QColor(66, 134, 244),  # 蓝色
            "gaze_x": QColor(244, 67, 54),  # 红色
            "gaze_y": QColor(76, 175, 80),  # 绿色
            "ear": QColor(255, 152, 0),  # 橙色
            "head": QColor(156, 39, 176),  # 紫色
            "grid": QColor(200, 200, 200, 100),  # 网格线
            "background": QColor(245, 245, 245)  # 背景
        }

    def update_data(self, attention_state, emotion_state=None):
        """更新数据"""
        try:
            # 注意力分数
            self.attention_scores.append(attention_state.get("attention_score", 0))

            # 视线数据
            self.gaze_x_values.append(attention_state.get("gaze_x", 0))
            self.gaze_y_values.append(attention_state.get("gaze_y", 0))

            # EAR值
            ear_avg = (attention_state.get("ear_left", 0) + attention_state.get("ear_right", 0)) / 2
            self.ear_values.append(ear_avg)

            # 头部姿态
            self.head_yaw_values.append(abs(attention_state.get("yaw", 0)))
            self.head_pitch_values.append(abs(attention_state.get("pitch", 0)))

        except Exception as e:
            print(f"更新图表数据错误: {e}")

    def draw_attention_chart(self, painter, x, y, width, height):
        """绘制注意力分数图表"""
        if not self.attention_scores:
            return self.draw_no_data(painter, x, y, width, height, "注意力分数")

        try:
            # 绘制背景
            painter.fillRect(x, y, width, height, self.colors["background"])

            # 绘制网格线
            pen = QPen(self.colors["grid"], 1)
            painter.setPen(pen)

            # 垂直网格线
            grid_x_count = 6
            for i in range(1, grid_x_count):
                grid_x = x + i * width // grid_x_count
                painter.drawLine(grid_x, y, grid_x, y + height)

            # 水平网格线 (0-100分)
            grid_y_count = 5
            for i in range(1, grid_y_count):
                grid_y = y + i * height // grid_y_count
                painter.drawLine(x, grid_y, x + width, grid_y)

            # 绘制分数参考线
            pen = QPen(QColor(255, 152, 0, 150), 2, Qt.DashLine)
            painter.setPen(pen)

            # 70分线（专注阈值）
            threshold_y = y + height - 70 * height // 100
            painter.drawLine(x, threshold_y, x + width, threshold_y)

            # 绘制坐标轴标签
            painter.setPen(Qt.black)
            painter.setFont(QFont("Microsoft YaHei", 8))

            # Y轴标签 (分数)
            for i in range(0, 101, 20):
                label_y = y + height - i * height // 100
                painter.drawText(x - 25, label_y + 4, f"{i}")

            # 绘制曲线
            if len(self.attention_scores) > 1:
                pen = QPen(self.colors["attention"], 3)
                painter.setPen(pen)

                points = []
                for i, score in enumerate(self.attention_scores):
                    # 计算点位置
                    point_x = x + i * width // (len(self.attention_scores) - 1) if len(self.attention_scores) > 1 else x
                    point_y = int(y + height - score * height // 100)
                    points.append(QPoint(point_x, point_y))

                # 绘制连线
                for i in range(len(points) - 1):
                    painter.drawLine(points[i], points[i + 1])

                # 绘制数据点
                painter.setBrush(QBrush(self.colors["attention"]))
                for point in points[-10:]:  # 只绘制最近10个点
                    painter.drawEllipse(point, 3, 3)

            # 添加标题
            painter.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))
            painter.setPen(Qt.darkBlue)
            painter.drawText(x + 10, y + 20, "注意力分数趋势")

            # 显示当前分数
            current_score = self.attention_scores[-1] if self.attention_scores else 0
            score_text = f"当前: {current_score:.1f}"
            painter.setFont(QFont("Microsoft YaHei", 9))

            if current_score >= 70:
                painter.setPen(Qt.darkGreen)
            elif current_score >= 50:
                painter.setPen(Qt.darkYellow)
            else:
                painter.setPen(Qt.darkRed)

            painter.drawText(x + width - 80, y + 20, score_text)

        except Exception as e:
            print(f"绘制注意力图表错误: {e}")
            self.draw_error(painter, x, y, width, height, "图表错误")

    def draw_gaze_chart(self, painter, x, y, width, height):
        """绘制视线追踪图表"""
        if not self.gaze_x_values or not self.gaze_y_values:
            return self.draw_no_data(painter, x, y, width, height, "视线追踪")

        try:
            # 绘制背景
            painter.fillRect(x, y, width, height, self.colors["background"])

            # 计算中心点
            center_x = x + width // 2
            center_y = y + height // 2

            # 绘制坐标系
            pen = QPen(Qt.black, 1)
            painter.setPen(pen)

            # X轴和Y轴
            painter.drawLine(x, center_y, x + width, center_y)
            painter.drawLine(center_x, y, center_x, y + height)

            # 绘制网格圆（视线容差范围）
            for radius in [height // 4, height // 2, 3 * height // 4]:
                pen = QPen(QColor(200, 200, 200, 100), 1)
                painter.setPen(pen)
                painter.drawEllipse(center_x - radius, center_y - radius, radius * 2, radius * 2)

            # 绘制视线轨迹
            if len(self.gaze_x_values) > 1 and len(self.gaze_y_values) > 1:
                # 绘制X轴视线
                pen = QPen(self.colors["gaze_x"], 2)
                painter.setPen(pen)

                for i in range(len(self.gaze_x_values)):
                    if i == 0:
                        continue

                    # 转换视线坐标为图表坐标
                    prev_x = center_x + self.gaze_x_values[i - 1] * width // 2
                    prev_y = center_y - self.gaze_y_values[i - 1] * height // 2
                    curr_x = center_x + self.gaze_x_values[i] * width // 2
                    curr_y = center_y - self.gaze_y_values[i] * height // 2

                    # 限制在图表范围内
                    prev_x = max(x, min(x + width, prev_x))
                    prev_y = max(y, min(y + height, prev_y))
                    curr_x = max(x, min(x + width, curr_x))
                    curr_y = max(y, min(y + height, curr_y))

                    painter.drawLine(int(prev_x), int(prev_y), int(curr_x), int(curr_y))

                # 绘制当前视线点
                if self.gaze_x_values and self.gaze_y_values:
                    curr_x = center_x + self.gaze_x_values[-1] * width // 2
                    curr_y = center_y - self.gaze_y_values[-1] * height // 2

                    # 绘制点
                    painter.setBrush(QBrush(self.colors["gaze_x"]))
                    painter.drawEllipse(int(curr_x) - 4, int(curr_y) - 4, 8, 8)

            # 添加标题
            painter.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))
            painter.setPen(Qt.darkBlue)
            painter.drawText(x + 10, y + 20, "视线追踪")

            # 显示当前视线坐标
            if self.gaze_x_values and self.gaze_y_values:
                gaze_text = f"X: {self.gaze_x_values[-1]:.2f}, Y: {self.gaze_y_values[-1]:.2f}"
                painter.setFont(QFont("Microsoft YaHei", 8))
                painter.setPen(Qt.darkGray)
                painter.drawText(x + width - 120, y + height - 10, gaze_text)

        except Exception as e:
            print(f"绘制视线图表错误: {e}")
            self.draw_error(painter, x, y, width, height, "图表错误")

    def draw_eye_chart(self, painter, x, y, width, height):
        """绘制眼部特征图表（带图例）"""
        if not self.ear_values:
            return self.draw_no_data(painter, x, y, width, height, "眼部特征")

        try:
            # 绘制背景
            painter.fillRect(x, y, width, height, self.colors["background"])

            # 绘制网格线
            pen = QPen(self.colors["grid"], 1)
            painter.setPen(pen)

            # 垂直网格线
            grid_x_count = 5
            for i in range(1, grid_x_count):
                grid_x = x + i * width // grid_x_count
                painter.drawLine(grid_x, y, grid_x, y + height)

            # 水平网格线 (0-0.4 EAR)
            grid_y_count = 5
            for i in range(1, grid_y_count):
                grid_y = y + i * height // grid_y_count
                painter.drawLine(x, grid_y, x + width, grid_y)

            # 绘制参考线（眨眼阈值 0.21）
            pen = QPen(QColor(244, 67, 54, 150), 2, Qt.DashLine)
            painter.setPen(pen)

            threshold_y = int(y + height - 0.21 * height // 0.4)
            painter.drawLine(x, threshold_y, x + width, threshold_y)

            # 绘制参考线标签
            painter.setPen(Qt.darkRed)
            painter.setFont(QFont("Microsoft YaHei", 7))
            painter.drawText(x + 5, threshold_y - 5, "Blink Threshold (0.21)")

            # 绘制坐标轴标签
            painter.setPen(Qt.black)
            painter.setFont(QFont("Microsoft YaHei", 7))

            # Y轴标签 (EAR值)
            for i in range(0, 5):
                ear_value = i * 0.1
                label_y = int(y + height - ear_value * height // 0.4)
                painter.drawText(x - 20, label_y + 3, f"{ear_value:.1f}")

            # 绘制EAR曲线
            ear_points = []
            if len(self.ear_values) > 1:
                pen = QPen(self.colors["ear"], 2)
                painter.setPen(pen)

                for i, ear in enumerate(self.ear_values):
                    # 计算点位置
                    point_x = x + i * width // (len(self.ear_values) - 1) if len(self.ear_values) > 1 else x
                    point_y = y + height - ear * height // 0.4
                    ear_points.append(QPoint(int(point_x), int(point_y)))

                # 绘制连线
                for i in range(len(ear_points) - 1):
                    painter.drawLine(ear_points[i], ear_points[i + 1])

            # 绘制头部姿态曲线（Yaw和Pitch）
            yaw_points = []
            pitch_points = []

            if len(self.head_yaw_values) > 1 and len(self.head_pitch_values) > 1:
                # Yaw (偏转) - 蓝色
                yaw_pen = QPen(QColor(30, 144, 255), 1.5)  # 蓝色
                painter.setPen(yaw_pen)

                for i, yaw_value in enumerate(self.head_yaw_values):
                    if i == 0:
                        continue
                    prev_x = x + (i - 1) * width // (len(self.head_yaw_values) - 1)
                    prev_y = y + height - yaw_value * 3  # 缩放因子
                    curr_x = x + i * width // (len(self.head_yaw_values) - 1)
                    curr_y = y + height - self.head_yaw_values[i] * 3
                    painter.drawLine(int(prev_x), int(prev_y), int(curr_x), int(curr_y))
                    yaw_points.append(QPoint(int(curr_x), int(curr_y)))

                # Pitch (俯仰) - 橙色
                pitch_pen = QPen(QColor(255, 165, 0), 1.5)  # 橙色
                painter.setPen(pitch_pen)

                for i, pitch_value in enumerate(self.head_pitch_values):
                    if i == 0:
                        continue
                    prev_x = x + (i - 1) * width // (len(self.head_pitch_values) - 1)
                    prev_y = y + height - pitch_value * 3  # 缩放因子
                    curr_x = x + i * width // (len(self.head_pitch_values) - 1)
                    curr_y = y + height - self.head_pitch_values[i] * 3
                    painter.drawLine(int(prev_x), int(prev_y), int(curr_x), int(curr_y))
                    pitch_points.append(QPoint(int(curr_x), int(curr_y)))

            # 添加图例
            legend_x = x + 10
            legend_y = y + 15

            # EAR图例
            painter.setPen(QPen(QColor(255, 152, 0), 2))
            painter.drawLine(legend_x, legend_y, legend_x + 30, legend_y)
            painter.setPen(Qt.black)
            painter.drawText(legend_x + 35, legend_y + 4, "EAR")

            # Yaw图例
            legend_y += 20
            painter.setPen(QPen(QColor(30, 144, 255), 2))
            painter.drawLine(legend_x, legend_y, legend_x + 30, legend_y)
            painter.setPen(Qt.black)
            painter.drawText(legend_x + 35, legend_y + 4, "Yaw")

            # Pitch图例
            legend_y += 20
            painter.setPen(QPen(QColor(255, 165, 0), 2))
            painter.drawLine(legend_x, legend_y, legend_x + 30, legend_y)
            painter.setPen(Qt.black)
            painter.drawText(legend_x + 35, legend_y + 4, "Pitch")

            # 添加标题
            painter.setFont(QFont("Microsoft YaHei", 9, QFont.Bold))
            painter.setPen(Qt.darkBlue)
            painter.drawText(x + 10, y + 15, "眼部与头部特征")

            # 显示当前EAR值
            current_ear = self.ear_values[-1] if self.ear_values else 0
            ear_status = "Open" if current_ear > 0.21 else "Closed"
            ear_color = Qt.darkGreen if current_ear > 0.21 else Qt.darkRed
            ear_text = f"EAR: {current_ear:.2f} ({ear_status})"
            painter.setFont(QFont("Microsoft YaHei", 8))
            painter.setPen(ear_color)
            painter.drawText(x + width - 120, y + 15, ear_text)

            # 显示当前头部姿态
            if self.head_yaw_values and self.head_pitch_values:
                current_yaw = self.head_yaw_values[-1]
                current_pitch = self.head_pitch_values[-1]
                head_text = f"Head: Yaw={current_yaw:.1f}°, Pitch={current_pitch:.1f}°"
                painter.setFont(QFont("Microsoft YaHei", 7))
                painter.setPen(Qt.darkGray)
                painter.drawText(x + width - 200, y + height - 10, head_text)

        except Exception as e:
            print(f"绘制眼部图表错误: {e}")
            self.draw_error(painter, x, y, width, height, "Chart Error")

    def draw_no_data(self, painter, x, y, width, height, title):
        """绘制无数据提示"""
        painter.fillRect(x, y, width, height, QColor(240, 240, 240))

        painter.setPen(Qt.gray)
        painter.setFont(QFont("Microsoft YaHei", 12))

        text_x = x + width // 2 - 100
        text_y = y + height // 2

        painter.drawText(text_x, text_y, f"等待{title}数据...")

    def draw_error(self, painter, x, y, width, height, message):
        """绘制错误提示"""
        painter.fillRect(x, y, width, height, QColor(255, 230, 230))

        painter.setPen(Qt.red)
        painter.setFont(QFont("Microsoft YaHei", 10))

        text_x = x + width // 2 - 40
        text_y = y + height // 2

        painter.drawText(text_x, text_y, message)

    def get_statistics(self):
        """获取统计信息"""
        if not self.attention_scores:
            return {}

        try:
            scores = list(self.attention_scores)
            gaze_x = list(self.gaze_x_values)
            gaze_y = list(self.gaze_y_values)
            ears = list(self.ear_values)

            stats = {
                "attention": {
                    "current": scores[-1] if scores else 0,
                    "average": np.mean(scores) if scores else 0,
                    "max": np.max(scores) if scores else 0,
                    "min": np.min(scores) if scores else 0,
                    "std": np.std(scores) if scores else 0
                },
                "gaze": {
                    "x_mean": np.mean(gaze_x) if gaze_x else 0,
                    "y_mean": np.mean(gaze_y) if gaze_y else 0,
                    "x_std": np.std(gaze_x) if gaze_x else 0,
                    "y_std": np.std(gaze_y) if gaze_y else 0
                },
                "eye": {
                    "ear_mean": np.mean(ears) if ears else 0,
                    "ear_std": np.std(ears) if ears else 0,
                    "blink_frames": sum(1 for ear in ears if ear < 0.21) if ears else 0
                }
            }

            return stats

        except Exception as e:
            print(f"获取统计信息错误: {e}")
            return {}

# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    # 设置应用信息
    app.setApplicationName("多动症儿童注意力与情绪检测系统")
    app.setApplicationDisplayName("多动症检测系统 v4.0")

    # 创建主窗口
    window = ADHDDetectionSystem()
    window.show()

    # 显示欢迎消息
    welcome_msg = """多动症儿童注意力与情绪检测系统 v4.0

    系统功能：
    1. 实时注意力分析（使用眼动追踪和头部姿态）
    2. 情绪识别（深度学习模型）
    3. 双输入模式：实时摄像头和视频上传
    4. 实时可视化显示和警报
    5. 全面的报告和统计信息
    6. 语音反馈系统

    使用说明：
    1. 点击'启动摄像头'进行实时分析
    2. 或点击'上传视频'分析录制的视频
    3. 在控制面板中调整显示设置
    4. 在右侧面板查看实时统计数据
    5. 导出全面的分析报告

    注意：为确保最佳效果，请确保良好的照明和正确的摄像头位置。
    """

    QMessageBox.information(window, "欢迎使用", welcome_msg)

    sys.exit(app.exec_())


if __name__ == '__main__':
    # 创建必要的目录
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    os.makedirs("recordings", exist_ok=True)

    main()