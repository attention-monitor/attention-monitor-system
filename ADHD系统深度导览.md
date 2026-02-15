# ADHD 注意力监测系统 · 深度导览文档

> 本文档是你理解整个系统的核心参考资料。覆盖三个主题：ADHD 特异性设计、论文-代码对应关系、注意力评分的完整拆解。
>
> 文件：`ui.py`（约 5050 行） | 版本：v4.0 | 核心类：9 个

---

## 一、ADHD 特异性设计——这个系统"专为 ADHD 而做"的地方

普通的注意力检测系统只关心"此刻学生是否在看屏幕"。本系统与之不同的地方，全部列在下面。

### 1.1 评分权重的病理学依据

评分权重不是拍脑袋定的，而是基于论文 Section 2.1 的 fMRI 分析。论文使用 ADHD-200 公开数据集（776 份 rs-fMRI 脑影像，其中 285 名 ADHD 患者），发现 ADHD 个体在**前额叶皮层**（对应注意力控制）和**运动皮层**（对应运动抑制）有显著异常。这决定了代码中的权重分配：

| 脑区异常 | 对应的外在行为 | 代码中的评分维度 | 权重占比 |
|---------|-------------|---------------|---------|
| 前额叶-默认模式网络连接减弱 | 眼睛状态异常（闭眼、半睁） | eye_openness | 25% |
| 前额叶执行功能下降 | 视线不稳定、频繁扫视 | gaze_stability | 20% |
| 前额叶注意力维持困难 | 无法长时间保持专注 | focus_duration | 20% |
| 前额叶-基底节连接异常 | 头部姿态偏离 | head_stability | 15% |
| 运动皮层抑制功能不足 | 眨眼模式异常 | blink_pattern | 10% |
| 运动皮层过度激活 | 身体微动、坐不住 | motor_restlessness | 10% |

**代码位置：** `OptimizedAttentionScoringSystem.__init__()` 第 3391 行

### 1.2 专门的 ADHD 行为模式检测

函数 `detect_adhd_features()`（第 3851 行）检测三种 DSM-5 定义的 ADHD 典型行为模式，并对评分进行额外惩罚：

**模式一：连续分心（对应 DSM-5 注意力缺陷）**
- 检测条件：最近 3 帧的注意力标签全部不是"专注"
- 惩罚：-5 分
- 较轻情况：最近 3 帧中有 2 帧不是"专注"→ -3 分
- 代码位置：第 3861-3866 行

**模式二：情绪波动（对应 DSM-5 情绪失调）**
- 检测条件：30 秒内出现 ≥3 次"生气"或"恐惧"情绪
- 惩罚：-4 分
- 原理：ADHD 儿童情绪调节困难，频繁的负面情绪波动是特征性表现
- 代码位置：第 3868-3884 行

**模式三：注意力频繁转移（对应 DSM-5 冲动性）**
- 检测条件：最近 20 帧中，"非专注→专注"的切换次数 ≥5 次
- 惩罚：-3 分
- 原理：不是"完全走神"，而是反复地分心又拉回、分心又拉回——这是 ADHD 特有的注意力"弹跳"现象
- 代码位置：第 3886-3900 行

**最大累计惩罚：-12 分**（三种模式同时触发）

### 1.3 持续专注时长作为独立高权重维度

函数 `calculate_duration_score_optimized()`（第 3718 行）

这是普通注意力系统不会有的设计。ADHD 的核心障碍之一是"注意力维持困难"——不是不能集中注意力，而是无法**持续**集中。因此代码把"已经连续专注了多久"单独拿出来作为满分 20 分的独立维度：

| 持续专注时长 | 得分 | 阈值参数名 |
|------------|-----|-----------|
| ≥ 10 秒 | 18 分 | long_focus_threshold |
| ≥ 5 秒 | 14 分 | medium_focus_threshold |
| ≥ 2 秒 | 9 分 | short_focus_threshold |
| < 2 秒 | 4 分 | — |
| 无数据 | 5 分（基础分） | — |

注意：满分是 20 分但最高只给 18 分，这是因为这个维度不设稳定性奖励。

**专注状态的判定与中断逻辑**（`update_focus_state_optimized()`，第 3945 行）：
- 进入专注：注意力标签为"专注" **且** 当前分数 ≥ 65 分
- 专注中断：一旦标签不是"专注"或分数 < 65，计时器立刻归零
- 系统会记录每次专注会话的时长和质量，以及中断次数

### 1.4 运动躁动度检测

函数 `calculate_motor_score()`（第 3781 行）

专门检测 ADHD 多动特征——不是大幅度的转头看别处，而是那种坐不住的**微小频繁晃动**：

- 微动定义：帧间头部移动幅度在 0.5°~5.0° 之间（太小是噪声，太大是主动转头）
- 检测窗口：最近 5 秒内的微动频率
- 评分逻辑：
  - 基础 8 分
  - 微动频率 > 3.0 次/秒 → -4 分（明显躁动），同时累加 hyperactivity_count
  - 微动频率 > 1.5 次/秒 → -2 分（中度躁动）
- 代码位置：第 3781-3819 行

### 1.5 眨眼簇检测

函数 `calculate_blink_score_optimized()`（第 3735 行）

ADHD 儿童在压力或焦虑下会出现连续快速眨眼（"眨眼簇"），这与普通的生理性眨眼不同：

- 眨眼簇定义：0.5 秒内连续眨眼 ≥5 次（`blink_cluster_threshold = 5`）
- 如果检测到眨眼簇 → 额外 -2 分
- 正常眨眼频率范围：10~30 次/分钟得 8 分；>40 次/分钟（过快）得 3 分；<5 次/分钟（过慢）得 4 分

### 1.6 自适应参数调整

函数 `update_adaptive_params()`（第 3919 行）

每个 ADHD 儿童的症状严重程度不同，系统用指数移动平均（学习率 α = 0.01）动态学习每个用户的个体基线：

```
新基线 = (1 - 0.01) × 旧基线 + 0.01 × 当前观测值
```

调整的三个基线参数：
- `user_ear_baseline`：个人眼睛睁开度基线（初始 0.22）
- `user_gaze_stability`：个人视线稳定性基线（初始 0.2）
- `user_head_stability`：个人头部稳定性基线（初始 10.0°）

### 1.7 情绪系数针对 ADHD 的特殊设定

函数 `calculate_emotion_adjustment_optimized()`（第 3821 行）

代码注释明确写了"考虑多动症儿童情绪调节困难"。情绪影响系数的设定比普通系统更极端——生气的惩罚高达 -10 分（乘以置信度），因为 ADHD 儿童在生气状态下注意力崩溃的程度比普通儿童严重得多。

---

## 二、论文核心部分与代码实现的对应

### 2.1 系统架构总对应（论文 Figure 1）

```
论文架构层                              代码实现
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
帧提取与噪声抑制层                       ADHDDetectionSystem
  └─ 视频流获取与预处理                   start_camera() / upload_video()
                                        update_frame() → 640×480 @ 15fps

深度行为挖掘引擎（Section 2.1）           FacialModeling 类（第 3184 行）
  ├─ 病理特征映射矩阵 R                  固定权重配置（第 3391 行）
  ├─ 动态权重 ω_i（公式 2）              update_adaptive_params()（第 3919 行）
  └─ ADHD 行为模式检测                   detect_adhd_features()（第 3851 行）

改进 Xception 情感识别（Section 2.2）     EmotionAnalyzer 类（第 422 行）
  ├─ DAF-Xception 网络                  video.h5 模型文件（16MB）
  ├─ 输入：48×48 灰度图                  shape_x=48, shape_y=48
  ├─ Dlib 人脸检测 + 68 点定位            face_landmarks.dat（99MB）
  └─ 7 类情绪 Softmax 输出               predict_emotion()（第 518 行）

STBP-AS 多模态追踪器（Section 2.3）      AttentionAnalyzer 类（第 92 行）
  ├─ MediaPipe FaceMesh（478 点）        mp_face_mesh.FaceMesh（第 97 行）
  ├─ EAR 眼睛纵横比                      eye_aspect_ratio()（第 134 行）
  ├─ SolvePnP 头部姿态                   head_pose()（第 149 行）
  ├─ 虹膜中心 + 视线向量                  iris_center() + gaze_vector()
  └─ 注意力标签判定                       attention_label()（第 219 行）

注意力评分与融合层（Section 2.3.2）       OptimizedAttentionScoringSystem（第 3387 行）
  ├─ 公式 13：δ 偏离强度                 6 个 calculate_*_score 子函数
  ├─ 公式 14：S_att 综合评分              calculate_attention_score()（第 3484 行）
  └─ 时间平滑                            deque 滑动窗口 + 历史记录

自适应学习支持层（Section 2.4）
  ├─ 个性化语音反馈                       VoiceReminderSystem（第 674 行）
  ├─ 校准系统                            CalibrationSystem（第 4346 行）
  ├─ 实时可视化                          RealTimeCharts（第 4584 行）
  └─ 学习报告导出                        export_report()
```

### 2.2 三个关键公式的代码实现

**公式 1：映射矩阵 B = f(R·Q + ε)**

论文描述：用 Pearson 相关系数矩阵 R 将 fMRI 神经特征 Q 映射为行为特征 B。

代码实现：这个映射过程在代码中被简化为固定权重——论文的理论分析告诉我们哪些行为和 ADHD 病理最相关，代码直接用预设的权重比例来体现这个结论，而不是运行时实际计算映射矩阵。

**公式 2：动态权重 ω_i**

论文描述：权重随着行为-病理相关性和统计显著性动态调整。

代码实现：`update_adaptive_params()`（第 3919 行），使用指数移动平均以 α=0.01 的学习率调整三个基线参数。这是一个简化但等价的实现——它不是调整"权重"本身，而是调整每个维度的"参考基线"，效果类似。

**公式 13-14：注意力评分 S_att(t) = S_base − Σδ_i,t − λΣη_j·ŷ_j(t)**

这是核心公式，完整拆解见下一章。

### 2.3 论文章节到代码函数的速查表

| 论文章节 | 核心概念 | 代码类 | 核心函数 | 行号 |
|---------|---------|-------|---------|-----|
| Section 2.1 | 病理-行为映射矩阵 | OptimizedAttentionScoringSystem | `__init__()` 权重配置 | 3391 |
| Section 2.1 | 动态权重 ω_i | OptimizedAttentionScoringSystem | `update_adaptive_params()` | 3919 |
| Section 2.1 | 行为特征向量 B | AttentionAnalyzer | `current_state` 字典 | 114 |
| Section 2.2 | DAF-Xception 网络 | EmotionAnalyzer | `predict_emotion()` | 518 |
| Section 2.2 | 7 类情绪分类 | EmotionAnalyzer | `emotion_labels` | 433 |
| Section 2.3 | EAR 眼睛纵横比 | AttentionAnalyzer | `eye_aspect_ratio()` | 134 |
| Section 2.3 | SolvePnP 头部姿态 | AttentionAnalyzer | `head_pose()` | 149 |
| Section 2.3 | 虹膜追踪与视线 | AttentionAnalyzer | `iris_center()` + `gaze_vector()` | 190-217 |
| Section 2.3.1 | 眨眼异常检测 | OptimizedAttentionScoringSystem | `calculate_blink_score_optimized()` | 3735 |
| Section 2.3.2 | 偏离强度 δ（公式 13） | OptimizedAttentionScoringSystem | 6 个子评分函数 | 3587-3819 |
| Section 2.3.2 | 综合评分（公式 14） | OptimizedAttentionScoringSystem | `calculate_attention_score()` | 3484 |
| Section 2.4 | 个性化反馈 | VoiceReminderSystem | `speak()` | 674 |
| Section 3 | CK+ 数据集实验 | EmotionAnalyzer | `video.h5` 模型 | 456 |

---

## 三、注意力评分完整拆解——从最小零件到最终分数

### 3.0 总入口

函数：`calculate_attention_score(attention_state, emotion_state)`（第 3484 行）

每一帧画面都会调用一次这个函数。它接收两个输入：
- `attention_state`：来自 AttentionAnalyzer，包含 ear_left, ear_right, yaw, pitch, gaze_x, gaze_y, attention_label, face_detected
- `emotion_state`：来自 EmotionAnalyzer，包含 emotion（字符串）, confidence（0~1 浮点数）

### 3.1 第一层：6 个子评分函数（对应公式 13 的 δ 偏离项）

#### ① 眼睛睁开度评分（满分 25 分）

函数：`calculate_eye_score_optimized(ear_left, ear_right, attention_label)`（第 3587 行）

**前置判断：** 如果 attention_label == "眼睛闭合" → 直接返回 0 分

**基础分（0~20 分）：** 取左右眼 EAR 均值

| EAR 均值 | 基础分 | 阈值参数 |
|---------|-------|---------|
| ≥ 0.22 | 20 分 | ear_optimal |
| ≥ 0.20 | 16 分 | ear_good_threshold |
| ≥ 0.18 | 12 分 | ear_fair_threshold |
| ≥ 0.16 | 6 分 | ear_bad_threshold |
| < 0.16 | 0 分 | — |

**左右眼不对称惩罚（0~-5 分）：**
- 触发条件：|EAR左 - EAR右| > 0.05（ear_asymmetry_threshold）
- 惩罚量：min(5, 不对称度 × 20)
- 含义：可能表示斜视或单侧疲劳

**EAR 稳定性奖励（0~+3 分）：**
- 条件：最近 30 帧的 EAR 标准差
- std < 0.02 → +3 分（非常稳定）
- std < 0.04 → +2 分（稳定）

**最终范围：max(0, 基础分 - 不对称惩罚 + 稳定性奖励) → 0~25 分**

#### ② 视线稳定性评分（满分 20 分）

函数：`calculate_gaze_score_optimized(gaze_x, gaze_y, attention_label)`（第 3623 行）

**前置判断：** 如果 attention_label ∈ {"视线偏离", "视线偏移"} → 直接返回 0 分

**基础分（0~15 分）：** 取视线偏移幅度 = √(gaze_x² + gaze_y²)

| 偏移幅度 | 基础分 | 阈值参数 |
|---------|-------|---------|
| ≤ 0.15 | 15 分 | gaze_optimal |
| ≤ 0.25 | 12 分 | gaze_good_threshold |
| ≤ 0.35 | 9 分 | gaze_fair_threshold |
| ≤ 0.50 | 5 分 | gaze_bad_threshold |
| > 0.50 | 0 分 | — |

**快速扫视惩罚（0~-5 分）：**
- 触发条件：当前帧与上一帧的视线幅度差 > 0.8（gaze_speed_threshold）
- 惩罚量：min(5, 速度差 × 3)
- 含义：快速扫视是注意力分散的信号

**视线长期稳定奖励（0~+3 分）：**
- 条件：最近 60 帧（约 2 秒）的视线标准差
- std < 0.1 → +3 分
- std < 0.2 → +2 分

**最终范围：max(0, 基础分 - 扫视惩罚 + 稳定奖励) → 0~20 分**

#### ③ 头部稳定性评分（满分 15 分）

函数：`calculate_head_score_optimized(yaw, pitch, attention_label)`（第 3665 行）

**前置判断：** 如果 attention_label == "视线偏离" → 直接返回 0 分

**基础分（0~12 分）：** 取头部偏移合成值 = √(yaw² + pitch²)

| 偏移角度 | 基础分 | 阈值参数 |
|---------|-------|---------|
| ≤ 8.0° | 12 分 | head_optimal |
| ≤ 15.0° | 10 分 | head_good_threshold |
| ≤ 25.0° | 8 分 | head_fair_threshold |
| ≤ 35.0° | 4 分 | head_bad_threshold |
| > 35.0° | 0 分 | — |

**快速转头惩罚（0~-4 分）：**
- 触发条件：帧间头部移动 > 10.0°（head_speed_threshold）
- 惩罚量：min(4, 移动距离 × 2)
- 含义：ADHD 典型的突然转头行为

**头部长期稳定奖励（0~+2 分）：**
- 条件：最近 60 帧的头部偏移标准差
- std < 5.0° → +2 分
- std < 10.0° → +1 分

**最终范围：max(0, 基础分 - 转头惩罚 + 稳定奖励) → 0~15 分**

#### ④ 持续专注时长评分（满分 20 分）⚠️ ADHD 核心

函数：`calculate_duration_score_optimized(attention_label, current_time)`（第 3718 行）

无子项拆分，直接按专注时长阶梯给分：

| 持续专注秒数 | 得分 |
|------------|-----|
| ≥ 10.0 秒 | 18 分 |
| ≥ 5.0 秒 | 14 分 |
| ≥ 2.0 秒 | 9 分 |
| < 2.0 秒 | 4 分 |
| 无数据（刚启动） | 5 分 |

**关键说明：** 专注状态的起止由 `update_focus_state_optimized()` 管理。进入专注的条件是 attention_label == "专注" **且** 当前总分 ≥ 65；一旦不满足，计时器归零、中断次数 +1。

#### ⑤ 眨眼模式评分（满分 10 分）

函数：`calculate_blink_score_optimized(ear_left, ear_right, current_time)`（第 3735 行）

**不足 30 帧历史数据时 → 返回 5 分（基础分）**

**眨眼检测机制：**
- 当 EAR左 < 0.16 且 EAR右 < 0.16 → 记录一次眨眼时间戳，累加当前眨眼簇计数
- 当 EAR 恢复 > 0.16 且距上次眨眼 > 0.5 秒 → 结束当前眨眼簇
- 如果一个眨眼簇内的连续眨眼次数 ≥ 5 → 记入 blink_clusters 列表

**基础分（3~8 分）：** 取最近 10 秒内的眨眼频率（换算为次/分钟）

| 眨眼频率（次/分钟） | 基础分 | 含义 |
|-------------------|-------|------|
| 10~30 | 8 分 | 正常范围 |
| 5~10 或 30~40 | 6 分 | 略偏 |
| > 40 | 3 分 | 过快（焦虑/疲劳） |
| < 5 | 4 分 | 过慢（过度专注/疲劳） |

**眨眼簇惩罚（0~-2 分）：**
- 触发条件：最近一次眨眼簇的连续眨眼数 ≥ 3
- 惩罚：-2 分

**最终范围：max(0, 基础分 - 眨眼簇惩罚) → 0~10 分**

#### ⑥ 运动躁动度评分（满分 10 分）⚠️ ADHD 特有

函数：`calculate_motor_score(yaw, pitch, current_time)`（第 3781 行）

**无上一帧数据时 → 返回 5 分（基础分）**

**微动检测：** 帧间头部移动幅度 = √((yaw差)² + (pitch差)²)
- 仅在 0.5°~5.0° 范围内计为一次"微动"（太小是噪声，太大是主动转头）

**基础分（4~8 分）：** 取最近 5 秒内的微动频率

| 微动频率（次/秒） | 基础分 | 含义 |
|-----------------|-------|------|
| ≤ 1.5 | 8 分 | 安静 |
| 1.5~3.0 | 6 分 | 中度躁动 |
| > 3.0 | 4 分 | 明显躁动（累加 hyperactivity_count） |

**最终范围：max(0, 基础分) → 0~10 分**

### 3.2 第二层：基础总分

```
base_score = eye_score + gaze_score + head_score + duration_score + blink_score + motor_score
```

| 维度 | 范围 | 满分 |
|-----|------|-----|
| eye_score | 0~25 | 25 |
| gaze_score | 0~20 | 20 |
| head_score | 0~15 | 15 |
| duration_score | 0~20 | 20 |
| blink_score | 0~10 | 10 |
| motor_score | 0~10 | 10 |
| **合计** | **0~100** | **100** |

**与论文的对应：** base_score 等价于 S_base − Σδ_i,t。论文是"从 100 开始扣分"，代码是"从 0 开始挣分"，数学等价。

### 3.3 第三层：情绪调整项

函数：`calculate_emotion_adjustment_optimized(emotion_state, attention_label)`（第 3821 行）

对应论文公式 14 的第三项：−λΣη_j·ŷ_j(t)

**计算方式：** 调整分 = η_j × ŷ_j(t)

- η_j = 情绪系数（下表）
- ŷ_j(t) = 该情绪的模型输出置信度（0~1）

| 情绪 | η_j（系数） | 置信度 0.9 时的实际调整 |
|-----|-----------|---------------------|
| 快乐 | +6 | +5.4 |
| 惊讶 | +3 | +2.7 |
| 中性 | 0 | 0 |
| 厌恶 | -4 | -3.6 |
| 悲伤 | -6 | -5.4 |
| 恐惧 | -8 | -7.2 |
| 生气 | -10 | -9.0 |

**额外奖励：** 如果 emotion == "快乐" 且 attention_label == "专注" → 再 +2 分

**简化说明：** 论文公式是对 7 类情绪求加权和（Σ），代码只取了 argmax 后的单一最强情绪。

**实际范围：约 -10 ~ +8 分**

### 3.4 第四层：ADHD 行为模式调整

函数：`detect_adhd_features(attention_state, emotion_state, current_time)`（第 3851 行）

这是论文中未单独列公式、代码额外增加的惩罚项。

| 检测项 | 条件 | 惩罚 |
|-------|------|-----|
| 连续分心 | 最近 3 帧全部非"专注" | -5 分 |
| 连续分心（轻） | 最近 3 帧中 2 帧非"专注" | -3 分 |
| 情绪波动 | 30 秒内 ≥3 次生气/恐惧 | -4 分 |
| 注意力频繁转移 | 20 帧内"非专注→专注"切换 ≥5 次 | -3 分 |

**最大累计惩罚：-12 分**

### 3.5 第五层：面部检测状态调整

```python
if not face_detected:
    base_score = max(0, base_score * 0.6)  # 未检测到面部时，基础分乘以 0.6
```

当摄像头没有检测到人脸时（比如学生离开了座位），基础分直接打六折。

### 3.6 第六层：非线性缩放

函数：`apply_nonlinear_scaling(score)`（第 3904 行）

对总分做最后一次变换，目的是让关键分数区间的差异更明显：

| 输入分数区间 | 变换规则 | 效果 |
|------------|---------|------|
| ≥ 80 | 80 + (score-80) × 0.8 | 高分段压缩，不容易满分 |
| 60~80 | 不变 | 中间段保持线性 |
| 40~60 | 40 + (score-40) × 1.2 | 低分段放大差异 |
| < 40 | score × 1.5 | 极低分段进一步放大 |

示例：输入 85 → 输出 84；输入 50 → 输出 52；输入 30 → 输出 45

### 3.7 第七层：最终输出

```python
total_score = base_score + emotion_adjustment + adhd_adjustment
total_score = apply_nonlinear_scaling(total_score)
final_score = max(0, min(100, total_score))
```

### 3.8 完整计算流程图

```
摄像头一帧画面
│
├─→ AttentionAnalyzer.analyze_frame()
│     输出: ear_left, ear_right, yaw, pitch, gaze_x, gaze_y, attention_label
│
├─→ EmotionAnalyzer.analyze_frame()
│     输出: emotion("快乐"等), confidence(0~1)
│
└─→ OptimizedAttentionScoringSystem.calculate_attention_score()
      │
      ├── ① eye_score    = f(EAR均值, 左右对称, 30帧稳定性)      → 0~25分
      ├── ② gaze_score   = f(视线偏移, 扫视速度, 60帧稳定性)     → 0~20分
      ├── ③ head_score   = f(头部偏转, 转头速度, 60帧稳定性)     → 0~15分
      ├── ④ duration_score = f(持续专注秒数)                     → 0~20分
      ├── ⑤ blink_score  = f(眨眼频率, 眨眼簇)                  → 0~10分
      ├── ⑥ motor_score  = f(5秒内微动频率)                     → 0~10分
      │
      ├── base_score = ①+②+③+④+⑤+⑥                           → 0~100分
      │
      ├── if 未检测到面部: base_score × 0.6
      │
      ├── ⑦ emotion_adjustment = η_j × confidence               → -10~+8分
      ├── ⑧ adhd_adjustment   = 三种ADHD模式惩罚                → -12~0分
      │
      ├── total = base_score + ⑦ + ⑧
      ├── total = 非线性缩放(total)
      └── final = clip(total, 0, 100)                           → 最终输出
```

---

## 四、你需要掌握的其他关键信息

### 4.1 三个模型文件

| 文件 | 大小 | 用途 | 由谁调用 | 技术细节 |
|-----|------|------|---------|---------|
| `Models/EmotionXCeption/video.h5` | 16MB | 情绪识别（改进的 Xception 网络） | EmotionAnalyzer 第 456 行 | 输入 48×48 灰度图 → 输出 7 维概率向量 |
| `Models/FaceDetect/haarcascade_frontalface_default.xml` | 0.9MB | OpenCV 人脸检测级联分类器 | 通过 dlib 间接使用 | 定位画面中的人脸区域 |
| `Models/Landmarks/face_landmarks.dat` | 99MB | dlib 的 68 点面部特征点定位 | EmotionAnalyzer 第 465 行 | 为情绪分析定位面部各器官 |

重要区分：**注意力检测用 MediaPipe（478 点），情绪识别用 dlib（68 点）+ Xception 深度学习模型**。两条管道独立运行，最后在评分层融合。

### 4.2 关键阈值速查

| 参数 | 值 | 含义 | 所在类 |
|-----|---|------|-------|
| ear_thresh | 0.21 | EAR 低于此值 → 判定闭眼 | AttentionConfig |
| yaw_thresh_deg | 20.0° | yaw 超过此值 → 判定转头 | AttentionConfig |
| pitch_thresh_deg | 20.0° | pitch 超过此值 → 判定抬头/低头 | AttentionConfig |
| gaze_off_center | 0.35 | 视线偏移超过此值 → 判定视线偏移 | AttentionConfig |
| reminder_cooldown | 15 秒 | 两次语音提醒的最小间隔 | VoiceReminderSystem |
| learning_rate | 0.01 | 自适应参数的学习率 | OptimizedAttentionScoringSystem |
| 专注判定阈值 | 分数 ≥ 65 | 低于此分数即使标签是"专注"也不计入专注时长 | update_focus_state_optimized |

### 4.3 代码中的简化与论文的差异

了解这些差异可以帮你在被追问时不会露馅：

1. **fMRI 映射矩阵：** 论文描述的是从 rs-fMRI 数据动态计算映射矩阵 R；代码中将论文分析的结论直接硬编码为固定权重。这是合理的工程简化——实时系统不可能接入脑影像数据。

2. **情绪求和 vs 取最大值：** 论文公式是对 7 类情绪的加权求和（Ση_j·ŷ_j）；代码只取 argmax 后的单一最强情绪计算调整值。这是因为 EmotionAnalyzer 只返回最高概率的那个情绪类别。

3. **非线性缩放：** 这是代码额外增加的后处理步骤，论文中没有描述。目的是让临界区域（40-60 分）的分数差异更明显。

4. **ADHD 行为模式检测：** detect_adhd_features() 中的三种模式检测是代码额外增加的，论文中有理论描述但没有单独的公式。

5. **动态权重的实现差异：** 论文的 ω_i 是调整评分权重本身；代码的 adaptive_params 是调整基线参考值。思路不同但目标一致——都是让系统适应个体差异。

### 4.4 语音提醒系统的技术细节

- 引擎：pyttsx3（离线 TTS，不需要联网）
- 运行方式：独立线程 + 消息队列（voice_queue），不阻塞主界面
- 语速：150 字/分钟（适合儿童）
- 音量：0.8
- 冷却时间：15 秒（避免反复提醒造成烦躁）
- 容错：pyttsx3 未安装时系统仍可正常运行，只是没有语音提醒

### 4.5 校准系统

CalibrationSystem（第 4346 行）提供 5 点校准流程：

1. 用户看屏幕中央 → 记录中性视线基线
2. 看左上、右上、左下、右下 → 记录视线范围
3. 每个点采集 60 帧数据，取后 30 帧平均
4. 校准结果保存到 calibration_data.json
5. 下次启动可以加载上次的校准数据

校准的目的是适应不同用户的面部特征差异（眼睛大小、瞳距、坐姿习惯等）。
