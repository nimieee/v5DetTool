from base_detector.v5_detector import yolo_detector
import json
import os
import cv2 as cv
import pathlib
import platform

if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath


def draw_result(frame, img_info, class_name):
    """在帧上绘制检测结果"""
    CLASS_COLORS = {
        'fire': (0, 0, 255),  # 红色
        'smoke': (128, 128, 128),  # 灰色
    }

    if img_info is None:
        return frame

    bboxs = img_info.get("bboxs", [])
    labels = img_info.get("labels", [])
    confs = img_info.get("confs", [])

    for i in range(len(bboxs)):
        x1, y1, x2, y2 = map(int, bboxs[i])
        cls_id = int(labels[i])

        if cls_id >= len(class_name):
            continue

        name = class_name[cls_id]
        conf = confs[i]
        color = CLASS_COLORS.get(name, (255, 255, 255))

        # 画框
        cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # 显示文本
        label_text = f"{name} {conf:.2f}"
        (tw, th), _ = cv.getTextSize(label_text, cv.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv.putText(frame, label_text, (x1 + 2, y1 - 4), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return frame


# ==================== 在这里修改你的视频路径 ====================
video_path = r"path/to/input.mp4"  # 输入视频路径
output_path = r"path/to/output.mp4"  # 输出视频路径
# ============================================================

# 加载模型
model_config = json.load(open("weights/model_config.json", "r", encoding="utf-8"))
weights = model_config["firesmoke"]["weights"]
conf_thres = model_config["firesmoke"]["conf_thres"]
iou_thres = model_config["firesmoke"]["iou_thres"]
img_size = model_config["firesmoke"]["img_size"]
device = 0

safety_detect = yolo_detector(
    weights=weights,
    device=device,
    conf_thres=conf_thres,
    iou_thres=iou_thres,
    img_size=img_size
)

class_name = ['fire', 'smoke']

# 打开视频
cap = cv.VideoCapture(video_path)
fps = int(cap.get(cv.CAP_PROP_FPS))
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

# 创建视频写入器
fourcc = cv.VideoWriter_fourcc(*'mp4v')
out = cv.VideoWriter(output_path, fourcc, fps, (width, height))

# 逐帧处理
while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_info = safety_detect.detect(frame)
    result_frame = draw_result(frame, img_info, class_name)
    out.write(result_frame)

# 释放资源
cap.release()
out.release()
print(f"处理完成，保存至: {output_path}")