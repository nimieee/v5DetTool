import cv2 as cv
import os
import json
from base_detector.v5_detector import yolo_detector

def draw_result(frame, img_info):
    # ====== 只画这些类别 ======
    allowed_class_names = ['car', 'person']
    allowed_class_ids = [0, 1]  # 对应的 class_id

    class_name = [
        'car',
        'person'
        # 'zhuitong',
        # 'chedaoxian',
        # 'paosawu',
        # 'yuanzhutong'
    ]

    CLASS_COLORS = {
        'car':       (0, 0, 255),     # 红
        'person':    (255, 0, 0),     # 蓝
        # 'zhuitong':       (0, 255, 255),   # 黄
        # 'chedaoxian':     (0, 165, 255),   # 橙
        # 'paosawu':        (0, 255, 0),     # 绿
        # 'yuanzhutong':    (0, 155, 0),     # 绿
    }

    if img_info is None:
        return frame

    bboxs = img_info.get("bboxs", [])
    labels = img_info.get("labels", [])
    confs = img_info.get("confs", [])

    for i in range(len(bboxs)):
        cls_id = int(labels[i])

        # ====== 只画允许的类别 ======
        if cls_id not in allowed_class_ids:
            continue

        if cls_id >= len(class_name):
            continue

        x1, y1, x2, y2 = map(int, bboxs[i])
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


if __name__ == '__main__':
    model_config = json.load(open("weights/model_config.json", "r", encoding="utf-8"))
    weights = model_config["shigong_det"]["weights"]
    conf_thres = model_config["shigong_det"]["conf_thres"]
    iou_thres = model_config["shigong_det"]["iou_thres"]
    img_size = model_config["shigong_det"]["img_size"]
    device = 0

    safety_detect = yolo_detector(
        weights=weights,
        device=device,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        img_size=img_size
    )

    # ---------- 视频路径 ----------
    video_path = r"path/to/input.mp4"
    save_video_path = r"path/to/output_det.mp4"

    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print("视频打开失败")
        exit(0)

    # ---------- 视频属性 ----------
    fps = cap.get(cv.CAP_PROP_FPS)
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    writer = cv.VideoWriter(save_video_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ---------- 检测 ----------
        img_info = safety_detect.detect(frame)

        # ---------- 画框 ----------
        draw_frame = draw_result(frame, img_info)

        # ---------- 写视频 ----------
        writer.write(draw_frame)

        # ---------- 显示（可选） ----------
        cv.imshow("xhf", draw_frame)
        if cv.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    writer.release()
    cv.destroyAllWindows()