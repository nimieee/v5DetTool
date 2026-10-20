import cv2 as cv
import os
import json
from base_detector.v5_detector import yolo_detector


def draw_result(frame, img_info):
    class_name = [
        'heiping',
        'hulansunhuai',
        'paosawu',
        'smokeFire',
        'jishui',
        'kengcao'
    ]

    CLASS_COLORS = {
        'heiping':       (0, 0, 255),     # 红
        'hulansunhuai':  (255, 0, 0),     # 蓝
        'paosawu':       (0, 255, 255),   # 黄
        'smokeFire':     (0, 165, 255),   # 橙
        'jishui':        (0, 255, 0),     # 绿
        'kengcao': (0, 155, 0),  # 绿
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

        color = CLASS_COLORS.get(name, (255, 255, 255))  # 默认白色

        # 画框
        cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # 显示文本
        label_text = f"{name} {conf:.2f}"

        (tw, th), _ = cv.getTextSize(
            label_text,
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            2
        )

        cv.rectangle(
            frame,
            (x1, y1 - th - 6),
            (x1 + tw + 4, y1),
            color,
            -1
        )

        cv.putText(
            frame,
            label_text,
            (x1 + 2, y1 - 4),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

    return frame

if __name__ == '__main__':
    model_config = json.load(open("weights/model_config.json", "r", encoding="utf-8"))
    weights = model_config["xhf"]["weights"]
    conf_thres = model_config["xhf"]["conf_thres"]
    iou_thres = model_config["xhf"]["iou_thres"]
    img_size = model_config["xhf"]["img_size"]
    device = 0

    safety_detect = yolo_detector(
        weights=weights,
        device=device,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        img_size=img_size
    )

    # ---------- 视频路径 ----------
    video_path = r"path/to/input.mp4"  # 输入视频路径
    save_video_path = r"path/to/output.mp4"  # 输出视频路径

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

        # # ---------- 显示（可选） ----------
        # cv.imshow("xhf", draw_frame)
        # if cv.waitKey(1) & 0xFF == 27:
        #     break

    cap.release()
    writer.release()
    cv.destroyAllWindows()
