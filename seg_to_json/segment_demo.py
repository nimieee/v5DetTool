#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分割模型自动标注脚本 —— 将分割结果保存为 LabelMe JSON 格式
遍历图片目录，对每张图运行分割模型，将分割结果保存为 labelme 兼容的 json 文件。

用法：
    1. 修改 main 中的配置（图片目录、模型配置、类别名称等）
    2. python segment_demo.py

依赖：
    yolo_seg_detector（base_detector/seg_detector.py）
"""

import json
import os
import cv2 as cv
import numpy as np
from tqdm import tqdm
import base64
import io
from PIL import Image

from base_detector.seg_detector import yolo_seg_detector, mask_to_polygons


def save_labelme_json(
    img_path,
    img_shape,
    seg_info,
    save_dir,
    class_name,
    save_image=False
):
    """
    将分割结果保存为 LabelMe JSON 格式。

    参数:
        img_path   : 原始图片路径
        img_shape  : (h, w, c)
        seg_info   : detect() 返回的结果字典
        save_dir   : json 保存目录
        class_name : 类别名称列表 ['cls0', 'cls1', ...]
        save_image : 是否将图片也复制到 save_dir 下（默认 False）
    """
    if seg_info is None:
        return

    polygons = seg_info.get("polygons", [])
    labels = seg_info.get("labels", [])

    if len(polygons) == 0:
        return

    os.makedirs(save_dir, exist_ok=True)

    h, w, c = img_shape
    filename = os.path.basename(img_path)
    base_name = os.path.splitext(filename)[0]
    json_name = base_name + ".json"
    json_path = os.path.join(save_dir, json_name)

    # ----- 读取图片并转 base64 -----
    with open(img_path, "rb") as f:
        image_data = f.read()
    image_base64 = base64.b64encode(image_data).decode("utf-8")

    # ----- 构建 shapes 列表 -----
    shapes = []
    for i in range(len(polygons)):
        cls_id = int(labels[i])
        if cls_id >= len(class_name):
            label_name = "unknown"
        else:
            label_name = class_name[cls_id]

        pts = polygons[i]  # [[x1,y1], [x2,y2], ...]
        if len(pts) < 3:
            continue

        shape = {
            "label": label_name,
            "points": pts,
            "group_id": None,
            "shape_type": "polygon",
            "flags": {}
        }
        shapes.append(shape)

    if not shapes:
        return

    # ----- 构建 LabelMe JSON -----
    labelme_json = {
        "version": "5.0.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": filename,
        "imageData": image_base64,
        "imageHeight": h,
        "imageWidth": w
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(labelme_json, f, ensure_ascii=False, indent=2)


def draw_result(frame, seg_info, class_name, class_colors=None):
    """
    在图像上绘制分割结果（多边形轮廓 + 标签）。
    可选：显示在原图上并等待按键。
    """
    if seg_info is None:
        return frame

    polygons = seg_info.get("polygons", [])
    labels = seg_info.get("labels", [])
    confs = seg_info.get("confs", [])

    default_colors = [
        (0, 0, 255),     # 红
        (255, 0, 0),     # 蓝
        (0, 255, 255),   # 黄
        (0, 165, 255),   # 橙
        (0, 255, 0),     # 绿
        (255, 0, 255),   # 紫
        (255, 255, 0),   # 青
        (128, 0, 128),   # 深紫
    ]
    if class_colors is None:
        class_colors = {}

    result = frame.copy()

    for i in range(len(polygons)):
        pts = polygons[i]
        if len(pts) < 3:
            continue

        cls_id = int(labels[i])
        name = class_name[cls_id] if cls_id < len(class_name) else "unknown"
        conf = confs[i] if i < len(confs) else 0.0

        # 获取颜色
        color = class_colors.get(name, default_colors[cls_id % len(default_colors)])

        # 画多边形
        pts_np = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
        cv.polylines(result, [pts_np], isClosed=True, color=color, thickness=2)
        cv.fillPoly(result, [pts_np], color=(color[0], color[1], color[2], 64))

        # 标签文字
        label_text = f"{name} {conf:.2f}"
        x1 = int(min(p[0] for p in pts))
        y1 = int(min(p[1] for p in pts))

        (tw, th), _ = cv.getTextSize(label_text, cv.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv.rectangle(result, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv.putText(result, label_text, (x1 + 2, y1 - 4),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return result


if __name__ == '__main__':
    # ========== 配置区 ==========
    # 1. 模型配置
    model_config = json.load(open("weights/model_config.json", "r", encoding="utf-8"))
    weights = model_config["seg"]["weights"]
    conf_thres = model_config["seg"]["conf_thres"]
    iou_thres = model_config["seg"]["iou_thres"]
    img_size = model_config["seg"]["img_size"]
    device = 0  # 0 表示 cuda:0，'cpu' 表示 CPU

    # 2. 创建分割检测器
    seg_detect = yolo_seg_detector(
        weights=weights,
        device=device,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        img_size=img_size
    )

    # 3. 图片目录（修改为你的测试图片路径）
    image_dir = r"path/to/images/"

    # 4. JSON 保存目录
    json_save_dir = r"path/to/labelme_jsons/"

    # 5. 类别名称（必须与模型训练时的类别顺序一致）
    class_name = [
        'fire',
        'smoke'
    ]

    # 6. 类别颜色（用于可视化，可选）
    CLASS_COLORS = {
        'fire':  (0, 0, 255),     # 红
        'smoke': (255, 0, 0),     # 蓝
    }

    # 7. 是否显示可视化画面（按 ESC 退出）
    show_vis = False

    # ========== 开始遍历 ==========
    image_list = sorted(os.listdir(image_dir))

    for name in tqdm(image_list, desc="Processing Images"):
        img_path = os.path.join(image_dir, name)
        img = cv.imread(img_path)

        if img is None:
            print(f"读取失败: {img_path}")
            continue

        # --- 分割推理 ---
        seg_info = seg_detect.detect(img)

        # --- 保存为 LabelMe JSON ---
        save_labelme_json(
            img_path=img_path,
            img_shape=img.shape,
            seg_info=seg_info,
            save_dir=json_save_dir,
            class_name=class_name,
            save_image=False
        )

        # --- 可视化（可选）---
        if show_vis:
            vis = draw_result(img, seg_info, class_name, CLASS_COLORS)
            cv.imshow("seg_result", vis)
            key = cv.waitKey(0)
            if key == 27:  # ESC
                break

    cv.destroyAllWindows()
    print("全部处理完成！")
