from base_detector.v5_detector import yolo_detector
import json
import os
import cv2 as cv
from tqdm import tqdm

import os
import xml.etree.ElementTree as ET
import pathlib
import platform

# ===== NumPy 1.x 兼容 NumPy 2.x 保存的模型 =====
import sys
import numpy as np

if not hasattr(np, '_core'):
    import numpy.core as _core
    sys.modules['numpy._core'] = _core
    sys.modules['numpy._core.multiarray'] = _core.multiarray
    sys.modules['numpy._core._multiarray_umath'] = _core._multiarray_umath
    sys.modules['numpy._core.numeric'] = _core.numeric
    sys.modules['numpy._core.umath'] = _core.umath
    sys.modules['numpy._core._exceptions'] = _core._exceptions
    sys.modules['numpy._core.shape_base'] = _core.shape_base
# ==============================================

if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath

def save_voc_xml(
    img_path,
    img_shape,
    img_info,
    save_dir,
    class_name,
    keep_classes=None  # 🆕 新增：只保留这些类别，None表示保存全部
):
    """
    img_path  : 原图路径
    img_shape : img.shape
    img_info  : detect 返回的结果
    save_dir  : xml 保存目录
    class_name: ['kengcao', ...]
    keep_classes: 🆕 只保留的类别列表，例如 ['car', 'person']，None表示保存全部
    """

    if img_info is None:
        return

    bboxs = img_info.get("bboxs", [])
    labels = img_info.get("labels", [])
    confs = img_info.get("confs", [])

    if len(bboxs) == 0:
        return

    # 🆕 如果指定了 keep_classes，只保留这些类别
    if keep_classes is not None:
        keep_set = set(keep_classes)
        filtered_bboxs = []
        filtered_labels = []
        filtered_confs = []
        for i in range(len(bboxs)):
            cls_id = int(labels[i])
            if cls_id < len(class_name) and class_name[cls_id] in keep_set:
                filtered_bboxs.append(bboxs[i])
                filtered_labels.append(labels[i])
                filtered_confs.append(confs[i])
        bboxs = filtered_bboxs
        labels = filtered_labels
        confs = filtered_confs

    if len(bboxs) == 0:
        return

    os.makedirs(save_dir, exist_ok=True)

    h, w, c = img_shape
    filename = os.path.basename(img_path)
    xml_name = os.path.splitext(filename)[0] + ".xml"
    xml_path = os.path.join(save_dir, xml_name)

    # -------- VOC 结构 --------
    annotation = ET.Element("annotation")

    folder = ET.SubElement(annotation, "folder")
    folder.text = os.path.basename(save_dir)

    fname = ET.SubElement(annotation, "filename")
    fname.text = filename

    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(w)
    ET.SubElement(size, "height").text = str(h)
    ET.SubElement(size, "depth").text = str(c)

    ET.SubElement(annotation, "segmented").text = "0"

    for i in range(len(bboxs)):
        cls_id = int(labels[i])
        if cls_id >= len(class_name):
            continue

        x1, y1, x2, y2 = map(int, bboxs[i])

        obj = ET.SubElement(annotation, "object")
        ET.SubElement(obj, "name").text = class_name[cls_id]
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"

        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(x1)
        ET.SubElement(bndbox, "ymin").text = str(y1)
        ET.SubElement(bndbox, "xmax").text = str(x2)
        ET.SubElement(bndbox, "ymax").text = str(y2)

    tree = ET.ElementTree(annotation)
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)



def draw_result(frame, img_info):
    class_name = [
        'car',
        'person',
        # 'zhuitong',
        # 'daodizhuitong',
        # 'miehuoqi',
        # 'yangqiping',
        # 'yuantongzhuitong'
    ]

    CLASS_COLORS = {
        'car':       (0, 0, 255),     # 红
        'person':  (255, 0, 0),     # 蓝
        'zhuitong':       (0, 255, 255),   # 黄
        'daodizhuitong':     (0, 165, 255),   # 橙
        'miehuoqi':        (0, 255, 0),     # 绿
        'yangqiping': (0, 255, 0),  # 绿
        'yuantongzhuitong': (0, 255, 0),  # 绿
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
    # 🆕 配置：只保留这些类别（None 表示保存全部）
    KEEP_CLASSES = [
        'helmet',
        'nohelmet',
        'reflective',
        'noreflectivez'
    ]  # 改成你想要保留的类别列表

    model_config = json.load(open("weights/model_config.json", "r", encoding="utf-8"))
    weights = model_config["vest_helmet"]["weights"]
    conf_thres = model_config["vest_helmet"]["conf_thres"]
    iou_thres = model_config["vest_helmet"]["iou_thres"]
    img_size = model_config["vest_helmet"]["img_size"]
    device = 0

    safety_detect = yolo_detector(
        weights=weights,
        device=device,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        img_size=img_size
    )

    image_dir = r"path/to/images/"

    xml_save_dir = r"path/to/xml/"

    # class_name = [
    #     'heiping',
    #     'hulansunhuai',
    #     'paosawu',
    #     'smokeFire',
    #     'jishui',
    # ]

    # class_name = [
    #     'car',
    #     'person',
    #     'zhuitong',
    #     'zhuitong_daofu',
    #     'biaoshipai',
    #     'miehuoqi',
    #     'yangqiping',
    #     'yuanzhuitong',
    #     'jikeng',
    #     'weilan',
    #     'yangcheng',
    #     'jishui',
    #     'duiliao',
    # ]

    # class_name = [
    #     'car',
    #     'person',
    #     'zhuitong',
    #     'zhuitong_daofu',
    #     'yuanzhuitong'
    # ]
    # class_name = [
    #     'fire',
    #     'smoke'
    # ]

    # class_name = [
    #     'car',
    #     'person',
    #     'zhuitong',
    #     'daodizhuitong',
    #     'miehuoqi',
    #     'yangqiping',
    #     'yuantongzhuitong'
    # ]

    # class_name = [
    #     'car',
    #     'person',
    #     'zhuitong',
    #     'yuanzhuitong',
    #     'zhuitong_daofu',
    #     'shuima_daofu',
    #     'jishui',
    #     'yangcheng'
    # ]

    class_name = [
        'helmet',
        'nohelmet',
        'reflective',
        'noreflectivez'
    ]

    image_list = sorted(os.listdir(image_dir))

    for name in tqdm(image_list, desc="Processing Images"):
        img_path = os.path.join(image_dir, name)
        img = cv.imread(img_path)

        if img is None:
            print("读取失败:", img_path)
            continue

        img_info = safety_detect.detect(img)
        print(img_info)
        # draw_frame = draw_result(img, img_info)

        # cv.imshow("img", img)
        save_voc_xml(
            img_path=img_path,
            img_shape=img.shape,
            img_info=img_info,
            save_dir=xml_save_dir,
            class_name=class_name,
            keep_classes=KEEP_CLASSES  # 🆕 传入只保留的类别
        )

        # key = cv.waitKey(0)  # 按任意键看下一张
        #
        # if key == 27:  # ESC 键退出
        #     break

    cv.destroyAllWindows()