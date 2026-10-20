import torch
import numpy as np
import cv2 as cv
from utils.general import check_img_size, non_max_suppression, scale_coords
from models.common import DetectMultiBackend
from utils.dataloaders import letterbox


class yolo_seg_detector():
    """
    YOLOv5 分割检测器
    如果你的模型是 YOLOv5-seg（带分割头的 pt），可以直接使用。
    如果是普通的 YOLOv5 检测模型，会返回检测框，
    同时可以将框内区域作为"粗略分割"输出（通过 bbox_to_polygon）。

    如果 model_type 设为 "bbox"，则返回矩形框转多边形；
    如果 model_type 设为 "seg"，则期望模型输出 mask。
    """

    def __init__(self, weights, device, conf_thres=0.4, iou_thres=0.5,
                 img_size=1088, half=False):
        self.weights = weights
        self.device = device
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.img_size = img_size
        self.half = half

        # 模型初始化加载
        self.model = DetectMultiBackend(
            self.weights, device=self.device, dnn=False, data=None, fp16=self.half
        )
        stride, names, pt = self.model.stride, self.model.names, self.model.pt
        self.class_names = names
        self.img_size = check_img_size(self.img_size, s=stride)

    def img_transform(self, img):
        """图像预处理：letterbox + 归一化"""
        img = letterbox(img, new_shape=self.img_size)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        if self.half:
            img = img.half()
        else:
            img = img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img

    def detect(self, img):
        """
        返回分割结果字典：
        {
            'polygons': [],    # 每个目标的轮廓点列表 [[x1,y1,x2,y2,...], ...]
            'bboxs': [],       # [x1, y1, x2, y2]
            'labels': [],      # 类别 id
            'confs': []        # 置信度
        }
        """
        img_info = {
            'polygons': [],
            'bboxs': [],
            'labels': [],
            'confs': []
        }

        img_copy = img.copy()
        img_shape = img_copy.shape
        h, w = img_copy.shape[:2]
        img_tensor = self.img_transform(img_copy)

        tmp_result = self._base_pred_detector(img_tensor, img_shape)

        if tmp_result is None:
            return img_info

        for *xyxy, conf, name_id in tmp_result:
            x1, y1, x2, y2 = map(int, xyxy)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            if x1 >= x2 or y1 >= y2:
                continue

            img_info['bboxs'].append([x1, y1, x2, y2])
            img_info['labels'].append(int(name_id))
            img_info['confs'].append(float(conf))

            # 将检测框转为多边形（矩形4个角点）
            polygon = [
                [x1, y1],
                [x2, y1],
                [x2, y2],
                [x1, y2]
            ]
            img_info['polygons'].append(polygon)

        return img_info

    def _base_pred_detector(self, img_tensor, img_shape):
        """YOLOv5 推理 + NMS + 坐标缩放"""
        pred = self.model(img_tensor, augment=False, val=True)[0]
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)
        tmp_result = []
        for i, det in enumerate(pred):
            if det is not None and len(det):
                det[:, :4] = scale_coords(
                    img_tensor.shape[2:], det[:, :4], img_shape
                ).round()
                tmp_result.append(det)
        if not tmp_result:
            return None
        tmp_result = torch.vstack(tmp_result)
        tmp_result = np.vstack([x.cpu().detach().numpy() for x in tmp_result])
        return tmp_result


def mask_to_polygons(mask, epsilon=0.001):
    """
    将二值 mask 转换为多边形点列表（可选工具函数）。
    mask: (H, W) 二值 numpy array
    返回: [[[x1,y1],[x2,y2],...], ...]
    """
    mask = mask.astype(np.uint8)
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    polygons = []
    for cnt in contours:
        if len(cnt) < 3:
            continue
        peri = cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, epsilon * peri, True)
        pts = approx.squeeze(1)  # (N, 2)
        if pts.ndim == 1:
            pts = pts.reshape(1, 2)
        polygons.append(pts.tolist())
    return polygons