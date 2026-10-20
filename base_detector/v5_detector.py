import torch
from utils.general import check_img_size, non_max_suppression, scale_coords
from models.common import DetectMultiBackend
from utils.dataloaders import letterbox
import numpy as np

import warnings

warnings.filterwarnings("ignore")



class yolo_detector():

    def __init__(self, weights, device, conf_thres=0.4, iou_thres=0.5, img_size=1088, half=False):
        self.weights = weights
        self.device = device
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.img_size = img_size
        self.half = half  # use FP16 half-precision inference
        # 模型初始化加载
        self.model = DetectMultiBackend(self.weights, device=self.device, dnn=False, data=None,
                                        fp16=self.half)
        # 获取模型的采样倍数，模型检测类别，模型是否为pt文件
        stride, names, pt = self.model.stride, self.model.names, self.model.pt
        self.img_size = check_img_size(self.img_size, s=stride)  # 查看模型输入大小是否为采样倍数的整数倍
        self.boundary_margin = 10  # 边界框像素调整，靠近边界框10个像素的话，认为框是不需要的

    # 输如图像预处理
    def img_transform(self, img):

        '''
             等比例缩放，以长边为主，长边长度够了后，短边缩放后需要进行填充，填充的规则是将短边缺少的像素填充至距离等长差后的32整数倍

             假设，图片为[2648,4864,3],则等比例缩放后为[720,1280,3],此时已经有一遍满足1280了，但是另一边不满足，即差560
                但是560不是32的整数倍，因此，需要对短边进行填充
                    填充的数值是，560对32取模，即填充16，即短边的总共补充16，则最终图片经过这个函数后得到的shape是[736,1280,3]

             总结 ： 缩放填充后，边必须变为32的整数倍
        '''
        img = letterbox(img, new_shape=self.img_size)[0]  # 得到的img是[h,w,c]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        # 将数组转化成张量，共享内存
        img = torch.from_numpy(img).to(self.device)

        if self.half:
            img = img.half()
        else:
            img = img.float()
        img /= 255.
        if img.ndimension() == 3:
            # 3维度 在0位置插入维度
            img = img.unsqueeze(0)

        return img

    def detect(self, img):
        img_info = {}
        img_info['bboxs'] = []
        img_info['labels'] = []
        img_info['confs'] = []

        img = img.copy()
        img_shape = img.shape
        h, w, _ = img_shape
        img = self.img_transform(img)
        tmp_result = self._base_pred_detector(img, img_shape)

        # boxes, scores = tmp_result[:, :4], tmp_result[:, 4]
        # visualize the box
        if tmp_result is None:
            return img_info
        for *xyxy, conf, name_id in tmp_result:
            xmin, ymin, xmax, ymax = xyxy
        #
        #     # # 边界框像素调整，靠近边界框10个像素的话，认为框是不需要的，
        #     # if xmin > self.boundary_margin and ymin > self.boundary_margin and xmax < w - self.boundary_margin and ymax < h - self.boundary_margin:
            img_info['bboxs'].append([xmin, ymin, xmax, ymax])
            img_info['labels'].append(name_id)
            img_info['confs'].append(conf)
        return img_info

    def _base_pred_detector(self, img, img_shape):
        pred = self.model(img, augment=False, val=True)[0]
        # NMS 非极大值抑制
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)
        tmp_result = []
        # 在原图的比例坐标
        for i, det in enumerate(pred):
            if det is not None and len(det):
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], img_shape).round()
                tmp_result.append(det)
        if not tmp_result:
            return None
        tmp_result = torch.vstack(tmp_result)
        # 垂直堆叠 转化成数组
        tmp_result = np.vstack([x.cpu().detach().numpy() for x in tmp_result])
        return tmp_result