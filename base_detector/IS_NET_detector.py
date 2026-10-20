import numpy as np
import cv2 as cv
import onnxruntime
import os


class CustomImagePreprocessor:
    def __init__(self, target_size=(1024, 800),
                 mean=[0.5, 0.5, 0.5],
                 std=[1.0, 1.0, 1.0]):
        self.target_size = target_size
        self.mean = mean
        self.std = std

    def normalize(self, image):
        image = image / 255.0
        image -= self.mean
        image /= self.std
        return image

    def __call__(self, *args, **kwargs):
        image = args[0]
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image = np.array(image, dtype=np.float32)
        image = cv.resize(image, self.target_size)
        # cv.imwrite("path/to/debug_input.jpg", image)

        image = self.normalize(image)
        image = np.expand_dims(np.transpose(image, (2, 0, 1)), axis=0).astype(np.float32)
        return image


class IS_NET_detector:
    def __init__(self, model_path, device):
        self.model_path = model_path
        self.device = device
        providers = ['CPUExecutionProvider']
        # providers = [
        #     ('CUDAExecutionProvider', {'device_id': self.device}),
        # ]
        # 加载模型
        self.net = onnxruntime.InferenceSession(self.model_path, providers=providers)
        self.input_name = self.net.get_inputs()[0].name
        # 数据预处理
        self.preprocessor = CustomImagePreprocessor(
            (self.net.get_inputs()[0].shape[3], self.net.get_inputs()[0].shape[2]))
        # 默认的图像宽和高
        self.height = 3888
        self.width = 5184

        # 切分时添加的pix
        self.split = False
        self.offsenum = 320
        self.row = 2
        self.column = 2

    def detect(self, image):
        img_copy = image.copy()
        res = self.base_detect(img_copy)
        # 传统后处理
        res = self.threshold_seg_and_erosion(res)
        # result_image, point = self.remove_invalid_box_to_img(img_copy, res)
        # return result_image, point

        return res

    def base_detect(self, image):
        height, width = image.shape[:2]
        self.height = height
        self.width = width

        if self.split:
            img_list = self.cut_img(image)
            res_list = []
            for img in img_list:
                result = self.process(img)
                res_list.append(result)
            res_img = self.merge_img(res_list)
        else:
            result = self.process(image)
            res_img = result
        return res_img

    def process(self, img):
        blob = self.preprocessor(img)
        outs = self.net.run(None, {self.net.get_inputs()[0].name: blob})
        result = np.array(outs[0]).squeeze()
        result *= 255
        return result

    # 切分图像
    def cut_img(self, image):
        img_list = []
        row_step = (int)(self.height / self.row)
        column_step = (int)(self.width / self.column)
        img = image[0:row_step * self.row, 0:column_step * self.column].copy()
        for i in range(self.row):
            for j in range(self.column):
                tmp_img = None
                if i == 0 and j == 0:
                    tmp_img = img[0:(row_step + self.offsenum), 0:column_step + self.offsenum]
                elif i == 0 and j == 1:
                    tmp_img = img[0:(row_step + self.offsenum),
                              (column_step - self.offsenum):(column_step) + column_step]
                elif i == 1 and j == 0:
                    tmp_img = img[(row_step - self.offsenum):(i * row_step + row_step),
                              (j * column_step):(j * column_step) + column_step + self.offsenum]
                else:
                    tmp_img = img[(i * row_step - self.offsenum):(i * row_step + row_step),
                              (j * column_step - self.offsenum):(j * column_step) + column_step]
                img_list.append(tmp_img)
        return img_list

    # 切分分割后的合并           
    def merge_img(self, res_list):
        row_step = (int)(self.height / self.row)
        column_step = (int)(self.width / self.column)
        merge_res = np.full((self.height, self.width), 114, dtype=np.uint8)  # base image with 4 tiles
        # print(merge_res.shape)
        for i in range(self.row):
            for j in range(self.column):
                if i == 0 and j == 0:
                    temp = cv.resize(res_list[i * self.column + j],
                                     dsize=(column_step + self.offsenum, row_step + self.offsenum),
                                     interpolation=cv.INTER_NEAREST)[0:row_step, 0:column_step].copy()
                    # print(temp.shape)
                    merge_res[(i * row_step):(i * row_step + row_step),
                    (j * column_step):(j * column_step) + column_step] = temp
                elif i == 0 and j == 1:
                    temp = cv.resize(res_list[i * self.column + j],
                                     dsize=(column_step + self.offsenum, row_step + self.offsenum),
                                     interpolation=cv.INTER_NEAREST)[0:row_step,
                           self.offsenum:column_step + self.offsenum]
                    # print(temp.shape)
                    merge_res[(i * row_step):(i * row_step + row_step),
                    (j * column_step):(j * column_step) + column_step] = temp.copy()
                elif i == 1 and j == 0:
                    temp = cv.resize(res_list[i * self.column + j],
                                     dsize=(column_step + self.offsenum, row_step + self.offsenum),
                                     interpolation=cv.INTER_NEAREST)[self.offsenum:row_step + self.offsenum:,
                           0:column_step]
                    # print(temp.shape)
                    merge_res[(i * row_step):(i * row_step + row_step),
                    (j * column_step):(j * column_step) + column_step] = temp.copy()
                else:
                    temp = cv.resize(res_list[i * self.column + j],
                                     dsize=(column_step + self.offsenum, row_step + self.offsenum),
                                     interpolation=cv.INTER_NEAREST)[self.offsenum:row_step + self.offsenum:,
                           self.offsenum:column_step + self.offsenum]
                    # print(temp.shape)
                    merge_res[(i * row_step):(i * row_step + row_step),
                    (j * column_step):(j * column_step) + column_step] = temp.copy()
        return merge_res

    # mask图像的后处理
    def threshold_seg_and_erosion(self, res):
        # 二值化
        _, res = cv.threshold(src=res,  # 要二值化的图片
                              thresh=127,  # 全局阈值
                              maxval=255,  # 大于全局阈值后设定的值
                              type=cv.THRESH_BINARY)  # 设定的二值化类型
        # 进行腐蚀操作
        k = np.ones((3, 3), dtype=np.uint8)
        res = cv.morphologyEx(res, cv.MORPH_OPEN, k)
        # 进行膨胀操作
        k2 = np.ones((3, 3), dtype=np.uint8)
        res = cv.morphologyEx(res, cv.MORPH_DILATE, k2)
        res = cv.convertScaleAbs(res)
        return res

    def remove_invalid_box_to_img(self, original_image, mask_img):
        num_labels, labels, stats, _ = cv.connectedComponentsWithStats(mask_img, connectivity=8)
        # 找到最大的连通区域标签
        max_label = 1
        max_size = 0
        for i in range(1, num_labels):
            if stats[i, cv.CC_STAT_AREA] > max_size:
                max_label = i
                max_size = stats[i, cv.CC_STAT_AREA]
        # 创建输出图像，显示最大连通区域

        output = np.zeros_like(mask_img, dtype=np.uint8)
        output[labels == max_label] = 255

        box = self.mask2box(output)
        if len(box) == 0:
            return []
        else:
            point = np.int32(cv.boxPoints(box[0]))

        return output, point

    # msak图像转box框
    def mask2box(self, mask):
        boxes = []
        contours, hechine = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        for c in range(len(contours)):
            epsilon = 0.01 * cv.arcLength(contours[c], True)
            box = cv.approxPolyDP(contours[c], epsilon, True)
            rect = cv.minAreaRect(box)  # 函数 cv2.minAreaRect() 返回一个Box2D结构 rect：（最小外接矩形的中心（x，y），（宽度，高度），旋转角度）
            boxes.append(rect)
        return boxes


if __name__ == "__main__":
    is_detector = IS_NET_detector("path/to/is-net-car01.onnx", 0)
    jpg_path = r"path/to/test.jpg"

    image = cv.imread(jpg_path)

    res = is_detector.detect(image)
    cv.imwrite("path/to/out.png", res)

#     save_path = "path/to/test_out"
#     if not os.path.exists(save_path):
#         os.makedirs(save_path)
#
#     jpg_path_list = os.listdir(jpg_path)
#     for img_name in jpg_path_list:
#         img_path = os.path.join(jpg_path, img_name)
#         # 读取图像
#         image = cv.imread(img_path)
#         points = is_detector.detecte(image)
#         draw_point_box(image, points, os.path.join(save_path, img_name))
#         print(img_path, points)