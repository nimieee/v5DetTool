from base_detector.IS_NET_detector import IS_NET_detector
import cv2 as cv
import numpy as np


class inference(object):
    def __init__(self, is_net_segment_path, v5_detecte_path):
        self.is_net_segment_path = is_net_segment_path
        self.v5_detecte_path = v5_detecte_path

        self.is_net_segment = IS_NET_detector(is_net_segment_path, 0)

    def start(self, video_path):
        cap = cv.VideoCapture(video_path)  # 打开视频或视频帧
        frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))  # 视频宽度
        frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))  # 视频高度
        fps = int(cap.get(cv.CAP_PROP_FPS))
        cap.set(cv.CAP_PROP_FPS, fps)
        fourcc = cv.VideoWriter_fourcc(*'mp4v')

        # # 视频输出头
        # self.out = cv2.VideoWriter(os.path.join(self.save_path_aqm_offline, f'{str(self.time_id)}.mp4'), fourcc,
        #                            self.desired_fps,
        #                            (int(self.frame_width / 2), int(self.frame_height / 2)))
        num = 0
        while True:

            ret, frame = cap.read()
            print(frame.dtype)
            if not ret:
                break
            # if num == 0:
            output, point = self.is_net_segment.detect(frame)

            # 创建一个填充颜色（例如：白色）
            fill_color = [0, 0, 0]  # 这里我们用白色填充
            # 使用掩码图像中的0值填充原图像
            result_image = np.where(output[:, :, None] == 0, fill_color, frame)
            # 找到包含这些点的最小矩形
            x, y, w, h = cv.boundingRect(point)
            # 裁剪图像
            cropped_image = result_image[y:y + h, x:x + w]
            result_image = result_image.astype('uint8')
            num += 1
            # print(frame.shape)

            # 在窗口中显示帧
            cv.imshow("Video", result_image)

            # 按下 'q' 键退出循环
            if cv.waitKey(25) & 0xFF == ord('q'):
                break

        # 释放视频对象和关闭窗口
        cap.release()
        cv.destroyAllWindows()


if __name__ == '__main__':
    is_net_segment_path = "path/to/is-net-car01.onnx"

    v5_detecte_path = ""
    video_path = "path/to/input.mp4"
    inference_api = inference(is_net_segment_path, v5_detecte_path)
    inference_api.start(video_path)
