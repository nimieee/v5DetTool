import json
import os
import pathlib
import platform
import sys

import cv2 as cv
import numpy as np
import torch

# ===== NumPy 1.x 兼容 NumPy 2.x 保存的模型（同 det_to_xml/detecte_demo.py）=====
# 环境中残留的 numpy/_core 空目录会让 hasattr(np, '_core') 误判，因此按主版本号判断
if np.__version__.startswith('1.'):
    import numpy.core as _core
    sys.modules['numpy._core'] = _core
    sys.modules['numpy._core.multiarray'] = _core.multiarray
    sys.modules['numpy._core._multiarray_umath'] = _core._multiarray_umath
    sys.modules['numpy._core.numeric'] = _core.numeric
    sys.modules['numpy._core.umath'] = _core.umath
    sys.modules['numpy._core._exceptions'] = _core._exceptions
    sys.modules['numpy._core.shape_base'] = _core.shape_base
# ==============================================

from base_detector.v5_detector import yolo_detector
from util import upfile_img, upfile_mp4, upfile_msg

# Linux 下训练保存的权重在 Windows 上加载需要
if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath

CONFIG_PATH = "config/address_configuration.json"

# 违规类别：未戴安全帽
VIOLATION_CLASSES = {'nohelmet'}


# 原子写入任务状态文件，避免界面读到写了一半的 JSON
def write_status(save_path, status):
    os.makedirs(save_path, exist_ok=True)
    tmp_path = os.path.join(save_path, "status.json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(status, f, ensure_ascii=False)
    os.replace(tmp_path, os.path.join(save_path, "status.json"))


class Helmet_offline(object):
    def __init__(self, taskId, log, time_id, videoPath,
                 aqm_upfile_url, aqm_img_msg_url, aqm_video_msg_url, aqm_msg_url):
        self.taskId = taskId
        self.log = log
        self.time_id = time_id
        self.videoPath = videoPath
        self.aqm_upfile_url = aqm_upfile_url  # 文件上传url地址
        self.aqm_img_msg_url = aqm_img_msg_url  # 告警图片信息url
        self.aqm_video_msg_url = aqm_video_msg_url  # 结果视频信息url
        self.aqm_msg_url = aqm_msg_url  # 任务结束信息url

        config = json.load(open(CONFIG_PATH, "r", encoding="utf-8"))["aqm_offline"]
        model_cfg = config["model"]
        self.frame_interval = config.get("frame_interval", 5)  # 每隔多少帧检测一次
        self.alarm_interval = config.get("alarm_interval", 10)  # 两次告警之间的最小间隔（秒）
        self.save_path = os.path.join(config.get("save_path", "result/aqm_offline"), str(taskId))
        os.makedirs(self.save_path, exist_ok=True)

        device = 0 if torch.cuda.is_available() else 'cpu'
        self.detector = yolo_detector(weights=model_cfg["weights"], device=device,
                                      conf_thres=model_cfg["conf_thres"], iou_thres=model_cfg["iou_thres"],
                                      img_size=model_cfg["img_size"])
        self.class_name = self.detector.model.names
        self.log.logger.info(f"模型加载完成: {model_cfg['weights']}, device={device}, 类别={self.class_name}")

        # 任务状态，写入 status.json 供可视化界面轮询
        self.status = {'taskId': taskId, 'videoPath': videoPath, 'state': 'running',
                       'frame': 0, 'total': 0, 'alarms': [], 'video': None}
        self.write_status()

    def write_status(self, **kwargs):
        self.status.update(kwargs)
        write_status(self.save_path, self.status)

    # 画框，返回是否存在未戴安全帽
    def draw_result(self, frame, img_info):
        no_helmet = False
        for bbox, label, conf in zip(img_info["bboxs"], img_info["labels"], img_info["confs"]):
            name = self.class_name[int(label)]
            if name in VIOLATION_CLASSES:
                no_helmet = True
                color = (0, 0, 255)
            elif name == 'helmet':
                color = (0, 255, 0)
            else:
                continue
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv.putText(frame, f"{name} {conf:.2f}", (x1, max(y1 - 5, 0)), cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame, no_helmet

    def detect(self):
        cap = cv.VideoCapture(self.videoPath)
        if not cap.isOpened():
            self.log.logger.error(f"视频打开失败: {self.videoPath}")
            self.write_status(state='error', error='视频打开失败')
            return
        fps = cap.get(cv.CAP_PROP_FPS) or 25
        width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.write_status(total=int(cap.get(cv.CAP_PROP_FRAME_COUNT)), fps=fps)

        video_name = f"{self.time_id}.mp4"
        video_path = os.path.join(self.save_path, video_name)
        # 优先用 H.264（avc1），浏览器可直接播放；不支持时退回 mp4v
        out = cv.VideoWriter(video_path, cv.VideoWriter_fourcc(*'avc1'), fps, (width, height))
        if not out.isOpened():
            out = cv.VideoWriter(video_path, cv.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        frame_id = 0
        alarm_count = 0
        last_alarm_time = -self.alarm_interval
        img_info = {"bboxs": [], "labels": [], "confs": []}
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_id % self.frame_interval == 0:
                img_info = self.detector.detect(frame)
            frame, no_helmet = self.draw_result(frame, img_info)
            out.write(frame)

            current_time = frame_id / fps
            if no_helmet and frame_id % self.frame_interval == 0 \
                    and current_time - last_alarm_time >= self.alarm_interval:
                last_alarm_time = current_time
                alarm_count += 1
                self.alarm(frame, frame_id, current_time)
            frame_id += 1
            if frame_id % 30 == 0:
                self.write_status(frame=frame_id)

        cap.release()
        out.release()
        self.log.logger.info(f"视频处理完成，共 {frame_id} 帧，告警 {alarm_count} 次，结果视频: {video_path}")
        self.write_status(state='finish', frame=frame_id, total=frame_id, video=video_name)

        msg, ok = upfile_mp4(self.aqm_upfile_url, video_path, video_name)
        self.log.logger.info(f"结果视频上传: ok={ok}, msg={msg}")
        if ok:
            msg, ok = upfile_msg(self.aqm_video_msg_url, {'taskId': self.taskId, 'videoUrl': msg})
            self.log.logger.info(f"视频信息上传: ok={ok}, msg={msg}")

        msg, ok = upfile_msg(self.aqm_msg_url, {'taskId': self.taskId, 'state': 'finish', 'alarmCount': alarm_count})
        self.log.logger.info(f"任务结束信息上传: ok={ok}, msg={msg}")

    # 未戴安全帽告警：本地保存截图并上传
    def alarm(self, frame, frame_id, current_time):
        frame_name = f"{self.time_id}_{frame_id}.jpg"
        cv.imwrite(os.path.join(self.save_path, frame_name), frame)
        self.log.logger.warning(f"发现未戴安全帽: 第 {frame_id} 帧（{current_time:.1f}s）")
        self.status['alarms'].append({'frame': frame_id, 'time': round(current_time, 2), 'image': frame_name})
        self.write_status(frame=frame_id)

        msg, ok = upfile_img(self.aqm_upfile_url, frame, frame_name)
        if not ok:
            self.log.logger.error(f"告警图片上传失败: {msg}")
            return
        msg, ok = upfile_msg(self.aqm_img_msg_url, {'taskId': self.taskId, 'imgUrl': msg,
                                                     'frameId': frame_id, 'time': round(current_time, 2)})
        self.log.logger.info(f"告警信息上传: ok={ok}, msg={msg}")
