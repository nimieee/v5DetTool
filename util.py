import cv2 as cv
import math
import warnings
import requests
import json
import ffmpeg
import logging
from logging import handlers

warnings.filterwarnings("ignore")


# 日志类
class Logger(object):
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }  # 日志级别关系映射

    def __init__(self, filename, level='info', when='D', backCount=3,
                 fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)  # 设置日志格式
        self.logger.setLevel(self.level_relations.get(level))  # 设置日志级别
        th = handlers.TimedRotatingFileHandler(filename=filename, when=when, backupCount=backCount,
                                               encoding='utf-8')  # 往文件里写入#指定间隔时间自动生成文件的处理器
        console_handler = logging.StreamHandler()
        # 实例化TimedRotatingFileHandler
        # interval是时间间隔，backupCount是备份文件的个数，如果超过这个个数，就会自动删除，when是间隔的时间单位，单位有以下几种：
        # S 秒
        # M 分
        # H 小时、
        # D 天、
        # W 每星期（interval==0时代表星期一）
        # midnight 每天凌晨
        th.setFormatter(format_str)  # 设置文件里写入的格式
        self.logger.addHandler(th)
        self.logger.addHandler(console_handler)


def id_draw_result(frame, img_info):
    class_name = ['kengcao']
    if not img_info is None:
        bboxs = img_info["bboxs"]
        labels = img_info["labels"]
        for i in range(len(bboxs)):
            if class_name[int(labels[i])] not in need_class_name:
                continue
            if class_name[int(labels[i])] == "aqm_red":
                color = (0, 0, 255)
            elif class_name[int(labels[i])] == "aqm_white":
                color = (255, 255, 255)
            elif class_name[int(labels[i])] == "aqm_yellow":
                color = (0, 255, 255)
            elif class_name[int(labels[i])] == "aqm_blue":
                color = (255, 255, 0)
            elif class_name[int(labels[i])] == "wcaqm":
                color = (0, 0, 0)
            cv.rectangle(frame, (int(bboxs[i][0]), int(bboxs[i][1])), (int(bboxs[i][2]), int(bboxs[i][3])), color,
                         2)
    return frame


def danger_draw_result(frame, img_info):
    class_name = ["zyaq_aqmbt", "person"]
    find_person = False
    if not img_info is None:
        bboxs = img_info["bboxs"]
        labels = img_info["labels"]
        for i in range(len(bboxs)):
            if class_name[int(labels[i])] == "person":
                find_person = True
                cv.rectangle(frame, (int(bboxs[i][0]), int(bboxs[i][1])), (int(bboxs[i][2]), int(bboxs[i][3])),
                             (0, 0, 255), 2)
    return frame, find_person


def anquammao_draw_result(frame, img_info):
    class_name = ["zyaq_aqmbt", "wcaqm", "aqmzc"]
    no_helmet = False

    if not img_info is None:
        bboxs = img_info["bboxs"]
        labels = img_info["labels"]
        for i in range(len(bboxs)):
            if class_name[int(labels[i])] == "aqmzc":
                cv.rectangle(frame, (int(bboxs[i][0]), int(bboxs[i][1])), (int(bboxs[i][2]), int(bboxs[i][3])),
                             (0, 255, 0), 2)
            elif class_name[int(labels[i])] == "wcaqm":
                no_helmet = True
                cv.rectangle(frame, (int(bboxs[i][0]), int(bboxs[i][1])), (int(bboxs[i][2]), int(bboxs[i][3])),
                             (0, 0, 255), 2)
    return frame, no_helmet


def split_video(input_path, output_path, start_frame, end_frame):
    video_capture = cv.VideoCapture(input_path)
    frame_count = int(video_capture.get(cv.CAP_PROP_FRAME_COUNT))

    if end_frame > frame_count:
        end_frame = frame_count

    video_capture.set(cv.CAP_PROP_POS_FRAMES, start_frame)

    fourcc = int(video_capture.get(cv.CAP_PROP_FOURCC))  # 获取原视频的fourcc值  # 视频编解码器
    frame_width = int(video_capture.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_capture.get(cv.CAP_PROP_FPS))

    out = cv.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    current_frame = start_frame
    while current_frame < end_frame:
        ret, frame = video_capture.read()
        if ret:
            out.write(frame)
        current_frame += 1

    video_capture.release()
    out.release()
    cv.destroyAllWindows()
    out_put_h264 = output_path[:-4] + "_h264" + ".mp4"
    # 使用ffmpeg对视频进行转码
    ffmpeg.input(output_path).output(out_put_h264, vcodec='h264').run()

    # 点在四边形(矩形)内?


def point_in_quadrilateral(pt_x, pt_y, corners):
    ab0 = corners[2] - corners[0]
    ab1 = corners[3] - corners[1]

    ad0 = corners[6] - corners[0]
    ad1 = corners[7] - corners[1]

    ap0 = pt_x - corners[0]
    ap1 = pt_y - corners[1]

    abab = ab0 * ab0 + ab1 * ab1
    abap = ab0 * ap0 + ab1 * ap1
    adad = ad0 * ad0 + ad1 * ad1
    adap = ad0 * ap0 + ad1 * ap1

    return abab >= abap and abap >= 0 and adad >= adap and adap >= 0

    # 相交后转化为直线交点


def line_segment_intersection(pts1, pts2, i, j):
    # pts1, pts2 为corners
    # i j 分别表示第几个交点，取其和其后一个点构成的线段
    # 返回为 tuple(bool, pts) bool=True pts为交点
    A, B, C, D, ret = [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]
    A[0] = pts1[2 * i]
    A[1] = pts1[2 * i + 1]

    B[0] = pts1[2 * ((i + 1) % 4)]
    B[1] = pts1[2 * ((i + 1) % 4) + 1]

    C[0] = pts2[2 * j]
    C[1] = pts2[2 * j + 1]

    D[0] = pts2[2 * ((j + 1) % 4)]
    D[1] = pts2[2 * ((j + 1) % 4) + 1]
    BA0 = B[0] - A[0]
    BA1 = B[1] - A[1]
    DA0 = D[0] - A[0]
    CA0 = C[0] - A[0]
    DA1 = D[1] - A[1]
    CA1 = C[1] - A[1]
    # 叉乘判断方向
    acd = DA1 * CA0 > CA1 * DA0
    bcd = (D[1] - B[1]) * (C[0] - B[0]) > (C[1] - B[1]) * (D[0] - B[0])
    if acd != bcd:
        abc = CA1 * BA0 > BA1 * CA0
        abd = DA1 * BA0 > BA1 * DA0
        # 判断方向
        if abc != abd:
            DC0 = D[0] - C[0]
            DC1 = D[1] - C[1]
            ABBA = A[0] * B[1] - B[0] * A[1]
            CDDC = C[0] * D[1] - D[0] * C[1]
            DH = BA1 * DC0 - BA0 * DC1
            Dx = ABBA * DC0 - BA0 * CDDC
            Dy = ABBA * DC1 - BA1 * CDDC
            ret[0] = Dx / DH
            ret[1] = Dy / DH
            return True, ret
    return False, ret

    # 顶点排序


def sort_vertex_in_convex_polygon(int_pts, num_of_inter):
    def _cmp(pt, center):
        vx = pt[0] - center[0]
        vy = pt[1] - center[1]
        d = math.sqrt(vx * vx + vy * vy)
        vx /= d
        vy /= d
        if vy < 0:
            vx = -2 - vx
        return vx

    if num_of_inter > 0:
        center = [0, 0]
        for i in range(num_of_inter):
            center[0] += int_pts[i][0]
            center[1] += int_pts[i][1]
        center[0] /= num_of_inter
        center[1] /= num_of_inter
        int_pts.sort(key=lambda x: _cmp(x, center))

    # 将多边形转化为多个三角形面积之和


def area(int_pts, num_of_inter):
    def _trangle_area(a, b, c):
        return ((a[0] - c[0]) * (b[1] - c[1]) - (a[1] - c[1]) *
                (b[0] - c[0])) / 2.0

    area_val = 0.0
    for i in range(num_of_inter - 2):
        area_val += abs(
            _trangle_area(int_pts[0], int_pts[i + 1],
                          int_pts[i + 2]))
    return area_val


def calculate_area(points):
    # 依次连接两个对角线顶点，得到矩形的两条边向量
    edge1 = [points[1][0] - points[0][0], points[1][1] - points[0][1]]
    edge2 = [points[2][0] - points[1][0], points[2][1] - points[1][1]]

    # 计算两个向量的叉积
    area = abs(edge1[0] * edge2[1] - edge1[1] * edge2[0])
    return area


def calculate_iou(box1, box2):
    box1 = [[point[0] * 10000, point[1] * 10000] for point in box1]
    box2 = [[point[0] * 10000, point[1] * 10000] for point in box2]

    box1_area = calculate_area(box1)
    box2_area = calculate_area(box2)

    corners1 = [item for sublist in box1 for item in sublist]
    corners2 = [item for sublist in box2 for item in sublist]

    pts, num_pts = [], 0
    for i in range(4):
        point = [corners1[2 * i], corners1[2 * i + 1]]
        if point_in_quadrilateral(point[0], point[1],
                                  corners2):
            num_pts += 1
            pts.append(point)
    for i in range(4):
        point = [corners2[2 * i], corners2[2 * i + 1]]
        if point_in_quadrilateral(point[0], point[1],
                                  corners1):
            num_pts += 1
            pts.append(point)
    for i in range(4):
        for j in range(4):
            ret, point = line_segment_intersection(corners1, corners2, i, j)
            if ret:
                num_pts += 1
                pts.append(point)
    sort_vertex_in_convex_polygon(pts, num_pts)
    polygon_area = area(pts, num_pts)

    iou = polygon_area / (box1_area + box2_area - polygon_area)

    return iou

    # 判断这个视频流是否有效


def check_video_stream(stream_url):
    cap = cv.VideoCapture(stream_url)
    if cap.isOpened():
        cap.release()
        return True
    else:
        cap.release()
        return False

    # 上传视频


def upfile_mp4(url, video_path, video_name):
    try:
        with open(video_path, 'rb') as video_file:
            files = {'upfile': (video_name, video_file, "video/mp4")}
            response = requests.post(url, files=files)
            if response.status_code == 200:
                data = response.json()
                if data['code'] == 200:
                    return data['msg'], True
                else:
                    return data['msg'], False
            else:
                return f"HTTP请求失败，状态码：{response.status_code}", False
    except Exception as e:
        return e, False

    # 上传图片


def upfile_img(url, frame, frame_name):
    image_data = cv.imencode('.jpg', frame)[1].tobytes()
    files = {'upfile': (frame_name, image_data, "image/jpeg")}
    try:
        response = requests.post(url, files=files)
        if response.status_code == 200:
            data = response.json()
            if data['code'] == 200:
                return data['msg'], True
            else:
                return data['msg'], False
        else:
            return f"HTTP请求失败，状态码：{response.status_code}", False
    except requests.exceptions.RequestException as e:
        # 处理请求异常
        return f"HTTP请求发生异常: {e}", False

    # 上传消息


def upfile_msg(url, data):
    # 上传最终json数据
    json_data = json.dumps(data)
    # 发送 HTTP 请求
    headers = {'Content-Type': 'application/json'}
    try:
        response = requests.post(url, headers=headers, data=json_data)
        if response.status_code == 200:
            return "数据发送成功", True
        else:
            return f"数据发送失败，错误代码: {response.status_code}", False
    except requests.RequestException as e:
        return f"数据发送异常，异常信息: {e}", False
