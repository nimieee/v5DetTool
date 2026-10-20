from flask import Flask, request, send_from_directory, abort
import json
from service.aqm_offline import Helmet_offline, write_status
from util import check_video_stream, Logger
import GPUtil
import multiprocessing
import os
import re
import threading
import webbrowser
from datetime import datetime

app = Flask(__name__)  # 接收http请求
app.json.ensure_ascii = False  # 返回的json中文正常显示


# 首页：服务状态和接口说明
@app.route('/', methods=['GET'])
def index():
    try:
        gpus = [{'name': gpu.name, 'memoryFree': f'{gpu.memoryFree:.0f}MB', 'memoryTotal': f'{gpu.memoryTotal:.0f}MB'}
                for gpu in GPUtil.getGPUs()]
    except Exception as e:
        gpus = f'获取GPU信息失败: {e}'
    return {'service': '安全帽离线检测',
            'status': 'running',
            'gpu': gpus,
            'ui': '/ui',
            'api': {'url': '/api/fault/aqmOffline',
                    'method': 'GET',
                    'params': {'taskId': '任务id', 'videoPath': '视频文件路径'},
                    'example': '/api/fault/aqmOffline?taskId=test1&videoPath=D:/test.mp4'}
            }


# 安全帽离线检测
@app.route('/api/fault/aqmOffline', methods=['GET'])
def aqm_Offline():
    # 检测是否有数据
    if request.args is None:
        print("getCheckStatus request is none!")
        return 'fail'
    # 任务id
    taskId = request.args.get('taskId', type=str)
    # 视频路径
    videoPath = request.args.get('videoPath', type=str)
    # 当有参数获取结果为None返回错误代码
    if taskId is None or videoPath is None:
        res_t = {'code': 400,
                 'msg': 'some parameters is None'
                 }
    else:
        # 视频文件是否正确
        stream_state = check_video_stream(videoPath)
        if stream_state:
            res_t = start_task(taskId, videoPath)
        else:
            res_t = {'code': 400,
                     'msg': 'error videoPath'
                     }
    return res_t


# 检查显存后启动后台检测进程
def start_task(taskId, videoPath):
    gpus = GPUtil.getGPUs()
    memoryFree = gpus[0].memoryFree
    if memoryFree <= 2000:
        return {'code': 400,
                'msg': 'gpu out of memory'
                }
    # 先写入排队状态，界面在模型加载完成前也能查到任务
    write_status(os.path.join(save_path, taskId), {'taskId': taskId, 'videoPath': videoPath, 'state': 'loading',
                                                   'frame': 0, 'total': 0, 'alarms': [], 'video': None})
    multiprocessing.Process(target=aqm_offline_function,
                            args=(taskId, videoPath, aqm_upfile_url, aqm_img_msg_url,
                                  aqm_video_msg_url, aqm_msg_url)).start()
    # 开始执行
    return {'code': 200,
            'msg': 'success',
            'taskId': taskId
            }


# ===================== 可视化界面 =====================
TASK_ID_PATTERN = re.compile(r'^[\w\-]+$')


# 界面：选择本地视频并查看检测结果
@app.route('/ui', methods=['GET'])
def ui():
    return send_from_directory('web', 'aqm_offline.html')


# 界面上传视频：保存到 upload 目录后启动检测
@app.route('/api/fault/aqmOffline/upload', methods=['POST'])
def aqm_offline_upload():
    file = request.files.get('video')
    if file is None or not file.filename:
        return {'code': 400, 'msg': '未选择视频文件'}
    taskId = datetime.now().strftime("%Y%m%d_%H%M%S")
    ext = os.path.splitext(file.filename)[1].lower() or '.mp4'
    os.makedirs(upload_path, exist_ok=True)
    videoPath = os.path.join(upload_path, taskId + ext)
    file.save(videoPath)
    if not check_video_stream(videoPath):
        os.remove(videoPath)
        return {'code': 400, 'msg': '无法解析该视频文件'}
    res_t = start_task(taskId, videoPath)
    res_t['fileName'] = file.filename
    return res_t


# 任务列表（按时间倒序）
@app.route('/api/fault/aqmOffline/tasks', methods=['GET'])
def aqm_offline_tasks():
    tasks = []
    if os.path.isdir(save_path):
        for taskId in os.listdir(save_path):
            status = read_status(taskId)
            if status is not None:
                tasks.append(status)
    tasks.sort(key=lambda t: t['taskId'], reverse=True)
    return {'code': 200, 'tasks': tasks}


# 单个任务的状态
@app.route('/api/fault/aqmOffline/task/<taskId>', methods=['GET'])
def aqm_offline_task(taskId):
    status = read_status(taskId)
    if status is None:
        return {'code': 404, 'msg': '任务不存在'}
    return {'code': 200, 'task': status}


# 结果文件（视频、告警截图）
@app.route('/result/<taskId>/<path:filename>', methods=['GET'])
def aqm_offline_result(taskId, filename):
    if not TASK_ID_PATTERN.match(taskId):
        abort(404)
    return send_from_directory(os.path.abspath(os.path.join(save_path, taskId)), filename)


def read_status(taskId):
    if not TASK_ID_PATTERN.match(taskId):
        return None
    status_file = os.path.join(save_path, taskId, 'status.json')
    if not os.path.isfile(status_file):
        return None
    try:
        with open(status_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def aqm_offline_function(taskId, videoPath, aqm_upfile_url, aqm_img_msg_url,
                         aqm_video_msg_url, aqm_msg_url):
    current_time = datetime.now()
    time_id = current_time.strftime("%Y-%m-%d")
    log_name = current_time.strftime("%H-%M-%S")
    log_path = os.path.join("log/aqm_offline", time_id)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log = Logger(os.path.join(log_path, log_name),
                 level='debug')  # 细节log日志
    log.logger.info(f'=====安全帽离线诊断线程开始执行,taskID is : {taskId} ,videoPath is :{videoPath}=====')

    aqm_Offline = Helmet_offline(taskId=taskId, log=log, time_id=current_time.strftime("%Y-%m-%d_%H-%M-%S"),
                                 videoPath=videoPath,
                                 aqm_upfile_url=aqm_upfile_url, aqm_img_msg_url=aqm_img_msg_url,
                                 aqm_video_msg_url=aqm_video_msg_url, aqm_msg_url=aqm_msg_url)

    try:
        aqm_Offline.detect()
    except Exception as e:
        log.logger.exception(f"检测异常: {e}")
        aqm_Offline.write_status(state='error', error=str(e))
    log.logger.info(f"=====安全帽离线诊断线程执行完毕,taskID is : {taskId} ,videoPath is :{videoPath}=====")


if __name__ == '__main__':
    '''
        相关配置文件和地址
    '''
    multiprocessing.set_start_method('spawn')
    config_url = json.load(open("config/address_configuration.json", "r"))

    app.config['HOST'] = config_url["aqm_offline"]["platform_url"]["host"]
    app.config['PORT'] = config_url["aqm_offline"]["platform_url"]["port"]

    aqm_upfile_url = config_url["aqm_offline"]["aqm_upfile_url"]  # 文件上传url地址
    aqm_img_msg_url = config_url["aqm_offline"]["aqm_img_msg_url"]  # 图片保存地址上传json的url
    aqm_video_msg_url = config_url["aqm_offline"]["aqm_video_msg_url"]  # 视频保存地址上传的Json的url
    aqm_msg_url = config_url["aqm_offline"]["aqm_msg_url"]  # 任务结束的信息
    save_path = config_url["aqm_offline"].get("save_path", "result/aqm_offline")  # 检测结果目录
    upload_path = "upload/aqm_offline"  # 界面上传的视频保存目录

    print(f" * 可视化界面: http://127.0.0.1:{app.config['PORT']}/ui")
    # 服务启动后自动在浏览器打开可视化界面
    threading.Timer(1.5, webbrowser.open, args=(f"http://127.0.0.1:{app.config['PORT']}/ui",)).start()

    '''
        第四步： 开启flask服务
    '''
    app.run(host=app.config['HOST'], port=app.config['PORT'], debug=False, threaded=True)
