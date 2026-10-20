from flask import Flask, request
import json
from service.aqm_offline import Helmet_offline
from util import check_video_stream, Logger
import GPUtil
import multiprocessing
import os
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
            gpus = GPUtil.getGPUs()
            memoryFree = gpus[0].memoryFree
            if memoryFree <= 2000:
                # 开始执行
                res_t = {'code': 400,
                         'msg': 'gpu out of memory'
                         }
            else:
                multiprocessing.Process(target=aqm_offline_function,
                                        args=(taskId, videoPath, aqm_upfile_url, aqm_img_msg_url,
                                              aqm_video_msg_url, aqm_msg_url)).start()
                # 开始执行
                res_t = {'code': 200,
                         'msg': 'success'
                         }
        else:
            res_t = {'code': 400,
                     'msg': 'error videoPath'
                     }
    return res_t


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

    aqm_Offline.detect()
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

    '''
        第四步： 开启flask服务
    '''
    app.run(host=app.config['HOST'], port=app.config['PORT'], debug=False, threaded=True)
