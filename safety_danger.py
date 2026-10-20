from flask import Flask, request
import json
from util import check_video_stream
import multiprocessing
import GPUtil
import os
from datetime import datetime
from util import Logger
from service.danger import Danger

app = Flask(__name__)  # 接收http请求


@app.route('/api/fault/danger', methods=['GET'])  # 接口名称
def area_invading():
    for key in list(process_library.keys()):
        if process_library[key].value == 0:
            del process_library[key]
    # 检测是否有数据
    if request.args is None:
        print("request is args none!")
        return 'fail'
    # 主任务id
    taskId = request.args.get('taskId', type=str)
    # 子任务id
    childId = request.args.get("childId", type=str)
    # 状态编码，state=1开始检测， state=0停止检测
    state = request.args.get('state', type=int)
    # 直播流地址
    stream = request.args.get('liveUrl', type=str)
    # 当有参数获取结果为None返回错误代码
    if taskId is None or state is None or childId is None or stream is None:
        res_t = {'code': 400,
                 'msg': 'some parameters is None'
                 }
    else:
        # 检查视频流是否正确
        stream_state = check_video_stream(stream)
        if stream_state:
            if state == 0:
                if childId in process_library:
                    process = process_library[childId]
                    process.value = False
                    del process_library[childId]
                    res_t = {'code': 200,
                             'msg': 'stop process success'
                             }
                else:
                    res_t = {'code': 400,
                             'msg': 'no this process'
                             }
            elif state == 1:
                if childId in process_library:
                    res_t = {'code': 400,
                             'msg': 'process childid repeated'
                             }
                else:
                    gpus = GPUtil.getGPUs()
                    memoryFree = gpus[0].memoryFree
                    if memoryFree <= 2000:
                        # 开始执行
                        res_t = {'code': 400,
                                 'msg': 'gpu out of memory'
                                 }
                    else:
                        process_start(taskId, childId, stream, process_library)
                        # 开始执行
                        res_t = {'code': 200,
                                 'msg': 'success'
                                 }
            else:
                res_t = {'code': 400,
                         'msg': 'invalid parameter state'
                         }
        else:
            res_t = {'code': 400,
                     'msg': 'error videoPath'
                     }
    return res_t


# 安全帽检测
def process_start(taskId, childId, stream, process_library):
    shared_bool = multiprocessing.Value('b', True)
    multiprocessing.Process(target=danger_function,
                            args=(taskId, childId, stream, shared_bool, danger_call_msg_url, danger_upfile_url,
                                  danger_img_msg_url, danger_video_msg_url)).start()
    process_library[childId] = shared_bool


def danger_function(taskId, childId, stream, shared_bool, danger_call_msg_url, danger_upfile_url, danger_img_msg_url,
                    danger_video_msg_url):
    current_time = datetime.now()
    time_id = current_time.strftime("%Y-%m-%d")
    log_name = current_time.strftime("%H-%M-%S")
    log_path = os.path.join("log/danger", time_id, taskId)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log = Logger(os.path.join(log_path, log_name),
                 level='debug')  # 细节log日志
    log.logger.info(f'=====区域入侵线程开始执行,taskId is : {taskId},childId is :{childId}=====')

    danger = Danger(taskId=taskId, childId=childId, stream=stream, log=log, state=shared_bool,
                    time_id=current_time.strftime("%Y-%m-%d_%H-%M-%S"), call_msg_url=danger_call_msg_url,
                    danger_upfile_url=danger_upfile_url, danger_img_msg_url=danger_img_msg_url,
                    danger_video_msg_url=danger_video_msg_url)

    danger.start()
    log.logger.info(f"=====区域入侵线程执行完毕,taskId is : {taskId},childId is : {childId}=====")


if __name__ == '__main__':
    '''
        相关配置文件和地址
    '''
    multiprocessing.set_start_method('spawn')
    config_url = json.load(open("config/address_configuration.json", "r"))

    app.config['HOST'] = config_url["danger"]["platform_url"]["host"]
    app.config['PORT'] = config_url["danger"]["platform_url"]["port"]

    danger_call_msg_url = config_url["danger"]["danger_call_msg_url"]
    danger_upfile_url = config_url["danger"]["danger_upfile_url"]
    danger_img_msg_url = config_url["danger"]["danger_img_msg_url"]
    danger_video_msg_url = config_url["danger"]["danger_video_msg_url"]

    process_library = {}
    '''
        第四步： 开启flask服务
    '''
    app.run(host=app.config['HOST'], port=app.config['PORT'], debug=False, threaded=True)
