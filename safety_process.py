from flask import Flask, request
import json
from util import Logger
from datetime import datetime
import os
import multiprocessing
import GPUtil
from service.process import Process

app = Flask(__name__)  # 接收http请求


# 光伏板施工进度
@app.route('/api/fault/process', methods=['POST'])  # 接口名称
def process():
    # 检测是否有数据
    if request.json is None:
        print("getCheckStatus request is none!")
        return 'fail'
    data = request.json  # 获取 JSON 格式的数据
    # 任务id
    taskId = data["taskId"]
    photoPathList = data["photoPathList"]

    if taskId is None or photoPathList is None:
        res_t = {'code': 400,
                 'msg': 'some parameters is None'
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
            multiprocessing.Process(target=process_function,
                                    args=(taskId, photoPathList, process_msg_url, process_upfile_url)).start()
            res_t = {'code': 200,
                     'msg': 'success'
                     }
    return res_t


def process_function(taskId, photoPathList, process_msg_url, process_upfile_url):
    current_time = datetime.now()
    time_id = current_time.strftime("%Y-%m-%d")
    log_name = current_time.strftime("%H-%M-%S")
    log_path = os.path.join("log/process", time_id)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log = Logger(os.path.join(log_path, log_name),
                 level='debug')  # 细节log日志
    log.logger.info(f'=====施工进度线程开始执行,taskID is : {taskId} ,num photo is :{len(photoPathList)}=====')
    process = Process(process_msg_url=process_msg_url, process_upfile_url=process_upfile_url, taskId=taskId,
                      log=log,
                      time_id=current_time.strftime("%Y-%m-%d_%H-%M-%S"),
                      photoPathList=photoPathList)

    process.start()
    log.logger.info(f"=====施工进度线程执行完毕,taskID is : {taskId} ,num photo is :{len(photoPathList)}=====")


if __name__ == '__main__':
    '''
        相关配置文件和地址
    '''
    multiprocessing.set_start_method('spawn')
    config_url = json.load(open("config/address_configuration.json", "r"))

    app.config['HOST'] = config_url["process"]["platform_url"]["host"]
    app.config['PORT'] = config_url["process"]["platform_url"]["port"]

    process_msg_url = config_url["process"]["process_msg_url"]
    process_upfile_url = config_url["process"]["process_upfile_url"]

    '''
        第四步： 开启flask服务
    '''
    app.run(host=app.config['HOST'], port=app.config['PORT'], debug=False, threaded=True)
