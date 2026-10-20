from flask import Flask, request
import json
from service.count import Count
from util import check_video_stream
from util import Logger
from datetime import datetime
import os
import multiprocessing
import GPUtil

app = Flask(__name__)  # 接收http请求


# 人员计数
@app.route('/api/fault/count', methods=['GET'])  # 接口名称
def count():
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
                multiprocessing.Process(target=count_function,
                                        args=(
                                            taskId, videoPath, count_upfile_url, count_msg_url)).start()
                # 开始执行
                res_t = {'code': 200,
                         'msg': 'success'
                         }
        else:
            res_t = {'code': 400,
                     'msg': 'error videoPath'
                     }
    return res_t


def count_function(taskId, videoPath, count_upfile_url, count_msg_url):
    current_time = datetime.now()
    time_id = current_time.strftime("%Y-%m-%d")
    log_name = current_time.strftime("%H-%M-%S")

    log_path = os.path.join("log/count", time_id)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log = Logger(os.path.join(log_path, log_name),
                 level='debug')  # 细节log日志

    log.logger.info(f'=====人员计数线程开始执行,taskID is : {taskId} ,videoPath is :{videoPath}=====')
    count = Count(count_upfile_url=count_upfile_url,
                  count_msg_url=count_msg_url,
                  log=log,
                  time_id=current_time.strftime("%Y-%m-%d_%H-%M-%S"),
                  taskId=taskId,
                  videoPath=videoPath)
    count.start()
    log.logger.info(f"=====人员计数线程执行完毕,taskID is : {taskId} ,videoPath is :{videoPath}=====")


if __name__ == '__main__':
    '''
        相关配置文件和地址
    '''
    multiprocessing.set_start_method('spawn')
    config_url = json.load(open("config/address_configuration.json", "r"))

    app.config['HOST'] = config_url["count"]["platform_url"]["host"]
    app.config['PORT'] = config_url["count"]["platform_url"]["port"]

    count_upfile_url = config_url["count"]["count_upfile_url"]
    count_msg_url = config_url["count"]["count_msg_url"]

    '''
        第四步： 开启flask服务
    '''
    app.run(host=app.config['HOST'], port=app.config['PORT'], debug=False, threaded=True)
