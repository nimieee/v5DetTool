from flask import Flask, request
import json
from service.aqm_online import Helmet
from util import check_video_stream

app = Flask(__name__)  # 接收http请求


# 安全帽检测
@app.route('/api/fault/getCheckStatus', methods=['GET'])  # 接口名称
def get_check_status():
    # 检测是否有数据
    if request.args is None:
        print("getCheckStatus request is none!")
        return 'fail'
    # 任务id
    taskId = request.args.get('taskId', type=str)
    # 状态编码，state=1开始检测， state=0停止检测
    state = request.args.get('state', type=int)
    # liveUrl直播流地址
    liveUrl = request.args.get('liveUrl', type=str)
    # 当有参数获取结果为None返回错误代码
    if taskId is None or state is None or liveUrl is None:
        res_t = {'code': 400,
                 'msg': 'some parameters is None'
                 }
    # 当多次执行调用时，状态码state = 0 表示 需要停止，state = 1 表示再次执行，则需要等待上次调用的完成后才能调用
    elif aqm_jiance.lock_proccess:
        if state == 0:
            # 将安全帽对象状态赋值为0
            aqm_jiance.state = 0
            res_t = {
                'code': 200,
                'msg': 'stop successfully. Please wait for a while before proceeding with further processing.'
            }
        elif state == 1:
            res_t = {'code': 400,
                     'msg': 'The previous helmet algorithm was not executed,You can set the state to 0, stop the previous algorithm, and wait for it to be executed'
                     }
        else:
            res_t = {'code': 400,
                     'msg': 'invalid parameter state'
                     }
    else:
        # 检查视频流是否正确
        stream_state = check_video_stream(liveUrl)
        if stream_state:
            # 赋值安全帽的视频流地址
            aqm_jiance.aqm_stream = liveUrl
            if state == 0:
                # 将安全帽对象状态赋值为1
                aqm_jiance.state = 0
                aqm_jiance.taskID = taskId
                res_t = {
                    'code': 200,
                    'msg': 'stop'
                }

            elif state == 1:
                # 将安全帽对象状态赋值为1
                aqm_jiance.state = 1
                aqm_jiance.taskID = taskId
                res_t = {
                    'code': 200,
                    'msg': 'start'
                }
            else:
                res_t = {'code': 400,
                         'msg': 'invalid parameter state'
                         }
        else:
            res_t = {'code': 400,
                     'msg': 'error liveUrl'
                     }
    return res_t


if __name__ == '__main__':
    '''
        相关配置文件和地址
    '''
    config_url = json.load(open("config/address_configuration.json", "r"))

    app.config['HOST'] = config_url["aqm_online"]["platform_url"]["host"]
    app.config['PORT'] = config_url["aqm_online"]["platform_url"]["port"]

    aqm_return_stearm = config_url["aqm_online"]["return_stream_url"]
    aqm_upfile_url = config_url["aqm_online"]["aqm_upfile_url"]
    aqm_img_msg_url = config_url["aqm_online"]["aqm_img_msg_url"]
    aqm_video_msg_url = config_url["aqm_online"]["aqm_video_msg_url"]
    aqm_call_msg_url = config_url["aqm_online"]["aqm_call_msg_url"]

    aqm_jiance = Helmet(return_steram=aqm_return_stearm, img_upfile_url=aqm_upfile_url, aqm_img_msg_url=aqm_img_msg_url,
                        aqm_video_msg_url=aqm_video_msg_url, aqm_call_msg_url=aqm_call_msg_url)

    '''
        第四步： 开启flask服务
    '''
    app.run(host=app.config['HOST'], port=app.config['PORT'], debug=False, threaded=True)
