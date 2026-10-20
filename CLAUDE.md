# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 环境

- 必须使用 conda 环境 `yolov5`（`D:\ANACONDA\envs\yolov5`，Python 3.12，torch 2.2 + CUDA）。**不要在 base 环境安装依赖**。非交互执行时可直接用 `D:/ANACONDA/envs/yolov5/python.exe`。
- 无 `requirements.txt`、无测试框架、无 lint 配置。验证改动的方式是 `python -m py_compile <files>` 加实际运行。
- 用户希望始终用中文回复。

## 常用命令

```bash
# 启动安全检测服务（Flask，端口见 config/address_configuration.json，默认 5002）
python safety.py

# 调用离线安全帽检测（后台进程执行，结果在 result/aqm_offline/<taskId>/，日志在 log/aqm_offline/）
curl "http://127.0.0.1:5002/api/fault/aqmOffline?taskId=test1&videoPath=D:/xxx.mp4"

# 自动标注工具：无命令行参数，直接改脚本 __main__ 中的配置；需在脚本目录运行且项目根在 PYTHONPATH
cd det_to_xml && PYTHONPATH=.. python detecte_demo.py
cd seg_to_json && PYTHONPATH=.. python segment_demo.py
```

## 架构

两套互相独立的功能，共用 `base_detector/` + YOLOv5 源码（`models/`、`utils/`、`export.py`，均为 Ultralytics 原版，一般不改）。

**1. 安全检测 HTTP 服务**
- `safety.py` 是总入口，用 `multiprocessing` + `subprocess`（`sys.executable`）分别拉起各 `safety_*.py`。目前只启用了 `safety_aqm_offline.py`，其余已注释。
- 每个 `safety_*.py` 是独立 Flask app：解析 GET 参数 → `check_video_stream` 校验视频 → `GPUtil` 检查显存（≤2000MB 拒绝）→ 起 `multiprocessing.Process` 后台执行 `service/<name>.py` 中的业务类 → 立即返回 `{"code":..., "msg":...}`。
- 服务地址、上传回调 URL、模型参数都读自 `config/address_configuration.json`（相对路径，须在项目根启动）。全局 URL 变量在 `__main__` 中赋值，所以必须通过 `python safety_xxx.py` 启动。
- 业务类完成后通过 `util.py` 的 `upfile_img` / `upfile_mp4` / `upfile_msg` 把截图、视频、结果回调给外部平台；当前配置中的 `127.0.0.1:8080` 是占位地址，上传失败不影响本地结果。
- **只有 `service/aqm_offline.py`（`Helmet_offline`）存在**，且是按推测重写的最小实现（`vest_helmet.pt`，类别 helmet/nohelmet/reflective/noreflectivez，`nohelmet` 触发告警）。`aqm_online`、`count`、`danger`、`process` 的 service 模块缺失，启用对应服务会 ImportError。

**2. 自动标注工具**
- `det_to_xml/`：`yolo_detector` 推理 → VOC XML（支持 `KEEP_CLASSES` 过滤）。
- `seg_to_json/`：`yolo_seg_detector` 推理 → LabelMe JSON 多边形。
- 模型名 → 权重/阈值/尺寸映射在各自的 `weights/model_config.json`，路径相对于脚本所在目录。

**检测器接口**：`yolo_detector.detect(img)` 返回 `{'bboxs': [[x1,y1,x2,y2],...], 'labels': [cls_id,...], 'confs': [...]}`；类别名从 `detector.model.names` 取。`IS_NET_detector` 走 onnxruntime。

## 已知坑

- 部分 `.pt` 权重是在 NumPy 2.x 下保存的，而 yolov5 环境是 NumPy 1.26，加载会报 `No module named 'numpy._core...'`。需在加载前注入 `sys.modules['numpy._core*']` 映射（见 `service/aqm_offline.py`、`det_to_xml/detecte_demo.py`）。环境里残留空的 `numpy/_core` 目录，所以要按 `np.__version__` 判断，不能用 `hasattr(np, '_core')`。
- Linux 训练的权重在 Windows 加载需 `pathlib.PosixPath = pathlib.WindowsPath`；`det_to_xml/fix_model.py` 用于处理这类权重。
- `det_to_xml/weights/model_config.json` 中 `vest_helmet` 的 `img_size` 为 128（疑似笔误），服务端用的是 config 中的 640。
- `util.Logger` 同时输出到文件和控制台，实际运行时控制台每条日志会打印两次（原因未查明）；日志文件为 UTF-8，控制台中文可能显示为乱码。

## Git

- 权重文件（`*.pt`、`*.onnx` 等）、`result/`、`log/`、`.idea/` 已在 `.gitignore` 中。
- `.continue/` 内含明文 API Key，必须保持忽略。
