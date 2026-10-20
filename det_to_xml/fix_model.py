import torch
import collections


def load_weights_only(model_path):
    """只加载模型权重，忽略其他元数据"""
    try:
        # 尝试正常加载
        checkpoint = torch.load(model_path, map_location='cpu')
    except NotImplementedError as e:
        if "PosixPath" in str(e):
            # 如果失败，使用更原始的方式
            print("检测到路径兼容性问题，尝试仅提取权重...")

            # 使用pickle的底层API
            import pickle
            import io

            with open(model_path, 'rb') as f:
                # 跳过PyTorch的包装层
                # PyTorch的.pt文件格式：首先是序列化器版本，然后是pickle数据
                f.read(4)  # 跳过魔术字
                unpickler = pickle.Unpickler(f)

                # 重写find_class来处理PosixPath
                def find_class(module, name):
                    if module == "pathlib" and name == "PosixPath":
                        import pathlib
                        return pathlib.WindowsPath
                    try:
                        return pickle.find_class(module, name)
                    except:
                        # 对于其他不兼容的类型，返回一个占位符
                        return collections.namedtuple('Dummy', [])

                unpickler.find_class = find_class
                checkpoint = unpickler.load()

    # 提取模型权重（假设checkpoint是标准的YOLOv5格式）
    if 'model' in checkpoint:
        return {'model': checkpoint['model']}
    else:
        # 如果不是标准格式，尝试直接返回
        return checkpoint


# 使用
ckpt = load_weights_only("weights/best_6cls.pt")