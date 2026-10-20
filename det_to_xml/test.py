import torch
import pathlib
import pickle

class WindowsCompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "pathlib" and name == "PosixPath":
            return pathlib.WindowsPath
        return super().find_class(module, name)

# 尝试加载
ckpt = torch.load("weights/best_6cls_fixed.pt",
                  map_location='cpu',
                  pickle_module=WindowsCompatUnpickler)
print("加载成功！")