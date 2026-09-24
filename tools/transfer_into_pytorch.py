# darknet53.conv.74 转换为 PyTorch 格式
import torch
import torch.nn as nn
from collections import OrderedDict
from models.darknet_yolov3_cbam import Darknet53_CBAM
import numpy as np

WEIGHTS_PATH = "weight"           # 原始 Darknet 权重
SAVE_PATH = "pytorch_weight"              # 转换后的 PyTorch 权重

def load_darknet_weights(model, weights_path):

    with open(weights_path, "rb") as f:
        header = np.fromfile(f, dtype=np.int32, count=5)
        buf = np.fromfile(f, dtype=np.float32)

    ptr = 0
    state_dict = model.state_dict()

    for key in state_dict.keys():
        if "num_batches_tracked" in key:
            continue

        # 卷积层
        if "conv" in key and "weight" in key:
            conv_shape = state_dict[key].shape
            num_params = state_dict[key].numel()

            state_dict[key].copy_(torch.from_numpy(buf[ptr:ptr+num_params]).view(conv_shape))
            ptr += num_params

        # BN 层
        elif "bn" in key:
            # bn.weight, bn.bias, bn.running_mean, bn.running_var
            for attr in ["weight", "bias", "running_mean", "running_var"]:
                bn_key = key.replace("weight", attr)
                if bn_key in state_dict:
                    num_params = state_dict[bn_key].numel()
                    shape = state_dict[bn_key].shape
                    state_dict[bn_key].copy_(torch.from_numpy(buf[ptr:ptr+num_params]).view(shape))
                    ptr += num_params

    print(f"Darknet weights loaded: {ptr}/{buf.size} parameters used.")
    return state_dict


if __name__ == "__main__":
    print("构建 Darknet53 模型（带 CBAM）...")
    model = Darknet53_CBAM()

    print("加载 Darknet conv.74 权重...")
    state_dict = load_darknet_weights(model, WEIGHTS_PATH)

    print("保存为 PyTorch 格式...")
    torch.save(state_dict, SAVE_PATH)

    print(f"转换成功！PyTorch checkpoint 保存到: {SAVE_PATH}")