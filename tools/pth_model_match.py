from models.darknet_yolov3_cbam import YOLOv3_CBAM
import torch

CHECKPOINT_PATH = "checkpoints/model.pth"

model = YOLOv3_CBAM(num_classes=1)
ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")

missing, unexpected = model.load_state_dict(ckpt, strict=False)

print("缺失参数（权重文件里没有的）：", missing)
print("多余参数（模型里不存在的）：", unexpected)