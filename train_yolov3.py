import os
import torch
import torch.optim as optim
from datetime import datetime
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset import CityPersonsTrain, CityPersonsVal
from models.darknet_yolov3 import YOLOv3
from models.darknet_yolov3_cbam import YOLOv3_CBAM
from models.darknet_yolov3_cbam_residual import YOLOv3_CBAM_RESIDUAL
from models.darknet_yolov3_eca_se_cbam import YOLOv3_ECA_SE_CBAM
from loss_yolov3 import YOLOLossDarknet

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print("使用设备:", DEVICE)

DATA_DIR = "CityPersons"
SAVE_DIR = "checkpoints"

IMG_SIZE = 416
BATCH_SIZE = 4
EPOCHS = 50
LEARNING_RATE = 1e-4

os.makedirs(SAVE_DIR, exist_ok=True)
LOSS_LOG_PATH = os.path.join(SAVE_DIR, "loss_log.txt")

PRETRAINED_BACKBONE = "backbone"
USE_PRETRAINED = True

def yolo_collate(batch):
    imgs = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]   
    return imgs, targets

def train():
    train_dataset = CityPersonsTrain(DATA_DIR)
    val_dataset   = CityPersonsVal(DATA_DIR)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=yolo_collate, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=yolo_collate, num_workers=0
    )

    #if USE_PRETRAINED:
    #    print("加载 Darknet-53 预训练权重...")
    #    model = YOLOv3(num_classes=1, pretrained_backbone=PRETRAINED_BACKBONE).to(DEVICE)
    #else:
    #    model = YOLOv3(num_classes=1).to(DEVICE)

    if USE_PRETRAINED:
        print("加载 Darknet-53 预训练权重...")
    
        # 初始化带 CBAM 的 YOLOv3 模型
        model = YOLOv3_ECA_SE_CBAM(num_classes=1).to(DEVICE)
        # 加载预训练权重
        pretrained_weights = torch.load(PRETRAINED_BACKBONE, map_location=DEVICE, weights_only=False)
        # 部分加载（忽略 CBAM 新增层）
        missing, unexpected = model.load_state_dict(pretrained_weights, strict=False)
    
        print("预训练权重加载完成。")
        print("未匹配的（CBAM 新层）参数数量：", len(missing))
        print("未使用的参数数量：", len(unexpected))
    
    else:
        print("不使用预训练权重，随机初始化 YOLOv3...")
        model = YOLOv3_CBAM(num_classes=1).to(DEVICE)
        
    # 使用内置 anchors 的 YOLOLossDarknet
    criterion = YOLOLossDarknet(num_classes=1, img_size=416).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_losses, val_losses = [], []

    with open(LOSS_LOG_PATH, "w") as f_log:
        f_log.write(f"YOLOv3 Log - ECA & SE & CBAM {datetime.now()}\n")
        f_log.write("Epoch\tTrain_Loss\tVal_Loss\n")

        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0.0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

            for imgs, targets in pbar:
                imgs = imgs.to(DEVICE)

                # 一次 forward 整个 batch
                preds = model(imgs)

                # 计算 loss
                loss = criterion(preds, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})


            avg_train_loss = total_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            print(f"Epoch [{epoch+1}] TrainLoss = {avg_train_loss:.4f}")

            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for imgs, targets in val_loader:
                    imgs = imgs.to(DEVICE)
                    preds = model(imgs)
                    loss = criterion(preds, targets)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            print(f"Epoch [{epoch+1}] ValLoss = {avg_val_loss:.4f}")

            f_log.write(f"{epoch+1}\t{avg_train_loss:.6f}\t{avg_val_loss:.6f}\n")
            f_log.flush()

            ckpt = os.path.join(SAVE_DIR,
                f"yolov3_epoch{epoch+1}_{datetime.now().strftime('%H-%M-%S')}.pth")
            torch.save(model.state_dict(), ckpt)
            print(f"✔ 模型已保存：{ckpt}\n")

    plt.figure(figsize=(8,6))
    plt.plot(train_losses, label="TrainLoss")
    plt.plot(val_losses, label="ValLoss")
    plt.legend()
    plt.grid()
    plt.title("YOLOv3 COCO Anchoers Loss Curve")
    plt.savefig(os.path.join(SAVE_DIR, "loss_curve.png"))
    print("训练完成！")


if __name__ == "__main__":
    train()