import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import DataLoader
from torchvision.ops import nms
from tqdm import tqdm
from dataset import CityPersonsTest
from models.darknet_yolov3 import YOLOv3
from models.darknet_yolov3_cbam import YOLOv3_CBAM
from models.darknet_yolov3_cbam_residual import YOLOv3_CBAM_RESIDUAL
from models.darknet_yolov3_eca_se_cbam import YOLOv3_ECA_SE_CBAM

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("使用设备:", DEVICE)

DATA_DIR = "CityPersons"
CHECKPOINT_PATH = "checkpoints/model.pth"
BASE_SAVE_DIR = "results"

os.makedirs(BASE_SAVE_DIR, exist_ok=True)

IMG_SIZE = 416
BATCH_SIZE = 4

ANCHORS = torch.tensor([
    [[11.20, 54.62], [14.20, 69.29], [17.66, 86.16]],
    [[21.87, 106.65], [26.33, 128.45], [32.10, 156.53]],
    [[38.87, 189.57], [46.49, 226.86], [62.91, 306.77]]
], device=DEVICE)


def yolo_collate_fn(batch):
    imgs = torch.stack([b[0] for b in batch], dim=0)
    targets = [b[1] for b in batch]
    return imgs, targets


def bbox_iou_xywh(box1, box2):
    if box1.numel() == 0 or box2.numel() == 0:
        return torch.zeros((box1.size(0), box2.size(0)), device=box1.device)

    b1_x1 = box1[:, 0] - box1[:, 2] / 2
    b1_y1 = box1[:, 1] - box1[:, 3] / 2
    b1_x2 = box1[:, 0] + box1[:, 2] / 2
    b1_y2 = box1[:, 1] + box1[:, 3] / 2

    b2_x1 = box2[:, 0] - box2[:, 2] / 2
    b2_y1 = box2[:, 1] - box2[:, 3] / 2
    b2_x2 = box2[:, 0] + box2[:, 2] / 2
    b2_y2 = box2[:, 1] + box2[:, 3] / 2

    inter_x1 = torch.max(b1_x1.unsqueeze(1), b2_x1.unsqueeze(0))
    inter_y1 = torch.max(b1_y1.unsqueeze(1), b2_y1.unsqueeze(0))
    inter_x2 = torch.min(b1_x2.unsqueeze(1), b2_x2.unsqueeze(0))
    inter_y2 = torch.min(b1_y2.unsqueeze(1), b2_y2.unsqueeze(0))

    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    return inter_area / (area1.unsqueeze(1) + area2.unsqueeze(0) - inter_area + 1e-6)


def decode_yolo_output(pred, anchors_pixels, img_size=416):
    B, C, S, _ = pred.shape
    pred = pred.view(B, 3, 6, S, S)

    tx = torch.sigmoid(pred[:, :, 0])
    ty = torch.sigmoid(pred[:, :, 1])
    tw = pred[:, :, 2]
    th = pred[:, :, 3]

    conf = torch.sigmoid(pred[:, :, 4])
    device = pred.device

    stride = img_size / S
    g = torch.arange(S, device=device)
    gy, gx = torch.meshgrid(g, g, indexing="ij")

    bx = (tx + gx) * stride / img_size
    by = (ty + gy) * stride / img_size
    bw = anchors_pixels[:, 0].view(3,1,1) * torch.exp(tw) / img_size
    bh = anchors_pixels[:, 1].view(3,1,1) * torch.exp(th) / img_size

    boxes = torch.stack([bx, by, bw, bh, conf], dim=-1)
    return boxes.view(B, -1, 5)


def visualize_prediction(img_tensor, gt_boxes, pred_boxes, conf, save_path):
    mean = torch.tensor([0.485, 0.456, 0.406], device=img_tensor.device).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=img_tensor.device).view(3,1,1)

    img = img_tensor * std + mean
    img = img.clamp(0,1).cpu().permute(1,2,0).numpy()

    plt.figure(figsize=(6,6))
    plt.imshow(img)
    ax = plt.gca()

    for gt in gt_boxes.cpu():
        x, y, w, h = gt * IMG_SIZE
        ax.add_patch(patches.Rectangle((x-w/2, y-h/2), w, h,
                       linewidth=2, edgecolor="lime", fill=False))

    for i, box in enumerate(pred_boxes.cpu()):
        x, y, w, h = box * IMG_SIZE
        ax.add_patch(patches.Rectangle((x-w/2, y-h/2), w, h,
                       linewidth=1.5, edgecolor="red", fill=False))
        ax.text(x-w/2, y-h/2-2, f"{float(conf[i]):.2f}",
                color="yellow", fontsize=8)

    plt.axis("off")
    plt.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0.1)
    plt.close()

def evaluate_model(conf_thresh, nms_thresh, save_root):
    dataset = CityPersonsTest(root=DATA_DIR, img_size=IMG_SIZE)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=yolo_collate_fn)

    model = YOLOv3_CBAM_RESIDUAL(num_classes=1).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()

    print(f"\n评估: CONF={conf_thresh}, NMS={nms_thresh}\n")

    vis_dir = os.path.join(save_root, "visuals")
    os.makedirs(vis_dir, exist_ok=True)

    total_gt = 0
    tps, fps, confs = [], [], []
    iou_list, conf_pred, iou_pred = [], [], []
    img_index = 0

    with torch.no_grad():
        for imgs, targets in tqdm(loader):
            imgs = imgs.to(DEVICE)
            preds = model(imgs)

            decoded = [decode_yolo_output(preds[i], ANCHORS[i], IMG_SIZE) for i in range(3)]
            decoded_boxes = torch.cat(decoded, dim=1)

            for b in range(imgs.size(0)):
                gt = targets[b][:, 1:].to(DEVICE)
                total_gt += len(gt)

                pred_boxes = decoded_boxes[b][:, :4]
                conf = decoded_boxes[b][:, 4]

                # ---- 阈值过滤 ----
                m = conf > conf_thresh
                pred_boxes, conf = pred_boxes[m], conf[m]

                if pred_boxes.numel() == 0:
                    continue

                # 转换为 xyxy
                xyxy = pred_boxes.clone()
                xyxy[:,0] = pred_boxes[:,0] - pred_boxes[:,2]/2
                xyxy[:,1] = pred_boxes[:,1] - pred_boxes[:,3]/2
                xyxy[:,2] = pred_boxes[:,0] + pred_boxes[:,2]/2
                xyxy[:,3] = pred_boxes[:,1] + pred_boxes[:,3]/2

                keep = nms(xyxy, conf, nms_thresh)
                pred_boxes, conf = pred_boxes[keep], conf[keep]

                if pred_boxes.numel() == 0:
                    continue

                # IoU
                ious = bbox_iou_xywh(pred_boxes, gt)
                max_iou_gt = ious.max(dim=0).values
                iou_list.extend(max_iou_gt.tolist())

                for i in range(pred_boxes.size(0)):
                    max_iou = ious[i].max().item()
                    conf_pred.append(conf[i].item())
                    iou_pred.append(max_iou)
                    if max_iou > 0.5:
                        tps.append(1)
                        fps.append(0)
                    else:
                        tps.append(0)
                        fps.append(1)
                    confs.append(conf[i].item())

                # 可视化
                save_path = os.path.join(vis_dir, f"vis_{img_index}.jpg")
                visualize_prediction(imgs[b], gt, pred_boxes, conf, save_path)
                img_index += 1

    confs = np.array(confs)
    tps = np.array(tps)
    fps = np.array(fps)

    order = np.argsort(-confs)
    tps, fps = tps[order], fps[order]

    cum_tp = np.cumsum(tps)
    cum_fp = np.cumsum(fps)

    recall = cum_tp / (total_gt + 1e-6)
    precision = cum_tp / (cum_tp + cum_fp + 1e-6)

    recall_points = np.linspace(0,1,11)
    ap = np.mean([ np.max(precision[recall>=r]) if np.any(recall>=r) else 0 for r in recall_points ])
    mean_iou = np.mean(iou_list)

    # ====== 保存图表 ======
    plt.figure(figsize=(6,5))
    plt.hist(iou_list, bins=30,edgecolor='black', linewidth=1.0)
    plt.title("IoU Distribution")
    plt.savefig(os.path.join(save_root, "iou_hist.png"))
    plt.close()

    plt.figure(figsize=(6,5))
    plt.plot(recall, precision)
    plt.title("Precision–Recall Curve")
    plt.savefig(os.path.join(save_root, "pr_curve.png"))
    plt.close()

    plt.figure(figsize=(6,5))
    plt.scatter(conf_pred, iou_pred, alpha=0.4)
    plt.title("Confidence vs IoU")
    plt.savefig(os.path.join(save_root, "conf_iou_scatter.png"))
    plt.close()

    return ap, mean_iou, precision[-1], recall[-1], np.sum(tps), np.sum(fps)

def sweep_thresholds():
    #conf_list = [0.10, 0.15, 0.20, 0.25, 0.30]
    nms_list  = [0.30, 0.45, 0.60]
    conf_list = [0.01, 0.05, 0.10]
    # conf_list = [0.2]
    # nms_list = [0.45]

    summary_path = os.path.join(BASE_SAVE_DIR, "metrics_summary.txt")
    f = open(summary_path, "w")
    f.write("Conf\tNMS\tmAP\tMeanIoU\tPrecision\tRecall\tTP\tFP\n")

    for conf in conf_list:
        for nms in nms_list:

            # 创建独立目录
            exp_dir = os.path.join(BASE_SAVE_DIR, f"conf_{conf}_nms_{nms}")
            os.makedirs(exp_dir, exist_ok=True)

            ap, miou, prec, rec, tp, fp = evaluate_model(conf, nms, exp_dir)

            f.write(f"{conf}\t{nms}\t{ap:.4f}\t{miou:.4f}\t{prec:.4f}\t{rec:.4f}\t{tp}\t{fp}\n")

    f.close()
    print("结果已写入 metrics_summary.txt")

if __name__ == "__main__":
    sweep_thresholds()