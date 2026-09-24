import torch
import torch.nn as nn

class YOLOLossDarknet(nn.Module):
    def __init__(self, num_classes=1, img_size=416):
        super().__init__()

        # 使用聚类得到的 9 个 Anchor，并按小-中-大分 3 组
        anchors = torch.tensor([
            [[11.20, 54.62], [14.20, 69.29], [17.66, 86.16]],
            [[21.87, 106.65], [26.33, 128.45], [32.10, 156.53]],
            [[38.87, 189.57], [46.49, 226.86], [62.91, 306.77]],
        ])
        
        # COCO Anchor 
        #anchors = torch.tensor([
        #    [[10,13], [16,30], [33,23]],
        #    [[30,61], [62,45], [59,119]],
        #    [[116,90], [156,198], [373,326]]
        #])

        # 注册为 buffer，自动随着模型迁移设备
        self.register_buffer("anchors", anchors.float())

        self.num_classes = num_classes
        self.img_size = img_size
        self.bce = nn.BCELoss(reduction="sum")
        self.mse = nn.MSELoss(reduction="sum")

    def forward(self, outputs, targets):
        """
        outputs: list[ 3 个尺度输出 ]
            每个元素形状: (B, 3*(5+C), S, S)
        targets: 长度为 B 的 list, 每个元素是 (N_gt, 5) = (cls, x, y, w, h) 归一化坐标
        """
        device = outputs[0].device
        B = outputs[0].shape[0]
        total_loss = 0.0

        for scale_id, pred in enumerate(outputs):
            # pred: (B, 3*(5+C), S, S)
            B, _, S, _ = pred.shape

            anchors = self.anchors[scale_id].to(device)   # pixel units！

            # 重新 reshape 为 (B,3,5+C,S,S)
            pred = pred.view(B, 3, 5 + self.num_classes, S, S)

            # 分解各个输出分量
            tx    = torch.sigmoid(pred[:, :, 0])          # (B,3,S,S)
            ty    = torch.sigmoid(pred[:, :, 1])
            tw    = pred[:, :, 2]
            th    = pred[:, :, 3]
            pconf = torch.sigmoid(pred[:, :, 4])          # (B,3,S,S)

            # 这里先得到 (B,3,C,S,S)，然后 permute 成 (B,3,S,S,C)
            pcls = torch.sigmoid(pred[:, :, 5:])          # (B,3,C,S,S)
            pcls = pcls.permute(0, 1, 3, 4, 2).contiguous()  # (B,3,S,S,C)

            # 构造 target 容器
            obj_mask   = torch.zeros(B, 3, S, S, device=device)
            noobj_mask = torch.ones (B, 3, S, S, device=device)

            tx_t  = torch.zeros_like(tx)
            ty_t  = torch.zeros_like(ty)
            tw_t  = torch.zeros_like(tw)
            th_t  = torch.zeros_like(th)
            tconf = torch.zeros_like(pconf)
            tcls  = torch.zeros_like(pcls)                # (B,3,S,S,C)

            # 遍历 batch, 为每个 GT 分配 anchor & cell
            for b in range(B):
                gt_tensor = targets[b]
                if gt_tensor.numel() == 0:
                    continue
            
                gt_tensor = gt_tensor.to(device)
            
                for gt in gt_tensor:
                    _, gx, gy, gw, gh = gt.tolist()
                    cls = 0
            
                    # 映射到当前特征图坐标
                    gx_s = gx * S
                    gy_s = gy * S
                    gi = min(max(int(gx_s), 0), S - 1)
                    gj = min(max(int(gy_s), 0), S - 1)
            
                    # ========= 关键：使用 pixel 宽高 =========
                    gw_pixel = gw * self.img_size
                    gh_pixel = gh * self.img_size
            
                    # 当前尺度的 anchors 全是 pixel 单位
                    anchors_pixel = self.anchors[scale_id]
            
                    # 计算与三个 anchor 的 IoU（pixel 空间）
                    gt_wh_pixel = torch.tensor([gw_pixel, gh_pixel], device=device)
                    ious = self.anchor_iou(anchors_pixel, gt_wh_pixel)   # (3,)
                    ai = torch.argmax(ious).item()
            
                    obj_mask[b, ai, gj, gi] = 1.0
                    noobj_mask[b, ai, gj, gi] = 0.0
            
                    # 中心偏移
                    tx_t[b, ai, gj, gi] = gx_s - gi
                    ty_t[b, ai, gj, gi] = gy_s - gj
            
                    # ========= 正确的 tw/th（pixel 空间）=========
                    tw_t[b, ai, gj, gi] = torch.log(gw_pixel / anchors_pixel[ai, 0] + 1e-16)
                    th_t[b, ai, gj, gi] = torch.log(gh_pixel / anchors_pixel[ai, 1] + 1e-16)
            
                    # 置信度、类别
                    tconf[b, ai, gj, gi] = 1.0
                    tcls[b, ai, gj, gi, cls] = 1.0

                # 构造 ignore 区域：与任意 GT IoU > 0.5 的预测不算负样本
                pred_boxes = self.decode_pred(tx[b], ty[b], tw[b], th[b], anchors, S)
                gt_boxes   = self.targets_to_xyxy(gt_tensor, S)

                if gt_boxes.numel() > 0:
                    ious_full = self.bbox_iou_matrix(pred_boxes, gt_boxes)
                    max_iou   = ious_full.max(dim=1)[0]
                    ignore    = max_iou > 0.5
                    noobj_mask[b].view(-1)[ignore] = 0.0

            # 坐标损失
            loss_xy = self.bce(tx[obj_mask == 1], tx_t[obj_mask == 1]) + \
                      self.bce(ty[obj_mask == 1], ty_t[obj_mask == 1])

            loss_wh = self.mse(tw[obj_mask == 1], tw_t[obj_mask == 1]) + \
                      self.mse(th[obj_mask == 1], th_t[obj_mask == 1])

            # 置信度损失（正负样本）
            loss_conf = self.bce(pconf[obj_mask == 1],   tconf[obj_mask == 1]) + \
                        self.bce(pconf[noobj_mask == 1], tconf[noobj_mask == 1])

            # 分类损失（只有 1 类，其实就是学一个“是不是行人”的概率）
            loss_cls = self.bce(pcls[obj_mask == 1], tcls[obj_mask == 1])

            total_loss += (loss_xy + loss_wh + loss_conf + loss_cls)

        return total_loss / B

    # ----------------- 工具函数 ----------------- #

    def anchor_iou(self, anchors, gt_wh):
        """anchor (w,h) 和 GT (w,h) 的 IoU，仅用于选 best anchor"""
        w1, h1 = anchors[:, 0], anchors[:, 1]
        w2, h2 = gt_wh
        inter = torch.min(w1, w2) * torch.min(h1, h2)
        union = w1 * h1 + w2 * h2 - inter + 1e-6
        return inter / union

    def decode_pred(self, tx, ty, tw, th, anchors, S):
        stride = self.img_size / S
    
        g = torch.arange(S, device=tx.device)
        gy, gx = torch.meshgrid(g, g, indexing="ij")
    
        # center 转 pixel
        bx = (tx + gx.unsqueeze(0)) * stride
        by = (ty + gy.unsqueeze(0)) * stride
    
        # wh 转 pixel
        bw = anchors[:, 0].view(3,1,1) * torch.exp(tw)
        bh = anchors[:, 1].view(3,1,1) * torch.exp(th)
    
        # pixel → norm
        bx /= self.img_size
        by /= self.img_size
        bw /= self.img_size
        bh /= self.img_size
    
        x1 = bx - bw/2
        y1 = by - bh/2
        x2 = bx + bw/2
        y2 = by + bh/2

        return torch.stack([x1,y1,x2,y2],dim=-1).reshape(-1,4)
    
    def targets_to_xyxy(self, gts, S):
        """将 YOLO 格式 (x,y,w,h) 归一化坐标转为 (x1,y1,x2,y2)"""
        if gts.numel() == 0:
            return torch.zeros((0, 4), device=gts.device)
        x, y, w, h = gts[:, 1], gts[:, 2], gts[:, 3], gts[:, 4]
        return torch.stack([x - w / 2, y - h / 2,
                            x + w / 2, y + h / 2], dim=1)

    def bbox_iou_matrix(self, boxes1, boxes2):
        """两组 boxes 的 IoU 矩阵，用于 ignore 区域构造"""
        boxes2 = boxes2.to(boxes1.device)
        boxes1 = boxes1.unsqueeze(1)
        boxes2 = boxes2.unsqueeze(0)

        ix1 = torch.max(boxes1[..., 0], boxes2[..., 0])
        iy1 = torch.max(boxes1[..., 1], boxes2[..., 1])
        ix2 = torch.min(boxes1[..., 2], boxes2[..., 2])
        iy2 = torch.min(boxes1[..., 3], boxes2[..., 3])

        inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)
        area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

        union = area1 + area2 - inter + 1e-6
        return inter / union