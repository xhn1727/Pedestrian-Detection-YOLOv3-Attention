import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from dataset import CityPersonsTrain

DATA_ROOT = "CityPersons"
IMG_SIZE = 416
K = 9
BASE_SAVE_DIR = "results"


def load_wh_from_targets(root_dir):
    """
    CityPersonsTrain 返回的 target 是 YOLO 归一化格式 (x,y,w,h)
    这里需要将 w,h * 416，才能用于 anchor 聚类
    """

    train_set = CityPersonsTrain(root_dir, img_size=IMG_SIZE)
    ws, hs = [], []

    print(f"统计训练集：{len(train_set)} 张图")

    for img, targets, img_path in train_set:
        # targets: [N, 5] with (cls, x, y, w, h) normalized to 0~1
        if targets.numel() == 0:
            continue

        w_norm = targets[:, 3].numpy()
        h_norm = targets[:, 4].numpy()

        # 映射回 416 输入尺度
        ws.extend(w_norm * IMG_SIZE)
        hs.extend(h_norm * IMG_SIZE)

    ws = np.array(ws)
    hs = np.array(hs)
    
    print(f"参与聚类的目标框数: {len(ws)}")
    return ws, hs


def kmeans_anchors(ws_px, hs_px, k=9, img_size=416):
    """
    对 (w_px, h_px) 聚类，不进行任何额外缩放
    """

    data = np.stack([ws_px, hs_px], axis=1)

    # KMeans 聚类
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=20).fit(data)

    anchors = kmeans.cluster_centers_
    anchors = anchors[np.argsort(anchors[:, 0] * anchors[:, 1])]  # 按面积排序
    anchors = np.round(anchors, 2)

    print("\n最终 Anchor：")
    for w, h in anchors:
        print(f"[{w:.1f}, {h:.1f}], ratio={h/w:.2f}")

    np.savetxt(SAVE_PATH, anchors, fmt="%.2f", delimiter=",")
    print(f"\nAnchor 已保存至: {SAVE_PATH}")

    # 可视化
    plt.figure(figsize=(6,6))
    plt.scatter(ws_px, hs_px, s=4, alpha=0.3)
    plt.scatter(anchors[:, 0], anchors[:, 1], c='red', s=80)
    plt.title("CityPersons Anchors (416×416)")
    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.savefig("./anchors_citypersons_plot.png", dpi=150)

    return anchors


if __name__ == "__main__":
    ws_px, hs_px = load_wh_from_targets(DATA_ROOT)
    anchors = kmeans_anchors(ws_px, hs_px, k=K, img_size=IMG_SIZE)