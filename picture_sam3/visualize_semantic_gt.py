from pathlib import Path

import cv2
import numpy as np


IMAGE_ROOT = Path(
    "/home/via/mai/datasets/images"
)

LABEL_ROOT = Path(
    "/home/via/mai/datasets/"
    "sam3_benchmark_v1/cvat_roundtrip_test10/"
    "gt_semantic"
)

OUTPUT_ROOT = Path(
    "/home/via/mai/datasets/"
    "sam3_benchmark_v1/cvat_roundtrip_test10/"
    "gt_semantic_vis"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


# BGR颜色
COLORS = {
    0:   (0, 0, 0),          # background
    1:   (255, 160, 0),      # road
    2:   (0, 165, 255),      # construction vehicle
    3:   (0, 255, 255),      # truck
    4:   (255, 0, 0),        # car
    5:   (255, 0, 255),      # bus
    6:   (255, 255, 0),      # motorcycle
    7:   (160, 255, 0),      # bicycle
    8:   (0, 100, 255),      # rider
    9:   (0, 0, 255),        # pedestrian
    255: (255, 0, 255),      # ignore/conflict，紫色
}


for label_path in sorted(
    LABEL_ROOT.rglob("*.png")
):
    relative_path = label_path.relative_to(
        LABEL_ROOT
    )

    label = cv2.imread(
        str(label_path),
        cv2.IMREAD_UNCHANGED,
    )

    if label is None:
        print("[Error] 无法读取:", label_path)
        continue

    height, width = label.shape

    color_label = np.zeros(
        (height, width, 3),
        dtype=np.uint8,
    )

    for class_id, color in COLORS.items():
        color_label[label == class_id] = color

    output_path = OUTPUT_ROOT / relative_path
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cv2.imwrite(
        str(output_path),
        color_label,
    )

    print(
        "[OK]",
        relative_path,
        "values=",
        np.unique(label).tolist(),
    )

print("输出目录:", OUTPUT_ROOT)