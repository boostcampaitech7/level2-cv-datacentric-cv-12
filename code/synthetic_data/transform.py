import numpy as np
import cv2
from PIL import Image

# Perspective 변환 함수
your_function = cv2.getPerspectiveTransform

# Bounding box 변환 함수
def your_vm_function(bbox, M):
    v = np.array(bbox).reshape(-1, 2).T
    v = np.vstack([v, np.ones((1, 4))])
    v = M @ v
    v = v[:2, :] / v[2, :]
    return v.T.flatten().tolist()

# 문서의 perspective 변환 함수
def perturb_document_inplace(document, pad=0, color=None):
    if color is None:
        color = [64, 64, 64]
    width, height = np.array(document["image"].size)
    magnitude_lb = 0
    magnitude_ub = 200
    src = np.array([[0, 0], [width, 0], [width, height], [0, height]], np.float32)
    perturb = np.random.uniform(magnitude_lb, magnitude_ub, (4, 2)) * np.array(
        [[1, 1], [-1, 1], [-1, -1], [1, -1]]
    )
    perturb = perturb.astype(np.float32)
    dst = src + perturb

    M = your_function(src, dst)
    out = cv2.warpPerspective(
        np.array(document["image"]),
        M,
        document["image"].size,
        flags=cv2.INTER_LINEAR,
        borderValue=color,
    )
    document["image"] = Image.fromarray(out)

    for word in document["words"]:
        word["bbox"] = your_vm_function(word["bbox"], M)

    return document
