import numpy as np
from PIL import Image
import cv2


def make_document(words, width=1080, height=1920) -> Document:
    image = Image.fromarray(
        np.random.normal(230, 6, (height, width, 3)).astype(np.uint8)
    )
    x, y = 0, 0
    for word in words:
        patch = word["patch"]
        size = word["size"]
        m = word["margin"]

        if x + size[0] > width:
            x = 0
            y += size[1]

        vs = (x + m, y + m, x + size[0] - m, y + size[1] - m)
        word["bbox"] = [vs[0], vs[1], vs[2], vs[1], vs[2], vs[3], vs[0], vs[3]]
        image.paste(patch, (x, y))
        x += size[0]

    return {"image": image, "words": words}


def pad_document_inplace(document: Document, pad=50, color=None) -> Document:
    if color is None:
        color = [64, 64, 64]
    image = cv2.copyMakeBorder(
        np.array(document["image"]),
        pad,
        pad,
        pad,
        pad,
        cv2.BORDER_CONSTANT,
        value=[64, 64, 64],
    )
    document["image"] = Image.fromarray(image)
    for word in document["words"]:
        word["bbox"] = [v + pad for v in word["bbox"]]
    return document


def partial_copy(document: Document) -> Document:
    image = document["image"].copy()
    words = [word.copy() for word in document["words"]]
    for word in words:
        word["bbox"] = word["bbox"].copy()
    return {"image": image, "words": words}