import os.path as osp
import math
import json
from PIL import Image

import torch
import numpy as np
import cv2
import albumentations as A
from torch.utils.data import Dataset
from shapely.geometry import Polygon


def cal_distance(x1, y1, x2, y2):
    '''
    유클리드 거리를 계산합니다.

    Args:
        x1 (float): 첫 번째 점의 x좌표.
        y1 (float): 첫 번째 점의 y좌표.
        x2 (float): 두 번째 점의 x좌표.
        y2 (float): 두 번째 점의 y좌표.

    Returns:
        float: 두 점 사이의 유클리드 거리.
    '''
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)


def move_points(vertices, index1, index2, r, coef):
    '''
    두 점을 이동시켜 가장자리를 축소합니다.

    Args:
        vertices (numpy.ndarray): 텍스트 영역의 꼭짓점들 (8,).
        index1 (int): 첫 번째 점의 인덱스 오프셋.
        index2 (int): 두 번째 점의 인덱스 오프셋.
        r (list): 각 가장자리의 길이를 나타내는 리스트 [r1, r2, r3, r4].
        coef (float): 축소 비율.

    Returns:
        numpy.ndarray: 축소된 텍스트 영역의 꼭짓점들.
    '''
    index1 = index1 % 4
    index2 = index2 % 4
    x1_index = index1 * 2 + 0
    y1_index = index1 * 2 + 1
    x2_index = index2 * 2 + 0
    y2_index = index2 * 2 + 1

    r1 = r[index1]
    r2 = r[index2]
    length_x = vertices[x1_index] - vertices[x2_index]
    length_y = vertices[y1_index] - vertices[y2_index]
    length = cal_distance(vertices[x1_index], vertices[y1_index], vertices[x2_index], vertices[y2_index])
    if length > 1:
        ratio = (r1 * coef) / length
        vertices[x1_index] += ratio * (-length_x)
        vertices[y1_index] += ratio * (-length_y)
        ratio = (r2 * coef) / length
        vertices[x2_index] += ratio * length_x
        vertices[y2_index] += ratio * length_y
    return vertices


def shrink_poly(vertices, coef=0.3):
    '''
    텍스트 영역을 축소합니다.

    Args:
        vertices (numpy.ndarray): 텍스트 영역의 꼭짓점들 (8,).
        coef (float, optional): 축소 비율. 기본값은 0.3.

    Returns:
        numpy.ndarray: 축소된 텍스트 영역의 꼭짓점들 (8,).
    '''
    x1, y1, x2, y2, x3, y3, x4, y4 = vertices
    r1 = min(cal_distance(x1, y1, x2, y2), cal_distance(x1, y1, x4, y4))
    r2 = min(cal_distance(x2, y2, x1, y1), cal_distance(x2, y2, x3, y3))
    r3 = min(cal_distance(x3, y3, x2, y2), cal_distance(x3, y3, x4, y4))
    r4 = min(cal_distance(x4, y4, x1, y1), cal_distance(x4, y4, x3, y3))
    r = [r1, r2, r3, r4]

    # 이동할 오프셋을 자동으로 얻기 위해 계산
    if cal_distance(x1, y1, x2, y2) + cal_distance(x3, y3, x4, y4) > \
       cal_distance(x2, y2, x3, y3) + cal_distance(x1, y1, x4, y4):
        offset = 0  # 두 개의 긴 가장자리는 (x1y1-x2y2) & (x3y3-x4y4)
    else:
        offset = 1  # 두 개의 긴 가장자리는 (x2y2-x3y3) & (x4y4-x1y1)

    v = vertices.copy()
    v = move_points(v, 0 + offset, 1 + offset, r, coef)
    v = move_points(v, 2 + offset, 3 + offset, r, coef)
    v = move_points(v, 1 + offset, 2 + offset, r, coef)
    v = move_points(v, 3 + offset, 4 + offset, r, coef)
    return v


def get_rotate_mat(theta):
    '''
    회전 매트릭스를 생성합니다. 양의 theta 값은 시계 방향 회전을 의미합니다.

    Args:
        theta (float): 회전 각도 (라디안 단위).

    Returns:
        numpy.ndarray: 2x2 회전 매트릭스.
    '''
    return np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])


def rotate_vertices(vertices, theta, anchor=None):
    '''
    지정된 앵커를 기준으로 꼭짓점들을 회전시킵니다.

    Args:
        vertices (numpy.ndarray): 텍스트 영역의 꼭짓점들 (8,).
        theta (float): 회전 각도 (라디안 단위). 양수는 시계 방향 회전.
        anchor (numpy.ndarray, optional): 회전의 기준점. 기본값은 None으로, 꼭짓점의 중심을 사용.

    Returns:
        numpy.ndarray: 회전된 꼭짓점들 (8,).
    '''
    v = vertices.reshape((4, 2)).T
    if anchor is None:
        anchor = v[:, :1]
    rotate_mat = get_rotate_mat(theta)
    res = np.dot(rotate_mat, v - anchor)
    return (res + anchor).T.reshape(-1)


def get_boundary(vertices):
    '''
    주어진 꼭짓점들 주변의 최소 경계를 계산합니다.

    Args:
        vertices (numpy.ndarray): 텍스트 영역의 꼭짓점들 (8,).

    Returns:
        tuple: (x_min, x_max, y_min, y_max).
    '''
    x1, y1, x2, y2, x3, y3, x4, y4 = vertices
    x_min = min(x1, x2, x3, x4)
    x_max = max(x1, x2, x3, x4)
    y_min = min(y1, y2, y3, y4)
    y_max = max(y1, y2, y3, y4)
    return x_min, x_max, y_min, y_max


def cal_error(vertices):
    '''
    기본 방향은 x1y1 : 좌상단, x2y2 : 우상단, x3y3 : 우하단, x4y4 : 좌하단입니다.
    꼭짓점들의 방향성과 기본 방향과의 차이를 계산합니다.

    Args:
        vertices (numpy.ndarray): 텍스트 영역의 꼭짓점들 (8,).

    Returns:
        float: 방향성 차이의 측정값.
    '''
    x_min, x_max, y_min, y_max = get_boundary(vertices)
    x1, y1, x2, y2, x3, y3, x4, y4 = vertices
    err = cal_distance(x1, y1, x_min, y_min) + cal_distance(x2, y2, x_max, y_min) + \
          cal_distance(x3, y3, x_max, y_max) + cal_distance(x4, y4, x_min, y_max)
    return err


def find_min_rect_angle(vertices):
    '''
    폴리곤을 회전시켜 최소 사각형을 얻기 위한 최적의 각도를 찾습니다.

    Args:
        vertices (numpy.ndarray): 텍스트 영역의 꼭짓점들 (8,).

    Returns:
        float: 최적의 회전 각도 (라디안 단위).
    '''
    angle_interval = 1
    angle_list = list(range(-90, 90, angle_interval))
    area_list = []
    for theta in angle_list:
        rotated = rotate_vertices(vertices, theta / 180 * math.pi)
        x1, y1, x2, y2, x3, y3, x4, y4 = rotated
        temp_area = (max(x1, x2, x3, x4) - min(x1, x2, x3, x4)) * \
                    (max(y1, y2, y3, y4) - min(y1, y2, y3, y4))
        area_list.append(temp_area)

    sorted_area_index = sorted(list(range(len(area_list))), key=lambda k: area_list[k])
    min_error = float('inf')
    best_index = -1
    rank_num = 10
    # 올바른 방향성을 가진 최적의 각도 찾기
    for index in sorted_area_index[:rank_num]:
        rotated = rotate_vertices(vertices, angle_list[index] / 180 * math.pi)
        temp_error = cal_error(rotated)
        if temp_error < min_error:
            min_error = temp_error
            best_index = index
    return angle_list[best_index] / 180 * math.pi


def is_cross_text(start_loc, length, vertices):
    '''
    잘린 이미지 영역이 텍스트 영역을 교차하는지 확인합니다.

    Args:
        start_loc (list or tuple): 잘라낼 이미지의 좌상단 위치 [x, y].
        length (int): 잘라낼 이미지의 한 변 길이.
        vertices (numpy.ndarray): 텍스트 영역들의 꼭짓점들 (n, 8).

    Returns:
        bool: 교차하면 True, 그렇지 않으면 False.
    '''
    if vertices.size == 0:
        return False
    start_w, start_h = start_loc
    # 잘라낼 이미지 영역의 꼭짓점들 정의
    a = np.array([start_w, start_h, start_w + length, start_h, start_w + length, start_h + length,
                  start_w, start_h + length]).reshape((4, 2))
    p1 = Polygon(a).convex_hull
    for vertice in vertices:
        p2 = Polygon(vertice.reshape((4, 2))).convex_hull
        inter = p1.intersection(p2).area
        if 0.01 <= inter / p2.area <= 0.99:
            return True
    return False


def crop_img(img, vertices, labels, length):
    '''
    이미지 패치를 잘라 배치하고 증강합니다.

    Args:
        img (PIL.Image): 원본 이미지.
        vertices (numpy.ndarray): 텍스트 영역의 꼭짓점들 (n, 8).
        labels (numpy.ndarray): 레이블 (1: 유효, 0: 무시) (n,).
        length (int): 잘라낼 이미지 영역의 길이.

    Returns:
        tuple:
            - PIL.Image: 잘라낸 이미지 영역.
            - numpy.ndarray: 잘라낸 이미지 영역 내의 새로운 꼭짓점들.
    '''
    h, w = img.height, img.width
    # 이미지의 가장 짧은 변이 잘라낼 길이보다 작은 경우 크기 조정
    if h >= w and w < length:
        img = img.resize((length, int(h * length / w)), Image.BILINEAR)
    elif h < w and h < length:
        img = img.resize((int(w * length / h), length), Image.BILINEAR)
    ratio_w = img.width / w
    ratio_h = img.height / h
    assert(ratio_w >= 1 and ratio_h >= 1)

    new_vertices = np.zeros(vertices.shape)
    if vertices.size > 0:
        new_vertices[:, [0, 2, 4, 6]] = vertices[:, [0, 2, 4, 6]] * ratio_w
        new_vertices[:, [1, 3, 5, 7]] = vertices[:, [1, 3, 5, 7]] * ratio_h

    # 랜덤한 위치 찾기
    remain_h = img.height - length
    remain_w = img.width - length
    flag = True
    cnt = 0
    while flag and cnt < 1000:
        cnt += 1
        start_w = int(np.random.rand() * remain_w)
        start_h = int(np.random.rand() * remain_h)
        flag = is_cross_text([start_w, start_h], length, new_vertices[labels == 1, :])
    box = (start_w, start_h, start_w + length, start_h + length)
    region = img.crop(box)
    if new_vertices.size == 0:
        return region, new_vertices

    new_vertices[:, [0, 2, 4, 6]] -= start_w
    new_vertices[:, [1, 3, 5, 7]] -= start_h
    return region, new_vertices


def rotate_all_pixels(rotate_mat, anchor_x, anchor_y, length):
    '''
    다음 단계에서 사용할 모든 픽셀의 회전된 위치를 얻습니다.

    Args:
        rotate_mat (numpy.ndarray): 회전 매트릭스.
        anchor_x (float): 회전의 기준점 x좌표.
        anchor_y (float): 회전의 기준점 y좌표.
        length (int): 이미지의 길이.

    Returns:
        tuple:
            - numpy.ndarray: 회전된 x 위치들 (length, length).
            - numpy.ndarray: 회전된 y 위치들 (length, length).
    '''
    x = np.arange(length)
    y = np.arange(length)
    x, y = np.meshgrid(x, y)
    x_lin = x.reshape((1, x.size))
    y_lin = y.reshape((1, y.size))
    coord_mat = np.concatenate((x_lin, y_lin), 0)
    rotated_coord = np.dot(rotate_mat, coord_mat - np.array([[anchor_x], [anchor_y]])) + \
                                                   np.array([[anchor_x], [anchor_y]])
    rotated_x = rotated_coord[0, :].reshape(x.shape)
    rotated_y = rotated_coord[1, :].reshape(y.shape)
    return rotated_x, rotated_y


def resize_img(img, vertices, size):
    '''
    이미지를 지정된 크기로 리사이즈하고 꼭짓점들을 조정합니다.

    Args:
        img (PIL.Image): 원본 이미지.
        vertices (numpy.ndarray): 텍스트 영역의 꼭짓점들 (n, 8).
        size (int): 리사이즈할 이미지의 최대 크기.

    Returns:
        tuple:
            - PIL.Image: 리사이즈된 이미지.
            - numpy.ndarray: 리사이즈된 이미지에 맞게 조정된 꼭짓점들.
    '''
    h, w = img.height, img.width
    ratio = size / max(h, w)
    if w > h:
        img = img.resize((size, int(h * ratio)), Image.BILINEAR)
    else:
        img = img.resize((int(w * ratio), size), Image.BILINEAR)
    new_vertices = vertices * ratio
    return img, new_vertices


def adjust_height(img, vertices, ratio=0.2):
    '''
    이미지의 높이를 조정하여 데이터 증강을 수행합니다.

    Args:
        img (PIL.Image): 원본 이미지.
        vertices (numpy.ndarray): 텍스트 영역의 꼭짓점들 (n, 8).
        ratio (float, optional): 높이 변경 비율. [0.8, 1.2] 범위 내에서 랜덤하게 변경. 기본값은 0.2.

    Returns:
        tuple:
            - PIL.Image: 높이가 조정된 이미지.
            - numpy.ndarray: 높이가 조정된 이미지에 맞게 조정된 꼭짓점들.
    '''
    ratio_h = 1 + ratio * (np.random.rand() * 2 - 1)
    old_h = img.height
    new_h = int(np.around(old_h * ratio_h))
    img = img.resize((img.width, new_h), Image.BILINEAR)

    new_vertices = vertices.copy()
    if vertices.size > 0:
        new_vertices[:, [1, 3, 5, 7]] = vertices[:, [1, 3, 5, 7]] * (new_h / old_h)
    return img, new_vertices


def rotate_img(img, vertices, angle_range=10):
    '''
    이미지를 회전시켜 데이터 증강을 수행합니다. [-10, 10]도 범위 내에서 랜덤하게 회전합니다.

    Args:
        img (PIL.Image): 원본 이미지.
        vertices (numpy.ndarray): 텍스트 영역의 꼭짓점들 (n, 8).
        angle_range (int, optional): 회전 범위 (도 단위). 기본값은 10.

    Returns:
        tuple:
            - PIL.Image: 회전된 이미지.
            - numpy.ndarray: 회전된 이미지에 맞게 조정된 꼭짓점들.
    '''
    center_x = (img.width - 1) / 2
    center_y = (img.height - 1) / 2
    angle = angle_range * (np.random.rand() * 2 - 1)  # -angle_range ~ angle_range
    img = img.rotate(angle, Image.BILINEAR)
    new_vertices = np.zeros(vertices.shape)
    for i, vertice in enumerate(vertices):
        new_vertices[i, :] = rotate_vertices(vertice, -angle / 180 * math.pi, np.array([[center_x], [center_y]]))
    return img, new_vertices


def generate_roi_mask(image, vertices, labels):
    '''
    잘라낸 이미지 영역에 대한 ROI 마스크를 생성합니다.

    Args:
        image (numpy.ndarray): 이미지 배열.
        vertices (numpy.ndarray): 텍스트 영역의 꼭짓점들 (n, 8).
        labels (numpy.ndarray): 레이블 (1: 유효, 0: 무시) (n,).

    Returns:
        numpy.ndarray: ROI 마스크 (행, 열), 무시할 영역은 0으로 설정.
    '''
    mask = np.ones(image.shape[:2], dtype=np.float32)
    ignored_polys = []
    for vertice, label in zip(vertices, labels):
        if label == 0:
            ignored_polys.append(np.around(vertice.reshape((4, 2))).astype(np.int32))
    # 무시할 영역을 0으로 채움
    cv2.fillPoly(mask, ignored_polys, 0)
    return mask


def filter_vertices(vertices, labels, ignore_under=0, drop_under=0):
    '''
    텍스트 영역의 크기에 따라 꼭짓점과 레이블을 필터링합니다.

    Args:
        vertices (numpy.ndarray): 텍스트 영역의 꼭짓점들 (n, 8).
        labels (numpy.ndarray): 레이블 (1: 유효, 0: 무시) (n,).
        ignore_under (int, optional): 이 면적 미만인 경우 레이블을 0으로 설정. 기본값은 0.
        drop_under (int, optional): 이 면적 미만인 경우 꼭짓점과 레이블을 제거. 기본값은 0.

    Returns:
        tuple:
            - numpy.ndarray: 필터링된 꼭짓점들.
            - numpy.ndarray: 필터링된 레이블들.
    '''
    if drop_under == 0 and ignore_under == 0:
        return vertices, labels

    new_vertices, new_labels = vertices.copy(), labels.copy()

    # 각 텍스트 영역의 면적 계산
    areas = np.array([Polygon(v.reshape((4, 2))).convex_hull.area for v in vertices])
    # ignore_under 미만인 경우 레이블을 0으로 설정
    labels[areas < ignore_under] = 0

    if drop_under > 0:
        # drop_under 미만인 영역을 제거
        passed = areas >= drop_under
        new_vertices, new_labels = new_vertices[passed], new_labels[passed]

    return new_vertices, new_labels


class SceneTextDataset(Dataset):
    '''
    장면 텍스트 감지를 위한 PyTorch 데이터셋 클래스.

    Attributes:
        _lang_list (list): 지원하는 언어 목록.
        root_dir (str): 데이터셋의 루트 디렉토리.
        split (str): 데이터셋의 분할 (예: 'train', 'val').
        anno (dict): 이미지와 단어에 대한 주석 정보.
        image_fnames (list): 이미지 파일 이름 리스트.
        image_size (int): 이미지 크기 조정 시 최대 크기.
        crop_size (int): 잘라낼 이미지 패치의 크기.
        color_jitter (bool): 컬러 지터링 사용 여부.
        normalize (bool): 정규화 사용 여부.
        drop_under_threshold (int): 이 면적 미만인 텍스트 영역을 제거.
        ignore_under_threshold (int): 이 면적 미만인 텍스트 영역을 무시.
    '''
    def __init__(self, root_dir,
                 split='train',
                 image_size=2048,
                 crop_size=1024,
                 ignore_under_threshold=10,
                 drop_under_threshold=1,
                 color_jitter=True,
                 normalize=True):
        '''
        데이터셋을 초기화합니다.

        Args:
            root_dir (str): 데이터셋의 루트 디렉토리.
            split (str, optional): 데이터셋의 분할 (예: 'train', 'val'). 기본값은 'train'.
            image_size (int, optional): 이미지 크기 조정 시 최대 크기. 기본값은 2048.
            crop_size (int, optional): 잘라낼 이미지 패치의 크기. 기본값은 1024.
            ignore_under_threshold (int, optional): 이 면적 미만인 텍스트 영역을 무시. 기본값은 10.
            drop_under_threshold (int, optional): 이 면적 미만인 텍스트 영역을 제거. 기본값은 1.
            color_jitter (bool, optional): 컬러 지터링 사용 여부. 기본값은 True.
            normalize (bool, optional): 정규화 사용 여부. 기본값은 True.
        '''
        self._lang_list = ['chinese', 'japanese', 'thai', 'vietnamese']
        self.root_dir = root_dir
        self.split = split
        total_anno = dict(images=dict())
        # 각 언어별로 주석 파일 로드
        for nation in self._lang_list:
            with open(osp.join(root_dir, '{}_receipt/ufo/{}.json'.format(nation, split)), 'r', encoding='utf-8') as f:
                anno = json.load(f)
            for im in anno['images']:
                total_anno['images'][im] = anno['images'][im]

        self.anno = total_anno
        self.image_fnames = sorted(self.anno['images'].keys())

        self.image_size, self.crop_size = image_size, crop_size
        self.color_jitter, self.normalize = color_jitter, normalize

        self.drop_under_threshold = drop_under_threshold
        self.ignore_under_threshold = ignore_under_threshold

    def _infer_dir(self, fname):
        '''
        이미지 파일 이름을 기반으로 해당 언어의 이미지 디렉토리를 추론합니다.

        Args:
            fname (str): 이미지 파일 이름.

        Returns:
            str: 해당 이미지 파일의 전체 경로.
        '''
        lang_indicator = fname.split('.')[1]
        if lang_indicator == 'zh':
            lang = 'chinese'
        elif lang_indicator == 'ja':
            lang = 'japanese'
        elif lang_indicator == 'th':
            lang = 'thai'
        elif lang_indicator == 'vi':
            lang = 'vietnamese'
        else:
            raise ValueError
        return osp.join(self.root_dir, f'{lang}_receipt', 'img', self.split)

    def __len__(self):
        '''
        데이터셋의 크기를 반환합니다.

        Returns:
            int: 데이터셋에 포함된 이미지의 수.
        '''
        return len(self.image_fnames)

    def __getitem__(self, idx):
        '''
        주어진 인덱스에 해당하는 데이터를 반환합니다.

        Args:
            idx (int): 데이터셋 내의 인덱스.

        Returns:
            tuple:
                - torch.Tensor: 전처리된 이미지 텐서.
                - numpy.ndarray: 텍스트 영역의 바운딩 박스들 (n, 4, 2).
                - numpy.ndarray: ROI 마스크 (행, 열).
        '''
        image_fname = self.image_fnames[idx]
        image_fpath = osp.join(self._infer_dir(image_fname), image_fname)

        vertices, labels = [], []
        # 해당 이미지의 모든 단어 정보에 대해 꼭짓점과 레이블 수집
        for word_info in self.anno['images'][image_fname]['words'].values():
            num_pts = np.array(word_info['points']).shape[0]
            if num_pts > 4:
                continue  # 꼭짓점이 4개를 초과하면 무시
            vertices.append(np.array(word_info['points']).flatten())
            labels.append(1)
        vertices, labels = np.array(vertices, dtype=np.float32), np.array(labels, dtype=np.int64)

        # 꼭짓점과 레이블 필터링
        vertices, labels = filter_vertices(
            vertices,
            labels,
            ignore_under=self.ignore_under_threshold,
            drop_under=self.drop_under_threshold
        )

        image = Image.open(image_fpath)
        # 이미지 리사이즈
        image, vertices = resize_img(image, vertices, self.image_size)
        # 이미지 높이 조정
        image, vertices = adjust_height(image, vertices)
        # 이미지 회전
        image, vertices = rotate_img(image, vertices)
        # 이미지 크롭
        image, vertices = crop_img(image, vertices, labels, self.crop_size)

        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = np.array(image)

        funcs = []
        if self.color_jitter:
            funcs.append(A.ColorJitter())
        if self.normalize:
            funcs.append(A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        transform = A.Compose(funcs)

        # 이미지 증강 및 정규화
        image = transform(image=image)['image']
        # 텍스트 영역의 바운딩 박스를 (n, 4, 2) 형태로 재구성
        word_bboxes = np.reshape(vertices, (-1, 4, 2))
        # ROI 마스크 생성
        roi_mask = generate_roi_mask(image, vertices, labels)

        return image, word_bboxes, roi_mask
