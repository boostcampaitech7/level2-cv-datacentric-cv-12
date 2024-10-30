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
from numba import njit

@njit
def cal_distance(x1, y1, x2, y2):
    '''유클리드 거리 계산
    
    입력:
        x1, y1: 첫 번째 점의 좌표
        x2, y2: 두 번째 점의 좌표
    출력:
        두 점 사이의 거리
    '''
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

@njit
def move_points(vertices, index1, index2, r, coef):
    '''두 점을 이동시켜 엣지를 축소
    
    입력:
        vertices: 텍스트 영역의 꼭지점들 <numpy.ndarray, (8,)>
        index1  : 첫 번째 점의 인덱스 오프셋
        index2  : 두 번째 점의 인덱스 오프셋
        r       : 논문에서의 [r1, r2, r3, r4]
        coef    : 논문에서의 축소 비율
    출력:
        축소된 엣지를 반영한 꼭지점들 <numpy.ndarray, (8,)>
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

@njit
def shrink_poly(vertices, coef=0.3):
    '''텍스트 영역을 축소
    
    입력:
        vertices: 텍스트 영역의 꼭지점들 <numpy.ndarray, (8,)>
        coef    : 축소 비율 (기본값: 0.3)
    출력:
        축소된 텍스트 영역의 꼭지점들 <numpy.ndarray, (8,)>
    '''
    x1, y1, x2, y2, x3, y3, x4, y4 = vertices
    r1 = min(cal_distance(x1,y1,x2,y2), cal_distance(x1,y1,x4,y4))
    r2 = min(cal_distance(x2,y2,x1,y1), cal_distance(x2,y2,x3,y3))
    r3 = min(cal_distance(x3,y3,x2,y2), cal_distance(x3,y3,x4,y4))
    r4 = min(cal_distance(x4,y4,x1,y1), cal_distance(x4,y4,x3,y3))
    r = [r1, r2, r3, r4]

    # move_points()를 자동으로 수행하기 위한 오프셋 결정
    if cal_distance(x1,y1,x2,y2) + cal_distance(x3,y3,x4,y4) > \
       cal_distance(x2,y2,x3,y3) + cal_distance(x1,y1,x4,y4):
        offset = 0 # 두 긴 엣지는 (x1y1-x2y2) & (x3y3-x4y4)
    else:
        offset = 1 # 두 긴 엣지는 (x2y2-x3y3) & (x4y4-x1y1)

    v = vertices.copy()
    v = move_points(v, 0 + offset, 1 + offset, r, coef)
    v = move_points(v, 2 + offset, 3 + offset, r, coef)
    v = move_points(v, 1 + offset, 2 + offset, r, coef)
    v = move_points(v, 3 + offset, 4 + offset, r, coef)
    return v

@njit
def get_rotate_mat(theta):
    '''회전 행렬 생성 (양수는 시계 방향 회전)
    
    입력:
        theta: 회전 각도 (라디안)
    출력:
        2x2 회전 행렬
    '''
    return np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])

def rotate_vertices(vertices, theta, anchor=None):
    '''앵커를 중심으로 꼭지점 회전
    
    입력:
        vertices: 텍스트 영역의 꼭지점들 <numpy.ndarray, (8,)>
        theta   : 회전 각도 (라디안)
        anchor  : 회전 시 고정 위치 (기본값: None, 이미지 중심)
    출력:
        회전된 꼭지점들 <numpy.ndarray, (8,)>
    '''
    v = vertices.reshape((4,2)).T
    if anchor is None:
        anchor = v[:,:1]
    rotate_mat = get_rotate_mat(theta)
    res = np.dot(rotate_mat, v - anchor)
    return (res + anchor).T.reshape(-1)

@njit
def get_boundary(vertices):
    '''주어진 꼭지점을 포함하는 최소 경계 박스 계산
    
    입력:
        vertices: 텍스트 영역의 꼭지점들 <numpy.ndarray, (8,)>
    출력:
        경계 박스의 최소 및 최대 x, y 값
    '''
    x1, y1, x2, y2, x3, y3, x4, y4 = vertices
    x_min = min(x1, x2, x3, x4)
    x_max = max(x1, x2, x3, x4)
    y_min = min(y1, y2, y3, y4)
    y_max = max(y1, y2, y3, y4)
    return x_min, x_max, y_min, y_max

@njit
def cal_error(vertices):
    '''기본 방향과의 차이 계산 (x1y1: 좌상단, x2y2: 우상단, x3y3: 우하단, x4y4: 좌하단)
    
    입력:
        vertices: 텍스트 영역의 꼭지점들 <numpy.ndarray, (8,)>
    출력:
        방향 차이 측정값
    '''
    x_min, x_max, y_min, y_max = get_boundary(vertices)
    x1, y1, x2, y2, x3, y3, x4, y4 = vertices
    err = cal_distance(x1, y1, x_min, y_min) + cal_distance(x2, y2, x_max, y_min) + \
          cal_distance(x3, y3, x_max, y_max) + cal_distance(x4, y4, x_min, y_max)
    return err

@njit
def find_min_rect_angle(vertices):
    '''최소 영역을 가지는 회전 각도 찾기
    
    입력:
        vertices: 텍스트 영역의 꼭지점들 <numpy.ndarray, (8,)>
    출력:
        최적의 회전 각도 (라디안)
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
    # 올바른 방향을 가진 최적의 각도 찾기
    for index in sorted_area_index[:rank_num]:
        rotated = rotate_vertices(vertices, angle_list[index] / 180 * math.pi)
        temp_error = cal_error(rotated)
        if temp_error < min_error:
            min_error = temp_error
            best_index = index
    return angle_list[best_index] / 180 * math.pi

def is_cross_text(start_loc, length, vertices):
    '''크롭 이미지가 텍스트 영역과 겹치는지 확인
    
    입력:
        start_loc: 크롭 시작 위치 (좌상단 좌표)
        length   : 크롭 이미지의 길이
        vertices : 텍스트 영역의 꼭지점들 <numpy.ndarray, (n,8)>
    출력:
        텍스트 영역과 겹치면 True, 아니면 False
    '''
    if vertices.size == 0:
        return False
    start_w, start_h = start_loc
    # 크롭 영역의 꼭지점 정의
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
    '''이미지 패치를 크롭하여 배치 및 증강 수행
    
    입력:
        img         : PIL 이미지
        vertices    : 텍스트 영역의 꼭지점들 <numpy.ndarray, (n,8)>
        labels      : 1->유효, 0->무시 <numpy.ndarray, (n,)>
        length      : 크롭할 이미지 영역의 길이
    출력:
        region      : 크롭된 이미지 영역
        new_vertices: 크롭된 영역 내의 새로운 꼭지점들
    '''
    h, w = img.height, img.width
    # 이미지의 가장 짧은 변이 크롭 길이보다 작으면 크기 조정
    if h >= w and w < length:
        img = img.resize((length, int(h * length / w)), Image.BILINEAR)
    elif h < w and h < length:
        img = img.resize((int(w * length / h), length), Image.BILINEAR)
    ratio_w = img.width / w
    ratio_h = img.height / h
    assert(ratio_w >= 1 and ratio_h >= 1)

    new_vertices = np.zeros(vertices.shape)
    if vertices.size > 0:
        new_vertices[:,[0,2,4,6]] = vertices[:,[0,2,4,6]] * ratio_w
        new_vertices[:,[1,3,5,7]] = vertices[:,[1,3,5,7]] * ratio_h

    # 랜덤 위치 찾기
    remain_h = img.height - length
    remain_w = img.width - length
    flag = True
    cnt = 0
    while flag and cnt < 1000:
        cnt += 1
        start_w = int(np.random.rand() * remain_w)
        start_h = int(np.random.rand() * remain_h)
        flag = is_cross_text([start_w, start_h], length, new_vertices[labels==1,:])
    box = (start_w, start_h, start_w + length, start_h + length)
    region = img.crop(box)
    if new_vertices.size == 0:
        return region, new_vertices

    new_vertices[:,[0,2,4,6]] -= start_w
    new_vertices[:,[1,3,5,7]] -= start_h
    return region, new_vertices

@njit
def rotate_all_pixels(rotate_mat, anchor_x, anchor_y, length):
    '''모든 픽셀의 회전된 위치 계산
    
    입력:
        rotate_mat: 회전 행렬
        anchor_x  : 고정된 x 위치
        anchor_y  : 고정된 y 위치
        length    : 이미지의 길이
    출력:
        rotated_x : 회전된 x 좌표들 <numpy.ndarray, (length,length)>
        rotated_y : 회전된 y 좌표들 <numpy.ndarray, (length,length)>
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
    '''이미지 크기 조정
    
    입력:
        img      : PIL 이미지
        vertices : 텍스트 영역의 꼭지점들 <numpy.ndarray, (n,8)>
        size     : 조정할 최대 크기
    출력:
        img         : 크기 조정된 PIL 이미지
        new_vertices: 크기 조정된 꼭지점들
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
    '''데이터 증강을 위한 이미지 높이 조정
    
    입력:
        img         : PIL 이미지
        vertices    : 텍스트 영역의 꼭지점들 <numpy.ndarray, (n,8)>
        ratio       : 높이 변경 비율 [0.8, 1.2] 범위
    출력:
        img         : 높이가 조정된 PIL 이미지
        new_vertices: 높이가 조정된 꼭지점들
    '''
    ratio_h = 1 + ratio * (np.random.rand() * 2 - 1)
    old_h = img.height
    new_h = int(np.around(old_h * ratio_h))
    img = img.resize((img.width, new_h), Image.BILINEAR)

    new_vertices = vertices.copy()
    if vertices.size > 0:
        new_vertices[:,[1,3,5,7]] = vertices[:,[1,3,5,7]] * (new_h / old_h)
    return img, new_vertices

def rotate_img(img, vertices, angle_range=10):
    '''이미지 회전하여 데이터 증강
    
    입력:
        img         : PIL 이미지
        vertices    : 텍스트 영역의 꼭지점들 <numpy.ndarray, (n,8)>
        angle_range : 회전 범위 (기본값: ±10도)
    출력:
        img         : 회전된 PIL 이미지
        new_vertices: 회전된 꼭지점들
    '''
    center_x = (img.width - 1) / 2
    center_y = (img.height - 1) / 2
    angle = angle_range * (np.random.rand() * 2 - 1)
    img = img.rotate(angle, Image.BILINEAR)
    new_vertices = np.zeros(vertices.shape)
    for i, vertice in enumerate(vertices):
        new_vertices[i,:] = rotate_vertices(vertice, -angle / 180 * math.pi, np.array([[center_x],[center_y]]))
    return img, new_vertices

def generate_roi_mask(image, vertices, labels):
    '''ROI 마스크 생성
    
    입력:
        image    : 이미지 배열
        vertices : 텍스트 영역의 꼭지점들 <numpy.ndarray, (n,8)>
        labels   : 각 텍스트 영역의 레이블 (1: 유효, 0: 무시)
    출력:
        마스크 이미지 (무시할 영역은 0, 나머지는 1)
    '''
    mask = np.ones(image.shape[:2], dtype=np.float32)
    ignored_polys = []
    for vertice, label in zip(vertices, labels):
        if label == 0:
            ignored_polys.append(np.around(vertice.reshape((4, 2))).astype(np.int32))
    cv2.fillPoly(mask, ignored_polys, 0)
    return mask

def filter_vertices(vertices, labels, ignore_under=0, drop_under=0):
    '''텍스트 영역 필터링
    
    입력:
        vertices        : 텍스트 영역의 꼭지점들 <numpy.ndarray, (n,8)>
        labels          : 각 텍스트 영역의 레이블 <numpy.ndarray, (n,)>
        ignore_under    : 무시할 최소 영역 크기
        drop_under      : 제거할 최소 영역 크기
    출력:
        필터링된 꼭지점들과 레이블들
    '''
    if drop_under == 0 and ignore_under == 0:
        return vertices, labels

    new_vertices, new_labels = vertices.copy(), labels.copy()

    areas = np.array([Polygon(v.reshape((4, 2))).convex_hull.area for v in vertices])
    labels[areas < ignore_under] = 0

    if drop_under > 0:
        passed = areas >= drop_under
        new_vertices, new_labels = new_vertices[passed], new_labels[passed]

    return new_vertices, new_labels

class SceneTextDataset(Dataset):
    """Scene Text Dataset 클래스
    
    데이터셋을 로드하고 전처리, 증강을 수행합니다.
    """
    def __init__(self, root_dir,
                 split='train',
                 image_size=2048,
                 crop_size=1024,
                 ignore_under_threshold=10,
                 drop_under_threshold=1,
                 color_jitter=True,
                 normalize=True):
        """
        초기화 함수

        입력:
            root_dir               : 데이터셋의 루트 디렉토리
            split                  : 데이터 분할 ('train', 'val', 'test' 등)
            image_size             : 이미지의 최대 크기
            crop_size              : 크롭할 이미지의 크기
            ignore_under_threshold : 무시할 최소 영역 크기
            drop_under_threshold   : 제거할 최소 영역 크기
            color_jitter           : 컬러 지터링 적용 여부
            normalize              : 정규화 적용 여부
        """
        self._lang_list = ['chinese', 'japanese', 'thai', 'vietnamese']
        self.root_dir = root_dir
        self.split = split
        total_anno = dict(images=dict())
        for nation in self._lang_list:
            anno_path = osp.join(root_dir, f'{nation}_receipt/ufo/{split}.json')
            with open(anno_path, 'r', encoding='utf-8') as f:
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
        '''파일 이름으로부터 이미지 디렉토리 추론
        
        입력:
            fname: 이미지 파일 이름
        출력:
            이미지가 저장된 디렉토리 경로
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
            raise ValueError("알 수 없는 언어 지시자")
        return osp.join(self.root_dir, f'{lang}_receipt', 'img', self.split)

    def __len__(self):
        '''데이터셋의 크기 반환'''
        return len(self.image_fnames)

    def __getitem__(self, idx):
        '''데이터셋의 특정 인덱스에 해당하는 데이터를 반환
        
        입력:
            idx: 데이터 인덱스
        출력:
            image      : 전처리된 이미지 배열
            word_bboxes: 단어의 바운딩 박스 <numpy.ndarray, (n,4,2)>
            roi_mask   : ROI 마스크
        '''
        image_fname = self.image_fnames[idx]
        image_fpath = osp.join(self._infer_dir(image_fname), image_fname)

        vertices, labels = [], []
        for word_info in self.anno['images'][image_fname]['words'].values():
            num_pts = np.array(word_info['points']).shape[0]
            if num_pts > 4:
                continue
            vertices.append(np.array(word_info['points']).flatten())
            labels.append(1)
        vertices, labels = np.array(vertices, dtype=np.float32), np.array(labels, dtype=np.int64)

        vertices, labels = filter_vertices(
            vertices,
            labels,
            ignore_under=self.ignore_under_threshold,
            drop_under=self.drop_under_threshold
        )

        image = Image.open(image_fpath)
        image, vertices = resize_img(image, vertices, self.image_size)
        image, vertices = adjust_height(image, vertices)
        image, vertices = rotate_img(image, vertices)
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

        image = transform(image=image)['image']
        word_bboxes = np.reshape(vertices, (-1, 4, 2))
        roi_mask = generate_roi_mask(image, vertices, labels)

        return image, word_bboxes, roi_mask

