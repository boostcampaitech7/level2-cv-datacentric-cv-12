import math

import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from numba import njit


@njit
def nb_meshgrid(x, y):
    """
    메쉬그리드 생성 함수 (Numba 최적화 버전).
    
    입력:
        x (1차원 배열): x축 값들
        y (1차원 배열): y축 값들
    
    출력:
        tuple: X와 Y 그리드 배열
    """
    # 행과 열의 수를 가져옴
    rows = len(y)
    cols = len(x)

    # 빈 배열 미리 할당
    X = np.empty((rows, cols), dtype=x.dtype)
    Y = np.empty((rows, cols), dtype=y.dtype)

    # X 배열 채우기: x 값을 행 단위로 반복
    for i in range(rows):
        for j in range(cols):
            X[i, j] = x[j]

    # Y 배열 채우기: y 값을 열 단위로 반복
    for i in range(rows):
        for j in range(cols):
            Y[i, j] = y[i]

    return X, Y


@njit
def nb_amin(arr, axis=0):
    """
    배열의 최소값을 찾는 함수 (Numba 최적화 버전).
    
    입력:
        arr (2차원 배열): 입력 배열
        axis (int): 축 (현재는 0만 지원)
    
    출력:
        1차원 배열: 각 열의 최소값
    """
    # 입력 배열의 행과 열 수를 확인
    rows, cols = arr.shape

    # 축이 0인 경우, 열별로 최소값을 찾음
    if axis == 0:
        # 최소값을 저장할 배열 초기화
        min_vals = np.empty(cols, dtype=arr.dtype)

        # 각 열을 순회하며 최소값 찾기
        for j in range(cols):
            # 첫 번째 요소를 초기 최소값으로 설정
            min_val = arr[0, j]

            # 나머지 요소들과 비교하여 최소값 업데이트
            for i in range(1, rows):
                if arr[i, j] < min_val:
                    min_val = arr[i, j]

            # 최소값 저장
            min_vals[j] = min_val

        return min_vals


@njit
def nb_amax(arr, axis=0):
    """
    배열의 최대값을 찾는 함수 (Numba 최적화 버전).
    
    입력:
        arr (2차원 배열): 입력 배열
        axis (int): 축 (현재는 0만 지원)
    
    출력:
        1차원 배열: 각 열의 최대값
    """
    # 입력 배열의 행과 열 수를 확인
    rows, cols = arr.shape

    # 축이 0인 경우, 열별로 최대값을 찾음
    if axis == 0:
        # 최대값을 저장할 배열 초기화
        max_vals = np.empty(cols, dtype=arr.dtype)

        # 각 열을 순회하며 최대값 찾기
        for j in range(cols):
            # 첫 번째 요소를 초기 최대값으로 설정
            max_val = arr[0, j]

            # 나머지 요소들과 비교하여 최대값 업데이트
            for i in range(1, rows):
                if arr[i, j] > max_val:
                    max_val = arr[i, j]

            # 최대값 저장
            max_vals[j] = max_val

        return max_vals


@njit
def nb_norm(arr, axis=0):
    """
    벡터의 노름(유클리드 거리)을 계산하는 함수 (Numba 최적화 버전).
    
    입력:
        arr (2차원 배열): 입력 배열
        axis (int): 축 (현재는 0만 지원)
    
    출력:
        1차원 배열: 각 열의 노름 값
    """
    rows, cols = arr.shape
    norms = np.zeros(cols, dtype=np.float64)
    for j in range(cols):
        col_vector = arr[:, j]
        norms[j] = np.linalg.norm(col_vector)
    return norms


@njit
def get_rotated_coords(h, w, theta, anchor):
    """
    이미지의 모든 픽셀 좌표를 회전시키는 함수 (Numba 최적화 버전).
    
    입력:
        h (int): 이미지의 높이
        w (int): 이미지의 너비
        theta (float): 회전 각도 (라디안)
        anchor (2차원 배열): 회전의 중심점 좌표 (2, 1)
    
    출력:
        tuple: 회전된 X와 Y 좌표 배열
    """
    anchor = anchor.reshape(2, 1)
    rotate_mat = get_rotate_mat(theta)
    x, y = nb_meshgrid(np.arange(w), np.arange(h))
    # x, y = np.meshgrid(np.arange(w), np.arange(h))  # 주석 처리된 기존 코드
    x_lin = x.reshape((1, x.size))
    y_lin = y.reshape((1, y.size))
    coord_mat = np.concatenate((x_lin, y_lin), 0)
    rotated_coord = np.dot(rotate_mat, coord_mat - anchor) + anchor
    rotated_x = rotated_coord[0, :].reshape(x.shape)
    rotated_y = rotated_coord[1, :].reshape(y.shape)
    return rotated_x, rotated_y


@njit
def shrink_bbox(bbox, coef=0.3, inplace=False):
    """
    바운딩 박스를 축소하는 함수 (Numba 최적화 버전).
    
    입력:
        bbox (2차원 배열): 바운딩 박스 좌표 배열 (4, 2)
        coef (float, optional): 축소 비율 (기본값: 0.3)
        inplace (bool, optional): 원본 배열을 수정할지 여부 (기본값: False)
    
    출력:
        2차원 배열: 축소된 바운딩 박스 좌표 배열 (4, 2)
    """
    # 각 변의 길이를 계산
    lens = [np.linalg.norm(bbox[i] - bbox[(i + 1) % 4], ord=2) for i in range(4)]
    # 각 꼭지점에서 축소할 거리 계산
    r = [min(lens[(i - 1) % 4], lens[i]) for i in range(4)]

    if not inplace:
        bbox = bbox.copy()

    # 축소를 위한 오프셋 결정
    offset = 0 if lens[0] + lens[2] > lens[1] + lens[3] else 1
    for idx in [0, 2, 1, 3]:
        p1_idx, p2_idx = (idx + offset) % 4, (idx + 1 + offset) % 4
        p1p2 = bbox[p2_idx] - bbox[p1_idx]
        dist = np.linalg.norm(p1p2)
        if dist <= 1:
            continue
        bbox[p1_idx] += p1p2 / dist * r[p1_idx] * coef
        bbox[p2_idx] -= p1p2 / dist * r[p2_idx] * coef
    return bbox


@njit
def get_rotated_coords_(h, w, theta, anchor):
    """
    모든 픽셀 좌표를 회전시키는 또 다른 함수 (Numba 최적화 버전).
    이 함수는 기존의 get_rotated_coords와 동일한 기능을 수행합니다.
    
    입력:
        h (int): 이미지의 높이
        w (int): 이미지의 너비
        theta (float): 회전 각도 (라디안)
        anchor (2차원 배열): 회전의 중심점 좌표 (2, 1)
    
    출력:
        tuple: 회전된 X와 Y 좌표 배열
    """
    anchor = anchor.reshape(2, 1)
    rotate_mat = get_rotate_mat(theta)
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x_lin = x.reshape((1, x.size))
    y_lin = y.reshape((1, y.size))
    coord_mat = np.concatenate((x_lin, y_lin), 0)
    rotated_coord = np.dot(rotate_mat, coord_mat - anchor) + anchor
    rotated_x = rotated_coord[0, :].reshape(x.shape)
    rotated_y = rotated_coord[1, :].reshape(y.shape)
    return rotated_x, rotated_y


@njit
def get_rotate_mat(theta):
    """
    회전 행렬을 생성하는 함수 (Numba 최적화 버전).
    양의 theta 값은 시계 방향 회전을 의미합니다.
    
    입력:
        theta (float): 회전 각도 (라디안)
    
    출력:
        2x2 배열: 회전 행렬
    """
    return np.array([[math.cos(theta), -math.sin(theta)],
                     [math.sin(theta), math.cos(theta)]])


@njit
def calc_error_from_rect(bbox):
    """
    바운딩 박스의 방향 오류를 계산합니다.
    기본 방향은 x1y1: 좌상단, x2y2: 우상단, x3y3: 우하단, x4y4: 좌하단입니다.
    
    입력:
        bbox (2차원 배열): 바운딩 박스 좌표 배열 (4, 2)
    
    출력:
        float: 방향 오류의 합
    """
    x_min, y_min = nb_amin(bbox, axis=0)
    x_max, y_max = nb_amax(bbox, axis=0)
    rect = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]],
                    dtype=np.float32)
    return nb_norm(bbox - rect, axis=0).sum()


@njit
def rotate_bbox(bbox, theta, anchor=None):
    """
    바운딩 박스를 회전시키는 함수 (Numba 최적화 버전).
    
    입력:
        bbox (2차원 배열): 바운딩 박스 좌표 배열 (4, 2)
        theta (float): 회전 각도 (라디안)
        anchor (2차원 배열, optional): 회전의 중심점 좌표 (2, 1). 기본값은 None (중심점 사용)
    
    출력:
        2차원 배열: 회전된 바운딩 박스 좌표 배열 (4, 2)
    """
    points = bbox.T
    if anchor is None:
        anchor = points[:, :1]
    rotated_points = np.dot(get_rotate_mat(theta), points - anchor) + anchor
    return rotated_points.T


def find_min_rect_angle(bbox, rank_num=10):
    """
    최소 영역을 가지는 회전 각도를 찾는 함수.
    
    입력:
        bbox (2차원 배열): 바운딩 박스 좌표 배열 (4, 2)
        rank_num (int, optional): 고려할 최솟값의 개수 (기본값: 10)
    
    출력:
        float: 최적의 회전 각도 (라디안)
    """
    # 각도 리스트 생성 (-90도부터 89도까지)
    angles = np.arange(-90, 90) / 180 * math.pi
    areas = []
    for theta in angles:
        rotated_bbox = rotate_bbox(bbox, theta)
        x_min, y_min = nb_amin(rotated_bbox, axis=0)
        x_max, y_max = nb_amax(rotated_bbox, axis=0)
        areas.append((x_max - x_min) * (y_max - y_min))

    best_angle, min_error = -1, float('inf')
    # 영역이 작은 순으로 정렬한 각도들 중 상위 rank_num 개를 검토
    for idx in np.argsort(areas)[:rank_num]:
        rotated_bbox = rotate_bbox(bbox, angles[idx])
        error = calc_error_from_rect(rotated_bbox)
        if error < min_error:
            best_angle, min_error = angles[idx], error

    return best_angle


def generate_score_geo_maps(image, word_bboxes, map_scale=0.5):
    """
    이미지와 단어 바운딩 박스를 기반으로 스코어 맵과 지오 메트리 맵을 생성합니다.
    
    입력:
        image (numpy.ndarray): 입력 이미지 배열
        word_bboxes (리스트): 단어 바운딩 박스 리스트 (각 박스는 (4, 2) 형태)
        map_scale (float, optional): 맵의 스케일 비율 (기본값: 0.5)
    
    출력:
        tuple: 스코어 맵과 지오 메트리 맵
    """
    img_h, img_w = image.shape[:2]
    map_h, map_w = int(img_h * map_scale), int(img_w * map_scale)
    inv_scale = int(1 / map_scale)

    # 스코어 맵과 지오 메트리 맵 초기화
    score_map = np.zeros((map_h, map_w, 1), np.float32)
    geo_map = np.zeros((map_h, map_w, 5), np.float32)

    word_polys = []

    for bbox in word_bboxes:
        # 바운딩 박스를 축소하고 스케일 조정
        poly = np.around(map_scale * shrink_bbox(bbox)).astype(np.int32)
        word_polys.append(poly)

        # 중심 마스크 생성
        center_mask = np.zeros((map_h, map_w), np.float32)
        cv2.fillPoly(center_mask, [poly], 1)

        # 최적의 회전 각도 찾기
        theta = find_min_rect_angle(bbox)
        rotated_bbox = rotate_bbox(bbox, theta) * map_scale
        x_min, y_min = nb_amin(rotated_bbox, axis=0)
        x_max, y_max = nb_amax(rotated_bbox, axis=0)

        # 회전 중심점 설정
        anchor = bbox[0] * map_scale
        rotated_x, rotated_y = get_rotated_coords(map_h, map_w, theta, anchor)

        # 지오 메트리 맵 업데이트
        d1, d2 = rotated_y - y_min, y_max - rotated_y
        d1[d1 < 0] = 0
        d2[d2 < 0] = 0
        d3, d4 = rotated_x - x_min, x_max - rotated_x
        d3[d3 < 0] = 0
        d4[d4 < 0] = 0
        geo_map[:, :, 0] += d1 * center_mask * inv_scale
        geo_map[:, :, 1] += d2 * center_mask * inv_scale
        geo_map[:, :, 2] += d3 * center_mask * inv_scale
        geo_map[:, :, 3] += d4 * center_mask * inv_scale
        geo_map[:, :, 4] += theta * center_mask

    # 스코어 맵에 단어 폴리곤 채우기
    cv2.fillPoly(score_map, word_polys, 1)

    return score_map, geo_map


class EASTDataset(Dataset):
    """
    EAST 모델을 위한 데이터셋 클래스.
    
    이 클래스는 이미지와 해당 이미지의 단어 바운딩 박스를 기반으로 스코어 맵과 지오 메트리 맵을 생성하여
    모델의 입력으로 사용됩니다.
    """
    def __init__(self, dataset, map_scale=0.5, to_tensor=True):
        """
        초기화 함수.
        
        입력:
            dataset (Dataset): 기본 데이터셋 객체
            map_scale (float, optional): 맵의 스케일 비율 (기본값: 0.5)
            to_tensor (bool, optional): 데이터를 텐서로 변환할지 여부 (기본값: True)
        """
        self.dataset = dataset
        self.map_scale = map_scale
        self.to_tensor = to_tensor

    def __getitem__(self, idx):
        """
        특정 인덱스에 해당하는 데이터를 반환합니다.
        
        입력:
            idx (int): 데이터 인덱스
        
        출력:
            tuple: 이미지, 스코어 맵, 지오 메트리 맵, ROI 마스크
        """
        # 기본 데이터셋에서 이미지와 바운딩 박스 가져오기
        image, word_bboxes, roi_mask = self.dataset[idx]
        # 스코어 맵과 지오 메트리 맵 생성
        score_map, geo_map = generate_score_geo_maps(image, word_bboxes, map_scale=self.map_scale)

        # 마스크 크기 조정
        mask_size = int(image.shape[0] * self.map_scale), int(image.shape[1] * self.map_scale)
        roi_mask = cv2.resize(roi_mask, dsize=mask_size)
        if roi_mask.ndim == 2:
            roi_mask = np.expand_dims(roi_mask, axis=2)

        if self.to_tensor:
            # 이미지를 텐서로 변환하고 채널을 첫 번째 차원으로 이동
            image = torch.Tensor(image).permute(2, 0, 1)
            score_map = torch.Tensor(score_map).permute(2, 0, 1)
            geo_map = torch.Tensor(geo_map).permute(2, 0, 1)
            roi_mask = torch.Tensor(roi_mask).permute(2, 0, 1)

        return image, score_map, geo_map, roi_mask

    def __len__(self):
        """
        데이터셋의 크기를 반환합니다.
        
        출력:
            int: 데이터셋의 총 샘플 수
        """
        return len(self.dataset)
