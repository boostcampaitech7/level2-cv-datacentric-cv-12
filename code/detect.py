import numpy as np
import torch
import lanms
import warnings
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.augmentations.geometric.resize import LongestMaxSize

from dataset import get_rotate_mat

MAX_BOX_PREDICTIONS = 1000


def print_warning(num_boxes):
    """
    경고 메시지를 출력합니다. 감지된 박스의 수가 최대 허용치를 초과할 경우 경고를 발생시킵니다.
    
    입력:
        num_boxes (int): 감지된 박스의 수
    """
    warnings.warn(
        f"Found {num_boxes} boxes. Only {MAX_BOX_PREDICTIONS} boxes will be kept. "
        "Model trained with insufficient epochs could largely increase "
        "the number of bounding boxes. Check if the model was trained sufficiently.",
        stacklevel=2)


def is_valid_poly(res, score_shape, scale):
    """
    다각형이 이미지 범위 내에 있는지 확인합니다.
    
    입력:
        res (numpy.ndarray): 원본 이미지에서 복원된 다각형 좌표, 형태는 (2, N)
        score_shape (tuple): 스코어 맵의 형태 (높이, 너비)
        scale (float): 피처 맵에서 이미지로의 스케일 비율
    출력:
        bool: 유효한 다각형이면 True, 그렇지 않으면 False
    """
    cnt = 0
    for i in range(res.shape[1]):
        # 다각형의 각 점이 이미지 범위를 벗어나는지 확인
        if (res[0, i] < 0 or res[0, i] >= score_shape[1] * scale or 
            res[1, i] < 0 or res[1, i] >= score_shape[0] * scale):
            cnt += 1
    # 하나 이하의 점만 벗어나면 유효한 것으로 간주
    return cnt <= 1


def restore_polys(valid_pos, valid_geo, score_shape, scale=2):
    """
    주어진 위치에서 피처 맵으로부터 다각형을 복원합니다.
    
    입력:
        valid_pos (numpy.ndarray): 잠재적인 텍스트 위치, 형태는 (n, 2)
        valid_geo (numpy.ndarray): 유효한 위치의 지오메트리 정보, 형태는 (5, n)
        score_shape (tuple): 스코어 맵의 형태 (높이, 너비)
        scale (float): 이미지와 피처 맵의 스케일 비율 (기본값: 2)
    출력:
        tuple:
            polys (numpy.ndarray): 복원된 다각형들, 형태는 (n, 8)
            index (list): 유효한 다각형의 인덱스 리스트
    """
    polys = []
    index = []
    valid_pos *= scale  # 위치를 원본 이미지 스케일로 변환
    d = valid_geo[:4, :]  # 높이, 너비 등의 지오메트리 정보 (4 x N)
    angle = valid_geo[4, :]  # 회전 각도 (N,)
    
    for i in range(valid_pos.shape[0]):
        x = valid_pos[i, 0]
        y = valid_pos[i, 1]
        y_min = y - d[0, i]
        y_max = y + d[1, i]
        x_min = x - d[2, i]
        x_max = x + d[3, i]
        rotate_mat = get_rotate_mat(-angle[i])  # 각도에 따른 회전 행렬 생성
        
        # 다각형의 각 꼭지점 좌표 계산
        temp_x = np.array([[x_min, x_max, x_max, x_min]]) - x
        temp_y = np.array([[y_min, y_min, y_max, y_max]]) - y
        coordidates = np.concatenate((temp_x, temp_y), axis=0)  # (2, 4)
        res = np.dot(rotate_mat, coordidates)  # 회전 적용
        res[0, :] += x
        res[1, :] += y
    
        # 유효한 다각형인지 확인
        if is_valid_poly(res, score_shape, scale):
            index.append(i)
            polys.append([res[0, 0], res[1, 0], res[0, 1], res[1, 1], 
                          res[0, 2], res[1, 2], res[0, 3], res[1, 3]])
    return np.array(polys), index


def get_bboxes(score, geo, score_thresh=0.9, nms_thresh=0.2):
    """
    피처 맵으로부터 최종 바운딩 박스를 추출합니다.
    
    입력:
        score (numpy.ndarray): 모델로부터의 스코어 맵, 형태는 (1, 행, 열)
        geo (numpy.ndarray): 모델로부터의 지오메트리 맵, 형태는 (5, 행, 열)
        score_thresh (float): 스코어 맵을 세그먼트하기 위한 임계값 (기본값: 0.9)
        nms_thresh (float): NMS에서 사용할 임계값 (기본값: 0.2)
    출력:
        numpy.ndarray or None: 최종 다각형 박스들, 형태는 (n, 9). 박스가 없으면 None 반환
    """
    score = score[0, :, :]  # 스코어 맵 추출
    xy_text = np.argwhere(score > score_thresh)  # 스코어가 임계값을 초과하는 위치 추출 (n x 2), [r, c] 형식
    if xy_text.size == 0:
        return None
    
    xy_text = xy_text[np.argsort(xy_text[:, 0])]  # y축 기준 정렬
    valid_pos = xy_text[:, ::-1].copy()  # [x, y] 형식으로 변환 (n x 2)
    valid_geo = geo[:, xy_text[:, 0], xy_text[:, 1]]  # 해당 위치의 지오메트리 정보 추출 (5 x n)
    polys_restored, index = restore_polys(valid_pos, valid_geo, score.shape)
    if polys_restored.size == 0:
        return None
    
    boxes = np.zeros((polys_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = polys_restored  # 다각형 좌표
    boxes[:, 8] = score[xy_text[index, 0], xy_text[index, 1]]  # 스코어 값
    boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thresh)  # LANMS를 사용한 NMS 적용
    num_boxes = len(boxes)
    
    if num_boxes > MAX_BOX_PREDICTIONS:
        print_warning(num_boxes)
        boxes = boxes[:MAX_BOX_PREDICTIONS]  # 최대 박스 수 제한
    
    return boxes


def detect(model, images, input_size, map_scale=0.5):
    """
    모델을 사용하여 이미지에서 텍스트를 감지합니다.
    
    입력:
        model (torch.nn.Module): 텍스트 감지 모델
        images (list of numpy.ndarray): 감지할 이미지 리스트
        input_size (int): 모델 입력 크기
        map_scale (float): 맵 스케일 비율 (기본값: 0.5)
    출력:
        list of numpy.ndarray: 각 이미지에 대한 감지된 바운딩 박스 리스트, 각 박스는 (4, 2) 형태
    """
    # 전처리 파이프라인 정의
    prep_fn = A.Compose([
        LongestMaxSize(input_size),  # 가장 긴 변을 input_size로 조정
        A.PadIfNeeded(min_height=input_size, min_width=input_size,
                     position=A.PadIfNeeded.PositionType.TOP_LEFT),  # 필요한 경우 패딩 추가
        A.Normalize(),  # 정규화
        ToTensorV2()  # 텐서로 변환
    ])
    device = list(model.parameters())[0].device  # 모델의 디바이스 확인
    
    batch, orig_sizes = [], []
    for image in images:
        orig_sizes.append(image.shape[:2])  # 원본 이미지 크기 저장
        batch.append(prep_fn(image=image)['image'])  # 전처리된 이미지 추가
    batch = torch.stack(batch, dim=0).to(device)  # 배치를 텐서로 스택하고 디바이스로 이동
    
    with torch.no_grad():
        score_maps, geo_maps = model(batch)  # 모델을 통해 스코어 맵과 지오 맵 예측
    score_maps, geo_maps = score_maps.cpu().numpy(), geo_maps.cpu().numpy()  # CPU로 이동하고 numpy 배열로 변환
    
    by_sample_bboxes = []
    for score_map, geo_map, orig_size in zip(score_maps, geo_maps, orig_sizes):
        # 맵의 마진 계산
        map_margin = int(abs(orig_size[0] - orig_size[1]) * map_scale * input_size / max(orig_size))
        if orig_size[0] == orig_size[1]:
            score_map, geo_map = score_map, geo_map
        elif orig_size[0] > orig_size[1]:
            score_map, geo_map = score_map[:, :, :-map_margin], geo_map[:, :, :-map_margin]
        else:
            score_map, geo_map = score_map[:, :-map_margin, :], geo_map[:, :-map_margin, :]
    
        bboxes = get_bboxes(score_map, geo_map)  # 바운딩 박스 추출
        if bboxes is None:
            bboxes = np.zeros((0, 4, 2), dtype=np.float32)  # 박스가 없으면 빈 배열
        else:
            bboxes = bboxes[:, :8].reshape(-1, 4, 2)  # 박스 좌표 재구성
            bboxes *= max(orig_size) / input_size  # 원본 이미지 크기에 맞게 스케일 조정
    
        by_sample_bboxes.append(bboxes)
    
    return by_sample_bboxes
