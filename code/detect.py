import numpy as np
import torch
import lanms
import warnings
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.augmentations.geometric.resize import LongestMaxSize

from dataset import get_rotate_mat

# 최대 바운딩 박스 개수 설정
MAX_BOX_PREDICTIONS = 1000


def print_warning(num_boxes):
    """
    최대 바운딩 박스 개수를 초과했을 때 경고 메시지를 출력합니다.

    Args:
        num_boxes (int): 현재 발견된 바운딩 박스의 개수.
    """
    warnings.warn(
        f"Found {num_boxes} boxes. Only {MAX_BOX_PREDICTIONS} boxes will be kept. "
        "Model trained with insufficient epochs could largely increase "
        "the number of bounding boxes. Check if the model was trained sufficiently.",
        stacklevel=2)


def is_valid_poly(res, score_shape, scale):
    """
    폴리곤이 이미지 범위 내에 있는지 확인합니다.

    Args:
        res (numpy.ndarray): 원본 이미지에서 복원된 폴리곤 좌표.
        score_shape (tuple): 스코어 맵의 형태 (행, 열).
        scale (float): 피처 맵에서 이미지로의 스케일 비율.

    Returns:
        bool: 폴리곤이 유효하면 True, 그렇지 않으면 False.
    """
    cnt = 0
    for i in range(res.shape[1]):
        # 폴리곤의 각 좌표가 이미지 범위를 벗어나는지 확인
        if (res[0, i] < 0 or res[0, i] >= score_shape[1] * scale or res[1, i] < 0 or
            res[1, i] >= score_shape[0] * scale):
            cnt += 1
    # 이미지 범위를 벗어나는 좌표가 1개 이하인 경우 유효한 폴리곤으로 간주
    return cnt <= 1


def restore_polys(valid_pos, valid_geo, score_shape, scale=2):
    """
    유효한 위치와 지오메트리 맵을 사용하여 폴리곤을 복원합니다.

    Args:
        valid_pos (numpy.ndarray): 잠재적인 텍스트 위치 (n x 2).
        valid_geo (numpy.ndarray): 유효한 위치에서의 지오메트리 정보 (5 x n).
        score_shape (tuple): 스코어 맵의 형태 (행, 열).
        scale (float, optional): 피처 맵에서 이미지로의 스케일 비율. Default는 2.

    Returns:
        tuple: 
            - numpy.ndarray: 복원된 폴리곤 좌표 (n x 8).
            - list: 유효한 폴리곤의 인덱스 리스트.
    """
    polys = []
    index = []
    # 유효한 위치를 스케일 비율에 맞게 조정
    valid_pos *= scale
    d = valid_geo[:4, :]  # 4 x N, 지오메트리 정보 (상, 하, 좌, 우)
    angle = valid_geo[4, :]  # N, 회전 각도

    for i in range(valid_pos.shape[0]):
        x = valid_pos[i, 0]
        y = valid_pos[i, 1]
        # 각 바운딩 박스의 최소/최대 x, y 좌표 계산
        y_min = y - d[0, i]
        y_max = y + d[1, i]
        x_min = x - d[2, i]
        x_max = x + d[3, i]
        # 회전 매트릭스 계산
        rotate_mat = get_rotate_mat(-angle[i])

        # 회전 전 좌표 설정
        temp_x = np.array([[x_min, x_max, x_max, x_min]]) - x
        temp_y = np.array([[y_min, y_min, y_max, y_max]]) - y
        coordidates = np.concatenate((temp_x, temp_y), axis=0)
        # 회전 매트릭스를 적용하여 좌표 복원
        res = np.dot(rotate_mat, coordidates)
        res[0, :] += x
        res[1, :] += y

        # 폴리곤 유효성 검사
        if is_valid_poly(res, score_shape, scale):
            index.append(i)
            # 복원된 폴리곤을 리스트에 추가 (x1, y1, x2, y2, x3, y3, x4, y4)
            polys.append([res[0, 0], res[1, 0], res[0, 1], res[1, 1], res[0, 2], res[1, 2],
                          res[0, 3], res[1, 3]])
    return np.array(polys), index


def get_bboxes(score, geo, score_thresh=0.9, nms_thresh=0.2):
    """
    스코어 맵과 지오메트리 맵에서 바운딩 박스를 추출합니다.

    Args:
        score (numpy.ndarray): 모델로부터의 스코어 맵 (1 x 행 x 열).
        geo (numpy.ndarray): 모델로부터의 지오메트리 맵 (5 x 행 x 열).
        score_thresh (float, optional): 스코어 맵을 이진화할 임계값. Default는 0.9.
        nms_thresh (float, optional): NMS (Non-Maximum Suppression) 임계값. Default는 0.2.

    Returns:
        numpy.ndarray or None: 최종 바운딩 박스들 (n x 9) 또는 바운딩 박스가 없으면 None.
                                 각 바운딩 박스는 [x1, y1, x2, y2, x3, y3, x4, y4, score] 형태.
    """
    # 첫 번째 채널의 스코어 맵 추출
    score = score[0, :, :]
    # 스코어가 임계값을 초과하는 위치 찾기
    xy_text = np.argwhere(score > score_thresh)  # n x 2, 형식은 [r, c]
    if xy_text.size == 0:
        return None

    # y축 기준으로 정렬
    xy_text = xy_text[np.argsort(xy_text[:, 0])]
    # x, y 순으로 복사
    valid_pos = xy_text[:, ::-1].copy()  # n x 2, [x, y]
    # 유효한 위치에서의 지오메트리 정보 추출
    valid_geo = geo[:, xy_text[:, 0], xy_text[:, 1]]  # 5 x n
    # 폴리곤 복원
    polys_restored, index = restore_polys(valid_pos, valid_geo, score.shape)
    if polys_restored.size == 0:
        return None

    # 바운딩 박스 초기화 (n x 9)
    boxes = np.zeros((polys_restored.shape[0], 9), dtype=np.float32)
    # 복원된 폴리곤 좌표를 상위 8개에 할당
    boxes[:, :8] = polys_restored
    # 스코어를 마지막 열에 할당
    boxes[:, 8] = score[xy_text[index, 0], xy_text[index, 1]]
    # LANMS를 사용하여 NMS 적용 후 최종 바운딩 박스 추출
    boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thresh)
    num_boxes = len(boxes)

    # 최대 바운딩 박스 개수를 초과하면 경고를 출력하고 제한
    if num_boxes > MAX_BOX_PREDICTIONS:
        print_warning(num_boxes)
        boxes = boxes[:MAX_BOX_PREDICTIONS]

    return boxes


def detect(model, images, input_size, map_scale=0.5):
    """
    모델을 사용하여 이미지들에서 텍스트 바운딩 박스를 감지합니다.

    Args:
        model (torch.nn.Module): 텍스트 감지 모델.
        images (list or numpy.ndarray): 입력 이미지들의 리스트 또는 배열.
        input_size (int): 모델 입력 크기.
        map_scale (float, optional): 피처 맵에서 이미지로의 스케일 비율. Default는 0.5.

    Returns:
        list: 각 이미지에 대한 감지된 바운딩 박스들의 리스트.
              각 바운딩 박스는 (4 x 2)의 좌표 배열 형태.
    """
    # 전처리 파이프라인 정의
    prep_fn = A.Compose([
        LongestMaxSize(input_size),  # 가장 긴 변을 input_size로 조정
        A.PadIfNeeded(min_height=input_size, min_width=input_size,
                      position=A.PadIfNeeded.PositionType.TOP_LEFT),  # 필요 시 패딩 추가
        A.Normalize(),  # 정규화
        ToTensorV2()  # 텐서로 변환
    ])
    # 모델이 있는 디바이스 가져오기
    device = list(model.parameters())[0].device

    batch, orig_sizes = [], []
    for image in images:
        # 원본 이미지 크기 저장
        orig_sizes.append(image.shape[:2])
        # 전처리 수행 후 배치에 추가
        batch.append(prep_fn(image=image)['image'])
    # 배치를 텐서로 변환하고 디바이스로 이동
    batch = torch.stack(batch, dim=0).to(device)

    # 모델을 사용하여 예측 수행 (재현율 맵과 지오메트리 맵)
    with torch.no_grad():
        score_maps, geo_maps = model(batch)
    # 예측 결과를 CPU로 이동하고 NumPy 배열로 변환
    score_maps, geo_maps = score_maps.cpu().numpy(), geo_maps.cpu().numpy()

    by_sample_bboxes = []
    for score_map, geo_map, orig_size in zip(score_maps, geo_maps, orig_sizes):
        # 스코어 맵 크기 계산
        map_margin = int(abs(orig_size[0] - orig_size[1]) * map_scale * input_size / max(orig_size))
        if orig_size[0] == orig_size[1]:
            score_map, geo_map = score_map, geo_map        
        elif orig_size[0] > orig_size[1]:
            # 이미지의 너비가 높이보다 클 경우, 스코어 맵과 지오메트리 맵의 오른쪽 부분 잘라내기
            score_map, geo_map = score_map[:, :, :-map_margin], geo_map[:, :, :-map_margin]
        else:
            # 이미지의 높이가 너비보다 클 경우, 스코어 맵과 지오메트리 맵의 아래쪽 부분 잘라내기
            score_map, geo_map = score_map[:, :-map_margin, :], geo_map[:, :-map_margin, :]

        # 바운딩 박스 추출
        bboxes = get_bboxes(score_map, geo_map)
        if bboxes is None:
            # 바운딩 박스가 없으면 빈 배열 생성
            bboxes = np.zeros((0, 4, 2), dtype=np.float32)
        else:
            # 바운딩 박스의 좌표 재구성 및 스케일 조정
            bboxes = bboxes[:, :8].reshape(-1, 4, 2)
            bboxes *= max(orig_size) / input_size

        by_sample_bboxes.append(bboxes)

    return by_sample_bboxes
