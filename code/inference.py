import os
import os.path as osp
import json
from argparse import ArgumentParser
from glob import glob
import numpy as np
import torch
import cv2
from torch import cuda
from model import EAST
from tqdm import tqdm

from detect import detect


CHECKPOINT_EXTENSIONS = ['.pth', '.ckpt']
LANGUAGE_LIST = ['chinese', 'japanese', 'thai', 'vietnamese']

def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--data_dir', default=os.environ.get('SM_CHANNEL_EVAL', 'data'))
    parser.add_argument('--model_dir', default=os.environ.get('SM_CHANNEL_MODEL', 'trained_models'))
    parser.add_argument('--output_dir', default=os.environ.get('SM_OUTPUT_DATA_DIR', 'predictions'))

    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--input_size', type=int, default=2048)
    parser.add_argument('--batch_size', type=int, default=5)

    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args



def remove_shadow(image):
    # RGB 채널 분리
    rgb_planes = cv2.split(image)

    result_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(norm_img)

    # 채널 합성하여 그림자 제거된 이미지 생성
    result_norm = cv2.merge(result_planes)
    return result_norm

'''
    현재 그림자 제거 -> Grasy Scale -> Otsu 알고리즘 적용 순으로 Test이미지를 전처리하고 있다.
'''
def preprocess_image(image, input_size):
    # 그림자 제거
    image = remove_shadow(image)
    # 그레이스케일로 변환
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Otsu 이진화 적용
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 크기 조정
    resized_image = cv2.resize(binary_image, (input_size, input_size))

    return resized_image



def do_inference(model, ckpt_fpath, data_dir, input_size, batch_size, split='test'):
    model.load_state_dict(torch.load(ckpt_fpath, map_location='cpu'))
    model.eval()

    image_fnames, by_sample_bboxes = [], []

    images = []
    
    for image_fpath in tqdm(sum([glob(osp.join(data_dir, f'{lang}_receipt/img/{split}/*')) for lang in LANGUAGE_LIST], [])):

        image_fnames.append(osp.basename(image_fpath))

        images = cv2.imread(image_fpath)[:, :, ::-1]
        '''
            Test 데이터가 모델에 들어가기 전에 전처리를 해서 image에 다시 넣어준다.
        '''
        image = preprocess_image(image, input_size)  # 전처리 함수 적용

        images.append(image)


        if len(images) == batch_size:
            by_sample_bboxes.extend(detect(model, images, input_size))
            images = []

    if len(images):
        by_sample_bboxes.extend(detect(model, images, input_size))

    ufo_result = dict(images=dict())
    for image_fname, bboxes in zip(image_fnames, by_sample_bboxes):
        words_info = {idx: dict(points=bbox.tolist()) for idx, bbox in enumerate(bboxes)}
        ufo_result['images'][image_fname] = dict(words=words_info)

    return ufo_result


def main(args):
    # Initialize model
    model = EAST(pretrained=False).to(args.device)

    # Get paths to checkpoint files 
    # 체크포인트 이름 바꾸기
    ckpt_fpath = osp.join(args.model_dir, 'latest.pth')

    if not osp.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print('Inference in progress')

    ufo_result = dict(images=dict())
    split_result = do_inference(model, ckpt_fpath, args.data_dir, args.input_size,
                                args.batch_size, split='test')
    ufo_result['images'].update(split_result['images'])

    output_fname = 'output.csv'
    with open(osp.join(args.output_dir, output_fname), 'w') as f:
        json.dump(ufo_result, f, indent=4)


if __name__ == '__main__':
    args = parse_args()
    main(args)



# import os
# import os.path as osp
# import json
# from argparse import ArgumentParser
# from glob import glob
# import csv

# import torch
# import cv2
# from torch import cuda
# from model import EAST
# from tqdm import tqdm

# from detect import detect

# import random
# import numpy as np
# import matplotlib.pyplot as plt

# CHECKPOINT_EXTENSIONS = ['.pth', '.ckpt']
# LANGUAGE_LIST = ['chinese', 'japanese', 'thai', 'vietnamese']

# def parse_args():
#     parser = ArgumentParser()

#     # Conventional args
#     parser.add_argument('--data_dir', default=os.environ.get('SM_CHANNEL_EVAL', 'data'))
#     parser.add_argument('--model_dir', default=os.environ.get('SM_CHANNEL_MODEL', 'trained_models'))
#     parser.add_argument('--output_dir', default=os.environ.get('SM_OUTPUT_DATA_DIR', 'predictions'))

#     parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
#     parser.add_argument('--input_size', type=int, default=2048)
#     parser.add_argument('--batch_size', type=int, default=5)
#     parser.add_argument('--apply_otsu', action='store_true', help='Apply Otsu thresholding after shadow removal and grayscale conversion')

#     args = parser.parse_args()

#     if args.input_size % 32 != 0:
#         raise ValueError('`input_size` must be a multiple of 32')

#     return args

# # 전처리 함수들
# def remove_shadow(image):
#     # RGB 채널 분리
#     rgb_planes = cv2.split(image)

#     result_planes = []
#     for plane in rgb_planes:
#         dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
#         bg_img = cv2.medianBlur(dilated_img, 21)
#         diff_img = 255 - cv2.absdiff(plane, bg_img)
#         norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
#         result_planes.append(norm_img)

#     # 채널 합성하여 그림자 제거된 이미지 생성
#     result_norm = cv2.merge(result_planes)
#     return result_norm

# def preprocess_image(image, apply_otsu=False):
#     """
#     이미지에 전처리를 적용합니다.
#     :param image: BGR 이미지 (numpy 배열)
#     :param apply_otsu: Otsu 이진화를 적용할지 여부
#     :return: 전처리된 이미지
#     """
#     # 그림자 제거
#     shadow_removed = remove_shadow(image)

#     # 그레이스케일로 변환
#     gray = cv2.cvtColor(shadow_removed, cv2.COLOR_BGR2GRAY)

#     if apply_otsu:
#         # Otsu 이진화 적용
#         _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#         return otsu
#     else:
#         # 모델이 그레이스케일 이미지를 기대하면 그대로 반환
#         # 그렇지 않다면 다시 BGR로 변환할 수도 있습니다.
#         return gray

# def do_inference(model, ckpt_fpath, data_dir, input_size, batch_size, apply_otsu=False, split='test'):
#     model.load_state_dict(torch.load(ckpt_fpath, map_location='cpu'))
#     model.eval()

#     image_fnames, by_sample_bboxes = [], []

#     images = []
    
#     for image_fpath in tqdm(sum([glob(osp.join(data_dir, f'{lang}_receipt/img/{split}/*')) for lang in LANGUAGE_LIST], [])):
#         image_fnames.append(osp.basename(image_fpath))

#         image = cv2.imread(image_fpath)
#         if image is None:
#             print(f"이미지를 열 수 없습니다: {image_fpath}")
#             continue

#         # 전처리 적용
#         preprocessed_image = preprocess_image(image, apply_otsu=apply_otsu)

#         # 모델이 기대하는 형식으로 변환 (예: 채널 추가)
#         if len(preprocessed_image.shape) == 2:
#             preprocessed_image = np.expand_dims(preprocessed_image, axis=2)  # H x W x 1
#             preprocessed_image = np.repeat(preprocessed_image, 3, axis=2)    # H x W x 3

#         images.append(preprocessed_image)

#         if len(images) == batch_size:
#             by_sample_bboxes.extend(detect(model, images, input_size))
#             images = []

#     if len(images):
#         by_sample_bboxes.extend(detect(model, images, input_size))

#     # CSV 저장을 위한 데이터 구조
#     csv_rows = []
#     for image_fname, bboxes in zip(image_fnames, by_sample_bboxes):
#         for idx, bbox in enumerate(bboxes):
#             # bbox는 [x1, y1, x2, y2, ...] 형태로 가정
#             # 필요한 좌표 수에 따라 확장 가능
#             row = {
#                 'image_filename': image_fname,
#                 'word_index': idx
#             }
#             # bbox의 각 좌표를 개별 열로 추가
#             for i, point in enumerate(bbox):
#                 row[f'point_{i+1}'] = point
#             csv_rows.append(row)

#     return csv_rows

# def main(args):
#     # Initialize model
#     model = EAST(pretrained=False).to(args.device)

#     # Get paths to checkpoint files 
#     # 체크포인트 이름 바꾸기
#     ckpt_fpath = osp.join(args.model_dir, 'latest.pth')

#     if not osp.exists(args.output_dir):
#         os.makedirs(args.output_dir)

#     print('Inference in progress')

#     # do_inference는 이제 CSV에 필요한 행 리스트를 반환합니다.
#     csv_rows = do_inference(model, ckpt_fpath, args.data_dir, args.input_size,
#                             args.batch_size, apply_otsu=args.apply_otsu, split='test')

#     output_fname = 'output.csv'
#     output_path = osp.join(args.output_dir, output_fname)

#     # CSV 파일에 저장할 필드 이름 결정
#     # 가장 긴 bbox를 찾아 필드 이름을 설정 (예: point_1, point_2, ...)
#     max_points = max(len(row) - 2 for row in csv_rows)  # image_filename과 word_index 제외
#     fieldnames = ['image_filename', 'word_index'] + [f'point_{i+1}' for i in range(max_points)]

#     with open(output_path, 'w', newline='') as csvfile:
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

#         writer.writeheader()
#         for row in csv_rows:
#             # 모든 필드가 채워지도록 보장
#             for i in range(max_points):
#                 key = f'point_{i+1}'
#                 if key not in row:
#                     row[key] = ''
#             writer.writerow(row)

#     print(f'Inference results saved to {output_path}')

# if __name__ == '__main__':
#     args = parse_args()
#     main(args)
