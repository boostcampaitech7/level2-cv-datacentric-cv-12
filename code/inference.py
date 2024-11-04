import os
import os.path as osp
import json
from argparse import ArgumentParser
from glob import glob

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch import cuda
from baseline.model import EAST
from tqdm import tqdm

from baseline.detect import detect

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
    """
    이미지에서 그림자를 제거하는 함수.
    """
    # RGB 채널 분리
    rgb_planes = cv2.split(image)

    result_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, 
                                 norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(norm_img)

    # 채널 합성하여 그림자 제거된 이미지 생성
    result_norm = cv2.merge(result_planes)
    return result_norm

# def apply_otsu_threshold(gray_image):
#     """
#     그레이스케일 이미지에 Otsu의 이진화 적용.
#     """
#     _, otsu_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     return otsu_image

# def apply_clahe(gray_image, clip_limit=2.0, tile_grid_size=(8, 8)):
#     """
#     그레이스케일 이미지에 CLAHE를 적용하여 대비를 향상시킵니다.
#     """
#     clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
#     return clahe.apply(gray_image)

def do_inference(model, ckpt_fpath, data_dir, input_size, batch_size, split='test', output_dir='predictions'):
    model.load_state_dict(torch.load(ckpt_fpath, map_location='cpu'))
    model.eval()

    # base_name 정의 (확장자 제거)
    base_name = osp.splitext(osp.basename(ckpt_fpath))[0]

    image_fnames, by_sample_bboxes = [], []

    images = []
    
    # 플래그 변수 추가
    first_image_processed = False

    for image_fpath in tqdm(sum([glob(osp.join(data_dir, f'{lang}_receipt/img/{split}/*')) for lang in LANGUAGE_LIST], [])):
        image_fnames.append(osp.basename(image_fpath))

        # 이미지 로드 (BGR -> RGB)
        image = cv2.imread(image_fpath)
        if image is None:
            print(f"이미지를 열 수 없습니다: {image_fpath}")
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 그림자 제거 전 그레이스케일 변환
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        
        # 그림자 제거
        shadow_removed = remove_shadow(cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB))
        
        # 다시 3채널로 변환하여 모델 입력 형식 유지
        shadow_removed_3ch = cv2.cvtColor(shadow_removed, cv2.COLOR_BGR2RGB)
        
        images.append(shadow_removed_3ch)

        # 첫 번째 이미지 시각화
        if not first_image_processed:
            # 파일로 저장
            sample_image_path = osp.join(output_dir, f'{base_name}_sample_input_image.png')
            cv2.imwrite(sample_image_path, cv2.cvtColor(shadow_removed_3ch, cv2.COLOR_RGB2BGR))
            print(f'Sample input image saved to {sample_image_path}')

            # 화면에 표시하려면 아래 주석을 제거하세요
            # plt.imshow(shadow_removed_3ch)
            # plt.title('Sample Input Image (Grayscale + Shadow Removed)')
            # plt.axis('off')
            # plt.show()

            first_image_processed = True

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

    # 체크포인트 파일 경로 설정
    ckpt_fpath = osp.join(args.model_dir, 'remove_line_baseData_aug_2024-11-02_16-14-10_training_log_epoch150.pth')

    if not osp.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print('Inference in progress')

    # 체크포인트 파일 이름에서 .pth를 제거하고 .csv로 변경
    base_name = osp.basename(ckpt_fpath).replace('.pth', '.csv')
    output_fname = osp.join(args.output_dir, base_name)

    ufo_result = dict(images=dict())
    split_result = do_inference(
        model, 
        ckpt_fpath, 
        args.data_dir, 
        args.input_size,
        args.batch_size, 
        split='test',
        output_dir=args.output_dir,
    )
    ufo_result['images'].update(split_result['images'])

    # 결과를 output_fname에 저장
    with open(output_fname, 'w') as f:
        json.dump(ufo_result, f, indent=4)

if __name__ == '__main__':
    args = parse_args()
    main(args)