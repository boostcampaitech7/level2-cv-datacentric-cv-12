import os
import os.path as osp
import json
from argparse import ArgumentParser
from glob import glob

import torch
import cv2
from torch import cuda
from model import EAST
from tqdm import tqdm

from detect import detect


# 체크포인트 파일 확장자 리스트
CHECKPOINT_EXTENSIONS = ['.pth', '.ckpt']
# 지원하는 언어 리스트
LANGUAGE_LIST = ['chinese', 'japanese', 'thai', 'vietnamese']


def parse_args():
    """
    명령줄 인자를 파싱하여 반환합니다.
    
    반환:
        argparse.Namespace: 파싱된 인자를 포함하는 네임스페이스 객체
    """
    parser = ArgumentParser()

    # 기본 인자들 설정
    parser.add_argument('--data_dir', default=os.environ.get('SM_CHANNEL_EVAL', 'data'),
                        help='평가 데이터가 저장된 디렉토리 경로')
    parser.add_argument('--model_dir', default=os.environ.get('SM_CHANNEL_MODEL', 'trained_models'),
                        help='모델 체크포인트가 저장된 디렉토리 경로')
    parser.add_argument('--output_dir', default=os.environ.get('SM_OUTPUT_DATA_DIR', 'predictions'),
                        help='예측 결과를 저장할 디렉토리 경로')

    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu',
                        help='모델을 실행할 디바이스 (cuda 또는 cpu)')
    parser.add_argument('--input_size', type=int, default=2048,
                        help='모델 입력 이미지의 크기 (32의 배수여야 함)')
    parser.add_argument('--batch_size', type=int, default=5,
                        help='한 번에 처리할 배치 크기')

    args = parser.parse_args()

    # input_size가 32의 배수가 아닌 경우 오류 발생
    if args.input_size % 32 != 0:
        raise ValueError('`input_size`는 32의 배수여야 합니다.')

    return args


def do_inference(model, ckpt_fpath, data_dir, input_size, batch_size, split='test'):
    """
    모델을 사용하여 주어진 데이터셋에 대한 추론을 수행합니다.
    
    입력:
        model (torch.nn.Module): EAST 모델 인스턴스
        ckpt_fpath (str): 모델 체크포인트 파일 경로
        data_dir (str): 이미지 데이터가 저장된 디렉토리 경로
        input_size (int): 모델 입력 이미지의 크기
        batch_size (int): 한 번에 처리할 배치 크기
        split (str, optional): 데이터셋 분할 ('test' 등). 기본값은 'test'
    
    출력:
        dict: 예측된 바운딩 박스를 포함하는 UFO 형식의 결과 딕셔너리
    """
    # 체크포인트 파일 로드
    model.load_state_dict(torch.load(ckpt_fpath, map_location='cpu'))
    model.eval()

    image_fnames, by_sample_bboxes = [], []

    images = []
    
    # 모든 지원 언어에 대해 이미지 파일 경로 수집
    for image_fpath in tqdm(sum([glob(osp.join(data_dir, f'{lang}_receipt/img/{split}/*')) for lang in LANGUAGE_LIST], [])):
        image_fnames.append(osp.basename(image_fpath))

        # 이미지를 BGR에서 RGB로 변환하여 읽기
        images.append(cv2.imread(image_fpath)[:, :, ::-1])
        if len(images) == batch_size:
            # 배치 단위로 감지 수행
            by_sample_bboxes.extend(detect(model, images, input_size))
            images = []

    # 남아있는 이미지에 대해 감지 수행
    if len(images):
        by_sample_bboxes.extend(detect(model, images, input_size))

    # UFO 형식의 결과 딕셔너리 초기화
    ufo_result = dict(images=dict())
    for image_fname, bboxes in zip(image_fnames, by_sample_bboxes):
        # 각 이미지에 대해 바운딩 박스를 포인트 리스트로 변환
        words_info = {idx: dict(points=bbox.tolist()) for idx, bbox in enumerate(bboxes)}
        ufo_result['images'][image_fname] = dict(words=words_info)

    return ufo_result


def main(args):
    """
    메인 함수. 모델 초기화 및 추론을 수행하고 결과를 저장합니다.
    
    입력:
        args (argparse.Namespace): 파싱된 명령줄 인자
    """
    # 모델 초기화 및 디바이스로 이동
    model = EAST(pretrained=False).to(args.device)

    # 체크포인트 파일 경로 설정
    ckpt_fpath = osp.join(args.model_dir, 'latest.pth')

    # 출력 디렉토리가 존재하지 않으면 생성
    if not osp.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print('Inference in progress')

    # 추론 수행
    ufo_result = dict(images=dict())
    split_result = do_inference(model, ckpt_fpath, args.data_dir, args.input_size,
                                args.batch_size, split='test')
    ufo_result['images'].update(split_result['images'])

    # 결과를 JSON 파일로 저장
    output_fname = 'output.json'
    with open(osp.join(args.output_dir, output_fname), 'w') as f:
        json.dump(ufo_result, f, indent=4)


if __name__ == '__main__':
    args = parse_args()
    main(args)
