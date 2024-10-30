import os
import os.path as osp
import time
import math
from datetime import timedelta
from argparse import ArgumentParser

import torch
from torch import cuda
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm

from east_dataset import EASTDataset
from dataset import SceneTextDataset
from model import EAST


# 체크포인트 파일의 확장자 리스트
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

    # 일반적인 인자들 설정
    parser.add_argument('--data_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', 'data'),
                        help='학습 데이터가 저장된 디렉토리 경로')
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR',
                                                                        'trained_models'),
                        help='모델 체크포인트를 저장할 디렉토리 경로')

    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu',
                        help='모델을 실행할 디바이스 (cuda 또는 cpu)')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='데이터 로딩 시 사용할 워커 수')

    parser.add_argument('--image_size', type=int, default=2048,
                        help='원본 이미지의 크기')
    parser.add_argument('--input_size', type=int, default=1024,
                        help='모델 입력 이미지의 크기 (32의 배수여야 함)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='한 번에 처리할 배치 크기')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='학습률')
    parser.add_argument('--max_epoch', type=int, default=150,
                        help='최대 에포크 수')
    parser.add_argument('--save_interval', type=int, default=5,
                        help='모델 체크포인트를 저장할 에포크 간격')

    args = parser.parse_args()

    # input_size가 32의 배수가 아닌 경우 오류 발생
    if args.input_size % 32 != 0:
        raise ValueError('`input_size`는 32의 배수여야 합니다.')

    return args


def do_training(data_dir, model_dir, device, image_size, input_size, num_workers, batch_size,
                learning_rate, max_epoch, save_interval):
    """
    EAST 모델을 학습하는 함수입니다.

    입력:
        data_dir (str): 학습 데이터가 저장된 디렉토리 경로
        model_dir (str): 모델 체크포인트를 저장할 디렉토리 경로
        device (str): 학습에 사용할 디바이스 ('cuda' 또는 'cpu')
        image_size (int): 원본 이미지의 크기
        input_size (int): 모델 입력 이미지의 크기
        num_workers (int): 데이터 로딩 시 사용할 워커 수
        batch_size (int): 한 번에 처리할 배치 크기
        learning_rate (float): 학습률
        max_epoch (int): 최대 에포크 수
        save_interval (int): 모델 체크포인트를 저장할 에포크 간격
    """
    # 데이터셋 생성
    dataset = SceneTextDataset(
        data_dir,
        split='train',
        image_size=image_size,
        crop_size=input_size,
    )
    # EASTDataset으로 래핑
    dataset = EASTDataset(dataset)
    num_batches = math.ceil(len(dataset) / batch_size)  # 총 배치 수 계산
    # 데이터로더 생성
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    # 디바이스 설정
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 모델 초기화 및 디바이스로 이동
    model = EAST()
    model.to(device)
    # 옵티마이저 설정 (Adam 사용)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # 학습률 스케줄러 설정 (에포크 절반 지점에서 학습률 0.1배 감소)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[max_epoch // 2], gamma=0.1)

    model.train()  # 모델을 학습 모드로 설정
    for epoch in range(max_epoch):
        epoch_loss, epoch_start = 0, time.time()  # 에포크 손실 초기화 및 시작 시간 기록
        # tqdm을 사용하여 진행 상황 표시
        with tqdm(total=num_batches) as pbar:
            for img, gt_score_map, gt_geo_map, roi_mask in train_loader:
                pbar.set_description('[Epoch {}]'.format(epoch + 1))  # 현재 에포크 표시

                # 학습 단계에서 손실 계산
                loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                optimizer.zero_grad()  # 옵티마이저의 기울기 초기화
                loss.backward()        # 역전파 수행
                optimizer.step()       # 옵티마이저 업데이트

                loss_val = loss.item()
                epoch_loss += loss_val  # 에포크 손실에 현재 배치 손실 추가

                pbar.update(1)  # 진행 바 업데이트
                val_dict = {
                    'Cls loss': extra_info['cls_loss'], 
                    'Angle loss': extra_info['angle_loss'],
                    'IoU loss': extra_info['iou_loss']
                }
                pbar.set_postfix(val_dict)  # 손실 값 표시

        scheduler.step()  # 학습률 스케줄러 업데이트

        # 에포크 종료 후 평균 손실과 소요 시간 출력
        print('Mean loss: {:.4f} | Elapsed time: {}'.format(
            epoch_loss / num_batches, timedelta(seconds=time.time() - epoch_start)))

        # 지정된 간격마다 모델 체크포인트 저장
        if (epoch + 1) % save_interval == 0:
            if not osp.exists(model_dir):
                os.makedirs(model_dir)

            ckpt_fpath = osp.join(model_dir, 'latest.pth')
            torch.save(model.state_dict(), ckpt_fpath)
            print(f'Model checkpoint saved at epoch {epoch + 1} to {ckpt_fpath}')


def main(args):
    """
    메인 함수. 인자로 받은 설정을 바탕으로 학습을 수행합니다.

    입력:
        args (argparse.Namespace): 파싱된 명령줄 인자
    """
    do_training(**args.__dict__)  # 인자를 해체하여 do_training 함수에 전달


if __name__ == '__main__':
    """
    스크립트의 진입점입니다.
    
    명령줄 인자를 파싱하고, 메인 함수를 호출하여 학습을 시작합니다.
    """
    args = parse_args()
    main(args)
