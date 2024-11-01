import os
import os.path as osp
import time
import math
from datetime import timedelta
from argparse import ArgumentParser
import numpy as np
import torch
from torch import cuda
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm

from baseline.east_dataset import EASTDataset
from dataset import SceneTextDataset
from baseline.model import EAST

# wandb 연동을 위한 라이브러리 설치
import wandb
import datetime
import random


def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--data_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', 'data'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR',
                                                                        'trained_models'))

    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--image_size', type=int, default=2048)
    parser.add_argument('--input_size', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--max_epoch', type=int, default=150)
    parser.add_argument('--save_interval', type=int, default=10)

    '''
        wandb 관련 parser를 정의합니다.
    '''
    parser.add_argument('--project_name', type=str, default='이름 미지정 프로젝트',
                        help='wandb 프로젝트 이름')
    parser.add_argument('--run_name', type=str, default=None,
                        help='wandb 실행 이름')

    # 학습 진행 상황 로그 및 체크포인트가 저장되는 폴더를 설정해줍니다.
    parser.add_argument('--log_checkpoint_dir', type=str, default=None,
                            help='로그와 체크포인트 파일 저장 경로. 지정하지 않으면 현재 시각 기반으로 생성됩니다.')

    args = parser.parse_args()

    '''
        만일 wandb run name을 지정하지 않았다면 현재 시간을 기준으로 이름을 설정하게 됩니다.
    '''
    if args.run_name is None:
        args.run_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    '''
        만일 log_checkpoint_dir 또는 model_dir이 지정되지 않았다면 현재 시각을 기반으로 디렉토리 이름을 생성합니다.
    '''
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if args.log_checkpoint_dir is None:
        args.log_checkpoint_dir = f"{timestamp}_checkpoint_log"


    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args

def do_training(data_dir, model_dir, device, image_size, input_size, num_workers, batch_size,
                learning_rate, max_epoch, save_interval, project_name, run_name, log_checkpoint_dir):

    # wandb 인스턴스 생성
    wandb.init(project=project_name, name=run_name)
    wandb.config.update({
        'data_dir': data_dir,
        'model_dir': model_dir,
        'device': device,
        'num_workers': num_workers,
        'image_size': image_size,
        'input_size': input_size,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'max_epoch': max_epoch,
        'save_interval': save_interval,
    })

    # 로그 및 체크포인트 디렉토리가 없는 경우 생성
    if not osp.exists(log_checkpoint_dir):
        os.makedirs(log_checkpoint_dir)
    if not osp.exists(model_dir):
        os.makedirs(model_dir)

    # 현재 시간을 문자열로 포맷팅
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # 로그 파일 이름을 run_name과 timestamp를 포함하여 생성
    log_file_name = f'{run_name}_{timestamp}_training_log.txt'
    log_file_path = osp.join(log_checkpoint_dir, log_file_name)
    log_file = open(log_file_path, 'a')

    # 데이터셋 로딩 (SceneTextDataset)
    dataset = SceneTextDataset(
        data_dir,
        split='train',
        image_size=image_size,
        crop_size=input_size,
    )
    dataset = EASTDataset(dataset)
    num_batches = math.ceil(len(dataset) / batch_size)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[max_epoch // 2], gamma=0.1)

    model.train()
    for epoch in range(max_epoch):
        epoch_loss, epoch_start = 0, time.time()

        with tqdm(total=num_batches, desc=f'[Epoch {epoch + 1}]', ncols=100) as pbar:
            for batch_idx, (img, gt_score_map, gt_geo_map, roi_mask) in enumerate(train_loader):
                img = img.to(device)
                gt_score_map = gt_score_map.to(device)
                gt_geo_map = gt_geo_map.to(device)
                roi_mask = roi_mask.to(device)

                loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_val = loss.item()
                epoch_loss += loss_val

                pbar.update(1)
                val_dict = {
                    'Cls loss': extra_info['cls_loss'],
                    'Angle loss': extra_info['angle_loss'],
                    'IoU loss': extra_info['iou_loss']
                }
                pbar.set_postfix(val_dict)

                # wandb에 손실 값 로깅
                wandb.log({
                    'train/total_loss': loss_val,
                    'train/cls_loss': extra_info['cls_loss'],
                    'train/angle_loss': extra_info['angle_loss'],
                    'train/iou_loss': extra_info['iou_loss'],
                    'epoch': epoch + 1
                })

                # 로그 파일에 기록
                if (batch_idx + 1) % 10 == 0:
                    log_message = (f'Epoch [{epoch + 1}/{max_epoch}], '
                                   f'Batch [{batch_idx + 1}/{num_batches}], '
                                   f'Loss: {loss_val:.4f}, '
                                   f'Cls loss: {extra_info["cls_loss"]:.4f}, '
                                   f'Angle loss: {extra_info["angle_loss"]:.4f}, '
                                   f'IoU loss: {extra_info["iou_loss"]:.4f}')
                    print(log_message, file=log_file)
                    log_file.flush()

        scheduler.step()

        epoch_duration = timedelta(seconds=time.time() - epoch_start)
        mean_loss = epoch_loss / num_batches
        epoch_message = f'Epoch [{epoch + 1}/{max_epoch}] Mean loss: {mean_loss:.4f} | Elapsed time: {epoch_duration}'
        print(epoch_message)
        print(epoch_message, file=log_file)
        log_file.flush()

        # wandb에 에폭당 손실 값 로깅
        wandb.log({'train/epoch_loss': mean_loss, 'epoch': epoch + 1})

        # 이미지 시각화 추가
        try:
            num_samples = 10  # 시각화할 이미지 수
            dataset_size = len(dataset)
            
            # 데이터셋 크기가 시각화할 이미지 수보다 작은 경우 처리
            if dataset_size < num_samples:
                num_samples = dataset_size
                print(f"데이터셋의 크기가 {num_samples}보다 작습니다. {num_samples}개의 이미지만 시각화합니다.")
            
            # 랜덤으로 이미지 인덱스 선택
            random_indices = random.sample(range(dataset_size), num_samples)
            sample_images = []
            
            for idx in random_indices:
                sample_img, word_bboxes, gt_geo_map, roi_mask = dataset[idx]  # 랜덤 인덱스의 샘플
                sample_img = sample_img.cpu().numpy().transpose(1, 2, 0)  # [C, H, W] -> [H, W, C]

                # 그레이스케일 이미지인 경우 채널 수가 1이므로 이를 처리
                if sample_img.shape[2] == 1:
                    # 정규화 되돌리기
                    mean = np.array([0.5])
                    std = np.array([0.5])
                    sample_img = (sample_img * std + mean) * 255.0
                    sample_img = sample_img.astype(np.uint8)
                else:
                    # 정규화 되돌리기
                    mean = np.array([0.5, 0.5, 0.5])
                    std = np.array([0.5, 0.5, 0.5])
                    sample_img = (sample_img * std + mean) * 255.0
                    sample_img = sample_img.astype(np.uint8)
                
                # wandb.Image 객체로 변환하고 캡션 추가
                img_caption = f"Sample {idx} - Epoch {epoch + 1}"
                sample_images.append(wandb.Image(sample_img, caption=img_caption))
            
            # wandb에 이미지 리스트 로깅
            wandb.log({"validation_sample_images": sample_images, "epoch": epoch + 1})

        except Exception as e:
            print(f"Error visualizing images: {e}")

        # 체크포인트 저장
        # 체크포인트 저장
        if (epoch + 1) % save_interval == 0:
            ckpt_fpath = osp.join(model_dir, f'{run_name}_{timestamp}_training_log_epoch{epoch + 1}.pth')
            torch.save(model.state_dict(), ckpt_fpath)
            
            # wandb.Artifact를 사용하여 체크포인트 업로드
            artifact = wandb.Artifact(f'model-checkpoint-epoch-{epoch + 1}', type='model')
            artifact.add_file(ckpt_fpath, name=f'checkpoint_epoch_{epoch + 1}.pth')
            wandb.log_artifact(artifact)
            
            print(f'Checkpoint saved and uploaded to wandb: {ckpt_fpath}')

    # 로그 파일 닫기 및 wandb 종료는 루프 밖에서 처리
    # 로그 파일 닫기
    log_file.close()
    # wandb 종료
    wandb.finish()




def main(args):
    do_training(**args.__dict__)

if __name__ == '__main__':
    args = parse_args()
    main(args)
