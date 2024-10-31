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

import wandb  # wandb 통합을 위해 추가

from east_dataset import EASTDataset
from dataset import SceneTextDataset
from model import EAST


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
    parser.add_argument('--save_interval', type=int, default=5)


    # wandb 관련 인자 추가
    parser.add_argument('--project_name', type=str, default='EAST_Text_Detection',
                        help='wandb 프로젝트 이름')
    parser.add_argument('--run_name', type=str, default=None,
                        help='wandb 실행 이름 (선택 사항)')

    # 로그와 체크포인트 저장 경로 추가
    parser.add_argument('--log_checkpoint_dir', type=str, default='logs_and_checkpoints',
                        help='로그와 체크포인트 파일 저장 경로')

    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args


def do_training(data_dir, model_dir, device, image_size, input_size, num_workers, batch_size,

                learning_rate, max_epoch, save_interval, project_name, run_name, log_checkpoint_dir):
    # wandb 초기화
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

    # 로그와 체크포인트 디렉토리 생성
    if not osp.exists(log_checkpoint_dir):
        os.makedirs(log_checkpoint_dir)

    # 로그 파일 설정
    log_file_path = osp.join(log_checkpoint_dir, 'training_log.txt')
    log_file = open(log_file_path, 'a')  # 이어쓰기 모드로 파일 열기


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


    device = torch.device(device)

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

        # 체크포인트 저장
        if (epoch + 1) % save_interval == 0:
            ckpt_fpath = osp.join(log_checkpoint_dir, f'epoch_{epoch + 1}.pth')
            torch.save(model.state_dict(), ckpt_fpath)
            wandb.save(ckpt_fpath)

    # 로그 파일 닫기
    log_file.close()
    # wandb 종료
    wandb.finish()



def main(args):
    do_training(**args.__dict__)

if __name__ == '__main__':
    args = parse_args()
    main(args)

