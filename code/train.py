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

def parse_args():
    """
    Parses command-line arguments for configuring the training process.

    Returns:
        argparse.Namespace: Parsed arguments with default values if not provided.
    
    Raises:
        ValueError: If `input_size` is not a multiple of 32.
    """
    parser = ArgumentParser()

    # Data and model directories with environment variable defaults
    parser.add_argument('--data_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', 'data'),
                        help='Directory containing training data.')
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR',
                                                                        'trained_models'),
                        help='Directory to save trained models.')

    # Device configuration: use CUDA if available, else CPU
    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu',
                        help='Device to use for training (cuda or cpu).')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of worker processes for data loading.')

    # Model and training hyperparameters
    parser.add_argument('--image_size', type=int, default=2048,
                        help='Size of the input images.')
    parser.add_argument('--input_size', type=int, default=1024,
                        help='Size of the cropped input for the model.')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Number of samples per training batch.')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate for the optimizer.')
    parser.add_argument('--max_epoch', type=int, default=150,
                        help='Maximum number of training epochs.')
    parser.add_argument('--save_interval', type=int, default=5,
                        help='Interval (in epochs) to save the model checkpoint.')
    
    args = parser.parse_args()

    # Ensure input_size is a multiple of 32 for the EAST model
    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args


def do_training(data_dir, model_dir, device, image_size, input_size, num_workers, batch_size,
               learning_rate, max_epoch, save_interval):
    """
    Executes the training loop for the EAST model on scene text detection.

    Args:
        data_dir (str): Directory containing the training data.
        model_dir (str): Directory to save trained model checkpoints.
        device (str): Device to perform computations on ('cuda' or 'cpu').
        image_size (int): Size of the input images.
        input_size (int): Size of the cropped input for the model.
        num_workers (int): Number of worker processes for data loading.
        batch_size (int): Number of samples per training batch.
        learning_rate (float): Learning rate for the optimizer.
        max_epoch (int): Maximum number of training epochs.
        save_interval (int): Interval (in epochs) to save the model checkpoint.
    """
    # Initialize the dataset for training
    dataset = SceneTextDataset(
        data_dir=data_dir,
        split='train',
        image_size=image_size,
        crop_size=input_size,
    )
    
    # Wrap the dataset with EAST-specific preprocessing
    dataset = EASTDataset(dataset)
    
    # Calculate the total number of batches per epoch
    num_batches = math.ceil(len(dataset) / batch_size)
    
    # Create a DataLoader for batching and shuffling the data
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    # Set the device for computation
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Initialize the EAST model and move it to the specified device
    model = EAST()
    model.to(device)
    
    # Define the optimizer (Adam) and the learning rate scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[max_epoch // 2], gamma=0.1)

    # Set the model to training mode
    model.train()
    
    # Iterate over each epoch
    for epoch in range(max_epoch):
        epoch_loss = 0  # Accumulates the total loss for the epoch
        epoch_start = time.time()  # Record the start time of the epoch
        
        # Initialize a progress bar for the current epoch
        with tqdm(total=num_batches) as pbar:
            pbar.set_description(f'[Epoch {epoch + 1}]')  # Set the epoch description

            # Iterate over batches in the DataLoader
            for img, gt_score_map, gt_geo_map, roi_mask in train_loader:
                # Forward and backward pass through the model
                loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                
                optimizer.zero_grad()  # Reset gradients
                loss.backward()        # Backpropagate the loss
                optimizer.step()       # Update model parameters

                loss_val = loss.item()  # Get the scalar loss value
                epoch_loss += loss_val  # Accumulate the loss

                pbar.update(1)  # Update the progress bar

                # Prepare additional metrics for display
                val_dict = {
                    'Cls loss': extra_info['cls_loss'],
                    'Angle loss': extra_info['angle_loss'],
                    'IoU loss': extra_info['iou_loss']
                }
                pbar.set_postfix(val_dict)  # Update progress bar with metrics

        scheduler.step()  # Update the learning rate scheduler

        # Calculate and display the mean loss and elapsed time for the epoch
        print('Mean loss: {:.4f} | Elapsed time: {}'.format(
            epoch_loss / num_batches, timedelta(seconds=time.time() - epoch_start)))
        
        # Save the model checkpoint at specified intervals
        if (epoch + 1) % save_interval == 0:
            if not osp.exists(model_dir):
                os.makedirs(model_dir)  # Create the model directory if it doesn't exist

            ckpt_fpath = osp.join(model_dir, 'latest.pth')  # Define the checkpoint file path
            torch.save(model.state_dict(), ckpt_fpath)  # Save the model state


def main(args):
    """
    The main function to start the training process with parsed arguments.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    do_training(**args.__dict__)  # Unpack arguments and pass to do_training


if __name__ == '__main__':
    args = parse_args()  # Parse command-line arguments
    main(args)           # Start the training process
