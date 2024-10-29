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

# Define acceptable checkpoint file extensions
CHECKPOINT_EXTENSIONS = ['.pth', '.ckpt']

# List of supported languages for processing
LANGUAGE_LIST = ['chinese', 'japanese', 'thai', 'vietnamese']


def parse_args():
    """
    Parses command-line arguments for configuring the inference process.

    Returns:
        argparse.Namespace: Parsed arguments with default values if not provided.

    Raises:
        ValueError: If `input_size` is not a multiple of 32.
    """
    parser = ArgumentParser()

    # Directories for data, model, and output
    parser.add_argument('--data_dir', default=os.environ.get('SM_CHANNEL_EVAL', 'data'),
                        help='Directory containing evaluation data.')
    parser.add_argument('--model_dir', default=os.environ.get('SM_CHANNEL_MODEL', 'trained_models'),
                        help='Directory containing trained model checkpoints.')
    parser.add_argument('--output_dir', default=os.environ.get('SM_OUTPUT_DATA_DIR', 'predictions'),
                        help='Directory to save inference predictions.')

    # Device configuration: use CUDA if available, else CPU
    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu',
                        help='Device to use for inference (cuda or cpu).')
    parser.add_argument('--input_size', type=int, default=2048,
                        help='Size of the input images for the model.')
    parser.add_argument('--batch_size', type=int, default=5,
                        help='Number of images to process in a batch.')

    args = parser.parse_args()

    # Ensure input_size is a multiple of 32 for the EAST model
    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args


def do_inference(model, ckpt_fpath, data_dir, input_size, batch_size, split='test'):
    """
    Performs inference using the EAST model on a given dataset split.

    Args:
        model (torch.nn.Module): The EAST model instance.
        ckpt_fpath (str): File path to the model checkpoint.
        data_dir (str): Directory containing the evaluation data.
        input_size (int): Size of the input images for the model.
        batch_size (int): Number of images to process in a batch.
        split (str, optional): Dataset split to process (e.g., 'test'). Defaults to 'test'.

    Returns:
        dict: A dictionary containing image filenames and their corresponding detected bounding boxes.
    """
    # Load the model's state dictionary from the checkpoint
    model.load_state_dict(torch.load(ckpt_fpath, map_location='cpu'))
    model.eval()  # Set the model to evaluation mode

    image_fnames, by_sample_bboxes = [], []  # Lists to store image filenames and bounding boxes

    images = []  # List to accumulate images for batch processing

    # Iterate over all image file paths for the specified languages and split
    for image_fpath in tqdm(
            sum([glob(osp.join(data_dir, f'{lang}_receipt/img/{split}/*')) for lang in LANGUAGE_LIST], []),
            desc='Processing images'
    ):
        image_fnames.append(osp.basename(image_fpath))  # Extract and store the image filename

        # Read the image using OpenCV and convert from BGR to RGB
        images.append(cv2.imread(image_fpath)[:, :, ::-1])

        # If the batch is full, perform detection
        if len(images) == batch_size:
            # Detect bounding boxes for the current batch of images
            by_sample_bboxes.extend(detect(model, images, input_size))
            images = []  # Reset the images list for the next batch

    # Process any remaining images that didn't fill a complete batch
    if len(images):
        by_sample_bboxes.extend(detect(model, images, input_size))

    # Prepare the result dictionary in UFO (Universal Format for Objects) structure
    ufo_result = dict(images=dict())
    for image_fname, bboxes in zip(image_fnames, by_sample_bboxes):
        # For each bounding box, assign an index and convert it to a list
        words_info = {idx: dict(points=bbox.tolist()) for idx, bbox in enumerate(bboxes)}
        ufo_result['images'][image_fname] = dict(words=words_info)

    return ufo_result


def main(args):
    """
    The main function to execute the inference process.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    # Initialize the EAST model without pretraining
    model = EAST(pretrained=False).to(args.device)

    # Define the path to the latest checkpoint file
    ckpt_fpath = osp.join(args.model_dir, 'latest.pth')

    # Create the output directory if it doesn't exist
    if not osp.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print('Inference in progress...')

    # Perform inference on the 'test' split of the dataset
    split_result = do_inference(
        model=model,
        ckpt_fpath=ckpt_fpath,
        data_dir=args.data_dir,
        input_size=args.input_size,
        batch_size=args.batch_size,
        split='test'
    )

    # Initialize the final result dictionary
    ufo_result = dict(images=dict())
    ufo_result['images'].update(split_result['images'])

    # Define the output filename
    output_fname = 'output.csv'

    # Save the inference results as a JSON file
    with open(osp.join(args.output_dir, output_fname), 'w') as f:
        json.dump(ufo_result, f, indent=4)

    print(f'Inference completed. Results saved to {osp.join(args.output_dir, output_fname)}')


if __name__ == '__main__':
    args = parse_args()  # Parse command-line arguments
    main(args)           # Execute the inference process
