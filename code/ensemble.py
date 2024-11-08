import json
import numpy as np
from shapely.geometry import Polygon
from tqdm import tqdm
import os
from datetime import datetime
import argparse
from glob import glob
from deteval import calc_deteval_metrics
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args

# Load ground truth data
with open('/data/ephemeral/home/level2-cv-datacentric-cv-12/code/test.json', 'r') as f:
    gt_data = json.load(f)

gt_bboxes_dict = {}
for img_name, img_data in gt_data['images'].items():
    bboxes = []
    for _, word_data in img_data['words'].items():
        bbox = [
            min([point[0] for point in word_data['points']]),
            min([point[1] for point in word_data['points']]),
            max([point[0] for point in word_data['points']]),
            max([point[1] for point in word_data['points']])
        ]
        bboxes.append(bbox)
    gt_bboxes_dict[img_name] = bboxes

def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IOU) between two boxes
    Box format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    """
    poly1 = Polygon(box1)
    poly2 = Polygon(box2)
    
    if not poly1.is_valid or not poly2.is_valid:
        return 0.0
        
    try:
        intersection = poly1.intersection(poly2).area
        union = poly1.union(poly2).area
        
        if union == 0:
            return 0.0
            
        return intersection / union
    except:
        return 0.0

def ensemble_detections(csv_paths, iou_threshold, vote_count):
    """
    Ensemble multiple model results in UFO format
    """
    # Load all results
    all_results = []
    for path in tqdm(csv_paths, desc="Loading files"):
        with open(path, 'r') as f:
            all_results.append(json.load(f))
    
    ensemble_result = {'images': {}}
    
    # Collect all image names
    all_image_names = set()
    for result in all_results:
        all_image_names.update(result['images'].keys())
    
    for image_name in tqdm(all_image_names, desc="Processing images"):
        # Collect all boxes for each image
        all_boxes = []
        for model_idx, result in enumerate(all_results):
            if image_name in result['images']:
                words_info = result['images'][image_name]['words']
                for word_info in words_info.values():
                    box = word_info['points']
                    all_boxes.append({
                        'points': box,
                        'model_idx': model_idx
                    })
        
        if not all_boxes:
            continue
            
        # Perform box merging
        keep_boxes = []
        while len(all_boxes) > 0:
            current_box = all_boxes.pop(0)
            overlapping_boxes = [current_box]
            
            i = 0
            while i < len(all_boxes):
                iou = calculate_iou(current_box['points'], all_boxes[i]['points'])
                if iou > iou_threshold:
                    overlapping_boxes.append(all_boxes.pop(i))
                else:
                    i += 1
            
            # Check if enough models agree
            unique_models = len(set(box['model_idx'] for box in overlapping_boxes))
            if unique_models >= vote_count:
                # Calculate average of points
                merged_points = np.mean([box['points'] for box in overlapping_boxes], axis=0)
                keep_boxes.append(merged_points.tolist())
        
        # Generate result in UFO format
        words_info = {str(idx): {'points': box} for idx, box in enumerate(keep_boxes)}
        ensemble_result['images'][image_name] = {'words': words_info}
    
    return ensemble_result

def calculate_f1(ensemble_result):
    """
    Calculate F1 score for the ensemble result
    """
    pred_bboxes_dict = {}

    for img_name, img_data in ensemble_result['images'].items():
        bboxes = []
        for _, word_data in img_data['words'].items():
            bbox = [
                min([point[0] for point in word_data['points']]),
                min([point[1] for point in word_data['points']]),
                max([point[0] for point in word_data['points']]),
                max([point[1] for point in word_data['points']])
            ]
            bboxes.append(bbox)
        pred_bboxes_dict[img_name] = bboxes

    metrics = calc_deteval_metrics(pred_bboxes_dict, gt_bboxes_dict)
    f1_score = metrics['total']['hmean']
    return f1_score

@use_named_args(dimensions=[Real(0.2, 0.9, name='iou_threshold'),
                           Integer(1, 3, name='vote_count')])
def objective(iou_threshold, vote_count):
    """
    Objective function for Bayesian optimization
    """
    print(f"Current parameters: iou_threshold={iou_threshold:.4f}, vote_count={vote_count}")

    input_dir = "/data/ephemeral/home/level2-cv-datacentric-cv-12/code/pred"
    csv_paths = sorted(glob(os.path.join(input_dir, '*.csv')))
    ensemble_result = ensemble_detections(csv_paths, iou_threshold, vote_count)
    f1_score = calculate_f1(ensemble_result)
    return -f1_score  # Negative for maximization problem

def run_optimization(args):
    """
    Run Bayesian optimization to find optimal parameters
    """
    input_dir = '/data/ephemeral/home/level2-cv-datacentric-cv-12/code/pred'
    csv_paths = sorted(glob(os.path.join(input_dir, '*.csv')))
    if not csv_paths:
        raise ValueError(f"No CSV files found in input directory: {input_dir}")
    
    res = gp_minimize(objective, 
                     [Real(0.3, 0.7, name='iou_threshold'), 
                      Integer(2, 4, name='vote_count')],
                     n_calls=100,
                     n_random_starts=5,
                     verbose=True)
    
    print(f"Optimal parameters: iou_threshold={res.x[0]:.2f}, vote_count={int(res.x[1])}")
    return res.x[0], int(res.x[1])

def run_ensemble(args):
    """
    Run ensemble with given or optimal parameters
    """
    input_dir = '/data/ephemeral/home/level2-cv-datacentric-cv-12/code/pred'
    csv_paths = sorted(glob(os.path.join(input_dir, '*.csv')))
    if not csv_paths:
        raise ValueError(f"No CSV files found in input directory: {input_dir}")
    
    ensemble_result = ensemble_detections(csv_paths, args.iou_threshold, args.vote_count)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = f"{timestamp}_ensemble_iou{args.iou_threshold:.2f}_vote{args.vote_count}.json"
    output_path = os.path.join(args.output_dir, output_name)
    with open(output_path, 'w') as f:
        json.dump(ensemble_result, f, indent=4)
    print(f"Results saved: {output_path}")

    f1_score = calculate_f1(ensemble_result)
    print(f"F1 Score: {f1_score:.4f}")

def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='Script to ensemble results from multiple models')
    parser.add_argument('--mode', type=str, required=True, choices=['opt', 'ensemble'], 
                        help='Mode of operation: opt for optimization, ensemble for ensembling')
    parser.add_argument('--input_dir', type=str, help='Directory containing input CSV files')
    parser.add_argument('--output_dir', type=str, help='Directory to save output files')
    parser.add_argument('--iou_threshold', type=float, help='IOU threshold for box merging')
    parser.add_argument('--vote_count', type=int, help='Minimum number of models required to keep a box')
    return parser.parse_args()

def main():
    args = parse_args()
    if args.mode == 'opt':
        iou_threshold, vote_count = run_optimization(args)
        args.iou_threshold = iou_threshold
        args.vote_count = vote_count
        run_ensemble(args)
    elif args.mode == 'ensemble':
        run_ensemble(args)

if __name__ == '__main__':
    main()