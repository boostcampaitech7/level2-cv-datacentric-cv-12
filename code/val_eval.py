import json
from deteval import calc_deteval_metrics

if __name__ == "__main__":
    # GT.json과 pred.json 파일을 읽어들입니다
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

    with open('/data/ephemeral/home/level2-cv-datacentric-cv-12/code/legend_8412.csv', 'r') as f:
        pred_data = json.load(f)

    pred_bboxes_dict = {}
    for img_name, img_data in pred_data['images'].items():
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

    # 성능 평가를 실행합니다
    metrics = calc_deteval_metrics(pred_bboxes_dict, gt_bboxes_dict)

    print(f"Precision: {metrics['total']['precision']:.4f}")
    print(f"Recall: {metrics['total']['recall']:.4f}")
    print(f"F1-score: {metrics['total']['hmean']:.4f}")