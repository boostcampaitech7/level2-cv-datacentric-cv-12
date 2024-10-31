import json
import os

def update_annotations(annotation_path, image_dir):
    # 변경된 이미지 파일 이름 목록 생성
    new_filenames = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
    # 주석 파일 로드
    with open(annotation_path, 'r', encoding='utf-8') as f:
        annotations = json.load(f)

    # 기존 이미지 파일 이름 목록 추출
    old_filenames = sorted(annotations['images'].keys())

    # 파일 이름 매핑 생성
    filename_mapping = dict(zip(old_filenames, new_filenames))

    # 이미지 파일 이름 업데이트
    new_images = {}
    for old_name, new_name in filename_mapping.items():
        new_images[new_name] = annotations['images'][old_name]

    # 주석 데이터 업데이트
    annotations['images'] = new_images

    # 업데이트된 주석 파일 저장
    with open(annotation_path, 'w', encoding='utf-8') as f:
        json.dump(annotations, f, ensure_ascii=False, indent=4)

    print("Annotation file updated successfully.")

if __name__ == "__main__":
    annotation_file = "/data/ephemeral/home/level2-cv-datacentric-cv-12/code/data/english_receipt/ufo/train.json"
    image_dir = "/data/ephemeral/home/level2-cv-datacentric-cv-12/code/data/english_receipt/img/"
    update_annotations(annotation_file, image_dir)
