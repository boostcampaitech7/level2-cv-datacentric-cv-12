import json
import os
import random

input_folder = '/data/ephemeral/home/level2-cv-datacentric-cv-12/code/data/classic_receipt/original'
output_folder = '/data/ephemeral/home/level2-cv-datacentric-cv-12/code/data/classic_receipt/ufo'

# 출력할 폴더가 없으면 생성
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 전체 데이터를 담을 딕셔너리 초기화
ufo_data = {
    "images": {}
}

# 입력 폴더의 모든 JSON 파일 리스트 가져오기
all_files = [filename for filename in os.listdir(input_folder) if filename.endswith('.json')]
# 사용할 파일 수 설정
num_files_to_use = 300

# 파일 리스트를 랜덤하게 섞기 (원하시는 경우 주석 해제)
random.shuffle(all_files)

# 앞에서부터 300개의 파일 선택
selected_files = all_files[:num_files_to_use]

# 입력 폴더의 모든 JSON 파일에 대해 처리
for filename in selected_files:
    if filename.endswith('.json'):
        input_path = os.path.join(input_folder, filename)
        # 원본 JSON 파일 읽기
        with open(input_path, 'r', encoding='utf-8') as f:
            original_data = json.load(f)

            # 이미지 파일명과 크기 가져오기
            image_filename = f"{original_data['Image_File_name']}.jpg"
            image_width = original_data['Image_Width']
            image_height = original_data['Image_Height']

            # words 데이터 생성
            words = {}
            word_id = 1

            for row in original_data["Text_Coord"]:
                for coord_data in row:
                    bbox_info, transcription = coord_data
                    x, y, w, h, _, _ = bbox_info
                    points = [
                        [x, y],
                        [x + w, y],
                        [x + w, y + h],
                        [x, y + h]
                    ]
                    words[f"{word_id:04d}"] = {
                        "transcription": transcription,
                        "points": points
                    }
                    word_id += 1
            
            # 각 이미지의 데이터를 ufo_data에 추가
            ufo_data["images"][image_filename] = {
                "paragraphs": {},
                "words": words,
                "img_w": image_width,
                "img_h": image_height
            }
            
# 변환된 데이터를 출력 폴더에 저장
output_path = os.path.join(output_folder, "train.json")
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(ufo_data, f, ensure_ascii=False, indent=2)

print(f"전체 데이터가 {output_path}에 저장되었습니다.")