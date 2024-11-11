import json
import os

# UFO JSON 형식 데이터 생성 함수
def make_ufo_json(ufo_data, document, image_name):
    if not ufo_data:
        ufo_data["images"] = {
            image_name: {
                "words": {},
                "img_w": document["image"].width,
                "img_h": document["image"].height,
            }
        }
    else:
        ufo_data["images"][image_name] = {
            "words": {},
            "img_w": document["image"].width,
            "img_h": document["image"].height,
        }

    for idx, word in enumerate(document["words"], start=1):
        word_id = f"{idx:04d}"  # '0001', '0002', 등
        word_data = {
            "transcription": word.get("transcription", ""),  # 실제 텍스트 데이터가 있을 경우 추가
            "points": [[float(coord[0]), float(coord[1])] for coord in zip(word["bbox"][::2], word["bbox"][1::2])]
        }
        ufo_data["images"][image_name]["words"][word_id] = word_data

    return ufo_data

# JSON 파일 저장 함수
def save_ufo_json(ufo_data, output_path):
    with open(output_path, 'w') as f:
        json.dump(ufo_data, f, ensure_ascii=False, indent=4)

# 여러 JSON 파일을 하나로 합치는 함수
def merge_json_files(input_folder_path, output_json_path):
    merged_data = []
    for file_name in os.listdir(input_folder_path):
        if file_name.endswith('.json'):
            json_path = os.path.join(input_folder_path, file_name)
            with open(json_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        merged_data.extend(data)
                    elif isinstance(data, dict):
                        merged_data.append(data)
                except json.JSONDecodeError:
                    print(f'파일을 읽는 중 에러 발생 : {json_path}')

    merged_data = merged_data[0]
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)

    print(f"합쳐진 데이터를 {output_json_path}에 저장했습니다.")
