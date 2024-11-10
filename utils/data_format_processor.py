import os
import json
from tqdm import tqdm

class DataProcessor:
    def __init__(self, input_folder_path=None, input_json_path=None, output_json_path=None, images_folder_path=None, start_number=None, type_name=None):
        self.input_folder_path = input_folder_path
        self.input_json_path = input_json_path
        self.output_json_path = output_json_path
        self.images_folder_path = images_folder_path
        self.start_number = start_number
        self.type_name = type_name
    
    def _load_json(self, file_path):
        '''
        JSON 파일을 로드하여 데이터를 반환하는 메서드
        '''
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _dump_json(self, file_path, data):
        '''
        데이터를 JSON 형식으로 파일에 저장합니다.
        '''
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f'Data saved to {file_path}')


    def merge_json_files(self):
        '''
        폴더 내 모든 JSON 파일을 병합하여 하나의 JSON 파일로 저장하는 메서드
        '''
        merged_data = []
        json_files = [f for f in os.listdir(self.input_folder_path) if f.endswith('.json')]
        
        for file_name in tqdm(json_files, desc="Merging JSON files"):
            json_path = os.path.join(self.input_folder_path, file_name)
            try:
                data = self._load_json(json_path)
                if isinstance(data, list):
                    merged_data.extend(data)
                elif isinstance(data, dict):
                    merged_data.append(data)
            except json.JSONDecodeError:
                print(f'Error reading file : {json_path}')
    
        self._dump_json(self.output_json_path, merged_data)

    def convert_to_ufo_format(self):
        '''
        기존 JSON 파일을 UFO 형식으로 변환하는 메서드
        '''
        original_data = self._load_json(self.input_json_path)
        
        # 전체 데이터를 담을 딕셔너리 초기화
        ufo_data = {
            "images": {}
        }

        for item in tqdm(original_data, desc="Converting JSON"):
            # 이미지 파일명과 크기 가져오기
            image_filename = f"{item['Image_File_name']}.jpg"
            image_width = item["Image_Width"]
            image_height = item["Image_Height"]

            # words 데이터 생성
            words = {}
            word_id = 1
            
            for row in item["Text_Coord"]:
                for coord_data in row:
                    bbox_info, transcription = coord_data
                    x, y, w, h, _, _ = bbox_info
                    points = [
                        [x, y],
                        [x + w, y + h],
                        [x + w, y + h],
                        [x, y, + h]
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
        
        self._dump_json(self.output_json_path, ufo_data)
    
    def rename_images_json(self):
        '''
        이미지 파일명과 JSON 데이터 이름을 변경하는 메서드
        '''
        data = self._load_json(self.output_json_path)

        images = data.get("images", {})
        new_images = {}
        number = self.start_number
        old_image_names = sorted(images.keys())
        image_files = os.listdir(self.images_folder_path)
        image_files_set = set(image_files)
        
        name_mapping = {}
        for old_name in tqdm(old_image_names, desc="Renaming image names in JSON data"):
            new_number_str = f"{number:06d}"
            extension = os.path.splitext(old_name)[1]
            new_image_name = f'extractor.{self.type_name}.in_house.appen_{new_number_str}_page0001{extension}'
            name_mapping[old_name] = new_image_name
            number += 1
        
        for old_name, image_data in images.items():
            new_name = name_mapping.get(old_name, old_name)
            new_images[new_name] = image_data
       
        data['images'] = new_images
        self._dump_json(self.output_json_path, data)

        for old_name, new_name in tqdm(name_mapping.items(), desc="Renaming image files"):
            old_image_path = os.path.join(self.images_folder_path, old_name)
            new_image_path = os.path.join(self.images_folder_path, new_name)

            if old_name in image_files_set:
                os.rename(old_image_path, new_image_path)
            else:
                print(f"Image file does not exist: {old_name}")
        
        print("Image file renaming completed.")

if __name__ == "__main__":

    # 경로 및 설정 초기화
    input_folder_path = 'path/to/json_folder'  # 병합할 JSON 파일들이 있는 폴더 경로
    input_json_path = 'path/to/input.json'     # 변환 또는 이름 변경을 위한 입력 JSON 파일 경로
    output_json_path = 'path/to/output.json'   # 결과가 저장될 출력 JSON 파일 경로
    images_folder_path = 'path/to/images'      # 이름을 변경할 이미지 파일들이 있는 폴더 경로
    start_number = 1                           # 이미지 이름의 시작 번호
    type_name = 'en'                           # 언어 지시자 (예: 'en', 'zh' 등)

    # DataProcessor 인스턴스 생성
    processor = DataProcessor(
        input_folder_path=input_folder_path,
        input_json_path=input_json_path,
        output_json_path=output_json_path,
        images_folder_path=images_folder_path,
        start_number=start_number,
        type_name=type_name
    )

    # JSON 파일 병합
    processor.merge_json_files()

    # # JSON 파일을 UFO 형식으로 변환
    processor.convert_to_ufo_format()

    # # 이미지와 JSON 파일의 이름 변경
    processor.rename_images_json()

