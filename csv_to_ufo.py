import csv
import json
import os
import re

# CSV 파일들이 위치한 디렉토리 경로를 지정합니다.
csv_directory = '/data/ephemeral/home/level2-cv-datacentric-cv-12/ICDAR-2019-SROIE-master/data/box'

# 변환된 JSON 파일을 저장할 디렉토리와 파일 이름을 지정합니다.
json_output_directory = '/data/ephemeral/home/level2-cv-datacentric-cv-12'
json_output_filename = 'output_sorted.json'
json_file_path = os.path.join(json_output_directory, json_output_filename)

def natural_sort_key(s):
    """파일 이름을 자연스러운 숫자 순서로 정렬하기 위한 키 함수"""
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', s)]

def convert_multiple_csv_to_single_ufo_json(csv_directory, json_file_path):
    ufo_data = {'images': {}}
    # 파일 이름을 숫자 순서로 정렬합니다.
    csv_files = [f for f in os.listdir(csv_directory) if f.endswith('.csv')]
    csv_files.sort(key=natural_sort_key)

    for filename in csv_files:
        csv_file_path = os.path.join(csv_directory, filename)
        image_filename = os.path.splitext(filename)[0] + '.jpg'  # 이미지 파일 이름 추정
        words = {}
        word_id = 1
        with open(csv_file_path, 'r', encoding='utf-8') as csvfile:
            csvreader = csv.reader(csvfile)
            for row in csvreader:
                if len(row) < 9:
                    continue  # 필요한 데이터가 부족한 행은 건너뜁니다.
                try:
                    # 좌표 추출
                    x1 = float(row[0])
                    y1 = float(row[1])
                    x2 = float(row[2])
                    y2 = float(row[3])
                    x3 = float(row[4])
                    y3 = float(row[5])
                    x4 = float(row[6])
                    y4 = float(row[7])
                    # 텍스트 추출 (콤마로 연결)
                    transcription = ','.join(row[8:]).strip()
                    # 고유한 단어 ID 생성
                    word_key = '{:04d}'.format(word_id)
                    # 단어 데이터 구성
                    words[word_key] = {
                        'transcription': transcription,
                        'points': [
                            [x1, y1],
                            [x2, y2],
                            [x3, y3],
                            [x4, y4]
                        ]
                    }
                    word_id += 1
                except ValueError:
                    continue  # 유효하지 않은 데이터가 있는 행은 건너뜁니다.
        # 이미지 데이터에 단어 추가
        ufo_data['images'][image_filename] = {
            'paragraphs': {},
            'words': words
        }
    # JSON 파일로 저장
    os.makedirs(os.path.dirname(json_file_path), exist_ok=True)
    with open(json_file_path, 'w', encoding='utf-8') as jsonfile:
        json.dump(ufo_data, jsonfile, ensure_ascii=False, indent=4)

# 함수 실행
convert_multiple_csv_to_single_ufo_json(csv_directory, json_file_path)
