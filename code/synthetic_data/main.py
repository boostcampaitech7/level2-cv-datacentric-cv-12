import os
from data_generator import generate_and_save_documents
from json_utils import merge_json_files

# 생성할 파일 경로 및 설정
language = "생성하고자 하는 언어"
total_words = "그 언어의 단어 개수"
words_per_image = "한 이미지에 몇개의 단어를 쓸 건지"

# 데이터 생성 및 저장
generate_and_save_documents(total_words, language, words_per_image)

# 모든 이미지의 json 정보 merge
output_dir = f"g{language}_your_path"
os.makedirs(output_dir, exist_ok=True)
path = os.path.join(output_dir, f'train.json')

merge_json_files(f"json_g{language}", path)