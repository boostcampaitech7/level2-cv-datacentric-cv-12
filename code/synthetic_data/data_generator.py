import json
import os
from trdg.generators import GeneratorFromStrings
from synth_utils import make_document, partial_copy
from transform import perturb_document_inplace
from json_utils import make_ufo_json, save_ufo_json
from PIL import Image

# 텍스트 이미지 생성 함수
def get_words(language, count=128):
    # 언어별 텍스트 추출
    json_path = f"/path/to/your_{language}_train.json"
    texts = []
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for image_data in data["images"].values():
            for word_data in image_data["words"].values():
                texts.append(word_data["transcription"])
    # 공백 제거
    texts = [x for x in texts if x]

    # 언어별 폰트 설정
    font_dict = {
        "chinese": ("/fonts/NotoSansSC-VariableFont_wght.ttf", "cn"),
        "japanese": ("/fonts/NotoSansJP-VariableFont_wght.ttf", "ja"),
        "thai": ("/fonts/Sarabun-Regular.ttf", "th"),
        "vietnamese": ("/fonts/NotoSans-Regular.ttf", "vi")
    }

    # 해당 언어에 맞는 폰트를 가져옴
    font_path, generator_language = font_dict.get(language, (None, None))

    # 폰트 파일 존재 여부 확인
    if not font_path or not os.path.exists(font_path):
        raise ValueError(f"Invalid or missing font for language: {language}")
    
    # TextRecognitionDataGenerator의 GeneratorFromStrings를 사용해 이미지 생성
    generator = GeneratorFromStrings(
        texts, 
        language=generator_language, 
        size=64, 
        count=count, 
        fonts=[font_path],  # 언어에 맞는 폰트를 설정
        margins=(1, 5, 5, 5)
    )
    words = [{"patch": patch, "text": text, "size": patch.size, "margin": 5} for patch, text in generator]
    return words

# 단어 리스트로 문서 이미지를 생성하고 저장하는 함수
def generate_and_save_documents(total_words, language, words_per_image):
    output_dir = f"g{language}_your_path"
    json_dir = f"json_g{language}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)

    language_dict = {
        "chinese": "gzh",
        "japanese": "gja",
        "thai": "gth",
        "vietnamese": "gvi"
        }
    
    # 전체 단어 리스트 가져오기
    all_words = get_words(language, total_words)

    # words_per_image개씩 분할하여 이미지 생성 및 저장
    for i in range(0, len(all_words), words_per_image):
        # 현재 세트의 단어들 가져오기
        words_subset = all_words[i:i + words_per_image]

        # 문서 이미지 생성
        document = make_document(words_subset)

        # PIL Image 객체로 변환하여 저장
        image_filename = f"{language_dict[language]}_yout_image_filename1.png"
        image_path = os.path.join(output_dir, image_filename)
        document['image'].save(image_path)

        # wrap perspective 변환
        doc2 = partial_copy(document)
        perturb_document_inplace(doc2)

        image_filename2 = f"{language_dict[language]}_yout_image_filename2.png"
        image_path = os.path.join(output_dir, image_filename2)
        doc2['image'].save(image_path)

        # UFO 형식 JSON 데이터 생성
        ufo_json = {}
        ufo_json = make_ufo_json(ufo_json, document, image_filename)
        ufo_json = make_ufo_json(ufo_json, doc2, image_filename2)

        # JSON 파일 저장
        json_filename = f"{i // words_per_image + 1}.json"
        json_path = os.path.join(json_dir, json_filename)
        save_ufo_json(ufo_json, json_path)

