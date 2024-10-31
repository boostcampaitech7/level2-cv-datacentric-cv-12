import os

def rename_files(directory):
    file_list = sorted(os.listdir(directory))
    for idx, filename in enumerate(file_list):
        if filename.endswith('.jpg'):
            new_filename = f"extractor.en.in_house.appen_{idx:06d}_page0001.jpg"
            src = os.path.join(directory, filename)
            dst = os.path.join(directory, new_filename)
            os.rename(src, dst)
            print(f"Renamed: {filename} -> {new_filename}")

if __name__ == "__main__":
    image_dir = "/data/ephemeral/home/level2-cv-datacentric-cv-12/code/data/english_receipt/img/"
    rename_files(image_dir)
