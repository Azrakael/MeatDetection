#%%
from sklearn.model_selection import train_test_split
import os
import shutil

# 폴더 경로 설정
base_dir = './Bounding_Meat'
train_dir = './dataset/train'
test_dir = './dataset/test'
valid_dir = './dataset/valid'

# 이미지 파일 목록 생성
all_files = [f for f in os.listdir(base_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

# train_test_split을 사용하여 데이터 분할
train_files, test_valid_files = train_test_split(all_files, test_size=0.3, random_state=42)
test_files, valid_files = train_test_split(test_valid_files, test_size=0.66, random_state=42)

# 필요한 폴더 생성
for folder in [train_dir, test_dir, valid_dir]:
    os.makedirs(os.path.join(folder, 'images'), exist_ok=True)
    os.makedirs(os.path.join(folder, 'labels'), exist_ok=True)

# 분할된 데이터를 디렉토리에 복사
def copy_files(files, dest_folder):
    images_folder = os.path.join(dest_folder, 'images')
    labels_folder = os.path.join(dest_folder, 'labels')
    for file in files:
        shutil.copy(os.path.join(base_dir, file), os.path.join(images_folder, file))
        txt_file = file.rsplit('.', 1)[0] + '.txt'
        shutil.copy(os.path.join(base_dir, txt_file), os.path.join(labels_folder, txt_file))

copy_files(train_files, train_dir)
copy_files(test_files, test_dir)
copy_files(valid_files, valid_dir)
