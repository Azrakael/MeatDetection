# %%
import torch
from PIL import Image
from pathlib import Path

# YOLOv5 모델 불러오기
model = torch.hub.load('ultralytics/yolov5', 'custom', path='./yolov5/runs/train/meat_yolov5s_results3/weights/best.pt')

# 이미지 불러오기
img = Image.open('./apple.jpg')

# 디텍션 수행
results = model(img)

# 디텍션 결과 저장 폴더 설정
save_dir = Path('./detect')

# 결과 저장 및 화면에 표시
results.save(save_dir=save_dir)
results.show() 

# 저장된 결과 경로 출력
save_path = save_dir / 'image1.jpg'
print(f"디텍션된 이미지가 {save_path}에 저장")


#%%
import torch
from pathlib import Path
from PIL import Image

# YOLOv5 모델 불러오기
model = torch.hub.load('ultralytics/yolov5', 'custom', path='./yolov5/runs/train/meat_yolov5s_results2/weights/best.pt')

# 이미지가 있는 폴더 경로
image_dir = Path('./dataset/test/images')

# 결과를 저장할 폴더 경로
save_dir = Path('./dataset/test/detected')

# 결과 저장 폴더가 없다면 생성
save_dir.mkdir(parents=True, exist_ok=True)

# 이미지 디렉토리에서 모든 이미지 파일을 순회
for image_path in image_dir.glob('*.jpg'):
    # 이미지 불러오기
    img = Image.open(image_path)

    # 디텍션 수행
    results = model(img)

    # 결과 이미지를 처리
    for img in results.render():
        # PIL 이미지 객체 생성
        output_image = Image.fromarray(img)
        
        # 결과 이미지 저장 이름 설정
        save_path = save_dir / f"{image_path.stem}_detected.jpg"
        
        # 이미지 저장
        output_image.save(save_path)

# 모든 이미지 처리 완료
print(f"모든 이미지가 {save_dir}에 저장")

# %%
import torch
from pathlib import Path
from PIL import Image

# YOLOv5 모델 불러오기
model = torch.hub.load('ultralytics/yolov5', 'custom', path='./yolov5/runs/train/meat_yolov5s_results3/weights/best.pt')

# 이미지가 있는 폴더 경로
image_dir = Path('./dataset/test/images')

# 결과를 저장할 폴더 경로
save_dir = Path('./dataset/test/detected2')

# 결과 저장 폴더가 없다면 생성
save_dir.mkdir(parents=True, exist_ok=True)

# 이미지 디렉토리에서 모든 이미지 파일을 순회
for image_path in image_dir.glob('*.jpg'):
    # 이미지 불러오기
    img = Image.open(image_path)

    # 디텍션 수행
    results = model(img)

    # 결과 이미지를 처리
    for img in results.render():
        # PIL 이미지 객체 생성
        output_image = Image.fromarray(img)
        
        # 결과 이미지 저장 이름 설정
        save_path = save_dir / f"{image_path.stem}_detected.jpg"
        
        # 이미지 저장
        output_image.save(save_path)

# 모든 이미지 처리 완료
print(f"모든 이미지가 {save_dir}에 저장")

# %%
