0. YOLOv5의 경우 >> 현재폴더에 !git clone https://github.com/ultralytics/yolov5.git으로 설치함
1. MEAT폴더내의 MEAT이미지들을 [Bordered.py]를 통해 바운딩 처리 및 라벨링파일생성
2. (1)을 통해 생성된 바운딩박스 및 라벨링된 이미지들(Bounding_Meat)를 [divide.py]를 통해 train,test,valid로 나누기.
3. (2)를 통해 생성된 (dataset)폴더 내부에 [meat_yaml.py]를 통해 yaml파일 생성
4. 프롬프트창 열어서 환경변수 끄기
set KMP_DUPLICATE_LIB_OK=TRUE
5. 학습진행
python train.py --img 416 --batch 16 --data C:/MeatDetection/dataset/meat.yaml --cfg C:/MeatDetection/yolov5/models/yolov5m.yaml --weights yolov5m.pt --name meat_yolov5s_results
