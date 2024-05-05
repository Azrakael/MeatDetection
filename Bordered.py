#%%
import cv2
import numpy as np
import os

def bounding_box(img):
    # HSV 색상 공간으로 변환
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 고기의 붉은색 부분에 대한 HSV 범위 정의
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    # 빨간색 부분을 마스킹
    mask_red1 = cv2.inRange(hsv_img, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv_img, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    blurred_mask_red = cv2.GaussianBlur(mask_red, (9, 9), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    morphed_mask_red = cv2.morphologyEx(blurred_mask_red, cv2.MORPH_CLOSE, kernel)
    morphed_mask_red = cv2.morphologyEx(morphed_mask_red, cv2.MORPH_OPEN, kernel)

    # 마스크로부터 컨투어 찾기
    contours, _ = cv2.findContours(morphed_mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 컨투어 중 가장 큰 컨투어만 찾아서 바운딩박스처리
    # 가장 큰 컨투어에 대한 바운딩박스 처리 & 가장 큰 컨투어만 반환
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 500:
            x, y, w, h = cv2.boundingRect(largest_contour)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 10)
            return img, [largest_contour]
    return img, []


# 욜로 포맷에는 다음정보가 포함되어야함(클래스번호, x_center, y_center, width, height)
# x_center는 x좌표의 중심점을 이미지 너비로 나눈 값
# y_center는 y좌표의 중심점을 이미지 높이로 나눈 값
# width는 바운딩박스의 너비를 이미지 너비로 나눈 값
# height는 바운딩박스의 높이를 이미지 높이로 나눈 값
# 클래스번호는 0으로 설정 >> 육류만 분류하기에 클래스 1개만 존재함.
def write_labels(txt_path, contours, img_width, img_height):
    with open(txt_path, 'w') as f:
        # YOLO 포맷으로 레이블링
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x_center = (x + w / 2) / img_width
            y_center = (y + h / 2) / img_height
            width = w / img_width
            height = h / img_height
            f.write(f'0 {x_center} {y_center} {width} {height}\n')


# 경로 설정
base_dir = "./MEAT"
bounding_dir = "./Bounding_Meat"

# Bounding_Meat 디렉토리가 없다면 생성
if not os.path.exists(bounding_dir):
    os.makedirs(bounding_dir)

# 이미지 처리
for file_name in os.listdir(base_dir):
    if file_name.endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(base_dir, file_name)
        img = cv2.imread(img_path)
        if img is not None:
            # bounding_box 함수를 이용하여 이미지 처리
            processed_img, contours = bounding_box(img) 
            # 이미지와 레이블 저장
            save_path = os.path.join(bounding_dir, file_name)
            txt_path = os.path.join(bounding_dir, file_name.rsplit('.', 1)[0] + '.txt')
            write_labels(txt_path, contours, img.shape[1], img.shape[0])
            cv2.imwrite(save_path, processed_img)
            print(f'이미지 처리됨: {save_path}')
            print(f'라벨 처리됨: {txt_path}')
        else:
            print(f'이미지 로드 실패: {img_path}')
# %%
