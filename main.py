import cv2
import numpy as np

# 모델 불러오기 : 커피머신 준비
net = cv2.dnn.readNetFromTorch('models/eccv16/eccv16/la_muse.t7')
net2 = cv2.dnn.readNetFromTorch('models/instance_norm/mosaic.t7')
# 이미지 불러오기  : 원두 준비
img = cv2.imread('imgs/lution1.png')

# 이미지 전처리
h, w, c = img.shape
# 가로: 500, 세로: 이미지 비율 유지 해서 리사이즈
img = cv2.resize(img, dsize=(500, int(h / w * 500)))
print(img.shape) # (325, 500, 3)


MEAN_VALUE = [103.939, 116.779, 123.680]
# 이미지를 전처리한다.  blobFromImage 차원변형을 도와준다.
blob = cv2.dnn.blobFromImage(img, mean=MEAN_VALUE)
print(blob.shape) # (1, 3, 325, 500)

# 전처리 후 결과 추론하기 Inference
net.setInput(blob)  # 전처리한 이미지(blob)를 모델에 넣고
output = net.forward()  # 추론해서 outpu 변수에 결과를 저장합니다.

# 후처리
output = output.squeeze().transpose((1, 2, 0)) # squeeze 로 차원을 줄여주고, transpose 로 차원변형을 다시 거꾸로 해준다.
output += MEAN_VALUE  # 전처리에서 빼준 MEAN_VALUE 를 다시 더해준다.

output = np.clip(output, 0, 255) # 0 에서 255 로 크기를 제한한다. 255가 넘으면 255로 바꿔줘라!
output = output.astype('uint8')  # 사람이 볼수 있는 형태로 바꾸기 위해서 정수 형태로 바꾸면 사람이 볼 수 있는 이미지 형태로 바뀐다.

# 전처리 후 결과 추론하기 Inference
net2.setInput(blob)  # 전처리한 이미지(blob)를 모델에 넣고
output2 = net2.forward()  # 추론해서 outpu 변수에 결과를 저장합니다.

# 후처리
output2 = output2.squeeze().transpose((1, 2, 0)) # squeeze 로 차원을 줄여주고, transpose 로 차원변형을 다시 거꾸로 해준다.
output2 += MEAN_VALUE  # 전처리에서 빼준 MEAN_VALUE 를 다시 더해준다.

output2 = np.clip(output2, 0, 255) # 0 에서 255 로 크기를 제한한다. 255가 넘으면 255로 바꿔줘라!
output2 = output2.astype('uint8')  # 사람이 볼수 있는 형태로 바꾸기 위해서 정수 형태로 바꾸면 사람이 볼 수 있는 이미지 형태로 바뀐다.


# 반반 이미지 자르기
# output = output[:, :250]
# output2 = output2[:, 250:500]
# output3 = np.concatenate([output, output2], axis=1)
# output3 = np.concatenate([output[:, :250], output2[:, 250:]], axis=1)

# output 이라는 윈도우에 output 을 출력해라
cv2.imshow('output', output)
cv2.imshow('output2', output2)
# cv2.imshow('output3', output3)

# 윈도우 창 생성해서 이미지 보기 
cv2.imshow('img', img)

# 무한대기창
cv2.waitKey(0)

# 이미지 저장 imwrite
cv2.imwrite('after_imgs/imgs.png', output2)