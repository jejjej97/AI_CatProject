# coding: utf-8
import sys
from pathlib import Path

import numpy as np
import os
import tensorflow as tf
import cv2
from PIL import Image

# sys.argv = [sys.argv[0], "C:/CatProject/cat/src/main/webapp/resources/ai_cat_img/", "20220408005911_ehdwn.jpg"]

def inferenceImg(save_path, img_path):

	# 가중치와 옵티마이저를 포함하여 정확히 동일한 모델을 다시 생성합니다
	new_model = tf.keras.models.load_model('C:/CatProject/AI_Python/cat_model.h5')

	# 모델 구조를 출력합니다
	new_model.summary()

	# 사진 자르기
	# crop(가로 시작점, 세로 시작점, 가로 범위, 세로 범위)
	# for root, dirs, files in os.walk('./'):
	# 	print("탐색중인 경로: ", dirs)
	# 	for idx, file in enumerate(files):
	# 		fname, ext = os.path.splitext(file)
			# if ext in ['.jpg', '.png', '.gif', '.heic', '.webp']:
				# full_dir = root + "/" + file
				# print("files", files)
				# print("root", root)
				# print("dirs", dirs)
				# print("full_dir", full_dir)

	im = Image.open(save_path+img_path)
	width, height = im.size
	print("Width: ", width, "px, Height: ", height, "px -> ", img_path)
	if width > height:  # 일단 가로로 긴 경우만 생각하자
		print("가로 긴 직사각형")
		crop_image = im.crop(((width / 2) - (height / 2), 0, (width / 2) + (height / 2), height))
		print((width / 2) - (height / 2), 0, (width / 2) + (height / 2), height)
		os.chdir(save_path)
		crop_image.save(os.path.basename(save_path+img_path))
		# print(Path('C:', '/', 'Users'))
	if width < height:  # 세로로 긴 경우
		print("세로 긴 직사각형")
		crop_image = im.crop((0, (height / 2) - (width / 2), width, (height / 2) + (width / 2)))
		print(0, (height / 2) - (width / 2), width, (height / 2) - (width / 2))
		os.chdir(save_path)
		crop_image.save(os.path.basename(save_path+img_path))
		# print(Path('C:', '/', 'Users'))
	else:
		print("정사각형")

	print("파일",img_path)

	# real_img =
	#new_model
	img_bgr = cv2.imread(filename=save_path+img_path)
	img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

	#resize
	img_resized = cv2.resize(img_rgb,dsize=(160,160), interpolation = cv2.INTER_CUBIC)

	#normalization 0~255 -> -1 ~ 1
	rescale = tf.keras.layers.Rescaling(1./127.5, offset=-1)

	#convert to tensor
	image1 = tf.convert_to_tensor(img_resized, dtype=tf.float32)
	label1 = 1

	print(label1)

	print(image1.shape)
	# expand dimension
	image1 = tf.expand_dims(image1, 0)
	print(image1.shape)

	prediction1 = new_model.predict(image1).flatten()
	print(prediction1)

	# Apply a sigmoid since our model returns logits
	prediction2 = tf.nn.sigmoid( prediction1 )
	# print(prediction2[0].numpy())
	print('비만 점수 : ', prediction2[0].numpy()*100)
	value = str(prediction2[0].numpy()*100)
	# print(prediction2[0])
	# print(type(prediction2))
	# prediction2 = tf.where(prediction2 < 0.5, 0, 1)
	#
	# print('Prediction:\n', prediction2[0].numpy())
	# #print(class_names[prediction2[0].numpy()])
	# #print('Label:\n', label1)
	# #print(class_names[label1])
	return value

def main(argv):
	print(argv[0])
	print(argv[1])
	print("Test text")
	print(inferenceImg(argv[1],argv[2]))


if __name__ == "__main__":
	main(sys.argv)