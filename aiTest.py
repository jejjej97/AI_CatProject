# coding: utf-8
import sys
import numpy as np
import os
import tensorflow as tf
import cv2

sys.argv = [sys.argv[0], "./img/cat2.jpg"]

def inferenceImg(img_path):
	# 가중치와 옵티마이저를 포함하여 정확히 동일한 모델을 다시 생성합니다
	new_model = tf.keras.models.load_model('C:/CatProject/AI_Python/cat_model.h5')

	# 모델 구조를 출력합니다
	new_model.summary()

	#new_model
	img_bgr = cv2.imread(filename=img_path)
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
	print(prediction2[0].numpy())
	print(prediction2[0])
	print(type(prediction2))
	prediction2 = tf.where(prediction2 < 0.5, 0, 1)

	print('Prediction : \n', prediction2[0].numpy())
	#print(class_names[prediction2[0].numpy()])
	#print('Label:\n', label1)
	#print(class_names[label1])

def main(argv):
    print(argv[0])
    print(argv[1])
    print("Test text")
    inferenceImg(argv[1])


if __name__ == "__main__":
    main(sys.argv)