# -*- coding: utf-8 -*-
"""
Updated at Dec 16 2023

@author: Alireza Norouzi
source : https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/losses/dssim.py
"""

import tensorflow as tf
import numpy as np
import keras.losses as Loss

class SSIM_MSE_LOSS():
	def __init__(self, ssim_relative_loss, mse_relative_loss, ssim_win_size=4):
		self.ssim_relative_loss = tf.convert_to_tensor(ssim_relative_loss/(ssim_relative_loss+mse_relative_loss), tf.float32)
		self.mse_relative_loss = tf.convert_to_tensor(mse_relative_loss / (ssim_relative_loss+mse_relative_loss), tf.float32)
		self.win_size = ssim_win_size

	def ssimmse_loss(self, y_true, y_pred):
		return 1.0 - (self.ssim_relative_loss)*(self.tf_ssim(y_true, y_pred, size=self.win_size)) + (self.mse_relative_loss)*(Loss.mean_squared_error(y_true, y_pred))


	def tf_ssim(self, img1, img2, cs_map=False, mean_metric=True, size=11, sigma=1.5):
		window = _tf_fspecial_gauss(size, sigma) # window shape [size, size]
		K1 = 0.01
		K2 = 0.03
		L = 1  # depth of image (255 in case the image has a differnt scale)
		C1 = (K1*L)**2
		C2 = (K2*L)**2
		mu1 = tf.nn.conv2d(img1, window, strides=[1, 1, 1, 1], padding='VALID')
		mu2 = tf.nn.conv2d(img2, window, strides=[1, 1, 1, 1], padding='VALID')
		mu1_sq = mu1*mu1
		mu2_sq = mu2*mu2
		mu1_mu2 = mu1*mu2
		sigma1_sq = tf.nn.conv2d(img1*img1, window, strides=[1,1,1,1], padding='VALID') - mu1_sq
		sigma2_sq = tf.nn.conv2d(img2*img2, window, strides=[1,1,1,1], padding='VALID') - mu2_sq
		sigma12 = tf.nn.conv2d(img1*img2, window, strides=[1,1,1,1], padding='VALID') - mu1_mu2
		if cs_map:
			value = (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
						(sigma1_sq + sigma2_sq + C2)),
					(2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
		else:
			value = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
						(sigma1_sq + sigma2_sq + C2))

		if mean_metric:
			value = tf.reduce_mean(value)
		return value


def _tf_fspecial_gauss(size, sigma):
	# 生成二维网格坐标
	x_data, y_data = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]

	# 扩展维度为 [size, size, 1, 1]
	x_data = np.expand_dims(x_data, axis=-1)
	x_data = np.expand_dims(x_data, axis=-1)
	y_data = np.expand_dims(y_data, axis=-1)
	y_data = np.expand_dims(y_data, axis=-1)

	# 转换为 TensorFlow 张量
	x = tf.constant(x_data, dtype=tf.float32)
	y = tf.constant(y_data, dtype=tf.float32)

	# 计算 Gaussian 分布
	g = tf.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))

	# 归一化，使滤波器总和为 1
	g = g / tf.reduce_sum(g)

	# 再次归一化，确保扩展后的滤波器总和为 1
	g = g / tf.reduce_sum(g)

	return g



# # -*- coding: utf-8 -*-
# """
# Updated at Dec 16 2023
#
# @author: Alireza Norouzi
# source: https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/losses/dssim.py
# """
#
# import tensorflow as tf
# import numpy as np
# import keras.losses as Loss
#
#
# class SSIM_MSE_LOSS():
# 	def __init__(self, ssim_relative_loss, mse_relative_loss, ssim_win_size=4):
# 		self.ssim_relative_loss = tf.convert_to_tensor(ssim_relative_loss / (ssim_relative_loss + mse_relative_loss),
# 													   tf.float32)
# 		self.mse_relative_loss = tf.convert_to_tensor(mse_relative_loss / (ssim_relative_loss + mse_relative_loss),
# 													  tf.float32)
# 		self.win_size = ssim_win_size
#
# 	def ssimmse_loss(self, y_true, y_pred):
# 		return 1.0 - (self.ssim_relative_loss) * (self.tf_ssim(y_true, y_pred, size=self.win_size)) + \
# 			(self.mse_relative_loss) * (Loss.mean_squared_error(y_true, y_pred))
#
# 	def tf_ssim(self, img1, img2, cs_map=False, mean_metric=True, size=11, sigma=1.5):
# 		# 生成 Gaussian 窗口，形状 [size, size, 3, 1]
# 		window = _tf_fspecial_gauss(size, sigma)
#
# 		K1 = 0.01
# 		K2 = 0.03
# 		L = 1  # 假设图像值范围 [0, 1]
# 		C1 = (K1 * L) ** 2
# 		C2 = (K2 * L) ** 2
#
# 		# 计算局部均值、方差和协方差
# 		mu1 = tf.nn.conv2d(img1, window, strides=[1, 1, 1, 1], padding='VALID')  # [batch, out_h, out_w, 3]
# 		mu2 = tf.nn.conv2d(img2, window, strides=[1, 1, 1, 1], padding='VALID')
# 		mu1_sq = mu1 * mu1
# 		mu2_sq = mu2 * mu2
# 		mu1_mu2 = mu1 * mu2
# 		sigma1_sq = tf.nn.conv2d(img1 * img1, window, strides=[1, 1, 1, 1], padding='VALID') - mu1_sq
# 		sigma2_sq = tf.nn.conv2d(img2 * img2, window, strides=[1, 1, 1, 1], padding='VALID') - mu2_sq
# 		sigma12 = tf.nn.conv2d(img1 * img2, window, strides=[1, 1, 1, 1], padding='VALID') - mu1_mu2
#
# 		# 计算 SSIM
# 		if cs_map:
# 			value = (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) /
# 					 ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)),
# 					 (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))
# 		else:
# 			value = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
#
# 		value = tf.reduce_mean(value)
#
#
#
# 		# 对高度、宽度和通道取平均
# 		if mean_metric:
# 			if cs_map:
# 				value = (tf.reduce_mean(value[0], axis=[1, 2, 3]), tf.reduce_mean(value[1], axis=[1, 2, 3]))
# 			else:
# 				# value = tf.reduce_mean(value, axis=[1, 2, 3])  # 平均所有维度，包括 3 个通道
# 				value = tf.reduce_mean(value)  # 平均所有维度，包括 3 个通道
#
#
# 		return value
#
#
#
#
# def _tf_fspecial_gauss(size, sigma):
# 	"""Generate a Gaussian filter for RGB input with shape [size, size, 3, 1].
#
#     Args:
#         size (int): Size of the Gaussian filter (e.g., 11 for an 11x11 filter).
#         sigma (float): Standard deviation of the Gaussian distribution.
#
#     Returns:
#         tf.Tensor: A [size, size, 3, 1] tensor for 3 input channels and 1 output channel.
#     """
# 	# 生成二维网格坐标
# 	x_data, y_data = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
# 	x_data = np.expand_dims(x_data, axis=-1)
# 	x_data = np.expand_dims(x_data, axis=-1)
# 	y_data = np.expand_dims(y_data, axis=-1)
# 	y_data = np.expand_dims(y_data, axis=-1)
#
# 	# 转换为 TensorFlow 张量
# 	x = tf.constant(x_data, dtype=tf.float32)
# 	y = tf.constant(y_data, dtype=tf.float32)
#
# 	# 计算 Gaussian 分布
# 	g = tf.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
# 	g = g / tf.reduce_sum(g)
#
# 	# 扩展为 [size, size, 3, 1]，适配 RGB 输入
# 	g = tf.tile(g, [1, 1, 3, 1])
# 	g = g / tf.reduce_sum(g)
#
# 	return g

# # 测试 RGB 输入
# img1 = tf.random.uniform([64, 32, 32, 3], 0, 1)  # 模拟 RGB 图像
# img2 = tf.random.uniform([64, 32, 32, 3], 0, 1)
#
# # a = Loss.mean_squared_error(img1, img2)
# # b = SSIM_MSE_LOSS(ssim_relative_loss=0.5, mse_relative_loss=0.5, ssim_win_size=11).tf_ssim(img1, img2)
#
# loss_fn = SSIM_MSE_LOSS(ssim_relative_loss=0.5, mse_relative_loss=0.5, ssim_win_size=11)
# loss_value = loss_fn.ssimmse_loss(img1, img2)
# print("Loss value:", loss_value)
# print("Output shape:", loss_value.shape)