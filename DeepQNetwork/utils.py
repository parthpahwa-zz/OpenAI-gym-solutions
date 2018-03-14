import numpy as np
import cv2

def rgb2gray(img):
    return np.dot(img[..., :3], [0.299, 0.587, 0.114])

def resize(img, width, height):
	return cv2.resize(img, (width, height))

def preprocess(img, width, height):
    return resize(rgb2gray(img), width, height)
