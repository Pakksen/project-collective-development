import cv2
import numpy as np
import time

clahefilter = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))


image = cv2.imread('4.jpg')
def overexposure(img):
    GLARE_MIN = np.array([0, 0, 50],np.uint8)
    GLARE_MAX = np.array([0, 0, 225],np.uint8)

    hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    #HSV
    frame_threshed = cv2.inRange(hsv_img, GLARE_MIN, GLARE_MAX)

    #INPAINT + HSV
    inpaintphsv_img = cv2.inpaint(img, frame_threshed, 0.1, cv2.INPAINT_TELEA) 


    #HSV+ INPAINT + CLAHE
    lab = cv2.cvtColor(inpaintphsv_img, cv2.COLOR_BGR2LAB)
    lab_planes = list(cv2.split(lab))
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return result