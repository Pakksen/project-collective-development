import cv2
import ChromaticAbberationCorrection as cac
import os
import numpy as np


image = cv2.imread("Path\To\image.jpeg")
os.chdir("Path/To/")
result = cac.RemoveChromaticAbberation(image)
Hori = np.concatenate((image, result), axis=1) 
cv2.imshow('HORIZONTAL', Hori) 
cv2.imwrite("result.jpeg", result) 
cv2.waitKey(0)
cv2.destroyAllWindows()
 