import cv2 as cv2

img = cv2.imread("godno.png")
median_image = cv2.medianBlur(img, 5)
cv2.imshow('MyPhoto', median_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
