from PIL import Image, ImageTk, ImageOps, ImageFilter
import cv2
#from skimage import io
import numpy as np
#import math

class ImageEdit:
    def __init__(self, image):
        self.original_image = image
        self.image = image.copy()

        self.canvas = None

        self.sel_start_x = 0
        self.sel_start_y = 0
        self.sel_stop_x = 0
        self.sel_stop_y = 0
        self.sel_react = None

    @property
    def image_tk(self):
        return ImageTk.PhotoImage(self.image)

    def update_image_on_canvas(self):
        if self.canvas is None:
            return RuntimeError("Canvas of image not given")

        image_tk = self.image_tk
        self.canvas.delete("all")
        self.canvas.configure(width=self.image.width, height=self.image.height)
        self.canvas.create_image(0, 0, image=image_tk, anchor="nw")
        self.canvas.image = image_tk

    def rotate(self, degrees):
        self.image = self.image.rotate(degrees)

    def flip(self, mode):
        self.image = self.image.transpose(mode)

    def resize(self, percents):
        w, h = self.image.size
        w = (w * percents) // 100
        h = (h * percents) // 100

        self.image = self.image.resize((w, h), Image.LANCZOS)

    def filter(self, filter_type):
        self.image = self.image.filter(filter_type)

    def apply_median_filter(self):  # Шум
        img = np.array(self.image)

        image_numpy = cv2.medianBlur(img, 5)
        self.image = Image.fromarray(np.uint8(image_numpy))

    def apply_blur(self): #Размытие
        img = np.array(self.image)

        sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpen = cv2.filter2D(img, -1, sharpen_kernel)

        self.image = Image.fromarray(np.uint8(sharpen))


    def apply_perspective(self): #Горизонт
        img = np.array(self.image)

        def DegreeTrans(theta):
            res = theta / np.pi * 180
            return res

        def rotateImage(src, degree):
            # Центр вращения является центром изображения
            h, w = src.shape[:2]
            # Рассчитать 2D повернутую матрицу аффинного преобразования
            RotateMatrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), degree, 1)
            print(RotateMatrix)
            # Аффинное преобразование, цвет фона заполнен белым
            rotate = cv2.warpAffine(src, RotateMatrix, (w, h), borderValue=(255, 255, 255))
            return rotate

        # Рассчитать угол с помощью преобразования Хафа
        def CalcDegree(srcImage):
            midImage = cv2.cvtColor(srcImage, cv2.COLOR_BGR2GRAY)
            dstImage = cv2.Canny(midImage, 50, 200, 3)
            lineimage = srcImage.copy()

            # Обнаружение прямых линий по преобразованию Хафа
            lines = cv2.HoughLines(dstImage, 1, np.pi / 180, 200)

            if lines is not None and len(lines) > 0:
                sum_theta = 0
                # Нарисуйте каждый отрезок по очереди
                for i in range(len(lines)):
                    for rho, theta in lines[i]:
                        a = np.cos(theta)
                        b = np.sin(theta)
                        x0 = a * rho
                        y0 = b * rho
                        x1 = int(round(x0 + 1000 * (-b)))
                        y1 = int(round(y0 + 1000 * a))
                        x2 = int(round(x0 - 1000 * (-b)))
                        y2 = int(round(y0 - 1000 * a))
                        sum_theta += theta
                        cv2.line(lineimage, (x1, y1), (x2, y2), (0, 0, 255), 1, cv2.LINE_AA)

                # Усредняем все углы, эффект вращения будет лучше
                average_theta = sum_theta / len(lines)
                angle = DegreeTrans(average_theta) - 90
                cv2.imshow("Imagelines", lineimage)
                return angle
            else:
                print("No lines detected.")
                return 0

        degree = CalcDegree(img)
        rotate = rotateImage(img, degree)

        self.image = Image.fromarray(np.uint8(rotate))

    def apply_bliky(self): #Переэкспонирование
        img = np.array(self.image)

        def overexposure(img):
            # Ограничение для контраста
            clahefilter = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))

            # Настраиваем минимальный и максимальный границы яркости
            GLARE_MIN = np.array([0, 0, 50], np.uint8)
            GLARE_MAX = np.array([0, 0, 225], np.uint8)

            # Переводим в HSV цветовой формат
            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # Аналог маски
            frame_threshed = cv2.inRange(hsv_img, GLARE_MIN, GLARE_MAX)

            # Красим по аналогу маски
            inpaintphsv_img = cv2.inpaint(img, frame_threshed, 0.1, cv2.INPAINT_TELEA)

            # Применение алгоритма CLAHE(Контраст)
            lab = cv2.cvtColor(inpaintphsv_img, cv2.COLOR_BGR2LAB)
            lab_planes = list(cv2.split(lab))
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab_planes[0] = clahe.apply(lab_planes[0])
            lab = cv2.merge(lab_planes)
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            return result

        image = overexposure(img)

        self.image = Image.fromarray(np.uint8(image))

    def apply_distorsia(self):
        img = np.array(self.image)

        self.image = Image.fromarray(np.uint8(img))

    def start_crop_selection(self):
        self.sel_react = self.canvas.create_rectangle(self.sel_start_x, self.sel_start_y, self.sel_stop_x, self.sel_stop_y,
                                                      dash=(10, 10), fill="cyan", width=1, stipple="gray25", outline="black")

        self.canvas.bind("<Button-1>", self._get_selection_start)
        self.canvas.bind("<B1-Motion>", self._update_selection_stop)

    def _get_selection_start(self, event):
        self.sel_start_x, self.sel_start_y = event.x, event.y

    def _update_selection_stop(self, event):
        self.sel_stop_x, self.sel_stop_y = event.x, event.y
        self.canvas.coords(self.sel_react, self.sel_start_x, self.sel_start_y, self.sel_stop_x, self.sel_stop_y)

    def crop_selected_area(self):
        if self.sel_react is None:
            return ValueError("Got no selection area from crop operation.")

        self.canvas.unbind("<Button-1>")
        self.canvas.unbind("<B1-Motion>")
        self.canvas.delete(self.sel_react)

        if self.sel_start_x > self.sel_stop_x:
            self.sel_start_x, self.sel_stop_x = self.sel_stop_x, self.sel_start_x
        if self.sel_start_y > self.sel_stop_y:
            self.sel_start_y, self.sel_stop_y = self.sel_stop_y, self.sel_start_y

        self.image = self.image.crop([self.sel_start_x, self.sel_start_y, self.sel_stop_x, self.sel_stop_y])

        self.sel_react = None

        self.sel_start_x, self.sel_start_y = 0, 0
        self.sel_stop_x, self.sel_stop_y = 0, 0









