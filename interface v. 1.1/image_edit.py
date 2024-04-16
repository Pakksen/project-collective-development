from PIL import Image, ImageTk, ImageOps, ImageFilter
import cv2
from matplotlib import pyplot as plt
#from skimage import io
from math import sqrt
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

        #x = float(input('Введите параметр x_distortion:\n'))
        #y = float(input('Введите параметр y_distortion:\n'))

        def get_new_coordinates(source_x, source_y, radius, x_distortion, y_distortion):
            '''
            Функция перерасчета положений пикселей:

            source_x: исходная координата x пикселя в исходном изображении.
            Это значение должно быть в диапазоне от 0 до ширины исходного изображения.
            source_y: исходная координата y пикселя в исходном изображении. Э
            то значение должно быть в диапазоне от 0 до высоты исходного изображения.
            radius: радиус линзы в пикселях.
            Это значение определяет, насколько сильно будут искажаться пиксели в зависимости от того,
            насколько они находятся от центра линзы.
            x_distortion: параметр искажения по оси x.
            Чем больше это значение, тем сильнее пиксели будут смещаться относительно центра изображения.
            y_distortion: параметр искажения по оси y.
            Чем больше это значение, тем сильнее пиксели будут смещаться относительно центра изображения.

            Функция возвращает нормализованные координаты x и y пикселя в новом преобразованном изображении.
            PS: эти координаты также могут быть за пределами границ искаженного изображения.
            '''
            if 1 - x_distortion * (radius ** 2) == 0:
                xn = source_x
            else:
                xn = source_x / (1 - (x_distortion * (radius ** 2)))

            if 1 - y_distortion * (radius ** 2) == 0:
                yn = source_y
            else:
                yn = source_y / (1 - (y_distortion * (radius ** 2)))
            return xn, yn

        def distortion(img, x_distortion, y_distortion, scale_x=1, scale_y=1):

            '''
            Функция реализует процедуру изменения исходного пиксельного расположения, делая тем самым
            операцию, соответсвующую аберрации оптической системы под названием дисторсия
            Функция на вход примимает исходное изображение в чб или rgb представлении (img). А также
            характеристики самой аберрации:

            x_distortion: параметр искажения по оси x.
            Чем больше это значение, тем сильнее пиксели будут смещаться относительно центра изображения.
            y_distortion: параметр искажения по оси y.
            Чем больше это значение, тем сильнее пиксели будут смещаться относительно центра изображения.

            Значения x_distortion, y_distortion со знаком минус говорят о том, что реализуется
            избавление исходного изображения от бочкообразности. Если же изображение исходное
            подушкообразное, то необходимо ставить данные значения со знаком +
            (Fx, Fy параметры в линзе)

            scale_x и scale_y по умолчанию равны 1. Этот параметр отвечает за степень
            сжатия иходного изображения по осям (a, b параметры в линзе)
            '''


            # Сохраним значения высоты и ширины исходного изображения
            w, h = img.shape[0], img.shape[1]
            w, h = float(w), float(h)

            # Для чб изображений сделаем копирование исходного канала 2 раза (h,w)->(h,w,3)
            if len(img.shape) == 2:
                bw_channel = np.copy(img)
                img = np.dstack((img, bw_channel))
                img = np.dstack((img, bw_channel))

            # Создадим массив нулей который будем перезаписывать в процессе выполнения функции
            result_img = np.zeros_like(img)

            # Делаем обход каждого пикселя в выходном изображении:
            for x in range(len(result_img)):
                for y in range(len(result_img[x])):

                    # Нормализация корродинат x, y чтобы были в пределах [-1, 1]
                    xnd, ynd = float((2 * x - w) / w), float((2 * y - h) / h)

                    # Теперь центр изображения имеет координату (0, 0), поэтому найти евклидово
                    # расстояние от центра до текущего пикселя можно так:
                    rd = sqrt(xnd ** 2 + ynd ** 2)

                    # Определение новых координат:
                    xdu, ydu = get_new_coordinates(xnd, ynd, rd, x_distortion, y_distortion)

                    # Добавление рескейлинга по осям (a, b параметры в линзе)
                    xdu, ydu = xdu * scale_y, ydu * scale_x

                    # Возвращение значений координат из [-1, 1] в исходные значения [h, w]
                    xu, yu = int(((xdu + 1) * w) / 2), int(((ydu + 1) * h) / 2)

                    # Если пиксель находится в пределах исходного размера изображения, то проводим
                    # замену значений. В новую координату пишем (r,g,b) значение старой координаты
                    if 0 <= xu and xu < img.shape[0] and 0 <= yu and yu < img.shape[1]:
                        result_img[x][y] = img[xu][yu]

            # Переведем значения пикселей в uint8
            return result_img.astype(np.uint8)

        def interctive_window(path, make_distortion=False):
            global xx, yy
            '''
            Функция для постройки интерактивного окна с возможностью подбирать
            коэффициенты для дисторсии с помощью слайдера
            На вход подается путь к файлу (исходное изображение)
            make_distortion = False -> коэффициенты со знаком - чтобы делать антидисторсию
            make_distortion = True -> коэффициенты со знаком + чтобы делать создать дисторсию
            '''

            def img_intensity_change_x(x):
                pass

            def img_intensity_change_y(y):
                pass

            # Считаем изображение:
            img_bgr = path
            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # Перевод и bgr в rgb

            # Создаем окно:
            cv2.namedWindow('LENS DISTORTION')
            val = -1  # переменная для хранения знака + или - для коэффициентов

            # Создадим sliders:
            cv2.createTrackbar('value x', 'LENS DISTORTION', 0, 20, img_intensity_change_x)
            cv2.createTrackbar('value y', 'LENS DISTORTION', 0, 20, img_intensity_change_y)
            key = 0
            while True:
                # Считаем данные со слайдера:
                x = cv2.getTrackbarPos('value x', 'LENS DISTORTION')
                y = cv2.getTrackbarPos('value y', 'LENS DISTORTION')
                # Переведем в значения меньшего диапазона и со знаком
                if make_distortion:
                    val = 1  # Поменяет знак на + при входе в функцию
                joint_1 = x * 0.05 * val
                joint_2 = y * 0.05 * val
                if x == 0:
                    joint_1 = 0
                if y == 0:
                    joint_2 = 0
                output_img = distortion(img, joint_1, joint_2)
                xx = joint_1
                yy = joint_2

                print(x, " ", y)

                # resize в 1.5 раза чтоб было больше иображение внешне в окне:
                dsize = (int(output_img.shape[1] * 1.5), int(output_img.shape[0] * 1.5))
                output_img = cv2.resize(output_img, dsize, interpolation=cv2.INTER_AREA)

                # Переод обратно в bgr:
                output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
                cv2.imwrite('images\output.png', output_img)

                # Добавление текста и отображение output_img:
                cv2.putText(output_img, 'x_distortion:' + str(round(joint_1, 3)), (20, 35),
                            fontFace=1, fontScale=3, color=(0, 0, 220), thickness=3)
                cv2.putText(output_img, 'y_distortion:' + str(round(joint_2, 3)), (20, 75),
                            fontFace=1, fontScale=3, color=(0, 0, 220), thickness=3)
                cv2.imshow('LENS DISTORTION', output_img)

                # Обновление каждый 10 мс пока не нажмется кнопка закрытие окна
                key = cv2.waitKey(10)
                if key == ord('q'):
                    # Вернем значения слайдеров при выходе из цикла
                    break
                elif key == 27:  # ASCII код клавиши ESC
                    # Закрываем окно при нажатии на клавишу ESC
                    break
            # Закрытие окна:
            if cv2.getWindowProperty('LENS DISTORTION', cv2.WND_PROP_VISIBLE) > 0:
                cv2.destroyAllWindows()



        interctive_window(img)
        print(xx, " ", yy)
        img = distortion(img, xx, yy)
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

    def chromatic_abberation(self):

        # Сатурация для избегания выхода значения за рамки
        def SaturateSum(A, B):
            C = A + B
            if C > 255:
                C = 255
            elif C < 0:
                C = 0
            return C

        def Saturate(A):
            if A > 255:
                A = 255
            elif A < 0:
                A = 0
            return A

        # Устранение, прогоняет каналы по одному положению (высота, ширина)
        def rmCA(threshold, arr, isRotated):
            # Получение высоты и ширины.
            height = arr[0].shape[0]
            width = arr[0].shape[1]

            for i in range(0, height):
                # Получение отдельных каналов
                bptr = np.ubyte(arr[0][i])
                gptr = np.ubyte(arr[1][i])
                rptr = np.ubyte(arr[2][i])
                for j in range(1, width - 1):
                    # Находим место абберации
                    if (abs(np.int64(gptr[j + 1]) - np.int64(gptr[j - 1])) >= threshold):
                        # Знак этого места
                        sign = 0
                        if (np.int64(gptr[j + 1]) - np.int64(gptr[j - 1]) > 0):
                            sign = 1
                        else:
                            sign = -1

                        # Поиск границ этой абберации
                        lpos = j - 1
                        rpos = j + 1
                        while (lpos > 0):
                            # Важно: градиент того же знака, что и место
                            ggrad = (np.int64(gptr[lpos + 1]) - np.int64(gptr[lpos - 1])) * sign
                            bgrad = (np.int64(bptr[lpos + 1]) - np.int64(bptr[lpos - 1])) * sign
                            rgrad = (np.int64(rptr[lpos + 1]) - np.int64(rptr[lpos - 1])) * sign
                            if (max(max(bgrad, ggrad), rgrad) < threshold):
                                break
                            lpos = lpos - 1
                        lpos -= 1
                        while (rpos < width - 1):
                            # Важно: градиент того же знака, что и место
                            ggrad = (np.int64(gptr[rpos + 1]) - np.int64(gptr[rpos - 1])) * sign
                            bgrad = (np.int64(bptr[rpos + 1]) - np.int64(bptr[rpos - 1])) * sign
                            rgrad = (np.int64(rptr[rpos + 1]) - np.int64(rptr[rpos - 1])) * sign
                            if (max(max(bgrad, ggrad), rgrad) < threshold):
                                break
                            rpos = rpos + 1
                        rpos += 1

                        if rpos >= width:
                            rpos = width - 1
                        if lpos <= 0:
                            lpos = 0

                        bgmaxVal = max(np.int64(bptr[lpos]) - np.int64(gptr[lpos]),
                                       np.int64(bptr[rpos]) - np.int64(gptr[rpos]))
                        bgminVal = min(np.int64(bptr[lpos]) - np.int64(gptr[lpos]),
                                       np.int64(bptr[rpos]) - np.int64(gptr[rpos]))
                        rgmaxVal = max(np.int64(rptr[lpos]) - np.int64(gptr[lpos]),
                                       np.int64(rptr[rpos]) - np.int64(gptr[rpos]))
                        rgminVal = min(np.int64(rptr[lpos]) - np.int64(gptr[lpos]),
                                       np.int64(rptr[rpos]) - np.int64(gptr[rpos]))

                        for k in range(lpos, rpos):
                            bdiff = np.int64(bptr[k]) - np.int64(gptr[k])
                            rdiff = np.int64(rptr[k]) - np.int64(gptr[k])
                            if (bdiff > bgmaxVal):
                                bptr[k] = SaturateSum(bgmaxVal, gptr[k])
                            elif (bdiff < bgminVal):
                                bptr[k] = SaturateSum(bgminVal, gptr[k])
                            else:
                                bptr[k] = Saturate(bptr[k])

                            if (rdiff > rgmaxVal):
                                rptr[k] = SaturateSum(rgmaxVal, gptr[k])
                            elif (rdiff < rgminVal):
                                rptr[k] = SaturateSum(rgminVal, gptr[k])
                            else:
                                rptr[k] = Saturate(rptr[k])
                        j = rpos - 2
                arr[0][i] = bptr
                arr[1][i] = gptr
                arr[2][i] = rptr
            return arr

        img = np.array(self.image)

        # разбиваем изображения на каналы RGB для более удобной работы
        arr = cv2.split(img)

        # Устанавливаем ограничение для нахождения границы в G канале
        threshold = 30

        # Горизонтальное устранение хроматических аббераций
        arr = rmCA(threshold, arr, False)

        # Поворот и вертикальное устранение хроматических аббераций
        rotate = cv2.merge(arr)
        rotate = cv2.transpose(rotate)
        arr = cv2.split(rotate)
        arr = rmCA(threshold, arr, True)

        # Соединение и соединение каналов в обработанное изображение
        rotate = cv2.merge(arr)
        result = cv2.transpose(rotate)

        self.image = Image.fromarray(np.uint8(result))

    def perspective(self):
        img = np.array(self.image)
        global coords
        plt.rcParams['toolbar'] = 'None'  # бирает меню в координатной плоскости

        # функция, которая позволяет после 4 кликов по углам изображения на координатной плоскоти,
        # записать координаты области которая подвержена перспективному искажению
        def onclick(event):

            ix, iy = event.xdata, event.ydata
            print(f'x = {ix}, y = {iy}')

            # Использование переменной coords
            coords.append((ix, iy))

            if len(coords) == 4:
                fig.canvas.mpl_disconnect(cid)
                plt.close()
            return coords

        # Load the image

        # Create a copy of the image
        img_copy = np.copy(img)

        fig = plt.figure()
        ax = fig.add_subplot()

        # Set titles for the figure and the subplot respectively
        fig.suptitle('Выберете крайние точки\nбудующего изображения в порядке:', fontsize=12, fontweight='bold')
        ax.set_title('ЛВ, ПВ, ЛН, ПН')  # (это крайние точки готового изображения
        # (Пример ЛВ(левая верхняя) точка конечного изображения))
        # ывод изображения на координатной плоскости

        # Convert to RGB so as to display via matplotlib
        # Using Matplotlib we can easily find the coordinates
        # of the 4 points that is essential for finding the
        # transformation matrix
        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)  # он нужен для правильной подачи цвета
        plt.imshow(img_copy)
        # массив координат начального изображения
        coords = []
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()

        pts1 = np.float32(coords)  # копия массива координат начального изображения
        pts2 = np.float32([[0, 0], [800, 0], [0, 600], [800, 600]])  # позиция выбранных координат,
        # на конечном изображении

        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        result = cv2.warpPerspective(img, matrix, (800, 600))  # размер конечного изображения

        self.image = Image.fromarray(np.uint8(result))











