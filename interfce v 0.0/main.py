from tkinter import *
from tkinter import filedialog as fd
from tkinter import messagebox as mb
from tkinter.ttk import Notebook
from PIL import Image, ImageTk, ImageOps, ImageFilter
import os
import json

import cv2
from skimage import io
import numpy as np
import math

CONFIG_FILE = "config.json"

class PyPhotoEditor:
    def __init__(self):
       self.root = Tk()
       self.image_tabs = Notebook(self.root)
       self.opened_images = []

       self.init()


    def init(self):
        self.root.title("Kulema")
        #self.root.iconphoto(True, PhotoImage(file = "")) #Иконка приложенияв кавычки нужно вставить путь к картинке
        self.image_tabs.enable_traversal()#Позволяет переключаться между вкладками

        self.root.bind("<Escape>", self._close)
        self.root.protocol("WM_DELETE_WINDOW", self._close)

        self.root.bind("<Control-s>", self.save_current_image)
        self.root.bind("<Control-x>", self.close_carrent_image)

        if not os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'w') as f:
                json.dump({"opened_images": []}, f)
        else:
            self.load_images_from_config()


    def run(self):
        self.drow_menu() #Прорисовка меню
        self.drow_widegets() #Прорисовка виджетов

        self.root.mainloop() #Звпуск основоного цикла программы


    def drow_menu(self):
        menu_bar = Menu(self.root)

        file_menu = Menu(menu_bar, tearoff = 0)
        file_menu.add_command(label = "Open", command=self.open_new_images)
        file_menu.add_command(label="Save", command=self.save_current_image)
        file_menu.add_command(label="Save as", command=self.save_image_as)
        file_menu.add_command(label="Save all", command=self.save_all_changes)
        file_menu.add_separator()
        file_menu.add_command(label="Close image", command=self.close_carrent_image)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._close)
        menu_bar.add_cascade(label = "File", menu = file_menu)

        edit_menu = Menu(menu_bar, tearoff = 0)
        transform_menu = Menu(edit_menu, tearoff = 0)
        rotate_menu = Menu(transform_menu, tearoff=0)
        flip_menu = Menu(edit_menu, tearoff = 0)
        resize_menu = Menu(edit_menu, tearoff = 0)
        filter_menu = Menu(edit_menu, tearoff = 0)

        rotate_menu.add_command(label="Rotate left by 90", command=lambda: self.rotate_current_image(90))
        rotate_menu.add_command(label="Rotate right by 90", command=lambda: self.rotate_current_image(-90))
        rotate_menu.add_command(label="Rotate left by 180", command=lambda: self.rotate_current_image(180))
        rotate_menu.add_command(label="Rotate right by 180", command=lambda: self.rotate_current_image(-180))
        transform_menu.add_cascade(label="Rotate", menu=rotate_menu)

        flip_menu.add_command(label="Flip horizontally", command=lambda: self.flip_current_image("horizontally"))
        flip_menu.add_command(label="Flip vertically", command=lambda: self.flip_current_image("vertically"))

        resize_menu.add_command(label="25% of original size", command=lambda: self.resize_current_image(25))
        resize_menu.add_command(label="50% of original size", command=lambda: self.resize_current_image(50))
        resize_menu.add_command(label="75% of original size", command=lambda: self.resize_current_image(75))
        resize_menu.add_command(label="100% of original size", command=lambda: self.resize_current_image(100))
        resize_menu.add_command(label="125% of original size", command=lambda: self.resize_current_image(125))
        resize_menu.add_command(label="150% of original size", command=lambda: self.resize_current_image(150))
        resize_menu.add_command(label="175% of original size", command=lambda: self.resize_current_image(175))
        resize_menu.add_command(label="200% of original size", command=lambda: self.resize_current_image(200))

        filter_menu.add_command(label="Blur", command=lambda: self.apply_filter_to_current_image(ImageFilter.BLUR)) #Размытие (размывает еще больше чем было)
        filter_menu.add_command(label="Sharpen", command=lambda: self.apply_filter_to_current_image(ImageFilter.SHARPEN))  # Повышение контрастности
        filter_menu.add_command(label="Contour", command=lambda: self.apply_filter_to_current_image(ImageFilter.CONTOUR))  # Повышение контрастности контура
        filter_menu.add_command(label="Detail", command=lambda: self.apply_filter_to_current_image(ImageFilter.DETAIL))  #Повышение детализированости
        filter_menu.add_command(label="Smooth", command=lambda: self.apply_filter_to_current_image(ImageFilter.SMOOTH))  #Что-то вроеде размытости

        filter_menu.add_command(label="Шум", command=self.apply_median_filter_to_current_image)  # Устраняет шум
        filter_menu.add_command(label="Размытие", command=self.apply_blur_to_current_image)  # Устраняет размытие
        filter_menu.add_command(label="Горизонт", command=self.apply_perspective_to_current_image)  # Устраняет Заваленный горизонт
        filter_menu.add_command(label="Переэкспонирование", command=self.apply_bliky_to_current_image)  # Устраняет Переэкспонирование
        filter_menu.add_command(label="Дисторсия", command=self.apply_distorsia_to_current_image)  # Устраняет дисторсию

        edit_menu.add_cascade(label="Transform", menu=transform_menu)
        edit_menu.add_cascade(label="Flip", menu=flip_menu)
        edit_menu.add_cascade(label="Resize", menu=resize_menu)
        edit_menu.add_cascade(label="Filter", menu=filter_menu)

        menu_bar.add_cascade(label="Edit", menu=edit_menu)

        self.root.configure(menu = menu_bar)


    def drow_widegets(self):
        self.image_tabs.pack(fill = "both", expand = 1)


    def load_images_from_config(self):
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)

        paths = config["opened_images"]
        for path in paths:
            self.add_new_image(path)


    def open_new_images(self):
        image_paths = fd.askopenfilenames(filetypes=(("Images", "*.jpeg; *.jpg; *.png"), ))
        for image_path in image_paths:
            self.add_new_image(image_path)


    def add_new_image(self, image_path):
        opened_images = [(path[:-1] if path[-1] == '*' else path) for (path, image) in self.opened_images]
        if image_path in opened_images:
            index = opened_images.index(image_path)
            self.image_tabs.select(index)
            return

        image = Image.open(image_path) #Чтобы можно было использовать библиотеку pl
        image_tk = ImageTk.PhotoImage(image)#К преобразованной картинке нельзя ее использовать
        self.opened_images.append([image_path, image])

        image_tab = Frame(self.image_tabs)

        #Если нужно будет выделять область на изображении замепенить Label на Canvas (10)
        image_label = Label(image_tab, image=image_tk)
        image_label.image = image_tk
        image_label.pack(side="bottom", fill="both", expand='yes')

        self.image_tabs.add(image_tab, text=image_path.split('/')[-1])
        self.image_tabs.select(image_tab)


    def get_current_working_data(self):
        current_tab = self.image_tabs.select()
        if not current_tab:
            return None, None, None
        tab_number = self.image_tabs.index(current_tab)
        path, image = self.opened_images[tab_number]

        return current_tab, path, image


    def save_current_image(self, event=None):
        current_tab, path, image = self.get_current_working_data()
        if not current_tab:
            return
        tab_number = self.image_tabs.index(current_tab)

        if path[-1] == '*':
            path = path[:-1]
            self.opened_images[tab_number][0] = path
            image.save(path)
            self.image_tabs.add(current_tab, text=path.split('/')[-1])


    def save_image_as(self):
        current_tab, path, image = self.get_current_working_data()
        if not current_tab:
            return
        tab_number = self.image_tabs.index(current_tab)

        old_path, old_ext = os.path.splitext(self.opened_images[tab_number][0])
        if old_ext[-1] == "*":
            old_ext = old_ext[:-1]

        new_path = fd.asksaveasfilename(initialdir=old_path, filetypes=(("Images", "*.jpeg; *.jpg; *.png"), ))
        if not new_path:
            return

        new_path, new_ext = os.path.splitext(new_path)
        if not new_ext:
            new_ext = old_ext
        elif old_ext != new_ext:
            mb.showerror("Incorrect extension", f"Got incorrect extension: {new_ext}. Old was: {old_ext}.")
            return

        image.save(new_path + new_ext)
        image.close()

        del self.opened_images[tab_number]
        self.image_tabs.forget(current_tab)

        self.add_new_image(new_path + new_ext)

    def save_all_changes(self):
        for index, (path, image) in enumerate(self.opened_images):
            if path[-1] != "*":
                continue
            path = path[:-1]
            self.opened_images[index][0] = path
            image.save(path)
            self.image_tabs.tab(index, text = path.split('/')[-1])


    def close_carrent_image(self, event=None):
        current_tab, path, image = self.get_current_working_data()
        if not current_tab:
            return
        index = self.image_tabs.index(current_tab)

        image.close()
        del self.opened_images[index]
        self.image_tabs.forget(current_tab)


    def update_image_inside_app(self, current_tab, image):
        tab_number = self.image_tabs.index(current_tab)
        tab_frame = self.image_tabs.children[current_tab[current_tab.rfind('!'):]]
        label = tab_frame.children['!label']

        self.opened_images[tab_number][1] = image

        image_tk = ImageTk.PhotoImage(image)
        label.configure(image=image_tk)
        label.image = image_tk

        image_path = self.opened_images[tab_number][0]
        if image_path[-1] != '*':
            image_path += '*'
            self.opened_images[tab_number][0] = image_path
            image_name = image_path.split('/')[-1]
            self.image_tabs.tab(current_tab, text=image_name)


    def rotate_current_image(self, degrees):
        current_tab, path, image = self.get_current_working_data()
        if not current_tab:
            return

        image = image.rotate(degrees)
        self.update_image_inside_app(current_tab, image)


    def flip_current_image(self, flip_type):
        current_tab, path, image = self.get_current_working_data()
        if not current_tab:
            return

        if flip_type == 'horizontally':
            image = ImageOps.mirror(image)
        elif flip_type == 'vertically':
            image = ImageOps.flip(image)

        self.update_image_inside_app(current_tab, image)


    def resize_current_image(self, percents):
        current_tab, path, image = self.get_current_working_data()
        if not current_tab:
            return

        w, h = image.size
        w = (w * percents) // 100
        h = (h * percents) // 100

        image = image.resize((w, h), Image.LANCZOS)
        self.update_image_inside_app(current_tab, image)

    def apply_filter_to_current_image(self, filter_type):
        current_tab, path, image = self.get_current_working_data()
        if not current_tab:
            return

        image = image.filter(filter_type)
        self.update_image_inside_app(current_tab, image)


    def apply_median_filter_to_current_image(self): #Шум
        current_tab, path, image = self.get_current_working_data()
        if not current_tab:
            return

        '''if path[-1] == '*':
            path = path[:-1]

        img = io.imread(path)'''

        img = np.array(image)

        image_numpy = cv2.medianBlur(img, 5)
        image = Image.fromarray(np.uint8(image_numpy))

        self.update_image_inside_app(current_tab, image)

    def apply_blur_to_current_image(self): #Размытие
        current_tab, path, image = self.get_current_working_data()
        if not current_tab:
            return

        img = np.array(image)

        sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpen = cv2.filter2D(img, -1, sharpen_kernel)

        image = Image.fromarray(np.uint8(sharpen))

        self.update_image_inside_app(current_tab, image)


    def apply_perspective_to_current_image(self):
        current_tab, path, image = self.get_current_working_data()
        if not current_tab:
            return

        img = np.array(image)

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

        image = Image.fromarray(np.uint8(rotate))

        self.update_image_inside_app(current_tab, image)


    def apply_bliky_to_current_image(self):
        current_tab, path, image = self.get_current_working_data()
        if not current_tab:
            return

        img = np.array(image)

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

        image = Image.fromarray(np.uint8(image))

        self.update_image_inside_app(current_tab, image)


    def apply_distorsia_to_current_image(self):
        current_tab, path, image = self.get_current_working_data()
        if not current_tab:
            return

        img = np.array(image)



        image = Image.fromarray(np.uint8(image))

        self.update_image_inside_app(current_tab, image)


    def save_images_to_config(self):
        paths = [(path[:-1] if path[-1] == '*' else path) for (path, image) in self.opened_images]
        images = {"opened_images": paths}
        with open(CONFIG_FILE, "w") as f:
            json.dump(images, f, indent=4)


    def unsave_images(self):
        #Неиспользованные переменнные называются нижнем подчеркиванием
        for path, _ in self.opened_images:
            if path[-1] == "*":
                return True
            return False



    def _close(self, event=None):
        if self.unsave_images():
            if not mb.askyesno("Unsave changes", "Got unsaved changes! Exit anyway?"):
                return

        self.save_images_to_config()
        self.root.quit()


if __name__ == "__main__":
    PyPhotoEditor().run()


