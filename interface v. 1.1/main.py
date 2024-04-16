from tkinter import *
from tkinter import filedialog as fd
from tkinter import messagebox as mb
from tkinter.ttk import Notebook
from PIL import Image, ImageFilter
import os
import json

from image_info import ImageInfo
CONFIG_FILE = "config.json"


class PyPhotoEditor:
    def __init__(self):
        self.root = Tk()
        self.image_tabs = Notebook(self.root)
        self.opened_images = []

        self.init()

    def init(self):
        self.root.title("Kulema")
        # self.root.iconphoto(True, PhotoImage(file = "")) #Иконка приложенияв кавычки нужно вставить путь к картинке
        self.image_tabs.enable_traversal()  # Позволяет переключаться между вкладками

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
        self.drow_menu()  # Прорисовка меню
        self.drow_widegets()  # Прорисовка виджетов

        self.root.mainloop()  # Звпуск основоного цикла программы

    def drow_menu(self):
        menu_bar = Menu(self.root)

        file_menu = Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="Open", command=self.open_new_images)
        file_menu.add_command(label="Save", command=self.save_current_image)
        file_menu.add_command(label="Save as", command=self.save_image_as)
        file_menu.add_command(label="Save all", command=self.save_all_changes)
        file_menu.add_separator()
        file_menu.add_command(label="Close image", command=self.close_carrent_image)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._close)
        menu_bar.add_cascade(label="File", menu=file_menu)

        edit_menu = Menu(menu_bar, tearoff=0)
        rotate_menu = Menu(edit_menu, tearoff=0)
        flip_menu = Menu(edit_menu, tearoff=0)
        resize_menu = Menu(edit_menu, tearoff=0)
        filter_menu = Menu(edit_menu, tearoff=0)
        crop_menu = Menu(edit_menu, tearoff=0)

        rotate_menu.add_command(label="Rotate left by 90", command=lambda: self.rotate_current_image(90))
        rotate_menu.add_command(label="Rotate right by 90", command=lambda: self.rotate_current_image(-90))
        rotate_menu.add_command(label="Rotate left by 180", command=lambda: self.rotate_current_image(180))
        rotate_menu.add_command(label="Rotate right by 180", command=lambda: self.rotate_current_image(-180))

        flip_menu.add_command(label="Flip horizontally", command=lambda: self.flip_current_image(Image.FLIP_LEFT_RIGHT))
        flip_menu.add_command(label="Flip vertically", command=lambda: self.flip_current_image(Image.FLIP_TOP_BOTTOM))

        resize_menu.add_command(label="25% of original size", command=lambda: self.resize_current_image(25))
        resize_menu.add_command(label="50% of original size", command=lambda: self.resize_current_image(50))
        resize_menu.add_command(label="75% of original size", command=lambda: self.resize_current_image(75))
        resize_menu.add_command(label="100% of original size", command=lambda: self.resize_current_image(100))
        resize_menu.add_command(label="125% of original size", command=lambda: self.resize_current_image(125))
        resize_menu.add_command(label="150% of original size", command=lambda: self.resize_current_image(150))
        resize_menu.add_command(label="175% of original size", command=lambda: self.resize_current_image(175))
        resize_menu.add_command(label="200% of original size", command=lambda: self.resize_current_image(200))

        filter_menu.add_command(label="Blur", command=lambda: self.apply_filter_to_current_image(
            ImageFilter.BLUR))  # Размытие (размывает еще больше чем было)
        filter_menu.add_command(label="Sharpen", command=lambda: self.apply_filter_to_current_image(
            ImageFilter.SHARPEN))  # Повышение контрастности
        filter_menu.add_command(label="Contour", command=lambda: self.apply_filter_to_current_image(
            ImageFilter.CONTOUR))  # Повышение контрастности контура
        filter_menu.add_command(label="Detail", command=lambda: self.apply_filter_to_current_image(
            ImageFilter.DETAIL))  # Повышение детализированости
        filter_menu.add_command(label="Smooth", command=lambda: self.apply_filter_to_current_image(
            ImageFilter.SMOOTH))  # Что-то вроеде размытости

        filter_menu.add_command(label="Шум", command=self.apply_median_filter_to_current_image)  # Устраняет шум
        filter_menu.add_command(label="Размытие", command=self.apply_blur_to_current_image)  # Устраняет размытие
        filter_menu.add_command(label="Горизонт",
                                command=self.apply_perspective_to_current_image)  # Устраняет Заваленный горизонт
        filter_menu.add_command(label="Переэкспонирование",
                                command=self.apply_bliky_to_current_image)  # Устраняет Переэкспонирование
        filter_menu.add_command(label="Дисторсия", command=self.apply_distorsia_to_current_image)  # Устраняет дисторсию
        filter_menu.add_command(label="Хроматические абберации", command=self.chromatic_abberation_to_current_image)
        filter_menu.add_command(label="Перспективные искажения", command=self.perspective_to_current_image)# Устраняет перспективные искажения

        crop_menu.add_command(label="Start area selection", command=self.start_crop_selection_of_current_image)
        crop_menu.add_command(label="Crop selected", command=self.crop_selection_of_current_image)

        edit_menu.add_cascade(label="Rotate", menu=rotate_menu)
        edit_menu.add_cascade(label="Flip", menu=flip_menu)
        edit_menu.add_cascade(label="Resize", menu=resize_menu)
        edit_menu.add_separator()
        edit_menu.add_cascade(label="Filter", menu=filter_menu)
        edit_menu.add_separator()
        edit_menu.add_cascade(label="Crop", menu=crop_menu)

        menu_bar.add_cascade(label="Edit", menu=edit_menu)

        self.root.configure(menu=menu_bar)

    def update_open_recent_menu(self):
        if self.open_recent_menu is None:
            return

        self.open_recent_menu.delete(0, "end")
        for path in self.last_viewed_images:
            self.open_recent_menu.add_command(label=path, command=lambda x=path: self.add_new_image(x))

    def drow_widegets(self):
        self.image_tabs.pack(fill="both", expand=1)

    def load_images_from_config(self):
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)

        paths = config["opened_images"]
        for path in paths:
            self.add_new_image(path)

    def open_new_images(self):
        image_paths = fd.askopenfilenames(filetypes=(("Images", "*.jpeg; *.jpg; *.png"),))
        for image_path in image_paths:
            self.add_new_image(image_path)

    def add_new_image(self, image_path):  #Изменено
        if not os.path.isfile(image_path):
            if image_path in self.last_viewed_images:
                self.last_viewed_images.remove(image_path)
                self.update_open_recent_menu()
            return
        opened_images = [info.path for info in self.opened_images]
        if image_path in opened_images:
            index = opened_images.index(image_path)
            self.image_tabs.select(index)
            return

        image = Image.open(image_path)  # Чтобы можно было использовать библиотеку pl
        image_tab = Frame(self.image_tabs)

        image_info = ImageInfo(image, image_path, image_tab)
        self.opened_images.append(image_info)

        image_tk = image_info.image_tk

        # Если нужно будет выделять область на изображении замепенить Label на Canvas (10)
        image_panel = Canvas(image_tab, width=image.width, height=image.height, bd=0, highlightthickness=0)
        image_panel.image = image_tk
        image_panel.create_image(0, 0, image=image_tk, anchor="nw")
        image_panel.pack(expand='yes')

        image_info.canvas = image_panel

        self.image_tabs.add(image_tab, text=image_info.filename())
        self.image_tabs.select(image_tab)

    def get_current_working_data(self):  #Изменено
        current_tab = self.image_tabs.select()
        if not current_tab:
            return None
        tab_number = self.image_tabs.index(current_tab)
        return self.opened_images[tab_number]

    def save_current_image(self, event=None):
        image = self.get_current_working_data()
        if not image:
            return
        if not image.unsaved:
            return

        image.save()
        self.image_tabs.add(image.tab, text=image.filename)

    def save_image_as(self):#Изменено
        image = self.get_current_working_data()
        if not image:
            return
        if not image.unsaved:
            return

        try:
            image.save_as()
            self.update_image_inside_app(image)
        except ValueError as e:
            mb.showerror("Sane as error", str(e))

    def save_all_changes(self): #Изменено
        for image_info in self.opened_images:
            if not image_info.unsaved:
                continue
            image_info.save()
            self.image_tabs.tab(image_info.tab, text=image_info.filename())

    def close_carrent_image(self, event=None): #Изменено
        image = self.get_current_working_data()
        if not image:
            return
        if image.unsaved:
            if not mb.askyesno("Unsaved changes", "Close without saving changes?"):
                return

        image.close()
        self.image_tabs.forget(image.tab)
        self.opened_images.remove(image)

    def delete_current_image(self): #Изменено
        image = self.get_current_working_data()
        if not image:
            return

        if not mb.askokcancel("Delete image", "Are you sure you want to delete image?\nThis operation is unrecoverable!"):
            return

        image.delete()
        self.image_tabs.forget(image.tab)
        self.opened_images.remove(image)

    def update_image_inside_app(self, image_info):  #Изменено
        image_info.update_image_on_canvas()
        self.image_tabs.tab(image_info.tab, text=image_info.filename())

    def rotate_current_image(self, degrees): #Изменено
        image = self.get_current_working_data()
        if not image:
            return

        image.rotate(degrees)
        image.unsaved = True
        self.update_image_inside_app(image)

    def flip_current_image(self, mode): #Изменено
        image = self.get_current_working_data()
        if not image:
            return

        image.flip(mode)
        image.unsaved = True
        self.update_image_inside_app(image)

    def resize_current_image(self, percents): #Изменено
        image = self.get_current_working_data()
        if not image:
            return

        image.resize(percents)
        image.unsaved = True
        self.update_image_inside_app(image)

    def apply_filter_to_current_image(self, filter_type):  #Изменено
        image = self.get_current_working_data()
        if not image:
            return

        image.filter(filter_type)
        image.unsaved = True
        self.update_image_inside_app(image)

    def apply_median_filter_to_current_image(self):  # Шум #Изменено
        image = self.get_current_working_data()
        if not image:
            return

        image.apply_median_filter()
        image.unsaved = True
        self.update_image_inside_app(image)

    def apply_blur_to_current_image(self):  # Размытие #Изменено
        image = self.get_current_working_data()
        if not image:
            return

        image.apply_blur()
        image.unsaved = True
        self.update_image_inside_app(image)

    def apply_perspective_to_current_image(self): #Изменено
        image = self.get_current_working_data()
        if not image:
            return

        image.apply_perspective()
        image.unsaved = True
        self.update_image_inside_app(image)

    def apply_bliky_to_current_image(self): #Изменено
        image = self.get_current_working_data()
        if not image:
            return

        image.apply_bliky()
        image.unsaved = True
        self.update_image_inside_app(image)

    def chromatic_abberation_to_current_image(self): #Изменено
        image = self.get_current_working_data()
        if not image:
            return

        image.chromatic_abberation()
        image.unsaved = True
        self.update_image_inside_app(image)

    def perspective_to_current_image(self): #Изменено
        image = self.get_current_working_data()
        if not image:
            return

        image.perspective()
        image.unsaved = True
        self.update_image_inside_app(image)

    def apply_distorsia_to_current_image(self): #Изменено
        image = self.get_current_working_data()
        if not image:
            return

        image.apply_distorsia()
        image.unsaved = True
        self.update_image_inside_app(image)

    def start_crop_selection_of_current_image(self):
        image = self.get_current_working_data()
        if not image:
            return

        image.start_crop_selection()

    def crop_selection_of_current_image(self):
        image = self.get_current_working_data()
        if not image:
            return

        try:
            image.crop_selected_area()
            image.unsaved = True
            self.update_image_inside_app(image)
        except ValueError as e:
            mb.showerror("Crop error", str(e))



    def save_images_to_config(self): #Изменено
        paths = [info.full_path(no_star=True) for info in self.opened_images]
        images = {"opened_images": paths}
        with open(CONFIG_FILE, "w") as f:
            json.dump(images, f, indent=4)

    def unsave_images(self): #Изменено
        # Неиспользованные переменнные называются нижнем подчеркиванием
        for info in self.opened_images:
            if info.unsaved:
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
