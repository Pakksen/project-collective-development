from image_edit import ImageEdit
from PIL import Image
from tkinter import filedialog as fd
import os

class ImageInfo(ImageEdit):
    def __init__(self, image, path, tab):
        super().__init__(image)

        self.path = path
        self.tab = tab

    @property #get
    def unsaved(self) -> bool:
        return self.path[-1] == "*"

    @unsaved.setter #set
    def unsaved(self, value: bool):
        if value and not self.unsaved:
            self.path += "*"
        elif not value and self.unsaved:
            self.path = self.path[:-1]

    def filename(self, no_star=False):
        name = os.path.split(self.path)[1]
        return name[:-1] if no_star and name[-1] == '*' else name

    def file_extension(self, no_star=False):
        ext = os.path.split(self.path)[1]
        return ext[:-1] if no_star and ext[-1] == '*' else ext

    def directory(self, no_star=False):
        dirname = os.path.split(self.path)[0]
        return dirname[:-1] if no_star and dirname[-1] == '*' else dirname

    def full_path(self, no_star=False):
        return self.path[:-1] if no_star and self.path[-1] == '*' else self.path

    def save(self):
        if not self.unsaved:
            return
        self.unsaved = False
        self.image.save(self.path)

    def save_as(self):
        old_ext = self.file_extension(no_star=True)
        new_path = fd.asksaveasfilename(initialdir=self.full_path(no_star=True), filetypes=[("Images", "*.jpeg", "*.png", "*.jpg"), ])
        if not new_path:
            return
        new_path, new_ext = os.path.split(new_path)
        if not new_ext:
            new_ext = old_ext
        elif new_ext != old_ext:
            raise ValueError(f"Got incorrect extension: '{new_ext}'. Old was '{old_ext}'")

        self.image.seve(new_path + new_ext)
        self.image.close

        self.path = new_path + new_ext
        self.unsaved = False

        self.image = Image.open(self.path)

    def close(self):
        self.image.close()
        self.original_image.close()

    def delete(self):
        self.close()
        os.remove(self.full_path(no_star=True))
