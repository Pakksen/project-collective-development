import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['toolbar'] = 'None'#бирает меню в координатной плоскости
# функция, которая позволяет после 4 кликов по углам изображения на координатной плоскоти,
# записать координаты области которая подвержена перспективному искажению
def onclick(event):
    global ix, iy
    ix, iy = event.xdata, event.ydata
    print(f'x = {ix}, y = {iy}')

    global coords
    coords.append((ix, iy))

    if len(coords) == 4:
        fig.canvas.mpl_disconnect(cid)
        plt.close()
    return coords

# Load the image
img = cv2.imread('table.jpg')

# Create a copy of the image
img_copy = np.copy(img)


fig = plt.figure()
ax = fig.add_subplot()

# Set titles for the figure and the subplot respectively
fig.suptitle('Выберете крайние точки\nбудующего изображения в порядке:', fontsize=12, fontweight='bold')
ax.set_title('ЛВ, ПВ, ЛН, ПН')#(это крайние точки готового изображения
                                # (Пример ЛВ(левая верхняя) точка конечного изображения))
#ывод изображения на координатной плоскости

# Convert to RGB so as to display via matplotlib
# Using Matplotlib we can easily find the coordinates
# of the 4 points that is essential for finding the
# transformation matrix
img_copy = cv2.cvtColor(img_copy,cv2.COLOR_BGR2RGB)#он нужен для правильной подачи цвета
plt.imshow(img_copy)
#массив координат начального изображения
coords = []
cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()

pts1 = np.float32(coords)#копия массива координат начального изображения
pts2 = np.float32([[0, 0], [800, 0], [0, 600], [800, 600]])#позиция выбранных координат,
                                                            # на конечном изображении

matrix = cv2.getPerspectiveTransform(pts1, pts2)
result = cv2.warpPerspective(img, matrix, (800, 600))#размер конечного изображения
cv2.imshow("Perspective transformation", result)

cv2.waitKey(0)
cv2.destroyAllWindows()