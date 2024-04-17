import cv2
import numpy

#Основной алгоритм для вызова
def RemoveChromaticAbberation(image):
    #разбиваем изображения на каналы RGB для более удобной работы
    arr= cv2.split(image)
    
    #Устанавливаем ограничение для нахождения границы в G канале
    threshold = 30

    #Горизонтальное устранение хроматических аббераций
    arr = __RemoveChromaticAbberation(threshold,arr,False)
    #Поворот и вертикальное устранение хроматических аббераций
    arr = __Rotate(arr)
    arr = __RemoveChromaticAbberation(threshold, arr, True)

    #Соединение и соединение каналов в обработанное изображение
    arr = cv2.merge(arr)
    arr = __Rotate(arr)
    result=cv2.transpose(arr)
    return result

#Поворот массива RGB
def __Rotate(arr):
    rotate = cv2.merge(arr)
    rotate=cv2.transpose(rotate)
    result=cv2.split(rotate)
    return result

#Сатурация для избегания выхода значения за рамки
def __Saturate(A,B=0):
    C=A+B
    if C>255:
        C=255
    elif C<0:
        C=0
    return C

#Устранение, прогоняет каналы по одному положению (высота, ширина)
def __RemoveChromaticAbberation(threshold,arr):
    #Получение высоты и ширины.
    height =arr[0].shape[0]
    width =arr[0].shape[1]
    
    for i in range (0,height):
        #Получение отдельных каналов
        bptr =numpy.ubyte( arr[0][i])
        gptr= numpy.ubyte(arr[1][i])
        rptr =numpy.ubyte(arr[2][i])
        for j in range(1,width-1):
            #Находим место абберации
            if (abs(numpy.int64(gptr[j + 1]) - numpy.int64(gptr[j - 1])) >= threshold):
                #Знак этого места
                sign = 0
                if (numpy.int64(gptr[j + 1]) - numpy.int64(gptr[j - 1]) > 0):
                    sign = 1
                else:
                    sign = -1

                #Поиск границ этой абберации
                lpos = j-1
                rpos = j+1
                while(lpos > 0):
                    #Важно: градиент того же знака, что и место
                    ggrad = (numpy.int64(gptr[lpos + 1]) - numpy.int64(gptr[lpos - 1]))*sign
                    bgrad = (numpy.int64(bptr[lpos + 1]) - numpy.int64(bptr[lpos - 1]))*sign
                    rgrad = (numpy.int64(rptr[lpos + 1]) - numpy.int64(rptr[lpos - 1]))*sign
                    if (max(max(bgrad, ggrad), rgrad) < threshold):
                        break
                    lpos=lpos-1
                lpos-=1
                while(rpos <width-1):
                    #Важно: градиент того же знака, что и место
                    ggrad = (numpy.int64(gptr[rpos + 1]) - numpy.int64(gptr[rpos - 1]))*sign
                    bgrad = (numpy.int64(bptr[rpos + 1]) - numpy.int64(bptr[rpos - 1]))*sign
                    rgrad = (numpy.int64(rptr[rpos + 1]) - numpy.int64(rptr[rpos - 1]))*sign
                    if (max(max(bgrad, ggrad), rgrad) < threshold):
                        break
                    rpos=rpos+1
                rpos+=1

                if rpos>=width:
                    rpos=width-1
                if lpos<=0:
                    lpos=0


                bgmaxVal = max(numpy.int64(bptr[lpos]) - numpy.int64(gptr[lpos]), numpy.int64(bptr[rpos]) - numpy.int64(gptr[rpos]))
                bgminVal = min(numpy.int64(bptr[lpos]) - numpy.int64(gptr[lpos]), numpy.int64(bptr[rpos]) - numpy.int64(gptr[rpos]))
                rgmaxVal = max(numpy.int64(rptr[lpos]) - numpy.int64(gptr[lpos]), numpy.int64(rptr[rpos]) - numpy.int64(gptr[rpos]))
                rgminVal = min(numpy.int64(rptr[lpos]) - numpy.int64(gptr[lpos]), numpy.int64(rptr[rpos]) - numpy.int64(gptr[rpos]))
                
                for k in range(lpos, rpos):
                    bdiff = numpy.int64(bptr[k]) - numpy.int64(gptr[k])
                    rdiff = numpy.int64(rptr[k]) - numpy.int64(gptr[k])
                    if (bdiff > bgmaxVal): 
                        bptr[k] = __Saturate(bgmaxVal , gptr[k])
                    elif (bdiff < bgminVal):
                        bptr[k] = __Saturate(bgminVal , gptr[k])
                    else:
                        bptr[k] = __Saturate(bptr[k])
                    
                    if (rdiff > rgmaxVal):
                        rptr[k] = __Saturate(rgmaxVal, gptr[k])
                    elif (rdiff < rgminVal):
                        rptr[k] = __Saturate(rgminVal , gptr[k])
                    else:
                        rptr[k] = __Saturate(rptr[k])
                j = rpos - 2
        arr[0][i] = bptr
        arr[1][i] = gptr
        arr[2][i] = rptr
    return arr	