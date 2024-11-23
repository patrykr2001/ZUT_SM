import os

import numpy as np
import matplotlib.pyplot as plt
import cv2

def show_img(img):
    plt.imshow(img)
    plt.show()

def show_img_grayscale(img):
    plt.imshow(img, cmap=plt.cm.gray, vmin=np.min(img), vmax=np.max(img))
    plt.show()

def load_img_plt(path):
    img = plt.imread(path)
    print(img.dtype)
    print(img.shape)
    print(np.min(img), np.max(img))
    return img

def load_img_cv2(path):
    img = cv2.imread(path)
    print(img.dtype)
    print(img.shape)
    print(np.min(img), np.max(img))
    return img

def img_to_uint8(img):
    if np.issubdtype(img.dtype,np.unsignedinteger):
        return img
    elif np.issubdtype(img.dtype,np.integer):
        return img.astype(np.uint8)
    elif np.issubdtype(img.dtype,np.floating):
        return (img * 255).astype(np.uint8)
    else: raise ValueError(f'Cannot convert img data type {img.dtype} '
                      f'to uint8!')

def img_to_float(img):
    if np.issubdtype(img.dtype, np.floating):
        return img
    elif (np.issubdtype(img.dtype, np.integer)
          or np.issubdtype(img.dtype, np.unsignedinteger)):
        return (img / 255.0).astype(np.float32)
    else: raise ValueError(f'Cannot convert img data type {img.dtype} '
                      f'to float32!')

def img_to_grayscale_cv2_y1(img_BGR):
    img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
    R = img_RGB[:, :, 0]
    G = img_RGB[:, :, 1]
    B = img_RGB[:, :, 2]

    return 0.299 * R + 0.587 * G + 0.114 * B

def img_to_grayscale_cv2_y2(img_BGR):
    img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
    R = img_RGB[:, :, 0]
    G = img_RGB[:, :, 1]
    B = img_RGB[:, :, 2]

    return 0.2126 * R + 0.7152 * G + 0.0722 * B


imgPath = './img/'

def zad_1():
    for file in os.listdir(imgPath):
        img = load_img_cv2(imgPath + file)
        grayscale = img_to_grayscale_cv2_y1(img)
        show_img_grayscale(grayscale)


def zad_2(fileName, plotFileName='temp.png'):
    img = load_img_cv2(imgPath + fileName)
    plt.subplot(3, 3, 1)
    plt.title('Oryginalny')
    plt.imshow(cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB))

    plt.subplot(3, 3, 2)
    plt.title('Y1')
    plt.imshow(img_to_grayscale_cv2_y1(img.copy()), cmap=plt.cm.gray, vmin=np.min(img), vmax=np.max(img))

    plt.subplot(3, 3, 3)
    plt.title('Y2')
    plt.imshow(img_to_grayscale_cv2_y2(img.copy()), cmap=plt.cm.gray, vmin=np.min(img), vmax=np.max(img))

    img_RGB = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
    R = img_RGB[:, :, 0]
    G = img_RGB[:, :, 1]
    B = img_RGB[:, :, 2]

    plt.subplot(3, 3, 4)
    plt.title('R')
    plt.imshow(R, cmap=plt.cm.gray, vmin=np.min(R), vmax=np.max(R))

    plt.subplot(3, 3, 5)
    plt.title('G')
    plt.imshow(G, cmap=plt.cm.gray, vmin=np.min(G), vmax=np.max(G))

    plt.subplot(3, 3, 6)
    plt.title('B')
    plt.imshow(B, cmap=plt.cm.gray, vmin=np.min(B), vmax=np.max(B))

    img_R = img_RGB.copy()
    img_R[:, :, 1] = 0
    img_R[:, :, 2] = 0

    plt.subplot(3, 3, 7)
    plt.title('R', color='red')
    plt.imshow(img_R)

    img_G = img_RGB.copy()
    img_G[:, :, 0] = 0
    img_G[:, :, 2] = 0

    plt.subplot(3, 3, 8)
    plt.title('G', color='green')
    plt.imshow(img_G)

    img_B = img_RGB.copy()
    img_B[:, :, 0] = 0
    img_B[:, :, 1] = 0

    plt.subplot(3, 3, 9)
    plt.title('B', color='blue')
    plt.imshow(img_B)

    plt.savefig(plotFileName)
    plt.show()
