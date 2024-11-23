import numpy as np
import matplotlib.pyplot as plt
import cv2

"""
Oblicza najbliższe dopasowanie koloru.

Parametry:
    colorValue (array-like): Wartość koloru do dopasowania. Może to być pojedyncza wartość w skali szarości lub trójka RGB.
    colorRange (array-like): Zakres kolorów do dopasowania. Może to być lista wartości w skali szarości lub lista trójek RGB.

Zwraca:
    array: Najbliższy dopasowany kolor z colorRange.
"""
def colorFit(pixel, pallet):
    distances = np.linalg.norm(pallet - pixel, axis=1)
    return pallet[np.argmin(distances)]


def kwant_colorFit(img, pallet):
    out_img = img.copy()
    for w in range(img.shape[0]):
        for k in range(img.shape[1]):
            out_img[w, k] = colorFit(img[w, k], pallet)

    return out_img

def random_dithering(image, pallet):
    # 1. Generate a matrix of random values
    random_matrix = np.random.rand(*image.shape)

    # 2. Compare the image with random values
    binary_image = image >= random_matrix

    # 3. Convert the logical matrix to numerical
    dithered_image = binary_image.astype(np.float32)

    # 4. Apply the colorFit function to map the dithered image to the palette
    for w in range(dithered_image.shape[0]):
        for k in range(dithered_image.shape[1]):
            dithered_image[w, k] = colorFit(dithered_image[w, k], pallet)

    return dithered_image


M2 = np.array([
    [0, 8, 2, 10,],
    [12, 4, 14, 6,],
    [3, 11, 1, 9,],
    [15, 7, 13, 5,],
])


def ordered_dithering(image, pallet, r=1.0):
    # 1. Przygotowanie tablicy Mpre z wartościami <-0.5, 0.5>
    Mpre = (M2 + 1) / (2 * 2) ** 2 - 0.5

    # 2. Dla każdego piksela obrazu
    out_img = image.copy()
    for w in range(image.shape[0]):
        for k in range(image.shape[1]):
            # Znajdź odpowiadający piksel w tablicy Mpre
            m = Mpre[w % 4, k % 4]

            # 3. Oblicz wartość tymczasową piksela
            temp_pixel = image[w, k] + m * r

            # 4. Znajdź nowe wartości koloru przy użyciu funkcji colorFit
            out_img[w, k] = colorFit(temp_pixel, pallet)

    return out_img


def floyd_steinberg_dithering(image, pallet):
    """
    Zastosowanie ditheringu metodą Floyda-Steinberga do obrazu.

    Parametry:
        image (array-like): Obraz do przetworzenia.
        pallet (array-like): Zakres kolorów do dopasowania. Może to być lista wartości w skali szarości lub lista trójek RGB.

    Zwraca:
        array: Obraz po zastosowaniu ditheringu.
    """
    out_img = image.copy()
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            old_pixel = out_img[y, x]
            new_pixel = colorFit(old_pixel, pallet)
            quant_error = old_pixel - new_pixel
            out_img[y, x] = new_pixel

            if x + 1 < image.shape[1]:
                out_img[y, x + 1] = np.clip(out_img[y, x + 1] + quant_error * 7 / 16, 0, 1)
            if x - 1 >= 0 and y + 1 < image.shape[0]:
                out_img[y + 1, x - 1] = np.clip(out_img[y + 1, x - 1] + quant_error * 3 / 16, 0, 1)
            if y + 1 < image.shape[0]:
                out_img[y + 1, x] = np.clip(out_img[y + 1, x] + quant_error * 5 / 16, 0, 1)
            if x + 1 < image.shape[1] and y + 1 < image.shape[0]:
                out_img[y + 1, x + 1] = np.clip(out_img[y + 1, x + 1] + quant_error * 1 / 16, 0, 1)

    return out_img


pallet8 = np.array([
    [0.0, 0.0, 0.0,],
    [0.0, 0.0, 1.0,],
    [0.0, 1.0, 0.0,],
    [0.0, 1.0, 1.0,],
    [1.0, 0.0, 0.0,],
    [1.0, 0.0, 1.0,],
    [1.0, 1.0, 0.0,],
    [1.0, 1.0, 1.0,],
])

pallet16 = np.array([
    [0.0, 0.0, 0.0,],
    [0.0, 1.0, 1.0,],
    [0.0, 0.0, 1.0,],
    [1.0, 0.0, 1.0,],
    [0.0, 0.5, 0.0,],
    [0.5, 0.5, 0.5,],
    [0.0, 1.0, 0.0,],
    [0.5, 0.0, 0.0,],
    [0.0, 0.0, 0.5,],
    [0.5, 0.5, 0.0,],
    [0.5, 0.0, 0.5,],
    [1.0, 0.0, 0.0,],
    [0.75, 0.75, 0.75,],
    [0.0, 0.5, 0.5,],
    [1.0, 1.0, 1.0,],
    [1.0, 1.0, 0.0,],
])


def load_img_cv2(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(img.dtype)
    print(img.shape)
    print(np.min(img), np.max(img))
    return img

def img_to_float(img):
    if np.issubdtype(img.dtype, np.floating):
        return img
    elif (np.issubdtype(img.dtype, np.integer)
          or np.issubdtype(img.dtype, np.unsignedinteger)):
        return (img / 255.0).astype(np.float32)
    else: raise ValueError(f'Cannot convert img data type {img.dtype} '
                      f'to float32!')

def img_to_uint8(img):
    if np.issubdtype(img.dtype,np.unsignedinteger):
        return img
    elif np.issubdtype(img.dtype,np.integer):
        return img.astype(np.uint8)
    elif np.issubdtype(img.dtype,np.floating):
        return (img * 255).astype(np.uint8)
    else: raise ValueError(f'Cannot convert img data type {img.dtype} '
                      f'to uint8!')

def img_to_grayscale_cv2_y1(img):
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]

    return 0.299 * R + 0.587 * G + 0.114 * B

# paleta = np.linspace(0, 1, 3).reshape(3, 1)
# print(colorFit(0.43, paleta))
# print(colorFit(0.66, paleta))
# print(colorFit(0.8, paleta))
#
# print(colorFit(np.array([0.25, 0.25, 0.5]), pallet8))
# print(colorFit(np.array([0.25, 0.25, 0.5]), pallet16))
#
# print((M2 + 1) / (2 * 2) ** 2 - 0.5)

def grayscale():
    imgPath = './IMG_GS/'

    img = load_img_cv2(imgPath + 'GS_0001.tif')
    img = img_to_float(img)
    img = img_to_grayscale_cv2_y1(img)

    plt.subplot(1, 4, 1)
    plt.title('Original')
    plt.imshow(img, cmap=plt.cm.gray, vmin=np.min(img), vmax=np.max(img))

    paleta = np.array([[0], [1]])
    plt.subplot(1, 4, 2)
    plt.title('Pallet 1-bit')
    plt.imshow(kwant_colorFit(img, paleta), cmap=plt.cm.gray, vmin=np.min(img), vmax=np.max(img))

    paleta = np.array([[0], [0.33], [0.67], [1]])
    plt.subplot(1, 4, 3)
    plt.title('Pallet 2-bit')
    plt.imshow(kwant_colorFit(img, paleta), cmap=plt.cm.gray, vmin=np.min(img), vmax=np.max(img))

    paleta = np.array([[0], [0.2], [0.4], [0.6], [0.8], [1]])
    plt.subplot(1, 4, 4)
    plt.title('Pallet 4-bit')
    plt.imshow(kwant_colorFit(img, paleta), cmap=plt.cm.gray, vmin=np.min(img), vmax=np.max(img))

    plt.show()

    img = load_img_cv2(imgPath + 'GS_0002.png')
    img = img_to_float(img)

    plt.subplot(1, 4, 1)
    plt.title('Original')
    plt.imshow(img, cmap=plt.cm.gray, vmin=np.min(img), vmax=np.max(img))

    paleta = np.array([[0], [1]])
    plt.subplot(1, 4, 2)
    plt.title('Pallet 1-bit')
    plt.imshow(kwant_colorFit(img, paleta), cmap=plt.cm.gray, vmin=np.min(img), vmax=np.max(img))

    paleta = np.array([[0], [0.33], [0.67], [1]])
    plt.subplot(1, 4, 3)
    plt.title('Pallet 2-bit')
    plt.imshow(kwant_colorFit(img, paleta), cmap=plt.cm.gray, vmin=np.min(img), vmax=np.max(img))

    paleta = np.array([[0], [0.2], [0.4], [0.6], [0.8], [1]])
    plt.subplot(1, 4, 4)
    plt.title('Pallet 4-bit')
    plt.imshow(kwant_colorFit(img, paleta), cmap=plt.cm.gray, vmin=np.min(img), vmax=np.max(img))

    plt.show()

    img = load_img_cv2(imgPath + 'GS_0003.png')
    img = img_to_float(img)

    plt.subplot(1, 4, 1)
    plt.title('Original')
    plt.imshow(img, cmap=plt.cm.gray, vmin=np.min(img), vmax=np.max(img))

    paleta = np.array([[0], [1]])
    plt.subplot(1, 4, 2)
    plt.title('Pallet 1-bit')
    plt.imshow(kwant_colorFit(img, paleta), cmap=plt.cm.gray, vmin=np.min(img), vmax=np.max(img))

    paleta = np.array([[0], [0.33], [0.67], [1]])
    plt.subplot(1, 4, 3)
    plt.title('Pallet 2-bit')
    plt.imshow(kwant_colorFit(img, paleta), cmap=plt.cm.gray, vmin=np.min(img), vmax=np.max(img))

    paleta = np.array([[0], [0.2], [0.4], [0.6], [0.8], [1]])
    plt.subplot(1, 4, 4)
    plt.title('Pallet 4-bit')
    plt.imshow(kwant_colorFit(img, paleta), cmap=plt.cm.gray, vmin=np.min(img), vmax=np.max(img))

    plt.show()

def color():
    imgPath = './IMG_SMALL/'

    img = load_img_cv2(imgPath + 'SMALL_0001.tif')
    img = img_to_float(img)

    plt.subplot(1, 3, 1)
    plt.title('Original')
    plt.imshow(img, vmin=np.min(img), vmax=np.max(img))

    plt.subplot(1, 3, 2)
    plt.title('pallet8')
    plt.imshow(kwant_colorFit(img, pallet8), vmin=np.min(img), vmax=np.max(img))

    plt.subplot(1, 3, 3)
    plt.title('pallet16')
    plt.imshow(kwant_colorFit(img, pallet16), vmin=np.min(img), vmax=np.max(img))

    plt.show()

    img = load_img_cv2(imgPath + 'SMALL_0004.jpg')
    img = img_to_float(img)

    plt.subplot(1, 3, 1)
    plt.title('Original')
    plt.imshow(img, vmin=np.min(img), vmax=np.max(img))

    plt.subplot(1, 3, 2)
    plt.title('pallet8')
    plt.imshow(kwant_colorFit(img, pallet8), vmin=np.min(img), vmax=np.max(img))

    plt.subplot(1, 3, 3)
    plt.title('pallet16')
    plt.imshow(kwant_colorFit(img, pallet16), vmin=np.min(img), vmax=np.max(img))

    plt.show()

    img = load_img_cv2(imgPath + 'SMALL_0005.jpg')
    img = img_to_float(img)

    plt.subplot(1, 3, 1)
    plt.title('Original')
    plt.imshow(img, vmin=np.min(img), vmax=np.max(img))

    plt.subplot(1, 3, 2)
    plt.title('pallet8')
    plt.imshow(kwant_colorFit(img, pallet8), vmin=np.min(img), vmax=np.max(img))

    plt.subplot(1, 3, 3)
    plt.title('pallet16')
    plt.imshow(kwant_colorFit(img, pallet16), vmin=np.min(img), vmax=np.max(img))

    plt.show()


def dithering_grayscale(imgUrl):
    imgPath = './IMG_GS/'

    img = load_img_cv2(imgPath + imgUrl)
    img = img_to_float(img)
    img = img_to_grayscale_cv2_y1(img)

    plt.subplot(2, 3, 1)
    plt.title('Original')
    plt.imshow(img, vmin=np.min(img), cmap=plt.cm.gray, vmax=np.max(img))

    paleta = np.array([[0], [1]])
    plt.subplot(2, 3, 2)
    plt.title('Kwantyzacja')
    newimg = kwant_colorFit(img, paleta)
    plt.imshow(newimg, cmap=plt.cm.gray, vmin=np.min(newimg), vmax=np.max(newimg))

    plt.subplot(2, 3, 3)
    plt.title('Dithering Zorganizowany')
    newimg = ordered_dithering(img, paleta)
    plt.imshow(newimg, cmap=plt.cm.gray, vmin=np.min(newimg), vmax=np.max(newimg))

    plt.subplot(2, 3, 4)
    plt.title('Dithering Losowy')
    newimg = random_dithering(img, paleta)
    plt.imshow(newimg, cmap=plt.cm.gray, vmin=np.min(newimg), vmax=np.max(newimg))

    plt.subplot(2, 3, 5)
    plt.title('Dithering Floyd-Steinberg')
    newimg = floyd_steinberg_dithering(img, paleta)
    plt.imshow(newimg, cmap=plt.cm.gray, vmin=np.min(newimg), vmax=np.max(newimg))

    plt.show()

    paleta = np.array([[0], [0.33], [0.67], [1]])

    plt.subplot(2, 2, 1)
    plt.title('Original')
    plt.imshow(img, vmin=np.min(img), cmap=plt.cm.gray, vmax=np.max(img))

    plt.subplot(2, 2, 2)
    plt.title('Dithering Zorganizowany')
    newimg = ordered_dithering(img, paleta)
    plt.imshow(newimg, vmin=np.min(newimg), cmap=plt.cm.gray, vmax=np.max(newimg))

    plt.subplot(2, 2, 3)
    plt.title('Kwantyzacja')
    newimg = kwant_colorFit(img, paleta)
    plt.imshow(newimg, vmin=np.min(newimg), cmap=plt.cm.gray, vmax=np.max(newimg))

    plt.subplot(2, 2, 4)
    plt.title('Dithering Floyd-Steinberg')
    newimg = floyd_steinberg_dithering(img, paleta)
    plt.imshow(newimg, vmin=np.min(newimg), cmap=plt.cm.gray, vmax=np.max(newimg))

    plt.show()

    paleta = np.array([[0], [0.2], [0.4], [0.6], [0.8], [1]])

    plt.subplot(2, 2, 1)
    plt.title('Original')
    plt.imshow(img, vmin=np.min(img), cmap=plt.cm.gray, vmax=np.max(img))

    plt.subplot(2, 2, 2)
    plt.title('Dithering Zorganizowany')
    newimg = ordered_dithering(img, paleta)
    plt.imshow(newimg, vmin=np.min(newimg), cmap=plt.cm.gray, vmax=np.max(newimg))

    plt.subplot(2, 2, 3)
    plt.title('Kwantyzacja')
    newimg = kwant_colorFit(img, paleta)
    plt.imshow(newimg, vmin=np.min(newimg), cmap=plt.cm.gray, vmax=np.max(newimg))

    plt.subplot(2, 2, 4)
    plt.title('Dithering Floyd-Steinberg')
    newimg = floyd_steinberg_dithering(img, paleta)
    plt.imshow(newimg, vmin=np.min(newimg), cmap=plt.cm.gray, vmax=np.max(newimg))

    plt.show()

def dithering_color(imgUrl):
    imgPath = './IMG_SMALL/'

    img = load_img_cv2(imgPath + imgUrl)
    img = img_to_float(img)

    plt.subplot(2, 2, 1)
    plt.title('Original')
    plt.imshow(img, vmin=np.min(img), vmax=np.max(img))

    plt.subplot(2, 2, 2)
    plt.title('Dithering Zorganizowany')
    newimg = ordered_dithering(img, pallet8)
    plt.imshow(newimg, vmin=np.min(newimg), vmax=np.max(newimg))

    plt.subplot(2, 2, 3)
    plt.title('Kwantyzacja')
    newimg = kwant_colorFit(img, pallet8)
    plt.imshow(newimg, vmin=np.min(newimg), vmax=np.max(newimg))

    plt.subplot(2, 2, 4)
    plt.title('Dithering Floyd-Steinberg')
    newimg = floyd_steinberg_dithering(img, pallet8)
    plt.imshow(newimg, vmin=np.min(newimg), vmax=np.max(newimg))

    plt.show()

    plt.subplot(2, 2, 1)
    plt.title('Original')
    plt.imshow(img, vmin=np.min(img), cmap=plt.cm.gray, vmax=np.max(img))

    plt.subplot(2, 2, 2)
    plt.title('Dithering Zorganizowany')
    newimg = ordered_dithering(img, pallet16)
    plt.imshow(newimg, vmin=np.min(newimg), vmax=np.max(newimg))

    plt.subplot(2, 2, 3)
    plt.title('Kwantyzacja')
    newimg = kwant_colorFit(img, pallet16)
    plt.imshow(newimg, vmin=np.min(newimg), vmax=np.max(newimg))

    plt.subplot(2, 2, 4)
    plt.title('Dithering Floyd-Steinberg')
    newimg = floyd_steinberg_dithering(img, pallet16)
    plt.imshow(newimg, vmin=np.min(newimg), vmax=np.max(newimg))

    plt.show()


# grayscale()
# color()
dithering_grayscale('GS_0003.png')
dithering_grayscale('GS_0001.tif')
dithering_grayscale('GS_0002.png')
dithering_color('SMALL_0001.tif')
dithering_color('SMALL_0004.jpg')
dithering_color('SMALL_0005.jpg')
