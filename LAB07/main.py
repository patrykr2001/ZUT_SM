import numpy as np
import cv2
import scipy.fftpack
import matplotlib.pyplot as plt


class Ver2:
    def __init__(self,Y,Cb,Cr,OGShape,Ratio="4:4:4",QY=np.ones((8,8)),QC=np.ones((8,8))):
        self.shape = OGShape
        self.Y=Y
        self.Cb=Cb
        self.Cr=Cr
        self.ChromaRatio=Ratio
        self.QY=QY
        self.QC=QC

def to_y_cr_cb(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb).astype(int)

def to_rgb(y_cr_cb):
    return cv2.cvtColor(y_cr_cb.astype(np.uint8), cv2.COLOR_YCrCb2BGR)

def dct2(a):
    return scipy.fftpack.dct( scipy.fftpack.dct( a.astype(float), axis=0, norm='ortho' ), axis=1, norm='ortho' )

def idct2(a):
    return scipy.fftpack.idct( scipy.fftpack.idct( a.astype(float), axis=0 , norm='ortho'), axis=1 , norm='ortho')

def zigzag(a):
    template= np.array([
            [0,  1,  5,  6,  14, 15, 27, 28],
            [2,  4,  7,  13, 16, 26, 29, 42],
            [3,  8,  12, 17, 25, 30, 41, 43],
            [9,  11, 18, 24, 31, 40, 44, 53],
            [10, 19, 23, 32, 39, 45, 52, 54],
            [20, 22, 33, 38, 46, 51, 55, 60],
            [21, 34, 37, 47, 50, 56, 59, 61],
            [35, 36, 48, 49, 57, 58, 62, 63],
            ])
    if len(a.shape)==1:
        b=np.zeros((8,8))
        for r in range(0,8):
            for c in range(0,8):
                b[r,c]=a[template[r,c]]
    else:
        b=np.zeros((64,))
        for r in range(0,8):
            for c in range(0,8):
                b[template[r,c]]=a[r,c]
    return b

def compress_jpeg(rgb, ratio="4:4:4", qy=np.ones((8, 8)), qc=np.ones((8, 8))):
    y_cr_cb=to_y_cr_cb(rgb)
    jpeg=Ver2(y_cr_cb[:,:,0],y_cr_cb[:,:,1],y_cr_cb[:,:,2], rgb.shape, ratio, qy, qc)
    # Tu chroma subsampling
    jpeg.Y=compress_layer(jpeg.Y,jpeg.QY)
    jpeg.Cr=compress_layer(jpeg.Cr,jpeg.QC)
    jpeg.Cb=compress_layer(jpeg.Cb,jpeg.QC)
    # tu dochodzi kompresja bezstratna
    return jpeg

def decompress_jpeg(jpeg):
    # dekompresja bezstratna
    y=decompress_layer(jpeg.Y, jpeg.QY, jpeg.shape[0:2])
    cr=decompress_layer(jpeg.Cr, jpeg.QC, jpeg.shape[0:2])
    cb=decompress_layer(jpeg.Cb, jpeg.QC, jpeg.shape[0:2])
    # Tu chroma resampling
    y_cr_cb=np.dstack([y,cr,cb]).clip(0,255).astype(np.uint8)
    rgb=to_rgb(y_cr_cb)
    return rgb

def compress_block(block, q):
    block = dct2(block)
    block = np.round(block / q).astype(int)
    vector=zigzag(block)
    return vector

def decompress_block(vector, q):
    vector = zigzag(vector)
    vector = vector * q
    block = idct2(vector)
    return block

## podział na bloki
# L - warstwa kompresowana
# S - wektor wyjściowy
def compress_layer(l, q):
    s=np.array([])
    for w in range(0, l.shape[0], 8):
        for k in range(0, l.shape[1], 8):
            block= l[w:(w + 8), k:(k + 8)]
            s=np.append(s, compress_block(block, q))
    return s

## wyodrębnianie bloków z wektora
# L - warstwa o oczekiwanym rozmiarze
# S - długi wektor zawierający skompresowane dane
def decompress_layer(s, q, shape):
    l = np.zeros(shape)
    for idx,i in enumerate(range(0, s.shape[0], 64)):
        vector= s[i:(i + 64)]
        m=l.shape[1]/8
        k=int((idx%m)*8)
        w=int((idx//m)*8)
        l[w:(w+8),k:(k+8)]=decompress_block(vector, q)
    return l

def draw_plt(przed_rgb, po_rgb):
    fig, axs = plt.subplots(4, 2, sharey=True)
    fig.set_size_inches(9, 13)
    # obraz oryginalny
    axs[0, 0].imshow(przed_rgb)  # RGB
    przed_y_cr_cb = cv2.cvtColor(przed_rgb, cv2.COLOR_RGB2YCrCb)
    axs[1, 0].imshow(przed_y_cr_cb[:, :, 0], cmap=plt.cm.gray)
    axs[2, 0].imshow(przed_y_cr_cb[:, :, 1], cmap=plt.cm.gray)
    axs[3, 0].imshow(przed_y_cr_cb[:, :, 2], cmap=plt.cm.gray)

    # obraz po dekompresji
    axs[0, 1].imshow(po_rgb)  # RGB
    po_y_cr_cb = cv2.cvtColor(po_rgb, cv2.COLOR_RGB2YCrCb)
    axs[1, 1].imshow(po_y_cr_cb[:, :, 0], cmap=plt.cm.gray)
    axs[2, 1].imshow(po_y_cr_cb[:, :, 1], cmap=plt.cm.gray)
    axs[3, 1].imshow(po_y_cr_cb[:, :, 2], cmap=plt.cm.gray)
    plt.show()

if __name__ == '__main__':
    print("Hello World")
    BGR = cv2.imread("sample-5.png")
    RGB = cv2.cvtColor(BGR, cv2.COLOR_BGR2RGB)
    QY = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 36, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ])
    QC = np.array([
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
    ])

    JPEG = compress_jpeg(RGB, qy=QY, qc=QC)
    RGB2 = decompress_jpeg(JPEG)
    draw_plt(RGB, RGB2)