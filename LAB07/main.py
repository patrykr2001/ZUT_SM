import numpy as np
import cv2
import scipy.fftpack

## wybrać jeden kontener i nie umieszczać w nim dodatkowych funkcji
class Ver1:
    Y=np.array([])
    Cb=np.array([])
    Cr=np.array([])
    ChromaRatio="4:4:4"
    QY=np.ones((8,8))
    QC=np.ones((8,8))
    shape=(0,0,3)

class Ver2:
    def __init__(self,Y,Cb,Cr,OGShape,Ratio="4:4:4",QY=np.ones((8,8)),QC=np.ones((8,8))):
        self.shape = OGShape
        self.Y=Y
        self.Cb=Cb
        self.Cr=Cr
        self.ChromaRatio=Ratio
        self.QY=QY
        self.QC=QC


def to_YCrCb(RGB):
    return cv2.cvtColor(RGB,cv2.COLOR_RGB2YCrCb).astype(int)

def to_RGB(YCrCb):
    return cv2.cvtColor(YCrCb.astype(np.uint8),cv2.COLOR_YCrCb2RGB)

def dct2(a):
    return scipy.fftpack.dct( scipy.fftpack.dct( a.astype(float), axis=0, norm='ortho' ), axis=1, norm='ortho' )

def idct2(a):
    return scipy.fftpack.idct( scipy.fftpack.idct( a.astype(float), axis=0 , norm='ortho'), axis=1 , norm='ortho')

def zigzag(A):
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
    if len(A.shape)==1:
        B=np.zeros((8,8))
        for r in range(0,8):
            for c in range(0,8):
                B[r,c]=A[template[r,c]]
    else:
        B=np.zeros((64,))
        for r in range(0,8):
            for c in range(0,8):
                B[template[r,c]]=A[r,c]
    return B

def compress_JPEG(RGB,Ratio="4:4:4",QY=np.ones((8,8)),QC=np.ones((8,8))):
    YCrCb=to_YCrCb(RGB)
    JPEG=Ver2(YCrCb[:,:,0],YCrCb[:,:,1],YCrCb[:,:,2],RGB.shape,Ratio,QY,QC)
    # Tu chroma subsampling
    JPEG.Y=compress_layer(JPEG.Y,JPEG.QY)
    JPEG.Cr=compress_layer(JPEG.Cr,JPEG.QC)
    JPEG.Cb=compress_layer(JPEG.Cb,JPEG.QC)
    # tu dochodzi kompresja bezstratna
    return JPEG

def decompress_JPEG(JPEG):
    # dekompresja bezstratna
    Y=decompress_layer(JPEG.Y,JPEG.QY)
    Cr=decompress_layer(JPEG.Cr,JPEG.QC)
    Cb=decompress_layer(JPEG.Cb,JPEG.QC)
    # Tu chroma resampling
    YCrCb=np.dstack([Y,Cr,Cb]).clip(0,255).astype(np.uint8)
    RGB=to_RGB(YCrCb)
    return RGB

def compress_block(block,Q):
    block = dct2(block)
    # tu kwantyzacja
    vector=zigzag(block)
    return vector

def decompress_block(vector,Q):
    vector = zigzag(vector)
    # tu dekwantyzacja
    block = idct2(vector)
    return block

## podział na bloki
# L - warstwa kompresowana
# S - wektor wyjściowy
def compress_layer(L,Q):
    S=np.array([])
    for w in range(0,L.shape[0],8):
        for k in range(0,L.shape[1],8):
            block=L[w:(w+8),k:(k+8)]
            S=np.append(S, compress_block(block,Q))
    return S

## wyodrębnianie bloków z wektora
# L - warstwa o oczekiwanym rozmiarze
# S - długi wektor zawierający skompresowane dane
def decompress_layer(S,Q):
    L= np.zeros((S.shape[0]*8,S.shape[0]*8))
    for idx,i in enumerate(range(0,S.shape[0],64)):
        vector=S[i:(i+64)]
        m=L.shape[1]/8
        k=int((idx%m)*8)
        w=int((idx//m)*8)
        L[w:(w+8),k:(k+8)]=decompress_block(vector,Q)
    return L

if __name__ == '__main__':
    print("Hello World")



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
