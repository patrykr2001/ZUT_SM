import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
import scipy.fftpack

soundPath = './sounds/'

def zad_1():
    data, fs = sf.read(soundPath + 'sound1.wav', dtype='float32')
    print(data.dtype)
    print(data.shape)

    # sd.play(data, fs)
    # status = sd.wait()

    sf.write('sound_L.wav', data[:,0], fs)
    sf.write('sound_R.wav', data[:,1], fs)
    mixData = (data[:,0] + data[:,1])/2.0
    sf.write('sound_mix.wav', mixData, fs)

    x = np.arange(0, data.shape[0])
    x = x/fs

    plt.subplot(3, 1, 1)
    plt.title('L')
    plt.xlabel('Time (s)')
    plt.plot(x, data[:, 0])

    plt.subplot(3, 1, 2)
    plt.title('P')
    plt.xlabel('Time (s)')
    plt.plot(x, data[:, 1])

    plt.subplot(3, 1, 3)
    plt.title('MIX')
    plt.xlabel('Time (s)')
    plt.plot(x, mixData)

    plt.savefig('wykres.png')

    fsize = 2 ** 8
    data, fs = sf.read(soundPath + 'sin_440Hz.wav', dtype=np.int32)

    plt.figure(dpi=300)
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(0, data.shape[0]) / fs, data, linewidth=0.5)

    plt.subplot(2, 1, 2)
    yf = scipy.fftpack.fft(data, fsize)
    plt.plot(np.arange(0, fs/2, fs/fsize), 20*np.log10(np.abs(yf[:fsize//2])), linewidth=0.5)
    plt.savefig('widmo.png')
