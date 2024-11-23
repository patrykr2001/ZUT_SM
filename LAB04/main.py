import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy.io import wavfile

def change_bit_resolution(data, original_bits, target_bits):
    """
    Change the bit resolution of the given data.

    :param data: Input data as a numpy array.
    :param original_bits: Original bit resolution of the data.
    :param target_bits: Target bit resolution.
    :return: Data with changed bit resolution.
    """
    # Calculate the maximum value for the original and target bit resolutions
    original_max = (1 << original_bits) - 1
    target_max = (1 << target_bits) - 1

    # Normalize the data to the range [0, 1]
    normalized_data = data / original_max

    # Scale the normalized data to the target bit resolution
    scaled_data = np.round(normalized_data * target_max)

    return scaled_data.astype(np.int32)


def decimate_data(data, interval):
    """
    Decimate the data by a given interval.

    :param data: Input data as a numpy array.
    :param interval: Interval for decimation.
    :return: Decimated data.
    """
    return data[::interval]


def interpolate_data(data, original_rate, new_rate, method='linear'):
    """
    Interpolate the data from the original sampling rate to a new sampling rate.

    :param data: Input data as a numpy array.
    :param original_rate: Original sampling rate of the data.
    :param new_rate: New sampling rate.
    :param method: Interpolation method ('linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic').
    :return: Interpolated data.
    """
    from scipy import interpolate

    # Calculate the time points for the original and new sampling rates
    original_time = np.arange(0, len(data)) / original_rate
    new_time = np.arange(0, len(data) * new_rate / original_rate) / new_rate

    # Create the interpolation function
    interpolator = interpolate.interp1d(original_time, data, kind=method)

    # Interpolate the data
    interpolated_data = interpolator(new_time)

    return interpolated_data


# Funkcja do generowania wykresów
def plot_signal_and_spectrum(data, sample_rate, title):
    time = np.arange(len(data)) / sample_rate
    plt.figure(figsize=(12, 6))

    # Wykres sygnału w czasie
    plt.subplot(2, 1, 1)
    plt.plot(time, data)
    plt.title(f'{title} - Signal in Time Domain')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')

    # Wykres widma w skali dB
    plt.subplot(2, 1, 2)
    spectrum = np.abs(fft(data))[:len(data) // 2]
    spectrum_db = 20 * np.log10(spectrum + np.finfo(np.float32).eps)
    freqs = np.fft.fftfreq(len(data), 1 / sample_rate)[:len(data) // 2]
    plt.plot(freqs, spectrum_db)
    plt.title(f'{title} - Spectrum in dB')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude [dB]')
    plt.tight_layout()
    plt.show()


def zad1():
    # Wczytaj dane z plików
    basePath = './SIN/'
    file_names = [basePath + 'sin_60Hz.wav', basePath + 'sin_440Hz.wav', basePath + 'sin_8000Hz.wav',
                  basePath + 'sin_combined.wav']
    data_list = [wavfile.read(file_name) for file_name in file_names]

    for wav_data in data_list:
        sample_rate = wav_data[0]
        data = wav_data[1]

        # Zmiana rozdzielczości bitowej
        for bits in [4, 8, 16, 24]:
            modified_data = change_bit_resolution(data, 16, bits)
            plot_signal_and_spectrum(modified_data, sample_rate, f'Bit Resolution {bits}')

        # Decymacja
        for interval in [2, 4, 6, 10, 24]:
            decimated_data = decimate_data(data, interval)
            plot_signal_and_spectrum(decimated_data, sample_rate // interval, f'Decimation Interval {interval}')

        # Interpolacja
        for new_rate in [2000, 4000, 8000, 11999, 16000, 16953, 24000, 41000]:
            for method in ['linear', 'cubic']:
                interpolated_data = interpolate_data(data, sample_rate, new_rate, method)
                plot_signal_and_spectrum(interpolated_data, new_rate, f'Interpolation to {new_rate} Hz ({method})')


def zad2():
    # Wczytaj dane z plików
    basePath = './SING/'
    file_names = [basePath + 'sing_low1.wav', basePath + 'sing_medium1.wav', basePath + 'sing_high1.wav']
    data_list = [wavfile.read(file_name) for file_name in file_names]

    for wav_data in data_list:
        sample_rate = wav_data[0]
        data = wav_data[1]

        # Zmiana rozdzielczości bitowej
        for bits in [4, 8]:
            modified_data = change_bit_resolution(data, 16, bits)
            plot_signal_and_spectrum(modified_data, sample_rate, f'Bit Resolution {bits}')
            # Odsłuch
            wavfile.write(f'{basePath}output_bit_{bits}.wav', sample_rate, modified_data)

        # Decymacja
        for interval in [4, 6, 10, 24]:
            decimated_data = decimate_data(data, interval)
            plot_signal_and_spectrum(decimated_data, sample_rate // interval, f'Decimation Interval {interval}')
            # Odsłuch
            wavfile.write(f'{basePath}output_decimation_{interval}.wav', sample_rate // interval, decimated_data)

        # Interpolacja
        for new_rate in [4000, 8000, 11999, 16000, 16953]:
            for method in ['linear', 'cubic']:
                interpolated_data = interpolate_data(data, sample_rate, new_rate, method)
                plot_signal_and_spectrum(interpolated_data, new_rate, f'Interpolation to {new_rate} Hz ({method})')
                # Odsłuch
                wavfile.write(f'{basePath}output_interpolation_{new_rate}_{method}.wav', new_rate, interpolated_data)



# # Example usage
# data = np.array([0, 128, 255], dtype=np.int32)  # Example 8-bit data
# new_data = change_bit_resolution(data, 8, 16)  # Change from 8-bit to 16-bit
# print(new_data)
# new_data = change_bit_resolution(data, 8, 4)  # Change from 8-bit to 16-bit
# print(new_data)
#
# # Example usage of decimation
# decimated_data = decimate_data(data, 2)  # Decimate data by an interval of 2
# print(decimated_data)

# zad1()
zad2()
