import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile


def quantize_signal(signal, num_bits=8):
    """Quantize the signal to the specified number of bits."""
    # Calculate the number of quantization levels
    num_levels = 2 ** num_bits

    # Scale the signal to the range [0, num_levels - 1]
    scaled_signal = (signal + 1) * (num_levels / 2)

    # Quantize the signal
    quantized_signal = np.round(scaled_signal)

    # Scale the quantized signal back to the range [-1, 1]
    quantized_signal = (quantized_signal / (num_levels / 2)) - 1

    return quantized_signal


def a_law_encode(signal, A=87.6):
    """A-law encoding."""
    abs_signal = np.abs(signal)
    encoded_signal = np.where(abs_signal < 1/A, A * abs_signal / (1 + np.log(A)), (1 + np.log(A * abs_signal)) / (1 + np.log(A)))
    encoded_signal = np.sign(signal) * encoded_signal
    return encoded_signal

def a_law_decode(encoded_signal, A=87.6):
    """A-law decoding."""
    abs_encoded_signal = np.abs(encoded_signal)
    decoded_signal = np.where(abs_encoded_signal < 1 / (1 + np.log(A)), abs_encoded_signal * (1 + np.log(A)) / A,
                              np.exp(abs_encoded_signal * (1 + np.log(A)) - 1) / A)
    decoded_signal = np.sign(encoded_signal) * decoded_signal
    return decoded_signal

def u_law_encode(signal, mu=255.0):
    """u-law encoding."""
    abs_signal = np.abs(signal)
    encoded_signal = np.sign(signal) * np.log(1 + mu * abs_signal) / np.log(1 + mu)
    return encoded_signal

def u_law_decode(encoded_signal, mu=255.0):
    """u-law decoding."""
    abs_encoded_signal = np.abs(encoded_signal)
    decoded_signal = np.sign(encoded_signal) * (1 / mu) * ((1 + mu) ** abs_encoded_signal - 1)
    return decoded_signal

# def normalize_signal(signal):
#     """Normalize the signal to the range [-1, 1]."""
#     return 2 * (signal - np.min(signal)) / (np.max(signal) - np.min(signal)) - 1

def dpcm_encode(signal, bit=8):
    """DPCM encoding without prediction."""
    encoded_signal = np.zeros(signal.shape)
    e = 0
    for i in range(0, signal.shape[0]):
        encoded_signal[i] = quantize_signal(signal[i] - e, bit)
        e += encoded_signal[i]
    return encoded_signal

def dpcm_decode(encoded_signal):
    """DPCM decoding without prediction."""
    decoded_signal = np.zeros(encoded_signal.shape)
    e = 0
    for i in range(0, encoded_signal.shape[0]):
        decoded_signal[i] = encoded_signal[i] + e
        e = decoded_signal[i]
    return decoded_signal

def dpcm_encode_with_prediction(signal, bit=8, predictor=np.mean, n=1):
    """DPCM encoding with linear prediction (1 element)."""
    encoded_signal = np.zeros(signal.shape)
    xp = np.zeros(signal.shape)
    e = 0
    for i in range(1, signal.shape[0]):
        encoded_signal[i] = quantize_signal(signal[i] - e, bit)
        xp[i] = encoded_signal[i] + e
        idx = (np.arange(i - n, i, 1, dtype=int) + 1)
        idx = np.delete(idx, idx < 0)
        e = predictor(xp[idx])
    return encoded_signal

def dpcm_decode_with_prediction(encoded_signal, predictor=np.mean, n=1):
    """DPCM decoding with linear prediction (1 element)."""
    decoded_signal = np.zeros(encoded_signal.shape)
    xp = np.zeros(encoded_signal.shape)
    decoded_signal[0] = encoded_signal[0]  # Set the first value to the first value of the encoded signal
    e = decoded_signal[0]
    for i in range(1, encoded_signal.shape[0]):
        decoded_signal[i] = encoded_signal[i] + e
        xp[i] = decoded_signal[i]
        idx = (np.arange(i - n, i, 1, dtype=int) + 1)
        idx = np.delete(idx, idx < 0)
        e = predictor(xp[idx])
    return decoded_signal

def test_compression():
    x = np.linspace(-1, 1, 1000)

    # A-law encoding
    a_law_encoded = a_law_encode(x)
    a_law_quantized = quantize_signal(a_law_encoded)

    # u-law encoding
    u_law_encoded = u_law_encode(x)
    u_law_quantized = quantize_signal(u_law_encoded)
    # A-law decoding
    a_law_decoded = a_law_decode(a_law_quantized)

    # u-law decoding
    u_law_decoded = u_law_decode(u_law_quantized)

    # Quantized original signal
    original_quantized = quantize_signal(x)

    # DPCM encoding and decoding without prediction
    dpcm_encoded = dpcm_encode(x)
    dpcm_decoded = dpcm_decode(dpcm_encoded)

    # DPCM encoding and decoding with linear prediction
    dpcm_encoded_pred = dpcm_encode_with_prediction(x)
    dpcm_decoded_pred = dpcm_decode_with_prediction(dpcm_encoded_pred)

    # Plot compression curves
    plt.figure(figsize=(10, 4))
    plt.plot(x, a_law_encoded, label='A-law Encoded')
    plt.plot(x, a_law_quantized, label='A-law Quantized', linestyle='--')
    plt.plot(x, u_law_encoded, label='u-law Encoded')
    plt.plot(x, u_law_quantized, label='u-law Quantized', linestyle='--')
    plt.title('Compression Curves')
    plt.xlabel('Input Signal')
    plt.ylabel('Encoded Signal')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot decompression curves
    plt.figure(figsize=(10, 4))
    plt.plot(x, x, label='Original Signal')
    plt.plot(x, a_law_decoded, label='A-law Decoded (Quantized)')
    plt.plot(x, u_law_decoded, label='u-law Decoded (Quantized)')
    plt.plot(x, original_quantized, label='Original Quantized', linestyle='--')
    plt.plot(x, dpcm_decoded, label='DPCM Decoded (No Prediction)')
    plt.plot(x, dpcm_decoded_pred, label='DPCM Decoded (With Prediction)')
    plt.title('Decompression Curves')
    plt.xlabel('Input Signal')
    plt.ylabel('Output Signal')
    plt.legend()
    plt.grid(True)
    plt.show()

def test_compression2():
    x = np.linspace(-1, 1, 1000)
    y = 0.9 * np.sin(np.pi * x * 4)

    # A-law encoding
    a_law_encoded = a_law_encode(y)
    a_law_quantized = quantize_signal(a_law_encoded, 4)

    # u-law encoding
    u_law_encoded = u_law_encode(y)
    u_law_quantized = quantize_signal(u_law_encoded, 4)
    # A-law decoding
    a_law_decoded = a_law_decode(a_law_quantized)

    # u-law decoding
    u_law_decoded = u_law_decode(u_law_quantized)

    # Quantized original signal
    original_quantized = quantize_signal(y)

    # DPCM encoding and decoding without prediction
    dpcm_encoded = dpcm_encode(y, 4)
    dpcm_decoded = dpcm_decode(dpcm_encoded)

    # DPCM encoding and decoding with linear prediction
    dpcm_encoded_pred = dpcm_encode_with_prediction(y, 4, np.mean, 4)
    dpcm_decoded_pred = dpcm_decode_with_prediction(dpcm_encoded_pred)

    plt.figure(figsize=(10, 20))

    plt.subplot(5, 1, 1)
    plt.plot(x, y)
    plt.title('Original Signal')
    plt.grid(True)

    plt.subplot(5, 1, 2)
    plt.plot(x, a_law_decoded)
    plt.title('A-law Compression')
    plt.grid(True)

    plt.subplot(5, 1, 3)
    plt.plot(x, u_law_decoded)
    plt.title('u-law Compression')
    plt.grid(True)

    plt.subplot(5, 1, 4)
    plt.plot(x, dpcm_decoded)
    plt.title('DPCM Compression (No Prediction)')
    plt.grid(True)

    plt.subplot(5, 1, 5)
    plt.plot(x, dpcm_decoded_pred)
    plt.title('DPCM Compression (With Prediction)')
    plt.grid(True)

    plt.tight_layout()
    plt.show()




def load_audio(file_path):
    rate, data = wavfile.read(file_path)
    return rate, data.astype(np.float32) / np.max(np.abs(data))


def save_audio(file_path, rate, data):
    wavfile.write(file_path, rate, (data * 32767).astype(np.int16))


def evaluate_compression(file_path, num_bits):
    rate, signal = load_audio(file_path)

    # A-law encoding and decoding
    a_law_encoded = a_law_encode(signal)
    a_law_quantized = quantize_signal(a_law_encoded, num_bits)
    a_law_decoded = a_law_decode(a_law_quantized)

    # u-law encoding and decoding
    u_law_encoded = u_law_encode(signal)
    u_law_quantized = quantize_signal(u_law_encoded, num_bits)
    u_law_decoded = u_law_decode(u_law_quantized)

    # DPCM encoding and decoding
    dpcm_encoded = dpcm_encode(signal, num_bits)
    dpcm_decoded = dpcm_decode(dpcm_encoded)

    # Save results
    save_audio(f'a_law_decoded_{num_bits}bit.wav', rate, a_law_decoded)
    save_audio(f'u_law_decoded_{num_bits}bit.wav', rate, u_law_decoded)
    save_audio(f'dpcm_decoded_{num_bits}bit.wav', rate, dpcm_decoded)

if __name__ == '__main__':
    # test_compression()
    # test_compression2()
    file_paths = [
        # './SING/sing_high1.wav',
        # './SING/sing_medium1.wav',
        './SING/sing_low1.wav'
    ]
    for file_path in file_paths:
        for num_bits in range(8, 1, -1):
            evaluate_compression(file_path, num_bits)
