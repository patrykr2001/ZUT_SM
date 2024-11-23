import numpy as np
from numpy import dtype
from tqdm import tqdm

def rle_encode(arr):
    # Flatten the array and get its shape
    flat_arr = arr.flatten()
    shape = arr.shape

    # Initialize the encoded data list
    encoded_data = []

    # Perform RLE encoding with progress bar
    prev_value = flat_arr[0]
    count = 1
    for value in tqdm(flat_arr[1:], desc="Encoding"):
        if value == prev_value:
            count += 1
        else:
            encoded_data.append(prev_value)
            encoded_data.append(count)
            prev_value = value
            count = 1
    encoded_data.append(prev_value)
    encoded_data.append(count)

    # Encode the shape at the beginning
    shape_encoded = np.array([len(shape)] + list(shape))
    encoded_data = np.concatenate([shape_encoded, encoded_data])

    return encoded_data


def rle_decode(encoded_data):
    # Extract the shape information
    shape_length = int(encoded_data[0])
    shape = tuple(encoded_data[1:1 + shape_length].astype(int))
    encoded_data = encoded_data[1 + shape_length:]

    # Initialize the decoded array
    decoded_arr = []

    # Perform RLE decoding with progress bar
    for i in tqdm(range(0, len(encoded_data), 2), desc="Decoding"):
        value = encoded_data[i]
        count = encoded_data[i + 1]
        decoded_arr.extend([value] * count)

    # Convert the list back to a NumPy array and reshape it
    decoded_arr = np.array(decoded_arr).reshape(shape)

    return decoded_arr


def byterun_encode(arr):
    # Flatten the array and get its shape
    flat_arr = arr.flatten()
    shape = arr.shape

    # Initialize the encoded data list
    encoded_data = []

    # Perform byte run-length encoding with progress bar
    prev_value = flat_arr[0]
    count = 1
    for value in tqdm(flat_arr[1:], desc="ByteRun Encoding"):
        if value == prev_value and count < 255:
            count += 1
        else:
            encoded_data.append(prev_value)
            encoded_data.append(count)
            prev_value = value
            count = 1
    encoded_data.append(prev_value)
    encoded_data.append(count)

    # Encode the shape at the beginning
    shape_encoded = np.array([len(shape)] + list(shape))
    encoded_data = np.concatenate([shape_encoded, encoded_data])

    return encoded_data


def byterun_decode(encoded_data):
    # Extract the shape information
    shape_length = int(encoded_data[0])
    shape = tuple(encoded_data[1:1 + shape_length].astype(int))
    encoded_data = encoded_data[1 + shape_length:]

    # Initialize the decoded array
    decoded_arr = []

    # Perform byte run-length decoding with progress bar
    for i in tqdm(range(0, len(encoded_data), 2), desc="ByteRun Decoding"):
        value = encoded_data[i]
        count = encoded_data[i + 1]
        decoded_arr.extend([value] * count)

    # Convert the list back to a NumPy array and reshape it
    decoded_arr = np.array(decoded_arr).reshape(shape)

    return decoded_arr


def calculate_compression_metrics(original, encoded):
    original_size = original.nbytes
    encoded_size = encoded.nbytes
    cr = original_size / encoded_size
    # pr = (1 - (encoded_size / original_size)) * 100
    pr = (encoded_size / original_size) * 100
    return cr, pr


tests = (
        np.array([1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1]),
        np.array([1, 2, 3, 1, 2, 3, 1, 2, 3]),
        np.array([5, 1, 5, 1, 5, 5, 1, 1, 5, 5, 1, 1, 5]),
        np.array([-1, -1, -1, -5, -5, -3, -4, -2, 1, 2, 2, 1]),
        np.zeros((1, 520), dtype=np.int64),
        np.arange(0, 521, 1),
        np.eye(7, dtype=np.int64),
        np.dstack([np.eye(7, dtype=np.int64), np.eye(7, dtype=np.int64), np.eye(7, dtype=np.int64)]),
        np.ones((1, 1, 1, 1, 1, 1, 10), dtype=np.int64)
    )

def przypadki_testowe_rle():
    for test in tests:
        encoded_data = rle_encode(test)
        decoded_arr = rle_decode(encoded_data)

        # print("Original array:")
        # print(test)
        print("Encoded data:")
        print(encoded_data)
        # print("Decoded array:")
        # print(decoded_arr)
        print("Arrays are equal:", np.array_equal(test, decoded_arr))
        print()


def przypadki_testowe_byterun():
    for test in tests:
        encoded_data = byterun_encode(test)
        decoded_arr = byterun_decode(encoded_data)

        # print("Original array:")
        # print(test)
        print("Encoded data:")
        print(encoded_data)
        # print("Decoded array:")
        # print(decoded_arr)
        print("Arrays are equal:", np.array_equal(test, decoded_arr))
        print()


def test_compression_methods():
    for test in tests:
        print("Original array:")
        print(test)

        # RLE
        encoded_rle = rle_encode(test)
        decoded_rle = rle_decode(encoded_rle)
        cr_rle, pr_rle = calculate_compression_metrics(test, encoded_rle)
        print("RLE Encoded data:")
        print(encoded_rle)
        print("RLE Decoded array:")
        print(decoded_rle)
        print("RLE Arrays are equal:", np.array_equal(test, decoded_rle))
        print(f"RLE Compression Ratio (CR): {cr_rle:.2f}")
        print(f"RLE Percentage Reduction (PR): {pr_rle:.2f}%")
        print()

        # ByteRun
        encoded_byterun = byterun_encode(test)
        decoded_byterun = byterun_decode(encoded_byterun)
        cr_byterun, pr_byterun = calculate_compression_metrics(test, encoded_byterun)
        print("ByteRun Encoded data:")
        print(encoded_byterun)
        print("ByteRun Decoded array:")
        print(decoded_byterun)
        print("ByteRun Arrays are equal:", np.array_equal(test, decoded_byterun))
        print(f"ByteRun Compression Ratio (CR): {cr_byterun:.2f}")
        print(f"ByteRun Percentage Reduction (PR): {pr_byterun:.2f}%")
        print()


test_images = (
    './images/dokument.png',
    './images/kolorowy.jpeg',
    './images/techniczny.png',
)


def test_compression_methods_with_test_images():
    for image in test_images:
        print(f"Testing image: {image}")

        # Load the image
        import PIL.Image as Image
        img = np.array(Image.open(image), dtype=np.int64)

        # RLE
        encoded_rle = rle_encode(img)
        decoded_rle = rle_decode(encoded_rle)
        cr_rle, pr_rle = calculate_compression_metrics(img, encoded_rle)
        print(f"RLE Compression Ratio (CR): {cr_rle:.2f}")
        print(f"RLE Percentage Reduction (PR): {pr_rle:.2f}%")
        print("RLE Arrays are equal:", np.array_equal(img, decoded_rle))
        print()

        # ByteRun
        encoded_byterun = byterun_encode(img)
        decoded_byterun = byterun_decode(encoded_byterun)
        cr_byterun, pr_byterun = calculate_compression_metrics(img, encoded_byterun)
        print(f"ByteRun Compression Ratio (CR): {cr_byterun:.2f}")
        print(f"ByteRun Percentage Reduction (PR): {pr_byterun:.2f}%")
        print("ByteRun Arrays are equal:", np.array_equal(img, decoded_byterun))
        print()


# print('RLE')
# przypadki_testowe_rle()
# print('ByteRun')
# przypadki_testowe_byterun()

print('Testy z metrykami')
# test_compression_methods()
test_compression_methods_with_test_images()