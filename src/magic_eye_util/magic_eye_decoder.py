import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import convolve


def decode_magic_eye(magic_image):
    bw_magic_image = ImageOps.grayscale(magic_image)
    img_arr = np.asarray(bw_magic_image)

    depth_width = estimate_depth_width(img_arr)

    result_depth_map_arr = np.zeros(img_arr.shape)
    depth_range = np.linspace(0, 256, depth_width-1)
    for shift_idx, diff_zeros_smoothed in get_depth_zeros(range(depth_width-1), img_arr):
        result_depth_map_arr[diff_zeros_smoothed] = depth_range[shift_idx]

    return Image.fromarray(result_depth_map_arr.astype(np.uint8))


def estimate_depth_width(img_arr):
    shift_zero_count = []

    for shift_idx, diff_zeros_smoothed in get_depth_zeros(range(img_arr.shape[1]), img_arr):
        shift_zero_count.append(sum(sum(diff_zeros_smoothed)))
    return autocorr(np.array(shift_zero_count))


def autocorr(x):
    n = x.size
    norm = (x - np.mean(x))
    result = np.correlate(norm, norm, mode='same')
    acorr = result[n//2 + 1:] / (x.var() * np.arange(n-1, n//2, -1))
    lag = np.abs(acorr).argmax() + 1
    r = acorr[lag-1]
    if np.abs(r) > 0.5:
        return lag
    raise Exception('Could not decode image')


def get_depth_zeros(depth_range, img_arr):
    weights = np.array([[1, 1, 1],
                        [1, 2, 1],
                        [1, 1, 1]], dtype=np.float)
    weights = weights / np.sum(weights[:])

    for shift_idx, depth in enumerate(depth_range):
        print(shift_idx)
        diff_zeros = (img_arr - np.roll(img_arr, -shift_idx, axis=1)) == 0
        diff_zeros_smoothed = convolve(diff_zeros, weights, mode='constant')
        yield shift_idx, diff_zeros_smoothed



