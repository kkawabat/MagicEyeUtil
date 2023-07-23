import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import convolve
from scipy.signal import find_peaks, peak_widths


def decode_magic_eye(magic_image):
    bw_magic_image = ImageOps.grayscale(magic_image)
    img_arr = np.asarray(bw_magic_image, dtype='int16')

    candidate_indices = get_candidate_indices(img_arr)

    result_depth_map_arr = np.zeros(img_arr.shape)
    depth_range = np.linspace(128, 256, len(candidate_indices))

    weights = np.array([[1, 1, 1],
                        [1, 2, 1],
                        [1, 1, 1]], dtype=float)
    weights = weights / np.sum(weights[:])

    for i, (shift_idx, shift_diff) in enumerate(get_shift_diff(candidate_indices, img_arr)):
        shift_diff = np.roll(shift_diff, shift_idx, axis=1)
        diff_zeros_smoothed = convolve(abs(shift_diff) < 5, weights, mode='constant')
        result_depth_map_arr[diff_zeros_smoothed] = depth_range[i]
    img = Image.fromarray(result_depth_map_arr)
    return img


def get_candidate_indices(img_arr):
    shift_zero_count = []
    for shift_idx, shift_diff in get_shift_diff(range(img_arr.shape[1]//2), img_arr):
        shift_zero_count.append(abs(shift_diff).sum())
    indicies = get_high_correlation_indicies(np.array(shift_zero_count))

    return indicies


def get_high_correlation_indicies(x):
    n = x.size
    norm = (x - np.mean(x))
    result = np.correlate(norm, norm, mode='same')
    acorr = result[n//2 + 1:] / (x.var() * np.arange(n-1, n//2, -1))
    peaks, _ = find_peaks(acorr, height=.1)
    if len(peaks) == 0:
        raise Exception("could not decode magic eye image")
    best_peak = sorted(peaks, key=lambda x: acorr[x], reverse=True)[0]
    return range(max(int(best_peak-(best_peak/2)), 0), int(best_peak+(best_peak/2)))


def get_shift_diff(shift_range, img_arr):
    for shift_idx in shift_range:
        print(shift_idx)
        shift_diff = img_arr - np.roll(img_arr, -shift_idx, axis=1)
        yield shift_idx, shift_diff
