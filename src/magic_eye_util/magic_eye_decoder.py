import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import convolve
from scipy.signal import find_peaks


def decode_magic_eye(magic_image, threshold=5, sensitivity=.8):
    bw_magic_image = ImageOps.grayscale(magic_image)
    img_arr = np.asarray(bw_magic_image, dtype='int16')

    candidate_indices, candidate_weights = get_candidate_indices(img_arr)

    result_depth_map_arr = np.zeros(img_arr.shape)
    depth_range = np.linspace(64, 192, len(candidate_indices))

    for i, (shift_idx, shift_diff) in enumerate(get_shift_diff(candidate_indices, img_arr)):
        print(shift_idx)
        shift_diff = np.roll(shift_diff, shift_idx, axis=1)
        shift_diff_smoothed = smooth_image(abs(shift_diff) < threshold)
        activated_depth_pixels = shift_diff_smoothed >= sensitivity
        result_depth_map_arr[activated_depth_pixels] = depth_range[i]
    img = Image.fromarray(result_depth_map_arr)
    return img


def get_candidate_indices(img_arr):
    # this function returns a list of indices that highly correlates alone the x-axis used to determine the tiling intervals of the magic eye image
    shift_zero_count = []
    for shift_idx, shift_diff in get_shift_diff(range(img_arr.shape[1] // 2), img_arr):
        shift_zero_count.append(abs(shift_diff).sum())
    indices, weights = get_high_correlation_indices(np.array(shift_zero_count))
    return indices, weights


def get_high_correlation_indices(x):
    n = x.size
    norm = (x - np.mean(x))
    result = np.correlate(norm, norm, mode='same')
    acorr = result[n // 2 + 1:] / (x.var() * np.arange(n - 1, n // 2, -1))
    peaks, _ = find_peaks(acorr, height=.1)
    if len(peaks) == 0:
        raise Exception("could not decode magic eye image")
    best_peak = sorted(peaks, key=lambda x: acorr[x], reverse=True)[0]
    indices = range(max(int(best_peak - (best_peak / 2)), 0), int(best_peak + (best_peak / 2)))
    indices_weights = acorr[indices]
    return indices, indices_weights


def get_shift_diff(shift_range, img_arr):
    for shift_idx in shift_range:
        shift_diff = img_arr - np.roll(img_arr, -shift_idx, axis=1)
        yield shift_idx, shift_diff


SMOOTHING_WEIGHTS = np.array([[.1, .1, .1],
                              [.1, .2, .1],
                              [.1, .1, .1]], dtype=float)


def smooth_image(img_arr):
    smooth_img_arr = convolve(img_arr.astype(float), SMOOTHING_WEIGHTS, mode='constant')
    return smooth_img_arr


def running_avg(cur_avg, new_frame, i):
    if cur_avg is None:
        return new_frame

    new_avg = cur_avg * (i / (i + 1)) + (new_frame / (i + 1))
    return new_avg
