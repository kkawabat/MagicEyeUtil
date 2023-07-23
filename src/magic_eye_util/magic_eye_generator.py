import numpy as np
from PIL import Image


def generate_magic_eye(depth_map_image, texture_map_image, depth_factor=.1, num_strips=10, strip_width=100):
    texture_arr = resize_texture_img(texture_map_image, strip_width)
    depth_arr = resize_depth_map(depth_map_image, num_strips, strip_width)
    magic_eye_arr = gen_depth_offset_map(texture_arr, depth_arr, num_strips, depth_factor)
    return Image.fromarray(magic_eye_arr, mode="RGB")


def resize_texture_img(img, strip_width):
    w, h = img.size
    img_resized = img.resize((strip_width, int(strip_width * (h / w))))
    arr_resized = np.array(img_resized)
    if len(arr_resized.shape) == 2:
        arr_resized = arr_resized[:, :, None]
    return arr_resized


def resize_depth_map(img, num_strips, strip_width):
    bw_img = img.convert('L')
    w, h = bw_img.size
    bw_img_resized = bw_img.resize((strip_width * num_strips, int(strip_width * num_strips * (h / w))), resample=Image.BOX)
    arr_resized = np.array(bw_img_resized) / 255
    return arr_resized


def gen_depth_offset_map(texture_arr, depth_arr, num_strips, depth_factor):
    th, tw, t_channels = texture_arr.shape[0:3]
    dh, dw = depth_arr.shape[0:2]
    depth_normed = (depth_arr * depth_factor * tw).astype(int)

    tile_range_x = np.tile(range(tw), (dh, 1))
    tile_range_y = np.tile(range(th), (tw, int(dh / th) + 1)).T[:dh, :]

    result_map = np.empty((dh, tw * (num_strips + 1), t_channels))
    result_map[:, :tw, :] = texture_arr[tile_range_y, tile_range_x, :]

    tile_range_x_tmp = np.tile(range(result_map.shape[1]), (dh, 1))
    tmp = np.empty((dh, tw * (num_strips + 1)))

    row_idc = np.arange(result_map.shape[0])

    for i in range(dw):
        result_map[row_idc, i + tw, :] = result_map[row_idc, i + depth_normed[:, i], :].reshape(-1, t_channels)
        tmp[row_idc, i + tw] = tile_range_x_tmp[row_idc, i + depth_normed[:, i]]

    result_map = result_map[:, :, :3].astype(np.uint8)
    return result_map
