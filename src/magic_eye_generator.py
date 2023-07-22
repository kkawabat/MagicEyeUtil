import numpy as np
from PIL import Image


def magic_eye_generate(depth_map_image, texture_map_image, depth_factor=.1, num_strips=10, strip_width=100):
    texture_map_arr = resize_texture_img(texture_map_image, strip_width)
    depth_map_arr = resize_depth_map(depth_map_image, num_strips, strip_width)
    result_map = gen_depth_offset_map(texture_map_arr, depth_map_arr, num_strips, depth_factor)
    return Image.fromarray(result_map, mode="RGB")


def resize_texture_img(texture_map_im, strip_width):
    w, h = texture_map_im.size
    r = h / w
    texture_map_resized = texture_map_im.resize((strip_width, int(strip_width * r)))
    texture_map_data = np.array(texture_map_resized)
    return texture_map_data


def resize_depth_map(depth_map_im, num_strips, strip_width):
    bw_depth_map_image = depth_map_im.convert('L')
    w, h = bw_depth_map_image.size
    r = h / w
    depth_map_resized = bw_depth_map_image.resize((strip_width * num_strips, int(strip_width * num_strips * r)), resample=Image.BOX)
    depth_map_data = np.array(depth_map_resized) / 255
    return depth_map_data


def gen_depth_offset_map(texture_map_data, depth_map_data, num_strips, depth_factor):
    if len(texture_map_data.shape) == 2:
        num_texture_channels = 1
    elif len(texture_map_data.shape) == 3:
        num_texture_channels = texture_map_data.shape[2]
    else:
        raise ValueError("expected at most 3 dimension for texture_map (width, height, #channel)")

    th, tw = texture_map_data.shape[0:2]
    dh, dw = depth_map_data.shape[0:2]
    depth_normed = (depth_map_data * depth_factor * tw).astype(int)

    tile_range_x = np.tile(range(tw), (dh, 1))
    tile_range_y = np.tile(range(th), (tw, int(dh / th) + 1)).T[:dh, :]

    result_map = np.empty((dh, tw * (num_strips + 1), num_texture_channels))
    tile_range_x_tmp = np.tile(range(result_map.shape[1]), (dh, 1))
    tmp = np.empty((dh, tw * (num_strips + 1)))
    result_map[:, :tw, :] = texture_map_data[tile_range_y, tile_range_x, :]

    row_idc = np.arange(result_map.shape[0])

    for i in range(dw):
        result_map[row_idc, i + tw, :] = result_map[row_idc, i + depth_normed[:, i], :].reshape(-1, num_texture_channels)
        tmp[row_idc, i + tw] = tile_range_x_tmp[row_idc, i + depth_normed[:, i]]

    result_map = result_map[:, :, :3].astype(np.uint8)
    return result_map
