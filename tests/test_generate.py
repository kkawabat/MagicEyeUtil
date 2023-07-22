import cProfile
import os
import pstats

from PIL import Image

from src.magic_eye_generator import magic_eye_generate


def test_generate():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    texture_map_path_ = os.path.join(data_dir, 'squirrel.png')
    texture_map_image = Image.open(texture_map_path_)

    depth_map_path_ = os.path.join(data_dir, 'flower.png')
    depth_map_image = Image.open(depth_map_path_)

    depth_factor_ = .05
    num_strips_ = 10

    result_map_ = magic_eye_generate(texture_map_image, depth_map_image, depth_factor_, num_strips_)
    result_map_.show()


def profile_sis():
    profile = cProfile.Profile()
    profile.runcall(test_generate)
    ps = pstats.Stats(profile)
    ps.sort_stats('cumtime')
    ps.print_stats()


if __name__ == '__main__':
    test_generate()
    # profile_sis()
