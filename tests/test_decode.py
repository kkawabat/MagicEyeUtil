import os

from PIL import Image

from src.magic_eye_decoder import decode_magic_eye


def test_decode():
    sample_magic_eye_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'squirrel_magic_eye.png')
    sample_magic_eye_img = Image.open(sample_magic_eye_path)
    result_depth_map_ = decode_magic_eye(sample_magic_eye_img)
    result_depth_map_.show()


if __name__ == '__main__':
    test_decode()

