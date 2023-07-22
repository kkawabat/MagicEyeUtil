from setuptools import setup

setup(
    name='magic_eye_util',
    version='1.0.0',
    packages=['magic-eye-util'],
    package_dir={'magic-eye-util': 'src'},
    url='',
    license='',
    author='kkawabat',
    author_email='kkawabat@asu.edu',
    description='Utility functions used to generate magic eye images',
    install_requires=['numpy', 'pillow']
)
