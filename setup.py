from setuptools import setup, find_packages


REQUIRED = [
    'numpy>=1.21',
    'trimesh[easy]>=3.16.0',
    'scipy',
    'vedo',
    'PyQt5',
    'image_grid',
    'loguru',
    # 'opencv-python-headless'
]

setup(
    name='balanna',
    version='2.1.0',
    description='Visualization Tools for 2D & 3D stuff working out-of-the-box',
    author='Simon Schaefer & Dorian Henning',
    author_email='simon.k.schaefer@gmail.com',
    packages=find_packages(),
    install_requires=REQUIRED,
    python_requires='>=3.8'
)
