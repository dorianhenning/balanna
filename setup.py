from setuptools import setup, find_packages


REQUIRED = [
    'numpy>=1.21',
    'trimesh[easy]>=3.5.0',
    'vedo',
    'PyQt5',
    # 'opencv-python-headless'
]

setup(
    name='balanna',
    version='0.1',
    description='Visualization Tools for Humans and Stuff',
    author='Simon Schaefer & Dorian Henning',
    author_email='dorian.henning@gmail.com',
    packages=find_packages(),
    install_requires=REQUIRED,
    python_requires='>=3.7'
)
