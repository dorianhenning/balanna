from setuptools import setup, find_packages


REQUIRED = [
    'numpy>=1.21'
    'trimesh[easy]>=3.5.0'
    'pyglet>=1.5',
    'glooey==0.3.6'
]

setup(
    name='balanna',
    version='0.1',
    description='Visualization Tools for Humans and Stuff',
    author='Simon Schaefer & Dorian Henning',
    author_email='dorian.henning@gmail.com',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    include_package_data=False,
    zip_safe=False,
    install_requires=REQUIRED,
    python_requires='>=3.7'
)

