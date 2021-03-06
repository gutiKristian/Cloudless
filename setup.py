from setuptools import setup, find_namespace_packages

with open(file="README.md", mode="r") as readme:
    long_description = readme.read()

setup(
    name='sentinel-pipeline',
    #   - MAJOR VERSION 0
    #   - MINOR VERSION 1
    #   - MAINTENANCE VERSION 0
    version='0.1.0',
    description='Pipeline for creating cloudless pictures from Sentinel-2 images',
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "colorama==0.4.4",
        "cycler==0.10.0",
        "kiwisolver==1.3.1",
        "llvmlite==0.35.0",
        "matplotlib==3.3.3",
        "numba==0.52.0",
        "numpy==1.19.4",
        "Pillow==8.0.1",
        "pyparsing==2.4.7",
        "python-dateutil==2.8.1",
        "Shapely==1.7.1",
        "six==1.15.0"
    ],
    packages=['Pipeline'],
    include_package_data=True,
    python_requires='>=3.7',
)
