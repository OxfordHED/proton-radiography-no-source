import os
from setuptools import setup, find_packages

setup(
    name='prns',
    version="1.0",
    description='Retrieving fields from proton radiography without source profiles',
    url='https://github.com/OxfordHED/proton-radiography-no-source/',
    author='mfkasim91',
    author_email='firman.kasim@gmail.com',
    license='MIT',
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=[
        "numpy>=1.8.2",
        "scipy>=0.15",
        "matplotlib>=1.5.3",
        "sunbear>=0.1",
        "scikit-learn>=0.21",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",

        "Programming Language :: Python :: 3.6"
    ],
    keywords="project library deep-learning",
    zip_safe=False
)
