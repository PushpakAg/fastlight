#!/usr/bin/env python3
"""
Setup script for FASTLight - A lightweight action tokenizer for robotics
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="fastlight",
    version="1.0.0",
    author="FASTLight Team",
    author_email="fastlight@example.com",
    description="A lightweight action tokenizer for robotics with Huffman encoding",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/PushpakAg/fastlight",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "flake8",
            "mypy",
        ],
        "examples": [
            "tensorflow-datasets",
            "transformers",
            "matplotlib",
            "scipy",
        ],
    },
    entry_points={
        "console_scripts": [
            "fastlight-demo=fastlight.examples.demo:main",
            "fastlight-compare=fastlight.examples.fast_comparison:main",
        ],
    },
    include_package_data=True,
    package_data={
        "fastlight": ["*.png", "*.jpg", "*.jpeg"],
    },
    keywords="robotics, tokenization, compression, huffman, dct, actions",
    project_urls={
        "Bug Reports": "https://github.com/PushpakAg/fastlight/issues",
        "Source": "https://github.com/PushpakAg/fastlight",
        "Documentation": "https://fastlight.readthedocs.io/",
    },
)
