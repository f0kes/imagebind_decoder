import os
from distutils.core import setup
from subprocess import getoutput

import setuptools


def get_version_tag() -> str:
    try:
        version = os.environ["CLIP_TEXT_DECODER_VERSION"]
    except KeyError:
        version = "1.0.0"

    return version


setup(
    name="clip-text-decoder",
    version=get_version_tag(),
    author="Frank Odom",
    author_email="frank.odom.iii@gmail.com",
    url="https://github.com/fkodom/clip-text-decoder",
    packages=setuptools.find_packages(exclude=["tests"]),
    description="Generate text captions for images from their CLIP embeddings.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    install_requires=[
        "evaluate",
        "gdown",
        "numpy==1.26.4",
        "pytorch-lightning==2.3.0",
        "spacy",
        "torch==2.0.0+cu117",
        "torchdata==0.6.0",
        "torchvision==0.15.0",
        "transformers==4.26.1",
        "wget",
    ],
    dependency_links=[
        "https://github.com/omegalabsinc/ImageBind/tarball/e2bfdec716b1c1d511d6bea806227b2b3dfcadee#egg=imagebind-0.0.1",
    ],
    extras_require={
        "test": [
            "black",
            "flake8",
            "isort",
            "pre-commit",
            "pytest",
            "pytest-cov",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
