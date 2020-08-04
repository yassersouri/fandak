import setuptools

__version__ = "0.1.3"

with open("README.md", "r") as f:
    long_description = f.read()


setuptools.setup(
    name="fandak",
    version=__version__,
    author="Yasser Souri",
    author_email="yassersouri@gmail.com",
    description="A Framework for ML Research Using PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yassersouri/fandak",
    packages=setuptools.find_packages(exclude=["docs", "examples", "tests", "scripts"]),
    install_requires=[
        "torch>=1.1",
        "torchvision>=0.3",
        "tqdm>=4.32",
        "matplotlib>=3.0.0",
        "tensorboard>=1.14.0",
        "future>=0.17.1",
        "yacs>=0.1.6",
        "click>=7.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Typing :: Typed",
    ],
)
