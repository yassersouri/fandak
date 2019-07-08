import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="fandak",
    version="0.1a1.dev2",
    author="Yasser Souri",
    author_email="yassersouri@gmail.com",
    description="A Framework for Deep Learning Research in PyTorch.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yassersouri/fandak",
    packages=setuptools.find_packages(),
    install_requires=[
        "torch>=1.0,<2",
        "torchvision>=0.3,<1",
        "tqdm>=4.32,<5"
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
