import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mltrainingtools_dmoranj",
    version="0.0.3",
    author="Daniel Morán Jiménez",
    author_email="dmoranj@gmail.com",
    description="Package containing some utilities for Machine Learning projects",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dmoranj/mltrainingtools",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
)
