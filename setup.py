import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="BroutonLab Deep Learning Course",
    version="0.0.1",
    long_description=long_description,
    author="BroutonLab team",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    author_email="hello@broutonlab.com",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    install_requires=[
        "pyzmq",
        "protobuf",
        "antlr4-python3-runtime",
        "numpy",
    ],
    python_requires=">=3.6",
)