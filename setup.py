from setuptools import setup, find_packages

setup(
    name="pleas-merging",
    version="0.1.0",
    author="anshuln2",
    author_email="anasery@uw.edu",
    description="PLeaS - Merging Models with Permutations and Least Squares",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/SewoongLab/PLeaS-Merging",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "tqdm>=4.62.0",
        "wandb>=0.12.0",
        "torchmetrics>=0.7.0",
        "scikit-learn>=1.0.0",
        "gurobipy>=9.5.0",  # Optional for QP optimization
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "flake8>=3.9.0",
            "black>=21.5b2",
            "isort>=5.9.0",
        ],
    },
)