from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="digital-image-forgery-detection",
    version="1.0.0",
    author="Mariam Abd Al Aal",
    author_email="abdmariam900@gmail.com",
    description="Digital image forgery detection using deep learning techniques",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Mariam6600/Digital-image-forgery-detection",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Graphics",
    ],
    python_requires=">=3.8",
    install_requires=[
        "tensorflow>=2.13.0",
        "torch>=2.0.1",
        "torchvision>=0.15.2",
        "scikit-learn>=1.3.0",
        "numpy>=1.24.3",
        "opencv-python>=4.8.0.74",
        "Pillow>=10.0.0",
        "pandas>=2.0.3",
        "matplotlib>=3.7.2",
        "seaborn>=0.12.2",
        "tqdm>=4.66.1",
    ],
    extras_require={
        "dev": [
            "jupyter>=1.0.0",
            "jupyter-lab>=3.6.5",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
            "pylint>=2.17.5",
        ],
        "gpu": [
            "cuda-toolkit>=11.8",
        ],
    },
    keywords="forgery detection deep learning image forensics",
    project_urls={
        "Bug Reports": "https://github.com/Mariam6600/Digital-image-forgery-detection/issues",
        "Documentation": "https://github.com/Mariam6600/Digital-image-forgery-detection",
        "Source Code": "https://github.com/Mariam6600/Digital-image-forgery-detection",
    },
)
