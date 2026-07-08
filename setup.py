from setuptools import find_packages, setup

setup(
    name="forgery-detection-ela-resnet101v2",
    version="1.0.0",
    description="Digital image forgery detection — Methodology 2: Single-Input ELA + ResNet101V2 (CASIA 2.0).",
    author="Mariam Abd Al Aal",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.9",
    install_requires=[
        "tensorflow>=2.15,<2.17",
        "numpy>=1.23,<2.0",
        "pillow>=9.0",
        "scikit-learn>=1.2",
        "matplotlib>=3.6",
        "seaborn>=0.12",
        "kagglehub>=0.2",
    ],
)
