from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="qmc_tfim_py",
    version="0.1.0",
    author="Yuntai Song",
    author_email="yuntais2@illinois.edu",
    description="A Python implementation of Quantum Monte Carlo for Transverse Field Ising Model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/qmc_tfim_py",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy",
        "scipy",
        "tqdm",
    ],
    extras_require={
        "dev": ["pytest", "flake8"],
    },
    entry_points={
        "console_scripts": [
            "qmc_tfim=qmc_tfim.main:main",
        ],
    },
)
