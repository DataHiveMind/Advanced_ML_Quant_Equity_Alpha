# setup.py

import os
from setuptools import setup, find_packages

# Helper function to read the README file.
# This is used for the 'long_description' field.
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

# Helper function to read the requirements.txt file
def get_requirements(fname):
    with open(os.path.join(os.path.dirname(__file__), fname)) as f:
        return f.read().splitlines()

setup(
    # Basic project metadata
    name="advanced_ml_quant_equity_alpha",  # Name of your package
    version="0.1.0",                        # Initial version of your package
    author="Kenneth LeGare",                   # Replace with your name
    author_email="kennethlegare5@gamil.com",# Replace with your email
    description="A senior project for systematic equity alpha generation using advanced ML and quant techniques.",
    long_description=read('README.md'),     # Use your project's README as the long description
    long_description_content_type="text/markdown", # Specify markdown content type
    url="https://github.com/[YourGitHubUsername]/Advanced_ML_Quant_Equity_Alpha", # Replace with your GitHub URL
    license="MIT",                          # Or your chosen license (e.g., Apache-2.0)

    # Automatically find all packages in the 'src' directory
    # This assumes your main code is within 'src/' subdirectories.
    packages=find_packages(where="src"),
    package_dir={"": "src"}, # Tell setuptools that packages are under 'src'

    # List of dependencies required by your package
    install_requires=get_requirements('requirements.txt'),

    # Classifiers help users find your project on PyPI and understand its purpose.
    # Choose appropriate ones from https://pypi.org/classifiers/
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha", # Indicate project maturity
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Financial Analysis",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Framework :: TensorFlow",
        "Framework :: PyTorch",
    ],
    # Specify the minimum Python version required
    python_requires='>=3.9',
    # Include non-code files (e.g., config, data samples) if needed within the package.
    # If your data and config are outside the package root, you might not need this.
    # include_package_data=True,
    # package_data={
    #     'your_package_name': ['data/*.csv', 'config/*.yaml'],
    # },
)