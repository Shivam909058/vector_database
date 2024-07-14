from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="my-vector-database",
    version="0.1.0",
    author="Shivam singh",
    author_email="shivamatvit@gmail.com",
    description="A simple vector database using transformers and cosine similarity",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/vector_database",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
        "scikit-learn",
        "numpy"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
