from setuptools import setup, find_packages

with open("README.md") as readme_file:
    README = readme_file.read()

with open("HISTORY.md") as history_file:
    HISTORY = history_file.read()

setup_args = dict(
    name="nlpbaselines",
    version="0.0.45",
    description="Quickly establish strong baselines for NLP tasks",
    long_description_content_type="text/markdown",
    long_description=README,
    include_package_data=True,
    license="Apache Licence 2.0",
    packages=find_packages(),
    author="Xiaoou WANG",
    author_email="xiaoouwangfrance@gmail.com",
    keywords=[
        "text mining",
        "nlp",
        "corpus",
        "french",
        "bert",
        "transformers",
        "camembert",
    ],
    url="https://github.com/xiaoouwang/nlpbaselines",
    download_url="https://pypi.org/project/nlpbaselines/",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

install_requires = [
    "accelerate",
    "datasets",
    "matplotlib",
    "torch",
    "transformers",
    "scikit-learn",
    "numpy",
    "pandas",
    "py7zr",
    "sentencepiece",
    "sacremoses",
    "numba",
]


if __name__ == "__main__":
    setup(**setup_args, install_requires=install_requires)
