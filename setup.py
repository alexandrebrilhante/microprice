import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setuptools.setup(
    name="microprice",
    version="0.1.1",
    author="Alexandre Brilhante",
    author_email="alexandre.brilhante@gmail.com",
    description="A Jax implementation of Micro-Price: A High Frequency Estimator of Future Prices.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/alexandrebrilhante/quantnet",
    project_urls={
        "Bug Tracker": "https://github.com/alexandrebrilhante/microprice/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.8",
)
