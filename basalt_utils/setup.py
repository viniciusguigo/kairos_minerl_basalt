import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="basalt_utils",
    version="0.0.1",
    author="Cody Wild",
    author_email="codywild@berkeley.edu",
    description="Package for baselines utilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="None",
    project_urls={
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)