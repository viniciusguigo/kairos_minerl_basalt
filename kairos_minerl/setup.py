import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="kairos_minerl",
    version="1.0.1",
    author="KAIROS Lab",
    author_email="vinicius.goecks@gmail.com, nick.waytowich@gmail.com, davidwatkins@cs.columbia.edu, bhp1@umbc.edu",
    description="KAIROS learning agent for NeurIPS 2021 MineRL BASALT competition.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.aicrowd.com/vinicius_g_goecks",
    project_urls={
        "Bug Tracker": "https://gitlab.aicrowd.com/vinicius_g_goecks/issues",
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