import os
from setuptools import setup, find_packages

HERE = os.path.abspath(os.path.dirname(__file__))


def read_requirements(path):
    reqs = []
    with open(os.path.join(HERE, path)) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            reqs.append(line)
    return reqs


def read_long_description():
    readme_path = os.path.join(HERE, "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, encoding="utf-8") as f:
            return f.read()
    return ""


setup(
    name="convective-nowcasting",
    version="0.1.0",
    description="Multi-horizon convective precipitation nowcasting from satellite and radar data",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    packages=find_packages(
        include=["src", "src.*", "train", "train.*", "visualization", "visualization.*"],
        exclude=["*.tests", "*.tests.*", "tests.*", "tests", "unittest", "unittest.*"],
    ),
    python_requires=">=3.10",
    install_requires=read_requirements("requirements.txt"),
    include_package_data=True,
)