from setuptools import setup, find_packages
import os
from pathlib import Path

folder = os.path.abspath(os.path.dirname(__file__))
versionpath = os.path.join(folder, "version.txt")

VERSION = Path(versionpath).read_text()

setup(
    name="pythia",
    version=VERSION,
    description="Pythia",
    url="https://github.com/GrowlingM1ke/MLP-CW2-2021",
    packages=find_packages(exclude=["test", "test.*"]),
    include_package_data=True
)
