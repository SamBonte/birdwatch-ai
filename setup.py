"""
Setup script for birdwatch-ai package.
"""

from setuptools import setup, find_packages

# Read requirements from requirements.txt
with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="birdwatch-ai",
    version="1.0.0",
    description="A production-ready bird species classification system using MobileNetV2",
    author="Sam Bonte",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "birdwatch-ai=birdwatch_ai.run:main",
        ],
    },
)

