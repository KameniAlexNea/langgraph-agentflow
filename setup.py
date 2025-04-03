from setuptools import setup, find_packages
import os

# Read package metadata from settings.py
metadata = {}
with open(os.path.join(os.path.dirname(__file__), "settings.py")) as f:
    exec(f.read(), metadata)

def read_requirements():
    with open(os.path.join(os.path.dirname(__file__), "requirements.txt")) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

def read_long_description():
    with open(os.path.join(os.path.dirname(__file__), "README.md"), encoding="utf-8") as f:
        return f.read()

setup(
    name=metadata.get("PACKAGE_NAME", "langgraph-agentflow"),
    version=metadata.get("VERSION", "0.0.1"),  # updated version
    description=metadata.get("DESCRIPTION", ""),
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    author=metadata.get("AUTHOR", ""),
    author_email=metadata.get("AUTHOR_EMAIL", ""),
    url=metadata.get("URL", ""),
    packages=find_packages(),
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "isort>=5.0.0",
            "mypy>=1.0.0",
        ],
        "examples": [
            "langchain-ollama",
            "pygraphviz",
            "jupyter",
            "pickleshare",
        ],
    },
    dependency_links=[
        "git+https://github.com/Nganga-AI/tumkwe-invest.git#egg=tumkwe-invest"
    ],
    python_requires=">=3.8",
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
