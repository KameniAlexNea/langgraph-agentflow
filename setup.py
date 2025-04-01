from setuptools import setup, find_packages

setup(
    name="langgraph-agentflow",
    version="0.1.0",
    description="AgentFlow library for orchestrating multi-step agent workflows",
    author="KameniAlexNea",
    author_email="Kamenialexnea@gmail.com",
    packages=find_packages(include=["langgraph_agentflow", "langgraph_agentflow.*"]),
    package_data={"langgraph_agentflow": ["py.typed"]},
    install_requires=[
        "langgraph",
        "langchain-core",
        "loguru",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "isort>=5.0.0",
            "mypy>=1.0.0",
        ],
        "examples": [
            "langchain-ollama",
        ],
    },
    python_requires=">=3.8",
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
