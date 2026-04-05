# from setuptools import setup, find_packages

# setup(
#     name="causal-context-pruning",
#     version="0.1.0",
#     author="Amit Kumar Patel",
#     description=(
#         "Causal Context Pruning (CCP): Training-Free Causal Necessity Scoring "
#         "for Context Management in Long-Horizon Agentic Systems"
#     ),
#     packages=find_packages(),
#     python_requires=">=3.10",
#     install_requires=[
#         "langgraph>=0.2.0",
#         "langchain>=0.2.0",
#         "langchain-openai>=0.1.0",
#         "openai>=1.0.0",
#         "tenacity>=8.2.0",
#         "tiktoken>=0.7.0",
#         "numpy>=1.26.0",
#         "tqdm>=4.66.0",
#     ],
#     extras_require={
#         "benchmark": ["appworld>=0.1.3", "matplotlib>=3.8.0", "pandas>=2.2.0"],
#         "dev":       ["pytest>=8.0.0", "pytest-mock>=3.12.0"],
#     },
#     entry_points={
#         "console_scripts": [
#             "ccp-run=experiments.run_experiment:main",
#         ],
#     },
# )

from setuptools import setup, find_packages

setup(
    name="causal-context-pruning",
    version="0.1.0",
    author="Amit Kumar Patel",
    description=(
        "Causal Context Pruning (CCP): Training-Free Causal Necessity Scoring "
        "for Context Management in Long-Horizon Agentic Systems"
    ),
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "langgraph>=0.2.0",
        "langchain>=0.2.0",
        "langchain-openai>=0.1.0",
        "openai>=1.0.0",
        "tenacity>=8.2.0",
        "tiktoken>=0.7.0",
        "numpy>=1.26.0",
        "tqdm>=4.66.0",
    ],
    extras_require={
        "benchmark": ["matplotlib>=3.8.0", "pandas>=2.2.0"],
        "dev":       ["pytest>=8.0.0", "pytest-mock>=3.12.0"],
    },
)