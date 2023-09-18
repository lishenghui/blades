from setuptools import find_packages, setup

requirements = [
    "torch>=2.0",
    "torchvision",
    "torchaudio",
    "numpy>=1.19.4",
    "ray>=2.4.0",
    "matplotlib>=3.4.1",
    "requests>=2.27.1",
    "setuptools",
    "ruamel.yaml",
    "tqdm",
    "wandb",
    "typer",
    "dm_tree",
    "scikit-learn",
    "pytest",
    "ray[rllib]",
    "pre-commit",
    "torchmetrics",
]
setup(
    name="blades",
    version="0.1.0",
    description="""
    A Unified Benchmark Suite for Byzantine Attacks and Defenses in Federated Learning
    """,
    # long_description=open('../README.rst').read(),
    # long_description_content_type="text/markdown",
    author="Blades Team",
    author_email="shenghui.li@it.uu.se",
    url="https://github.com/lishenghui/blades",
    python_requires=">=3.9",
    license="Apache License 2.0",
    zip_safe=False,
    # entry_points={
    #     'console_scripts': [""]
    # },
    install_requires=requirements,
    keywords="Federated learning",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
