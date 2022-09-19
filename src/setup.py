from setuptools import setup, find_packages

requirements = [
    'torch>=1.10.2',
    'torchvision>=0.11.3',
    'numpy>=1.19.4',
    'scipy>=1.5.4',
    'ray>=1.0.0',
    'sklearn>=0.0',
    'scikit-learn>=1.0.2',
    'matplotlib>=3.4.1',
    'requests>=2.27.1',
    'setuptools~=58.0.4',
    'ruamel.yaml',
]
setup(
    name='blades',
    version='0.1.1',
    description="""Blades: A simulator for Byzantine-robust federated Learning with Attacks and Defenses Experimental Simulation""",
    # long_description=open('../README.rst').read(),
    # long_description_content_type="text/markdown",
    author='Blades Team',
    author_email='shenghui.li@it.uu.se',
    url='https://bladesteam.github.io/',
    # py_modules=['blades'],
    python_requires='>=3.8',
    license='MIT license',
    zip_safe=False,
    # entry_points={
    #     'console_scripts': [""]
    # },
    install_requires=requirements,
    keywords='Federated learning',
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
