from setuptools import setup, find_packages

setup(
    name='blades',
    version='0.0.1',
    description="""ByzantineFL""",
    # long_description=open('blades/README.md').read(),
    # long_description_content_type="text/markdown",
    author='Shenghui Li',
    author_email='shenghui.li@it.uu.se',
    url='https://github.com/lishenghui',
    # py_modules=['blades'],
    python_requires='>=3.8',
    license='Apache 2.0',
    zip_safe=False,
    # entry_points={
    #     'console_scripts': ["fedn=cli:main"]
    # },
    keywords='Federated learning',
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
