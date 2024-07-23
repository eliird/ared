from setuptools import setup, find_packages

setup(
    name="your_package_name",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A brief description of your package",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/eliird/ared",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        # List your package dependencies here, for example:
        'numpy>=1.18.0',
        'opencv-python'
    ],
    entry_points={
        'console_scripts': [
            # 'your-command=your_package.module:function',
        ],
    },
    include_package_data=True,
    package_data={
        # If any package contains *.txt or *.rst files, include them:
        # '': ['*.txt', '*.rst'],
        # Include example files in the package:
        # 'your_package_name': ['examples/*'],
    },
)
