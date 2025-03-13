from setuptools import setup, find_packages

setup(
    name="UrbanAccessAnalyzer",
    version="0.1.0",
    description="A Python package to process isochrone service quality converage on open street network data.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Miguel Ureña Pliego",
    author_email="miguel.urena@upm.es",
    url="https://github.com/GeomaticsCaminosUPM/UrbanAccessAnalyzer/",
    license="Apache-2.0",
    packages=find_packages(),
    install_requires=[
        "geopandas>=1.0.0",
        "pandas>=1.3.0",
        "shapely>=1.8.0",
        "numpy>=1.21.0",
        "scipy>=1.0.0",
        "pyogrio>=0.1.0",
        "scikit-learn>=1.0.0",
        "networkx>=3.0.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache-2.0 License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
)
