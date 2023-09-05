from setuptools import setup, find_packages

dependencies = ["numpy", "scipy>=1.5", "matplotlib", "plotly"]

setup(name='jcl', version='1.0', packages=find_packages(), python_requires=">=3.6", install_requires=dependencies)
