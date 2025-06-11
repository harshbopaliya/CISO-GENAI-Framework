from setuptools import setup, find_packages

setup(
    name="ciso_genai",
    version="0.1.0",
    packages=find_packages(),
    install_requires=open("requirements.txt").read().splitlines(),
)
