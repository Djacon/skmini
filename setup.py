from setuptools import find_packages, setup

setup(
    name="skmini",
    version='0.1.0',
    author="Djacon",
    author_email="djaconfr@gmail.com",
    description="A simplified version of the sklearn library for ML",
    url="https://github.com/Djacon/skmini/",
    packages=find_packages(),
    install_requires=["numpy>=1.17.3"],
)

# python3 setup.py sdist bdist_wheel
