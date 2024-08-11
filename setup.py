from setuptools import find_packages, setup
from typing import List

EXCLUDE_PACKAGE = '-e .'

def load_requirements(path: str) -> List[str]:
    dependencies = []
    with open(path) as file:
        dependencies = file.readlines()
        dependencies = [dep.replace('\n', '') for dep in dependencies]
        if EXCLUDE_PACKAGE in dependencies:
            dependencies.remove(EXCLUDE_PACKAGE)
    return dependencies

setup(
    name='Curvetopia',
    packages=find_packages(),
    install_requires=load_requirements('requirements.txt')
)
