from setuptools import setup, find_packages
from os.path import join, dirname

setup(
    name='rct',
    version='1.0',
    packages= ['rct'],
    long_description=open(join(dirname(__file__), 'README.txt')).read()
    )
