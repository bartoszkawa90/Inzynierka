# setup.py

from setuptools import setup, Extension

module = Extension("example", sources=["example.c"])

setup(
    name="example",
    version="1.0",
    description="Example C Extension",
    ext_modules=[module]
)
