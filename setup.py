import os
from setuptools import setup, find_packages

__version__ = None

pth = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "descwl_coadd",
    "version.py"
)
with open(pth, 'r') as fp:
    exec(fp.read())

setup(
    name="descwl_coadd",
    version=__version__,
    packages=find_packages(),
    include_package_data=True,
    author='Erin S. Sheldon',
    author_email='erin.sheldon@gmail.com',
    url='https://github.com/LSSTDESC/descwl_coadd',
)
