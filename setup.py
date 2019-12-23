from setuptools import setup, find_packages

setup(
    name="descwl_coadd",
    version="0.1.0",
    packages=find_packages(),
    install_requires=['numpy'],
    include_package_data=True,
    author='Erin S. Sheldon',
    author_email='erin.sheldon@gmail.com',
    url='https://github.com/LSSTDESC/descwl_coadd',
)
