from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['tensorflow', 'numpy', 'sklearn', 'datetime', 'ipython-autotime', 'pathlib']

# removed matplotlib

setup(
    name='Meaningful',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='My trainer application package.'
)