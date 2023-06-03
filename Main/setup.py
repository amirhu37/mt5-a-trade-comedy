from setuptools import setup

setup(
    name='mt5-tool-kit',
    version='1.0.0',
    author='Amir hussain',
    author_email='amir.hussain37@yahoo.com',
    description='some toolkit to use algo trade with metatrader5',
    packages=['mt5tool\\'],
    install_requires=[
        # List any dependencies your package needs
        'MetaTrader5==5.0.37',
        'numpy',
        'pandas'
    ],
)
