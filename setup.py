import setuptools



setuptools.setup(
    name='TVB',
    version='0.0.1',
    author='Sathvik Udupa',
    author_email='sathvikudupa66@gmail.com',
    description='Tonic vs bursting package',
    url='https://github.com/bloodraven66/tonicBurstingPackage',
    license='MIT',
    packages=['TVB'],
    install_requires=[
                    'numpy',
                    'scipy',
                    'torch', 
                    'matplotlib',
                    'coloredlogs',
                    'attrdict',
                    'tqdm',
                    'huggingface_hub',
                    'sphinx==3.03',
                    'sphinx_rtd_theme==0.4.3',
                    ],
)
