import setuptools


setuptools.setup(name='ntflib',
                 version='0.0.1',
                 description='Non-negative Sparse Tensor Factorization Library',
                 long_description=open('README.md').read().strip(),
                 py_modules=['ntflib'],
                 install_requires=['numpy', 'numba'],
                 zip_safe=True)
