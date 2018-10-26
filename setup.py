import ast
import re
import os

from setuptools import setup

PACKAGE_NAME = 'neuralfold'

with open(os.path.join(PACKAGE_NAME, '__init__.py')) as f:
    match = re.search(r'__version__\s+=\s+(.*)', f.read())
version = str(ast.literal_eval(match.group(1)))

setup(
    # metadata
    name=PACKAGE_NAME,
    version=version,

    # options
    packages=[PACKAGE_NAME],
    include_package_data=True,
    zip_safe=False,
    python_requires='>=3.4',
    install_requires=[ 'numpy', 'pulp', 'chainer'],
    extras_require={
        'dev': [ ],
    },
    entry_points='''
        [console_scripts]
        {pkg}={pkg}.__main__:main
    '''.format(pkg=PACKAGE_NAME),
)