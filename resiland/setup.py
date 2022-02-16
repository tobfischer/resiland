from pkg_resources import Requirement
from setuptools import setup
from io import open 

requirements = [
    "jitcxde_common > 1.3",
    "symengine >= 0.3.1.dev0",
    "scipy",
    "numpy"
]

setup(
    name = "resiland",
    description = "Time series in potential landscape for resilience",
    long_description = open("README.md", encoding = "utf8").read(),
    author = "Tobias Fischer",
    author_email = "tobias.fischer@uni-bonn.de"
    url = "https://github.com/neurophysik/resiland",
    python_requires = ">=3.3",
    packages = ["resiland"],
    package_data = {"resiland": }
    install_requirements = requirements,
    setup_requires = ["setuptools_scm"],
    use_scm_version = {"write_to": "resiland/version.py"},
    classifiers = [
        "Development Status :: 4 - Beta",
        "Licence :: OSI Approved :: BSD Licence",
        "Operating System :: POSIX",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)

