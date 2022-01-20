# Copyright (c) 2019 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

import glob
import io
import sys

from setuptools import setup

# add mosaic_topog to our path in order to use the branch_scheme function
sys.path.append("mosaic_topog")
#from branch_scheme import branch_scheme  # noqa

with io.open("README.md", "r", encoding="utf-8") as readme_file:
    readme = readme_file.read()

setup_args = {
    "name": "mosaic_topog",
    "author": "Sierra Schleufer, Sabesan Lab",
    "url": "https://github.com/schleuf/Incubator-2022-Geometry-of-Color/mosaic_topog/",
    "license": "MIT",
    "description": "tools for analyzing the geometry of cones in the photoreceptor mosaic",
    "long_description": readme,
    "long_description_content_type": "text/markdown",
    "package_dir": {"mosaic_topog": "mosaic_topog"},
    "packages": ["mosaic_topog", "mosaic_topog.tests"],
    "scripts": glob.glob("scripts/*"),
    #"use_scm_version": {"local_scheme": branch_scheme},
    "include_package_data": True,
    "install_requires": [
        "numpy",
        "matplotlib",
        "scipy",
    ],

    "keywords": "cone photoreceptors",
}

if __name__ == "__main__":
    setup(**setup_args)