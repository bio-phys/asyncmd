# This file is part of asyncmd.
#
# asyncmd is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# asyncmd is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with asyncmd. If not, see <https://www.gnu.org/licenses/>.
import os
# always prefer setuptools over distutils!
from setuptools import setup, find_packages


###############################################################################

NAME = "asyncmd"
PACKAGES = find_packages(where="src")
KEYWORDS = ["molecular dynamics, high performance computing"]
PROJECT_URLS = {
  "Source Code": "https://gitea.kotspeicher.de/AIMMD/asyncmd",
}
CLASSIFIERS = [
  "Development Status :: 3 - Alpha"
  "Intended Audience :: Science/Research",
  "Natural Language :: English",
  "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
  "Operating System :: OS Independent",
  "Programming Language :: Python",
  "Programming Language :: Python :: 3",
  # py < 3.9 does not work because we e.g. use new style type annotations
  # "Programming Language :: Python :: 3.6",
  # "Programming Language :: Python :: 3.7",
  # "Programming Language :: Python :: 3.8",
  "Programming Language :: Python :: 3.9",
  "Programming Language :: Python :: 3.10",
  "Topic :: Scientific/Engineering",
  "Topic :: Scientific/Engineering :: Chemistry",
  "Topic :: Scientific/Engineering :: Physics",
  "Topic :: Software Development :: Libraries :: Python Modules",
]
SETUP_REQUIRES = []
INSTALL_REQUIRES = ["mdanalysis", "numpy", "aiofiles"]
EXTRAS_REQUIRE = {
    "docs": ["sphinx"],
    "tests": ["pytest", "pytest-asyncio"],
}
EXTRAS_REQUIRE["dev"] = (EXTRAS_REQUIRE["docs"] + EXTRAS_REQUIRE["tests"]
                         + ["pytest-cov", "coverage",
                            "flake8", "flake8-alfred", "flake8-comprehensions",
                            "flake8-docstrings", "flake8-if-statements",
                            "flake8-logging-format", "flake8-todo",
                            ]
                         )

###############################################################################

# Get the long description from the README file
HERE = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(HERE, 'README.md'), encoding='utf-8') as f:
    LONG_DESCRIPTION = f.read()
# Get version and other stuff from __about__.py
about_dct = {"__file__": __file__}
with open(os.path.join(HERE, "src", "asyncmd", "__about__.py"), 'r') as fp:
    exec(fp.read(), about_dct)

# the magic :)
setup(
        name=NAME,
        description=about_dct["__description__"],
        license=about_dct["__license__"],
        url=about_dct["__url__"],
        project_urls=PROJECT_URLS,
        version=about_dct["__version__"],
        author=about_dct["__author__"],
        author_email=about_dct["__author_email__"],
        maintainer=about_dct["__author__"],
        maintainer_email=about_dct["__author_email__"],
        keywords=KEYWORDS,
        long_description=LONG_DESCRIPTION,
        long_description_content_type="text/markdown",
        packages=PACKAGES,
        package_dir={"": "src"},
        python_requires=">=3.9",
        zip_safe=False,
        classifiers=CLASSIFIERS,
        setup_requires=SETUP_REQUIRES,
        install_requires=INSTALL_REQUIRES,
        extras_require=EXTRAS_REQUIRE,
        include_package_data=True,
    )
