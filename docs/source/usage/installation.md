# Installation

Independently of how you install asyncmd you will need a working installation of [GROMACS] and, if you want to submit jobs via the queuing system, [SLURM] on the machine you install asyncmd.

## pip install from PyPi

asyncmd is published on [PyPi] (since v0.3.2), installing is as easy as:

```bash
pip install asyncmd
```

## pip install directly from the repository

In case you you intend to run the tests or {doc}`example notebooks </examples_link/README>` yourself or if want to install the latest and greatest development version of asyncmd (see the {doc}`changelog </include_changelog>` for whats new) you will need to install asyncmd from the git repository.

```{important}
Please note that you need to have [git-lfs] (an open source git extension) setup to get all input files needed to run the notebooks in the `examples` folder (no [git-lfs] is needed to get a working version of the library).
```

This will clone the repository to the current working directory and install asyncmd into the current python environment:

```bash
git clone https://github.com/bio-phys/asyncmd.git
cd asyncmd
pip install .
```

(tests-installation)=
### Tests

Tests use [pytest]. To run them you can install asycmd with the tests requirements. All tests should either pass or be skipped.

```{important}
Please note that you need [git-lfs] to get all (input) files used in the tests.
```

This will clone the repository to the current working directory and install asyncmd with the tests requirement into the current python environment:

```bash
git clone https://github.com/bio-phys/asyncmd.git
cd asyncmd
pip install .\[tests\]
# or use
pip install .\[tests-all\]
# to also install optional dependencies needed to run all tests
```

you can then run the tests (against the installed version) as

```bash
pytest
# or use
pytest -v
# to get a more detailed report
```

(documentation-installation)=
### Documentation

The documentation can be build with [sphinx], use e.g. the following to build it in html format:

```{note}
Use ```pip install .\[docs\]``` to install the requirements needed to build the documentation.
```

```bash
cd asyncmd  # Need to be at the top folder of the repository for the next line to work
sphinx-build -b html docs/source docs/build/html
```

(developer-installation)=
## Developer installation

If you intend to contribute to asyncmd, it is recommended to use the dev extra and use an editable install to enable you to directly test your changes:

```bash
git clone https://github.com/bio-phys/asyncmd.git
cd asyncmd
pip install -e .\[dev\]
```

This will, in addition to the requirements to run the tests and to build the documentation, install [coverage] and its [pytest-cov] plugin such that you have an idea of the test coverage for your newly added code. To get a nice html coverage report you can run the tests as

```bash
pytest --cov=asyncmd --cov-report=html
```

[coverage]: https://pypi.org/project/coverage/
[git-lfs]: https://git-lfs.com/
[GROMACS]: https://www.gromacs.org/
[PyPi]: https://pypi.org/project/asyncmd/
[pytest]: https://docs.pytest.org/en/latest/
[pytest-cov]: https://pypi.org/project/pytest-cov/
[SLURM]: https://slurm.schedmd.com/documentation.html
[sphinx]: https://www.sphinx-doc.org/en/master/index.html