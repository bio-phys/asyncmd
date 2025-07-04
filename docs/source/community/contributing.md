# Contributing to asyncmd

There are many ways you can contribute to asyncmd, be it with bug reports or feature requests, by writing or expanding its documentation, or by submitting patches for new or fixed behavior.

## Bug reports and feature requests

If you have encountered a problem using asyncmd or have an idea for a new feature, please open a [Github issue].
For bug reports, please include the full error message and, if possible, a (minimal) example resulting in the bug.

## Contribute code

The asyncmd source code is managed using git and [hosted on Github][Github]. The recommended way for contributors to submit code is to fork this repository and open a [pull request][Github pr].

(contributing-getting-started)=
### Getting started

Before starting a patch, it is recommended to check for open [issues][Github issue] or [pull requests][Github pr] relating to the topic.

The basic steps to start developing on asyncmd are:

1. [Fork](https://github.com/bio-phys/asyncmd/fork) the repository on Github.
2. (Optional but highly recommended) Create and activate a python virtual environment using your favorite environment manager (e.g. virtualenv, conda).
3. Clone the repository and install it in editable mode using the dev target (see the [installation instructions](#developer-installation)).
4. Create a new working branch and write your code. Please try to write tests for your code and make sure that all tests pass (see [here](#tests-installation)).
5. Add a bullet point to `CHANGELOG.md` if the fix or feature is not trivial, then commit.
6. Push the changes and open a [pull request on Github][Github pr].

### Coding style

Please follow these guidelines when writing code for asyncmd:

- Try to use the same code style as the rest of the project.
- Update `CHANGELOG.md` for non-trivial changes. If your changes alter existing behavior, please document this.
- New features should be documented. If possible, also include them in the example notebooks or add a new example notebook showcasing the feature.
- Add appropriate unit tests.

## Contribute documentation

To contribute documentation you will need to modify the source files in the `doc/source` folder. To get started follow the steps in [Getting started](#contributing-getting-started), but instead of writing code (and tests) modify the documentation source files. You can then build the documentation locally (see the [installation instructions](#documentation-installation)) to check that everything works as expected. Additionally, after submitting a [pull request][Github pr], you can preview your changes as they will be rendered on readthedocs directly.

[Github]: https://github.com/bio-phys/asyncmd
[Github issue]: https://github.com/bio-phys/asyncmd/issues
[Github pr]: https://github.com/bio-phys/asyncmd/pulls
