# Contributing

Welcome to Edsger's contributor's guide.

This document focuses on getting any potential contributor familiarized with the development processes, but [other kinds of contributions](https://opensource.guide/how-to-contribute) are also appreciated.

If you are new to using git or have never collaborated in a project previously, please have a look at [contribution-guide.org](https://www.contribution-guide.org/). Other resources are also listed in the excellent [guide created by FreeCodeCamp](https://github.com/FreeCodeCamp/how-to-contribute-to-open-source).

Please notice, all users and contributors are expected to be **open, considerate, reasonable, and respectful**. When in doubt, [Python Software Foundation's Code of Conduct](https://www.python.org/psf/conduct/) is a good reference in terms of behavior guidelines.

## Issue Reports

If you experience bugs or general issues with Edsger, please have a look on the [issue tracker](https://github.com/aetperf/Edsger/issues). If you don't see anything useful there, please feel free to fire an issue report.

> **Tip:** Please don't forget to include the closed issues in your search. Sometimes a solution was already reported, and the problem is considered **solved**.

New issue reports should include information about your programming environment (e.g., operating system, Python version) and steps to reproduce the problem. Please try also to simplify the reproduction steps to a very minimal example that still illustrates the problem you are facing. By removing other factors, you help us to identify the root cause of the issue.

## Documentation Improvements

You can help improve Edsger docs by making them more readable and coherent, or by adding missing information and correcting mistakes.

Edsger documentation uses [Sphinx](https://www.sphinx-doc.org/en/master/) as its main documentation compiler. This means that the docs are kept in the same repository as the project code, and that any documentation update is done in the same way as a code contribution.

> **Tip:** The [GitHub web interface](https://docs.github.com/en/repositories/working-with-files/managing-files/editing-files) provides a quick way to propose changes in Edsger's files. While this mechanism can be tricky for normal code contributions, it works perfectly fine for contributing to the docs, and can be quite handy.
>
> If you are interested in trying this method out, please navigate to the `docs` folder in the [repository](https://github.com/aetperf/Edsger), find which file you would like to propose changes and click in the little pencil icon at the top, to open [GitHub's code editor](https://docs.github.com/en/repositories/working-with-files/managing-files/editing-files). Once you finish editing the file, please write a message in the form at the bottom of the page describing which changes have you made and what are the motivations behind them and submit your proposal.

When working on documentation changes in your local machine, you can compile them using:

```bash
cd docs
make html
```

and use Python's built-in web server for a preview in your web browser (`http://localhost:8000`):

```bash
python3 -m http.server --directory 'docs/_build/html'
```

## Code Contributions

Edsger is a Cython-based library for graph algorithms. The core algorithms are implemented in Cython (`.pyx` files) for performance, with a Python API layer for ease of use.

Key modules:

- `src/edsger/dijkstra.pyx` - Dijkstra's shortest path algorithm
- `src/edsger/bellman_ford.pyx` - Bellman-Ford algorithm
- `src/edsger/bfs.pyx` - Breadth-First Search
- `src/edsger/star.pyx` - CSR graph representation

### Submit an issue

Before you work on any non-trivial code contribution it's best to first create a report in the [issue tracker](https://github.com/aetperf/Edsger/issues) to start a discussion on the subject. This often provides additional considerations and avoids unnecessary work.

### Create an environment

Before you start coding, we recommend creating an isolated virtual environment to avoid any problems with your installed Python packages. This can easily be done via either `virtualenv`:

```bash
virtualenv <PATH TO VENV>
source <PATH TO VENV>/bin/activate
```

or Miniconda:

```bash
conda create -n edsger python=3.11
conda activate edsger
```

### Clone the repository

1. Create a user account on GitHub if you do not already have one.

2. Fork the project [repository](https://github.com/aetperf/Edsger): click on the *Fork* button near the top of the page. This creates a copy of the code under your account on GitHub.

3. Clone this copy to your local disk:

   ```bash
   git clone git@github.com:YourLogin/Edsger.git
   cd Edsger
   ```

4. Install the development dependencies and the package in editable mode:

   ```bash
   pip install -r requirements-dev.txt
   pip install -e .
   ```

5. Install `pre-commit`:

   ```bash
   pip install pre-commit
   pre-commit install
   ```

   Edsger comes with pre-commit hooks configured to automatically help the developer to check the code being written.

### Implement your changes

1. Create a branch to hold your changes:

   ```bash
   git checkout -b my-feature
   ```

   and start making changes. Never work on the main branch!

2. Start your work on this branch. Don't forget to add [docstrings](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html) to new functions, modules and classes, especially if they are part of public APIs.

3. Add yourself to the list of contributors in `AUTHORS.md`.

4. When you're done editing, do:

   ```bash
   git add <MODIFIED FILES>
   git commit
   ```

   to record your changes in git.

   Please make sure to see the validation messages from `pre-commit` and fix any eventual issues. This should automatically use [black](https://pypi.org/project/black/) to check/fix the code style and [cython-lint](https://github.com/MarcoGorelli/cython-lint) for Cython files.

   > **Important:** Don't forget to add unit tests and documentation in case your contribution adds an additional feature and is not just a bugfix.
   >
   > Moreover, writing a [descriptive commit message](https://chris.beams.io/posts/git-commit) is highly recommended. In case of doubt, you can check the commit history with:
   >
   > ```bash
   > git log --graph --decorate --pretty=oneline --abbrev-commit --all
   > ```
   >
   > to look for recurring communication patterns.

5. Please check that your changes don't break any unit tests with:

   ```bash
   pytest
   ```

### Submit your contribution

1. If everything works fine, push your local branch to GitHub with:

   ```bash
   git push -u origin my-feature
   ```

2. Go to the web page of your fork and click "Create pull request" to send your changes for review.

   Find more detailed information in [creating a PR](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request). You might also want to open the PR as a draft first and mark it as ready for review after the feedbacks from the continuous integration (CI) system or any required fixes.

## Troubleshooting

The following tips can be used when facing problems to build or test the package:

1. Make sure to fetch all the tags from the upstream [repository](https://github.com/aetperf/Edsger). The command `git describe --abbrev=0 --tags` should return the version you are expecting. If you are trying to run CI scripts in a fork repository, make sure to push all the tags. You can also try to remove all the egg files or the complete egg folder, i.e., `.eggs`, as well as the `*.egg-info` folders in the `src` folder or potentially in the root of your project.

2. If you have trouble building the Cython extensions, make sure you have a working C compiler installed. On Linux, install `build-essential`. On macOS, install Xcode Command Line Tools. On Windows, install Visual Studio Build Tools.

3. [Pytest can drop you](https://docs.pytest.org/en/stable/how-to/failures.html#using-python-library-pdb-with-pytest) in an interactive session in the case an error occurs. In order to do that you need to pass a `--pdb` option (for example by running `pytest -k <NAME OF THE FAILING TEST> --pdb`). You can also setup breakpoints manually instead of using the `--pdb` option.

## Maintainer tasks

### Releases

If you are part of the group of maintainers and have correct user permissions on [PyPI](https://pypi.org/), the following steps can be used to release a new version for Edsger:

1. Make sure all unit tests are successful.
2. Tag the current commit on the main branch with a release tag, e.g., `v1.2.3`.
3. Push the new tag to the upstream [repository](https://github.com/aetperf/Edsger), e.g., `git push upstream v1.2.3`
4. Create a release on GitHub from the tag. The CI will automatically build and publish wheels to PyPI.
