# Contributing to samplesar

Thanks for your interest. Bug reports, questions, documentation fixes and code
are all welcome. This page covers how to get set up and what a change needs
before it can be merged.

## Getting help and reporting problems

- **Bugs and feature requests:** open an issue at
  <https://github.com/jmcmeen/samplesar/issues>. For a bug, include the
  `samplesar` version, your Python version, a minimal example that reproduces
  it, and the full traceback.
- **Questions:** also an issue. There is no separate mailing list or chat.
- **Security concerns:** email the maintainer directly rather than opening a
  public issue.

The project has one maintainer. Expect a response to issues within a week.

## Development setup

Only the maintainer can push to this repository, so contributions come in as
pull requests from a fork. Fork
[jmcmeen/samplesar](https://github.com/jmcmeen/samplesar) on GitHub, then clone
your fork. The project uses [uv](https://docs.astral.sh/uv/) for environments
and `make` as a thin front end.

```bash
git clone https://github.com/YOUR-USERNAME/samplesar.git   # your fork
cd samplesar
git remote add upstream https://github.com/jmcmeen/samplesar.git
make install        # uv sync --extra dev
make check          # lint, format check, type check and tests, as CI runs them
```

## Making a change

1. Open an issue first for anything larger than a typo, so the design can be
   discussed before code is written.
2. In your fork, branch from an up-to-date `main`, naming the branch after the
   kind of change (`feat/`, `fix/`, `docs/`, `refactor/`, `test/`, `build/`,
   `ci/` or `chore/`):

   ```bash
   git fetch upstream
   git switch -c fix/short-description --no-track upstream/main
   ```

3. Make the change and add or update tests. New behavior needs a test; a bug
   fix needs a test that fails without the fix.
4. Add a line under `[Unreleased]` in `CHANGELOG.md` (Keep a Changelog
   sections: Added, Changed, Deprecated, Removed, Fixed).
5. Run `make fmt`, then `make check`.
6. Push the branch to your fork (`git push -u origin fix/short-description`)
   and open a pull request against `main` in this repository. Say what changed
   and why, and reference the issue (`Fixes #N`).

CI runs on the pull request. The maintainer reviews and merges it. If `main`
moves on while your pull request is open, update your branch with
`git fetch upstream && git merge upstream/main`.

## Releases

Merges to `main` that pass CI are released automatically and published to
PyPI, normally as one patch version per merge. A minor or major release is
triggered by hand when the maintainer decides one is due. The `[Unreleased]`
changelog entries become the release notes, so you don't need to touch
version numbers.
