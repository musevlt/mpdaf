#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["nox"]
# ///

import nox

nox.needs_version = ">= 2025.10.14"
nox.options.default_venv_backend = "uv|virtualenv"
nox.options.reuse_venv = "yes"

pytest_args = ("-rs", "--pyargs", "mpdaf")


def install(session):
    if "-v" in session.posargs:
        session.install(".[tests,all]", "-v")
    else:
        session.install(".[tests,all]")


@nox.session()
def tests(session):
    install(session)
    session.run("pytest", *pytest_args, *session.posargs)


@nox.session()
def coverage(session):
    install(session)
    session.run("coverage", "run", "-m", "pytest", *pytest_args, *session.posargs)
    session.run("coverage", "report")


@nox.session
def docs(session):
    session.install(".[docs]")
    with session.chdir("doc"):
        # fmt: off
        session.run(
            "python", "-m", "sphinx",
            "-T", "-E", "--keep-going",
            "-b", "html",
            "-d", "_build/doctrees",
            "-j", "auto",
            ".",
            "_build/html",
        )
        # fmt: on


if __name__ == "__main__":
    nox.main()
