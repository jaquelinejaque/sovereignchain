"""Opt-in test bypass for the Quorum license gate.

Usage (in tests/conftest.py):

    import quorum.testing  # noqa: F401

KNOWN LIMITATION (v0.2.1): importing ``quorum.testing`` does NOT currently
bypass the gate, because ``quorum/__init__.py`` runs ``check_license()`` at
package import time, which fires BEFORE Python can resolve the submodule.

Workaround until lazy-gate refactor lands: set the env var directly in your
test runner config or shell, e.g.::

    # pytest.ini / pyproject.toml [tool.pytest.ini_options]
    env = _QUORUM_TEST_BYPASS_INTERNAL=1

    # or shell
    export _QUORUM_TEST_BYPASS_INTERNAL=1
    pytest

Tracked: gate to become lazy in a future release, at which point the
``import quorum.testing`` pattern starts working as documented.
"""
import os

os.environ["_QUORUM_TEST_BYPASS_INTERNAL"] = "1"
