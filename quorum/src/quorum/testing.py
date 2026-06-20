"""Opt-in test bypass for the license gate. Import this in conftest.py."""
import os
os.environ["_QUORUM_TEST_BYPASS_INTERNAL"] = "1"
