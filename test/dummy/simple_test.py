import pytest


def test_always_succeed():
    assert(True)

@pytest.mark.skip(reason="This test will always fail")
def test_always_fail():
    assert(False)