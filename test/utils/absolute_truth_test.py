def test_always_succeed():
    assert(True)

def test_always_fail():
    try:
        print(1/0)
        assert(False)
    except ZeroDivisionError as exc:
        assert(True)
