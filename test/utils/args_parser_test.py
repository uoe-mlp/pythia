from pythia.utils import ArgsParser


def test_get_or_error():
    try:
        ArgsParser.get_or_error({}, 'test')
        assert(False)
    except KeyError as exc:
        assert(True)

    assert ArgsParser.get_or_error({'test': 1}, 'test') == 1

def test_get_or_default():
    assert ArgsParser.get_or_default({}, 'test', 1) == 1
    assert ArgsParser.get_or_default({'test': 2}, 'test', 1) == 2
