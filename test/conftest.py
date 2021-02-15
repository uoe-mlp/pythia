import os
import shutil


def pytest_sessionstart(session):
    """
    Called after the Session object has been created and
    before performing collection and entering the run test loop.
    """
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    return session


def pytest_sessionfinish(session, exitstatus):
    shutil.rmtree(os.path.join('test', '.tmp'))
