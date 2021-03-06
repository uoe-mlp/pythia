from pythia.experiment import StandardExperiment


def test_standard_analysis_three_args():
    analysis = StandardExperiment.initialise(
        path='/.',
        market=None,
        agent=None,
        journal=None,
        settings={},
        params={
        'train': 30,
        'val': 20,
        'test': 50
    })

    assert analysis.train == 0.30
    assert analysis.val == 0.20
    assert analysis.test == 0.50

def test_standard_analysis_two_args():
    analysis = StandardExperiment.initialise(
        path='/.',
        market=None,
        agent=None,
        journal=None,
        settings={},
        params={
        'train': 0.50,
        'val': 0.12
    })

    assert analysis.test == 0.38

def test_standard_analysis_one_args():
    analysis = StandardExperiment.initialise(
        path='/.',
        market=None,
        agent=None,
        journal=None,
        settings={},
        params={
        'train': 0.6
    })

    assert analysis.test == 0.20
    assert analysis.val == 0.20    

def test_standard_analysis_zero_args():
    analysis = StandardExperiment.initialise(
        path='/.',
        market=None,
        agent=None,
        journal=None,
        settings={},
        params={})

    assert analysis.test == 0.15
    assert analysis.val == 0.15
    assert analysis.train == 0.70
    