from pythia.experiment.analysis import StandardAnalysis


def test_standard_analysis_three_args():
    analysis = StandardAnalysis.initialise({
        'train': 30,
        'val': 20,
        'test': 50
    })

    assert analysis.train == 0.30
    assert analysis.val == 0.20
    assert analysis.test == 0.50

def test_standard_analysis_two_args():
    analysis = StandardAnalysis.initialise({
        'train': 0.50,
        'val': 0.12
    })

    assert analysis.test == 0.38

def test_standard_analysis_one_args():
    analysis = StandardAnalysis.initialise({
        'train': 0.6
    })

    assert analysis.test == 0.20
    assert analysis.val == 0.20    

def test_standard_analysis_zero_args():
    analysis = StandardAnalysis.initialise({})

    assert analysis.test == 0.15
    assert analysis.val == 0.15
    assert analysis.train == 0.70
    