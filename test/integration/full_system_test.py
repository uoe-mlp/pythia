import os

from pythia.experiment import ExperimentParser


def test_naivetrader_linearpredictor_dailyhistoricalmarket():
    experiment = ExperimentParser.parse(os.path.join('test', 'integration', 'data', 'simple_experiment.json'))
    experiment.run()
    assert True
    
def test_naivetrader_linearpredictor_livedailyhistoricalmarket():
    experiment = ExperimentParser.parse(os.path.join('test', 'integration', 'data', 'simple_experiment_live.json'))
    experiment.run()
    assert True
    
def test_chalvatzis():
    experiment = ExperimentParser.parse(os.path.join('test', 'integration', 'data', 'chalvatzis.json'))
    experiment.run()
    assert True