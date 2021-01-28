import os

from pythia.experiment import ExperimentParser


def test_naivetrader_linearpredictor_dailyhistoricalmarket():
    experiment = ExperimentParser.parse(os.path.join('test', 'integration', 'data', 'simple_experiment.json'))
    experiment.run()
    

