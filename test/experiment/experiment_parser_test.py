import os
from pythia.experiment.experiment import Experiment

from pythia.experiment import ExperimentParser


def test_experiment_parser_simple_experiment():
    experiment = ExperimentParser.parse(os.path.join('test', 'experiment', 'data', 'simple_experiment.json'))
    
    assert issubclass(type(experiment), Experiment)
    assert experiment.path == os.path.join('test', 'experiment', 'data', 'simple_experiment.json')
    assert experiment.test == 0.15
    assert experiment.val == 0.15
    assert experiment.train == 0.70
