import os
import pytest

from pythia.experiment import ExperimentParser


def test_experiment_parser_simple_experiment():
    experiment = ExperimentParser(os.path.join('test', 'experiment', 'data', 'simple_experiment.json'))
    
    assert isinstance(experiment, ExperimentParser)
    assert experiment.loaded
    assert experiment.path == os.path.join('test', 'experiment', 'data', 'simple_experiment.json')
    assert experiment.analysis.test == 0.15
    assert experiment.analysis.val == 0.15
    assert experiment.analysis.train == 0.70
