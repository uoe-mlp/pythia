from __future__ import annotations 
import json
from typing import Optional, Dict, cast

from pythia.utils import ArgsParser

from .analysis import Analysis
from .analysis import StandardAnalysis


class ExperimentParser(object):

    def __init__(self, path: str):
        """[summary]

        Args:
            path (str): [description]
        """
        self.path: str = path
        self.loaded: bool = False
        self.data: Optional[Dict] = None

        with open(path, 'r') as fp:
            self.data = json.load(fp)
            self.loaded = True

        self._analysis: Optional[Analysis] = None
        self._market: Optional[str] = None
        self._agent: Optional[str] = None
        self.parse_args(**self.data)
        
    @property
    def analysis(self) -> Analysis:
        if self._analysis is not None:
            return self._analysis
        else:
            raise ValueError('Property analysis has not been set')

    @property
    def market(self) -> str:
        if self._market is not None:
            return self._market
        else:
            raise ValueError('Property market has not been set')

    @property
    def agent(self) -> str:
        if self._agent is not None:
            return self._agent
        else:
            raise ValueError('Property agent has not been set')


    def parse_args(self, market: Dict={}, analysis: Dict={}, agent: Dict={}) -> None:
        # ---- ANALYSIS -----
        experiment_type = ArgsParser.get_or_default(analysis, 'type', 'standard')

        if experiment_type.lower() == 'standard':
            self._analysis = StandardAnalysis.initialise(cast(Dict, ArgsParser.get_or_default(analysis, 'params', {})))
        else:
            raise ValueError('Unexpected value for experiment_type: %s' % (experiment_type))

        # ---- MARKET -----
        # TODO: define market instrument

        # ---- AGENT -----
        # TODO: define agent instrument 
