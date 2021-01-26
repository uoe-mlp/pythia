from __future__ import annotations 
import json
from typing import Optional, Dict, cast

from pythia.utils import ArgsParser
from pythia.agent import Agent, SupervisedAgent

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
        self._market: Optional[str] = None                  # TODO: put Market object here
        self._agent: Optional[Agent] = None
        self._journal: Optional[str] = None                 # TODO: put Journal object here
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
    def agent(self) -> Agent:
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
        market_type = ArgsParser.get_or_default(market, 'type', 'daily-historical')
        input_size = 10
        output_size = 20

        # ---- AGENT -----
        agent_type = ArgsParser.get_or_default(agent, 'type', 'supervised')

        if agent_type.lower() == 'supervised':
            self._agent = SupervisedAgent.initialise(input_size, output_size, cast(Dict, ArgsParser.get_or_default(agent, 'params', {})))
        else:
            raise ValueError('Unexpected value for experiment_type: %s' % (experiment_type))
