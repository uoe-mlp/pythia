from __future__ import annotations 
import json
from pythia.journal.journal import Journal
from typing import Optional, Dict, cast

from pythia.utils import ArgsParser
from pythia.agent import Agent, SupervisedAgent
from pythia.market import Market, DailyHistoricalMarket
from pythia.journal import Journal

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
        self._market: Optional[Market] = None
        self._agent: Optional[Agent] = None
        self.journal: Journal = Journal()
        self.parse_args(**self.data)
        
    @property
    def analysis(self) -> Analysis:
        if self._analysis is not None:
            return self._analysis
        else:
            raise ValueError('Property analysis has not been set')

    @property
    def market(self) -> Market:
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
        market_type = ArgsParser.get_or_default(market, 'type', 'daily-historical')

        if market_type.lower() == 'daily-historical':
            self._market = DailyHistoricalMarket.initialise(cast(Dict, ArgsParser.get_or_default(market, 'params', {})))
        else:
            raise ValueError('Unexpected value for experiment_type: %s' % (experiment_type))
        input_size = self._market.input_size
        output_size = self._market.output_size

        # ---- AGENT -----
        agent_type = ArgsParser.get_or_default(agent, 'type', 'supervised')

        if agent_type.lower() == 'supervised':
            self._agent = SupervisedAgent.initialise(input_size, output_size, cast(Dict, ArgsParser.get_or_default(agent, 'params', {})))
        else:
            raise ValueError('Unexpected value for experiment_type: %s' % (experiment_type))
        