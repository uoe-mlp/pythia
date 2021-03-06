from __future__ import annotations
from typing import Dict, Optional
from abc import ABC, abstractmethod, abstractstaticmethod
from copy import deepcopy

from pythia.journal import Journal
from pythia.market import Market
from pythia.agent import Agent


class Experiment(ABC):

    def __init__(self, path: str, market: Market, agent: Agent, journal: Journal, benchmark: Optional[Agent], settings: Dict):
        self.path: str = path
        self.market: Market = market
        self.agent: Agent = agent
        self.journal: Journal = journal
        self.benchmark: Optional[Agent] = benchmark
        self.benchmark_journal: Optional[Journal] = deepcopy(journal) if benchmark else None
        self.settings: Dict = settings
        # TODO: add self.seed

    @staticmethod
    @abstractstaticmethod
    def initialise(path: str, market: Market, agent: Agent, journal: Journal, benchmark: Optional[Agent], settings: Dict, params: Dict=None) -> Experiment: pass

    @abstractmethod
    def run(self): pass