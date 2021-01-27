from __future__ import annotations
from typing import Dict, Optional
from abc import ABC, abstractmethod, abstractstaticmethod

from pythia.journal import Journal
from pythia.market import Market
from pythia.agent import Agent


class Experiment(ABC):

    def __init__(self, path: str, market: Market, agent: Agent, journal: Journal):
        self.path: str = path
        self.market: Market = market
        self.agent: Agent = agent
        self.journal: Journal = journal

    @staticmethod
    @abstractstaticmethod
    def initialise(path: str, market: Market, agent: Agent, journal: Journal, params: Dict=None) -> Experiment:
        raise NotImplementedError

    @abstractmethod
    def run(self):
        pass