from __future__ import annotations
from pythia.journal.journal import Journal
import json
from typing import Optional, Dict, cast

from pythia.utils import ArgsParser
from pythia.agent import Agent, SupervisedAgent
from pythia.market import Market, DailyHistoricalMarket
from pythia.journal import Journal

from .experiment import Experiment
from .standard_experiment import StandardExperiment


class ExperimentParser(object):

    @staticmethod
    def parse(path) -> Experiment:
        with open(path, 'r') as fp:
            data = json.load(fp)

        market = ArgsParser.get_or_error(data, 'market')
        agent = ArgsParser.get_or_error(data, 'agent')
        analysis = ArgsParser.get_or_error(data, 'analysis')

        # ---- MARKET -----
        market_type = ArgsParser.get_or_default(market, 'type', 'daily-historical')

        if market_type.lower() == 'daily-historical':
            market_obj = DailyHistoricalMarket.initialise(cast(Dict, ArgsParser.get_or_default(market, 'params', {})))
        else:
            raise ValueError('Unexpected value for market_type: %s' % (market_type))
        input_size = market_obj.input_size
        output_size = market_obj.output_size

        # ---- AGENT -----
        agent_type = ArgsParser.get_or_default(agent, 'type', 'supervised')

        if agent_type.lower() == 'supervised':
            agent_obj = SupervisedAgent.initialise(input_size, output_size, cast(Dict, ArgsParser.get_or_default(agent, 'params', {})))
        else:
            raise ValueError('Unexpected value for agent_type: %s' % (agent_type))
        
        # ---- ANALYSIS -----
        experiment_type = ArgsParser.get_or_default(analysis, 'type', 'standard')

        if experiment_type.lower() == 'standard':
            experiment: Experiment = StandardExperiment.initialise(
                path=path,
                market=market_obj,
                agent=agent_obj,
                journal=Journal(),
                params=cast(Dict, ArgsParser.get_or_default(analysis, 'params', {}))
            )
        else:
            raise ValueError('Unexpected value for experiment_type: %s' % (experiment_type))

        return experiment