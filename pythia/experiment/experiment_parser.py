from __future__ import annotations
from pythia.journal.journal import Journal
import json
from typing import Optional, Dict, cast
import os

from pythia.utils import ArgsParser
from pythia.agent import SupervisedAgent
from pythia.market import DailyHistoricalMarket, LiveDailyHistoricalMarket
from pythia.journal import Journal

from .experiment import Experiment
from .standard_experiment import StandardExperiment


class ExperimentParser(object):

    @staticmethod
    def parse(path) -> Experiment:
        with open(path, 'r') as fp:
            data: Dict = json.load(fp)

        market = ArgsParser.get_or_error(data, 'market')
        agent = ArgsParser.get_or_error(data, 'agent')
        benchmark = ArgsParser.get_or_default(data, 'benchmark', {})
        analysis = ArgsParser.get_or_error(data, 'analysis')

        # ---- MARKET -----
        market_type = ArgsParser.get_or_default(market, 'type', 'daily-historical')

        if market_type.lower() == 'daily-historical':
            market_obj = DailyHistoricalMarket.initialise(cast(Dict, ArgsParser.get_or_default(market, 'params', {})))
        elif market_type.lower() == 'live-daily-historical':
            market_obj = LiveDailyHistoricalMarket.initialise(cast(Dict, ArgsParser.get_or_default(market, 'params', {})))
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
        
        # ---- BENCHMARK -----
        if benchmark == {}:
            calc_benchmark = False
        else:
            calc_benchmark = True
            benchmark_type = ArgsParser.get_or_default(benchmark, 'type', 'supervised')

            if benchmark_type.lower() == 'supervised':
                benchmark_obj = SupervisedAgent.initialise(input_size, output_size, cast(Dict, ArgsParser.get_or_default(benchmark, 'params', {})))
            else:
                raise ValueError('Unexpected value for agent_type: %s' % (agent_type))

        # ---- ANALYSIS -----
        experiment_type = ArgsParser.get_or_default(analysis, 'type', 'standard')
        experiment_folder = ArgsParser.get_or_default(analysis, 'folder', os.path.join('data', 'experiments', 'default'))

        if experiment_type.lower() == 'standard':
            experiment: Experiment = StandardExperiment.initialise(
                path=path,
                market=market_obj,
                agent=agent_obj,
                journal=Journal(experiment_folder=experiment_folder),
                settings=data,
                benchmark=benchmark_obj if calc_benchmark else None,
                params=cast(Dict, ArgsParser.get_or_default(analysis, 'params', {}))
            )
        else:
            raise ValueError('Unexpected value for experiment_type: %s' % (experiment_type))

        return experiment