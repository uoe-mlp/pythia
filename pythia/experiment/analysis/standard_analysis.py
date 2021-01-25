from __future__ import annotations
from typing import Optional, List, Dict, cast

from .analysis import Analysis

from pythia.utils import ArgsParser


class StandardAnalysis(Analysis):

    DEFAULT_SPLIT: List[float] = [0.7, 0.15, 0.15]

    def __init__(self, train: float, val: float, test: float):
        self.train: float = train
        self.val: float = val
        self.test: float = test

    @staticmethod
    def initialise(params: Dict=None) -> Analysis:
        train: Optional[float] = ArgsParser.get_or_default(params if params is not None else {}, 'train', None)
        val: Optional[float] = ArgsParser.get_or_default(params if params is not None else {}, 'val', None)
        test: Optional[float] = ArgsParser.get_or_default(params if params is not None else {}, 'test', None)

        fractions: List[Optional[float]] = [train, val, test]

        available = sum([x for x in fractions if x is not None])
        if available > 1:
            fractions = [x / available if x is not None else None for x in fractions]

        if not any(fractions):
            # If non available, use default
            clean_fractions: List[float] = StandardAnalysis.DEFAULT_SPLIT
        elif not all(fractions):
            # If some available, fill missing following default proportions
            missing = 1 - available
            missing_defaults = sum([x for x, y in zip(StandardAnalysis.DEFAULT_SPLIT, fractions) if y is None])
            clean_fractions = [x if x is not None else y * missing / missing_defaults for x, y in zip(fractions, StandardAnalysis.DEFAULT_SPLIT)]
        else:
            # If all available, just use them. Had to use cast here cause linter could not pick the logic up. 
            clean_fractions = cast(List[float], fractions) 

        return StandardAnalysis(train=clean_fractions[0], val=clean_fractions[1], test=clean_fractions[2])

