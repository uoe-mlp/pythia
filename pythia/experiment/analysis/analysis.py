from __future__ import annotations
from typing import Dict, Optional
from abc import ABC


class Analysis(ABC):

    @staticmethod
    def initialise(params: Dict=None) -> Analysis:
        raise NotImplementedError