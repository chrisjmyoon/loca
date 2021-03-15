import pdb
import numpy as np
import pandas as pd
import os
import torch 
from bisect import bisect_left
from datetime import datetime
import functools

from . import base
import datasets
import agents

class SimCLRExp(base.BaseExp):
    def __init__(self, config, k):
        super().__init__(config, k)
           
    def get_agent(self):
        agent = agents.simclr.SimCLRAgent(self.config, self.loaders, self.k)
        return agent