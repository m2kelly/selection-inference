import json
import os
import sys
from matplotlib import pyplot as plt
from numpy.core.numeric import NaN
from scipy.optimize import curve_fit
from scipy.stats import poisson
from collections import Counter
import pandas as pd
import numpy as np

class GraphPoint():
    def __init__(self, file, label=''):
        self.filename = file
        self.label = label
        self.mue_frwrd = None
        self.mue_bckwrds = None
        self.pop_size = None
        self.parse_mue_values()

        self.sim_results = self.parse_df()

        self.mean_sub_rate = None
        self.sub_rates = None
        self.analyze_results()


    def parse_mue_values(self):
        with open(self.filename) as file: params = file.readlines()[0]
        values = json.loads(params.replace("#", ''))

        self.mue_frwrd = values['muef']
        self.mue_bckwrds = values['mueb']
        self.pop_size = values['pop_size']

    def parse_df(self):

        df = pd.read_csv(self.filename, header=1)
        return df

    def analyze_results(self):
        self.sim_results['difference'] = (self.sim_results['generation_of_fixation'] -
                                         self.sim_results['last_fixation_generation'])
                                         #self.sim_results['mutation_origin_generation'])

        self.sub_rates = 1/self.sim_results['difference']
        self.mean_sub_rate = self.sub_rates.mean()
        self.sub_rates = list(self.sub_rates)

    def __eq__(self, other):
        if round(self.mue_frwrd/other.mue_frwrd, 4)!=1: return False
        if round(self.mue_bckwrds/other.mue_bckwrds, 4)!=1: return False
        if self.pop_size != other.pop_size: return False
        return True

    @staticmethod
    def get_list_of_points_as_df(points):
        #df = pd.DataFrame(columns=['sub_rate', 'pop_size'])
        df = pd.DataFrame()
        rows = []
        for i, point in enumerate(points):
            rows.append(pd.Series({'sub_rate': point.sub_rate, 'pop_size': point.pop_size,
                        'obs_mu': point.mue_frwrd}))
            #df[i] = row
            #df = pd.concat([df,row], axis=1)
        df = pd.DataFrame(rows)
        return df

    @staticmethod
    def get_list_df_from_subs_rates(sub_rates, pop_size):
        #df = pd.DataFrame()
        rows = []
        for i, sub_rate in enumerate(sub_rates):
            rows.append(pd.Series({'sub_rate': sub_rate, 'pop_size': pop_size}))

            #df = pd.concat([df,row], axis=1)
        df = pd.DataFrame(rows)
        return df

class GraphPointAbstract():
    def __init__(self, mue_frwrd, mue_bckwrds, pop_size, label=None, mean_sub_rate=None, selection_coeff=0):
        self.mue_frwrd = mue_frwrd
        self.mue_bckwrds = mue_bckwrds
        self.pop_size = pop_size
        self.label = label
        self.mean_sub_rate = mean_sub_rate
        self.selection_coeff = selection_coeff
        self.hash = hash((self.mue_frwrd, self.mue_bckwrds, self.pop_size, self.selection_coeff))

    def __eq__(self, __o: object) -> bool:
        if self.mue_frwrd != __o.mue_frwrd: return False
        if self.mue_bckwrds != __o.mue_bckwrds: return False
        if self.pop_size != __o.pop_size: return False
        if self.selection_coeff != __o.selection_coeff: return False
        return True
