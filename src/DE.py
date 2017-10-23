from __future__ import print_function, division

__author__ = 'amrit'

import sys
sys.dont_write_bytecode = True

import collections
from random import random, randint, uniform, seed, choice
import numpy as np

__all__ = ['DE']
Individual = collections.namedtuple('Individual', 'ind fit')

class DE(object):
    def __init__(self, F=0.3, CR=0.7, NP=10, GEN=5, Goal="Max", termination="Early"):
        self.F=F
        self.CR=CR
        self.NP=NP
        self.GEN=5
        self.GOAL=Goal
        self.termination=termination
        seed(1)
        np.random.seed(1)

    def initial_pop(self):
        return [{self.para_dic.keys()[i]:self.calls[i](self.bounds[i]) for i in range(self.para_len)} for _ in range(self.NP)]


    ## Need a tuple for integer and continuous variable but need the whole list for category
    def randomisation_functions(self):
        l=[]
        for i in self.para_category:
            if i=='integer':
                l.append(self._randint)
            elif i=='continuous':
                l.append(self._randuniform)
            elif i=='categorical':
                l.append(self._randchoice)
        self.calls=l

    ## Paras will be keyword with default values, and bounds would be list of tuples
    def solve(self, fitness, paras={}, bounds=[], category=[]):
        self.para_len=len(paras.keys())
        self.para_dic=paras
        self.para_category=category
        self.bounds=bounds
        self.randomisation_functions()
        initial_population=self.initial_pop()


        self.cur_gen = [Individual(ind, fitness(ind)) for ind in
                              initial_population]

        if self.termination=='Early':
            self.early_termination(fitness)

        else:
            self.late_termination(fitness)

    def early_termination(self,fitness):
        for _ in range(self.GEN):
            trial_generation = []

            for ind in self.cur_gen:
                v = self._extrapolate(ind)
                trial_generation.append(Individual(v, fitness(v)))

            current_generation = self._selection(trial_generation)
            self.cur_gen=current_generation

        best_index = self._get_best_index()
        return self.cur_gen[best_index], self.cur_gen

    def late_termination(self,fitness):
        pass

    def _extrapolate(self,ind):
        if (random.random() < self.CR):
            l = self.select3others()
            mutated=[]
            for x,i in enumerate(self.para_category):
                if i=='continuous':
                    mutated.append(l[0][x]+self.F*(l[1][x]-l[1][x]))
                else:
                    mutated.append(self.calls[x](self.bounds[x]))

            check_mutated = []
            for i in range(self.para_len):
                if self.para_category[i]=='continuous':
                    check_mutated.append(mutated[i])
                else:
                    check_mutated.append(max(self.bounds[i][0], min(mutated[i], self.bounds[i][1])))

            return check_mutated
        else:
            return ind.ind

    def _select3others(self):
        l=[]
        for a in range(self.para_len):
            randint(0, len(self.cur_gen) - 1)
            x1 = self.cur_gen
            l.append(x1.ind)
        return l

    def _selection(self, trial_generation):
        generation = []

        for a, b in zip(self.cur_gen, trial_generation):
            if self.GOAL=='Max':
                if a.fit >= b.fit:
                    generation.append(a)
                else:
                    generation.append(b)
            else:
                if a.fit <= b.fit:
                    generation.append(a)
                else:
                    generation.append(b)

        return generation

    def _get_best_index(self):
        if self.GOAL=='Max':
            best = -float("inf")
            max_fitness=np
            for i, x in enumerate(self.cur_gen):
                if x.fit >= max_fitness:
                    best = i
                    max_fitness = x.fit
            return best
        else:
            best = float("inf")
            max_fitness = np
            for i, x in enumerate(self.cur_gen):
                if x.fit <= max_fitness:
                    best = i
                    max_fitness = x.fit
            return best

    def _randint(self,a):
        return randint(*a)

    def _randchoice(self,a):
        return choice(a)

    def _randuniform(self,a):
        return uniform(*a)


def main():
    de=DE()
    pop=de.solve(main,paras={'k':10,'x':5}, bounds=[(1,10),(1,10)],category=['integer','integer'])
    print(pop)
main()