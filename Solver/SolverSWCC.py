import pandas as pd
import numpy as np
from scipy.optimize import minimize

class SolverSWCC:

    def __init__(self, suc, sat, function):
        self.function = function

        self.suc = suc
        self.st = sat

    def bestFit(self):
        initClass = self.function(self.suc, self.st)

        guess = initClass.guess()
        
        bestFit = minimize(initClass.objective, guess, method='SLSQP', constraints=initClass.constraints(), bounds= initClass.bounds())
        
        coef = bestFit.x
        r2 = np.cumsum((self.st - initClass.saturationCalcule(self.suc, coef)) ** 2)[-1]
        r2_guess = np.cumsum((self.st - initClass.saturationCalcule(self.suc, guess)) ** 2)[-1]

        return r2, coef, guess, r2_guess


