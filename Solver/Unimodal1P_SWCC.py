import numpy as np
import random
from sklearn import tree

class Unimodal1P_SWCC:
    def __init__(self, suc=None, sat=None):
        self.suc = suc
        self.sat = sat


    def saturationCalcule(self, suction, params):
        sucB = params[0]
        a = params[1]

        lambda_ = np.arctan2(1, (np.log((10**6)/sucB)))
        teta = -lambda_/2
        r = np.tan(-teta)

        part1_num = np.tan(teta) * (1 + (r**2)) * np.log(suction/sucB)
        part1_den = 1 - ((r**2) * (np.tan(teta)**2))
        part1 = part1_num/part1_den


        part2_num = (1 + ((np.tan(teta))**2))
        part2 = part2_num/part1_den


        part3 = (r**2)*((np.log(suction/sucB))**2)

        part4 = ((a**2) * part1_den) / part2_num

        part3_4 = np.sqrt(part3 + part4)


        return part1 - (part2*part3_4) + 1


    def objective(self, params):
        quadratic = np.cumsum((self.sat - self.saturationCalcule(self.suc, params)) ** 2)
        optimize = quadratic[-1]
        return optimize


    def bounds(self):
        return ((0.01, 10**6), (0.02, 0.20))

    def constraints(self):
        constraints = ({'type': 'ineq', 'fun': lambda params: params[1]-0})

        return constraints

    def guess(self):
        guessList = [min(self.suc[self.sat < 1])]
        guessList.append(random.uniform(0.03, 0.19))
        
        return guessList