import numpy as np
import random
from sklearn import tree


class Unimodal2P_SWCC:
    def __init__(self, suc=None, sat=None):
        self.saturationInitial = 1
        self.saturationFinale = 0
        self.suctionFinale = 10**6
        self.lambdazero = 0
        self.suc = suc
        self.sat = sat


    def d_j(self, suction_aJ1, suction_aJ):
        #weight factors
        return 2 * np.exp(1 / np.log(suction_aJ1/suction_aJ))

    def lambdaI(self, sat_iA, sat_i1A, suc_i1A, suc_iA):
        #desaturation slope
        return np.arctan2((sat_iA - sat_i1A), (np.log(suc_i1A/suc_iA)))

    def rI(self, lambdaI_1, lambdaI):
        #aperture angles tangents
        return np.tan((lambdaI_1 - lambdaI)/2)

    def tetaI(self, lambdaI_1, lambdaI):
        #hyperbolas rotation angles
        return -(lambdaI_1 + lambdaI)/2

    def sI(self, i, tetaI, rI, suction, suc_iA, a, s_iA):
        part1_num = np.tan(tetaI) * (1 + (rI**2)) * np.log(suction/suc_iA)
        part1_den = 1 - ((rI**2) * (np.tan(tetaI)**2))
        part1 = part1_num/part1_den

        part2 = ((-1)**i)

        part3_num = (1 + ((np.tan(tetaI))**2))
        part3 = part3_num/part1_den


        part4 = (rI**2)*((np.log(suction/suc_iA))**2)

        part5 = ((a**2) * part1_den) / part3_num

        part4_5 = np.sqrt(part4 + part5)

        part6 = s_iA

        return part1 + (part2*part3*part4_5) + part6

    def saturationCalcule(self, suction, params):
        sucB = params[0]
        sucRes = params[1]
        satRes = params[2]
        a = params[3]

        s1a = 1
        s2a = satRes
        s3a = 0

        suc1a = sucB
        suc2a = sucRes
        suc3a = 10**6

        d1 = self.d_j(suc2a, suc1a)

        lambda0 = 0
        lambda1 = self.lambdaI(s1a, s2a, suc2a, suc1a)
        lambda2 = self.lambdaI(s2a, s3a, suc3a, suc2a)


        r1 = self.rI(lambda0, lambda1)
        r2 = self.rI(lambda1, lambda2)

        teta1 = self.tetaI(lambda0, lambda1)
        teta2 = self.tetaI(lambda1, lambda2)

        s1 = self.sI(1, teta1, r1, suction, suc1a, a, s1a)
        s2 = self.sI(2, teta2, r2, suction, suc2a, a, s2a)

        part1 = (s1 - s2) / (1 + ((suction / np.sqrt(sucB*sucRes))**d1))


        return part1 + s2

    def objective(self, params):
        quadratic = np.cumsum((self.sat - self.saturationCalcule(self.suc, params)) ** 2)
        optimize = quadratic[-1]
        return optimize

    def constraints(self):
        constraints = ({'type': 'ineq', 'fun': lambda params: params[1]-params[0]})

        return constraints


    def bounds(self):
        return ((0.01, 10**6), (0.01, 10**6), (0.001, 0.999), (0.05, 0.15))


    def guess(self):
        guessList = [min(self.suc[self.sat < 1])]

        suc = self.suc[self.sat < 1][:-1]
        st = self.sat[self.sat < 1][:-1]*100

        guessDecison_suction = tree.DecisionTreeRegressor(max_depth=1)
        guessDecison_suction.fit(st.reshape(-1,1), suc)
        
        guessDecison_saturation = tree.DecisionTreeRegressor(max_depth=1)
        guessDecison_saturation.fit(suc.reshape(-1,1), st)

        point1 = guessDecison_saturation.predict(np.asarray([.1]).reshape(-1,1))[0]/100
        point2 = guessDecison_saturation.predict(np.asarray([10**6]).reshape(-1,1))[0]/100

        vert1_x = max(self.suc[self.sat>=point1])
        vert2_x = min(self.suc[self.sat<point1])
        vert1_y = min(self.sat[self.sat>=point1])
        vert2_y = max(self.sat[self.sat<point1])

        a0 = (vert1_y-vert2_y)/np.log(vert1_x/vert2_x)
        b0 = vert1_y - a0*np.log(vert1_x)

        coef1 = [a0, b0]

        x1 = np.exp((point1-coef1[1])/coef1[0])
        x1 = max(x1, self.suc[1])
        
        guessList.append(x1)
        guessList.append(point1)
        guessList.append(random.uniform(0.06, 0.14))
        
        return guessList