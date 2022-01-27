import numpy as np
import random
from sklearn import tree


class BimodalSWCC:
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
        #hyperboles rotation angles
        return -(lambdaI_1 + lambdaI)/2


    def sI(self, i, tetaI, rI, suction, suc_iA, a, s_iA):
        part1_num = (np.tan(tetaI) * (1 + (rI**2)) * np.log(suction/suc_iA))
        part1_den = (1 - ((rI**2)*(np.tan(tetaI)**2)))
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
        s1a = 1
        s2a = params[2]#satRes1
        s3a = params[4]#satB
        s4a = params[6]#satRes2
        s5a = 0

        suc1a = params[0]#sucB1
        suc2a = params[1]#sucRes1
        suc3a = params[3]#sucB2
        suc4a = params[5]#sucRes2
        suc5a = 10**6

        a = params[7]

        d1 = self.d_j(suc2a, suc1a)
        d2 = self.d_j(suc3a, suc2a)
        d3 = self.d_j(suc4a, suc3a)

        lambda0 = 0
        lambda1 = self.lambdaI(s1a, s2a, suc2a, suc1a)
        lambda2 = self.lambdaI(s2a, s3a, suc3a, suc2a)
        lambda3 = self.lambdaI(s3a, s4a, suc4a, suc3a)
        lambda4 = self.lambdaI(s4a, s5a, suc5a, suc4a)

        r1 = self.rI(lambda0, lambda1)
        r2 = self.rI(lambda1, lambda2)
        r3 = self.rI(lambda2, lambda3)
        r4 = self.rI(lambda3, lambda4)

        teta1 = self.tetaI(lambda0, lambda1)
        teta2 = self.tetaI(lambda1, lambda2)
        teta3 = self.tetaI(lambda2, lambda3)
        teta4 = self.tetaI(lambda3, lambda4)

        s1 = self.sI(1, teta1, r1, suction, suc1a, a, s1a)
        s2 = self.sI(2, teta2, r2, suction, suc2a, a, s2a)
        s3 = self.sI(3, teta3, r3, suction, suc3a, a, s3a)
        s4 = self.sI(4, teta4, r4, suction, suc4a, a, s4a)

        part1 = (s1 - s2) / (1 + ((suction / np.sqrt(suc1a*suc2a))**d1))
        part2 = (s2 - s3) / (1 + ((suction / np.sqrt(suc2a * suc3a)) ** d2))
        part3 = (s3 - s4) / (1 + ((suction / np.sqrt(suc3a * suc4a)) ** d3))

        return part1 + part2 + part3 + s4


    def objective(self, params):
        quadratic = np.cumsum((self.sat - self.saturationCalcule(self.suc, params))**2)
        optimize = quadratic[-1]
        return optimize
       

    def constraints(self):
        constraints = ({'type': 'ineq', 'fun': lambda params: params[1] - params[0]},
                       {'type': 'ineq', 'fun': lambda params: params[3] - params[1]},
                       {'type': 'ineq', 'fun': lambda params: params[5] - params[3]},
                       {'type': 'ineq', 'fun': lambda params: params[2] - params[4]},
                       {'type': 'ineq', 'fun': lambda params: params[4] - params[6]})
                       
        return constraints


    def bounds(self):
        return ((0.01, 10**6), (0.01, 10**6), (0.001, 0.999), (0.01, 10**6), (0.001, 0.99), (0.01, 10**6), (0.001, 0.99), (0.02, 0.079))


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


        vert3_x = max(self.suc[self.sat>point2])
        vert4_x = min(self.suc[self.sat<=point2])
        vert3_y = min(self.sat[self.sat>point2])
        vert4_y = max(self.sat[self.sat<=point2])

        a0 = (vert3_y-vert4_y)/np.log(vert3_x/vert4_x)
        b0 = vert3_y - a0*np.log(vert4_x)

        coef2 = [a0, b0]

        x2 = np.exp((point2-coef2[1])/coef2[0])
        x2 = min(x2, max(self.suc))

        x3 = random.uniform(x2, 10**5)

        if x3<x2:
            x3 = random.uniform(x2, 5*(10**5))

        point3 = random.uniform(point2, 0.08)

        a = random.uniform(0.03, 0.079)
        
        guessList.append(x1)
        guessList.append(point1)
        guessList.append(x2)
        guessList.append(point2)
        guessList.append(x3)
        guessList.append(point3)
        guessList.append(a)

        return guessList