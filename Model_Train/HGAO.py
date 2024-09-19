# -*- coding: utf-8 -*-
"""
@name: HGAO
"""

import numpy as np
import DenseNet121 as md
import random
def fit_fun(param, X):  # Fitness function, here model training
    # Getting model parameters
    train_data = param['data']
    train_label = param['label']

    # Pass in the parameters to be optimized
    model = md.DenseNet121(X)
    res_model = model.model_create(X[0])
    history = res_model.fit(train_data, train_label, epochs=5, batch_size=16, validation_split=0.1)
    # Model parameters when the optimal loss value is obtained
    val_loss =min(history.history['val_loss'])
    val_loss = np.float64(val_loss)
    return val_loss

class HGAO:
    def __init__(self, model_param, hgao_param, beta1, beta2):
        # Getting model parameters
        self.model_param = model_param
        self.Max_iterations = hgao_param["Max_iter"]
        self.SearchAgents = hgao_param["SearchAgents"]
        self.lb = hgao_param['lb']
        self.ub = hgao_param['ub']
        self.dim = hgao_param['dim']

        # beta1 and beta2
        self.b1 = beta1
        self.b2 = beta2

    def INITIALIZATION(self):
        self.lowerbound = np.ones(self.dim) * self.lb  # Lower limit for variables
        self.upperbound = np.ones(self.dim) * self.ub  # Upper limit for variables
        X = np.random.uniform(low=self.lowerbound, high=self.upperbound,
                              size=(self.SearchAgents, self.dim))  # Initial population
        return X

    def newton_interpolation(self, x, y):
        x=np.array(x)
        y=np.array(y)
        n = len(x)
        lenx = len(x[0])
        l = []
        for k in range(lenx):
            c = y.copy()
            for j in range(1, n):
                for i in range(n - 1, j - 1, -1):
                    if x[i][k] != x[i - j][k]:
                        c[i] = (c[i] - c[i - 1]) / (x[i][k] - x[i - j][k])
                    else:
                        c[i] = 0
            l.append(c)
        l = np.array(l)
        return l

    def alpha_melanophore(self, fit, vMin, vMax):
        o = np.zeros(len(fit))
        for i in range(len(fit)):
            o[i] = (vMax - fit[i]) / (vMax - vMin)
        return o

    def getColor(self, colorPalette):
        band = True
        c1, c2 = 0, 0
        while band:
            c1 = colorPalette[np.random.randint(0, 30)]
            c2 = colorPalette[np.random.randint(0, 30)]
            if c1 != c2:
                band = False
        return c1, c2

    def getBinary(self):
        if np.random.random() < 0.5:
            val = 0
        else:
            val = 1
        return val

    def checkO(self, o):
        o = np.array(o)
        o[o < 0] = np.abs(o[o < 0])
        for i in range(len(o)):
            if o[i] < self.lb[i] or o[i] > self.ub[i]:
                if np.random.random()<0.5:
                    o[i] = self.X_best[i]
                else:
                    o[i] = o[i] = random.uniform(self.lb[i], self.ub[i])
        return o

    def R(self, NP):
        band = True
        r1, r2, r3, r4 = 0, 0, 0, 0
        while band:
            r1 = np.round(1 + (NP - 1) * np.random.rand())
            r2 = np.round(1 + (NP - 1) * np.random.rand())
            r3 = np.round(1 + (NP - 1) * np.random.rand())
            r4 = np.round(1 + (NP - 1) * np.random.rand())
            if r1 == NP: r1 -= 1
            if r2 == NP: r2 -= 1
            if r3 == NP: r3 -= 1
            if r4 == NP: r4 -= 1
            if (r1 != r2) and (r2 != r3) and (r1 != r3) and (r4 != r3) and (r4 != r2) and (r1 != r4):
                band = False
        return int(r1), int(r2), int(r3), int(r4)

    def mimicry(self, Xbest, X, Max_iter, SearchAgents_no, t):
        colorPalette = np.array(
            [0, 0.00015992, 0.001571596, 0.001945436, 0.002349794, 0.00353364, 0.0038906191, 0.003906191, 0.199218762,
             0.19999693, 0.247058824, 0.39999392, 0.401556397, 0.401559436, 0.498039216, 0.498046845, 0.499992341,
             0.49999997, 0.601556397, 0.8, 0.900000447, 0.996093809, 0.996109009, 0.996872008, 0.998039245, 0.998046875,
             0.998431444, 0.999984801, 0.999992371, 1])
        Delta = 2
        r1, r2, r3, r4 = self.R(SearchAgents_no)
        c1, c2 = self.getColor(colorPalette)
        o = Xbest + (Delta - Delta * t / Max_iter) * (
                c1 * ((np.sin(X[r1, :]) - np.cos(X[r2, :])) - ((-1) ** self.getBinary()) * c2 * np.cos(
            X[r3, :])) - np.sin(
            X[r4, :]))
        o = self.checkO(o)
        return o

    def shootBloodstream(self, Xbest, X, Max_iter, t):
        g = 0.009807  # 9.807 m/s2   a kilometros    =>  0.009807 km/s2
        epsilon = 1E-6
        Vo = 1  # 1E-2
        Alpha = np.pi / 2

        o = (Vo * np.cos(Alpha * t / Max_iter) + epsilon) * Xbest + (
                Vo * np.sin(Alpha - Alpha * t / Max_iter) - g + epsilon) * X
        o = self.checkO(o)
        return o

    def CauchyRand(self, m, c):
        cauchy = c * np.tan(np.pi * (np.random.rand() - 0.5)) + m
        return cauchy

    def randomWalk(self, Xbest, X):
        e = self.CauchyRand(0, 1)
        walk = -1 + 2 * np.random.rand()  # -1 < d < 1
        o = Xbest + walk * (0.5 - e) * X

        o = self.checkO(o)
        return o

    def Skin_darkening_or_lightening(self, Xbest, X, SearchAgents_no):
        darkening = [0.0, 0.4046661]
        lightening = [0.5440510, 1.0]

        dark1 = darkening[0] + (darkening[1] - darkening[0]) * np.random.rand()
        dark2 = darkening[0] + (darkening[1] - darkening[0]) * np.random.rand()
        light1 = lightening[0] + (lightening[1] - lightening[0]) * np.random.rand()
        light2 = lightening[0] + (lightening[1] - lightening[0]) * np.random.rand()

        r1, r2, r3, r4 = self.R(SearchAgents_no)

        if self.getBinary():
            o = Xbest + light1 * np.sin((X[r1, :] - X[r2, :]) / 2) - ((-1) ** self.getBinary()) * light2 * np.sin(
                (X[r3, :] - X[r4, :]) / 2)
        else:
            o = Xbest + dark1 * np.sin((X[r1, :] - X[r2, :]) / 2) - ((-1) ** self.getBinary()) * dark2 * np.sin(
                (X[r3, :] - X[r4, :]) / 2)
        o = self.checkO(o)
        return o

    def remplaceSearchAgent(self, Xbest, X, SearchAgents_no):
        band = True
        r1, r2 = 0, 0
        while band:
            r1 = np.round(1 + (SearchAgents_no - 1) * np.random.rand())
            r2 = np.round(1 + (SearchAgents_no - 1) * np.random.rand())
            if r1 == SearchAgents_no: r1 -= 1
            if r2 == SearchAgents_no: r2 -= 1
            if r1 != r2:
                band = False
        r1, r2 = int(r1), int(r2)

        o = Xbest + (X[r1, :] - ((-1) ** self.getBinary() * X[r2, :]) / 2)
        o = self.checkO(o)
        return o
    def run(self):
        self.X_best=self.lb

        # gao
        X = self.INITIALIZATION()
        fit = np.zeros(self.SearchAgents)
        for i in range(self.SearchAgents):
            fit[i] = fit_fun(self.model_param, X[i, :])
        self.xbest = 0
        self.fbest = 0

        # hloa
        Positions = np.random.uniform(self.lb, self.ub, (self.SearchAgents, self.dim))
        Positions = np.array(Positions)
        Fitness = np.zeros(self.SearchAgents)
        for i in range(Positions.shape[0]):
            Fitness[i] = fit_fun(self.model_param, Positions[i, :])

        minIdx = np.argmin(Fitness)
        self.vMin = Fitness[minIdx]
        self.theBestVct = Positions[minIdx, :]
        maxIdx = np.argmax(Fitness)
        self.vMax = Fitness[maxIdx]

        # Best position and fitness value for this iteration
        self.now_iter_x_best = self.theBestVct
        self.now_iter_y_best = self.vMin
        # The position of the previous best with the fitness value
        self.pre_iter_x_best = self.theBestVct
        self.pre_iter_y_best = self.vMin
        # The position points calculated by the quadratic interpolation formula and their fitness values
        self.qi_p_x_best = self.theBestVct
        self.qi_p_y_best = self.vMin

        Convergence_curve = np.zeros(self.Max_iterations)
        Convergence_curve[0] = self.vMin

        self.X_best = self.theBestVct


        alphaMelanophore = self.alpha_melanophore(Fitness, self.vMin, self.vMax)
        self.v = np.zeros((self.SearchAgents, self.dim))
        self.v = np.array(self.v)

        for t in range(self.Max_iterations):
            print(f"{t} iteration:")
            # update: BEST proposed solution
            Fbest = np.min(fit)
            blocation = np.argmin(fit)
            #  The position and fitness value of GAO global best
            self.xbest = X[blocation, :]
            self.fbest = Fbest
            if t == 0:
                # Best position and fitness value for this iteration
                self.now_iter_fbest,self.now_iter_xbest = self.fbest,self.xbest
                # The position of the previous best with the fitness value
                self.pre_iter_fbest, self.pre_iter_xbest = self.fbest,self.xbest

            for i in range(self.SearchAgents):
                # Phase 1: Attacking Termite Mounds (exploration phase)
                TM_location = np.where(fit < fit[i])[0]
                if TM_location.size == 0:
                    STM = self.xbest
                else:
                    K = np.random.randint(0, TM_location.size)
                    STM = X[TM_location[K], :]

                I = np.round(1 + np.random.rand())
                X_new_P1 = X[i, :] + np.random.rand() * (STM - I * X[i, :])
                X_new_P1 = self.checkO(X_new_P1)
                L = X_new_P1

                fit_new_P1 = fit_fun(self.model_param, L)
                if fit_new_P1 < fit[i]:
                    X[i, :] = X_new_P1
                    fit[i] = fit_new_P1

                # Phase 2: Digging in termite mounds (exploitation phase)
                X_new_P2 = X[i, :] + (1 - 2 * np.random.rand()) * (self.upperbound - self.lowerbound) / (t + 1)
                X_new_P2 = np.maximum(X_new_P2, self.lowerbound / (t + 1))
                X_new_P2 = np.minimum(X_new_P2, self.upperbound / (t + 1))

                X_new_P2 = self.checkO(X_new_P2)
                L = X_new_P2
                f_new = fit_fun(self.model_param, L)

                # The best fitness value and position in this population in this iteration are calculated for Newton interpolation
                self.now_iter_xbest = L
                self.now_iter_fbest = f_new

                x_known = [self.now_iter_xbest, self.pre_iter_xbest, self.xbest]
                y_known = [self.now_iter_fbest, self.pre_iter_fbest, self.fbest]

                c = self.newton_interpolation(x_known, y_known)
                x_min = []
                for i in range(len(c)):
                    a = c[i][2]
                    b = c[i][1]
                    if a == 0:
                        a = 1e-6
                    qi_x = -b / (2 * a)
                    x_min.append(qi_x)
                x_min=np.array(x_min)
                # Calculate the final result, Newton interpolation predicts the position and its corresponding fitness value
                self.ni_p_x_best = self.checkO(x_min)
                self.ni_p_y_best = fit_fun(self.model_param, self.ni_p_x_best)

                # The predicted value of Newton interpolation is compared with the best value of this iteration population,
                # and the best value is compared with the global best value to update the global best value
                if self.ni_p_y_best < self.now_iter_fbest:
                    self.now_iter_xbest = self.ni_p_x_best
                    self.now_iter_fbest = self.ni_p_y_best

                if self.now_iter_fbest <= fit[i]:
                    X[i, :] = self.now_iter_xbest
                    fit[i] = self.now_iter_fbest
                    if self.now_iter_fbest < self.fbest:
                        self.xbest = self.now_iter_xbest
                        self.fbest = self.now_iter_fbest
                self.pre_iter_xbest = self.now_iter_xbest
                self.pre_iter_fbest = self.now_iter_fbest

                # hloa

                if 0.5 < np.random.rand():
                    self.v[i, :] = self.mimicry(self.theBestVct, Positions, self.Max_iterations, self.SearchAgents, t)
                else:
                    if t % 2 == 1:
                        self.v[i, :] = self.shootBloodstream(self.theBestVct, Positions[i, :], self.Max_iterations, t)
                    else:
                        self.v[i, :] = self.randomWalk(self.theBestVct, Positions[i, :])
                Positions[maxIdx, :] = self.Skin_darkening_or_lightening(self.theBestVct, Positions,
                                                                         self.SearchAgents)

                self.v[i, :] = self.checkO(self.v[i, :])

                Fnew = fit_fun(self.model_param, self.v[i, :])

                if alphaMelanophore[i] <= 0.3:
                    x_new2 = self.remplaceSearchAgent(self.theBestVct, Positions, self.SearchAgents)
                    x_new2 = self.checkO(x_new2)
                    Fnew2 = fit_fun(self.model_param, x_new2)
                    if Fnew2 < Fnew:
                        Fnew = Fnew2
                        self.v[i, :] = x_new2

                self.now_iter_x_best = self.v[i, :]
                self.now_iter_y_best = Fnew

                # Now let's compute the lowest point of the quadratic interpolation and the corresponding fitness value.
                # pre_iter_x_best is the population best for the last iteration,
                # now_iter_x_best is the population best for the current iteration, and x_global_best is the global best
                # Calculating the molecular part
                numerator = (np.square(self.now_iter_x_best) - np.square(self.theBestVct)) * self.pre_iter_y_best + (
                        np.square(self.theBestVct) - np.square(self.pre_iter_x_best)) * self.now_iter_y_best + (
                                    np.square(self.pre_iter_x_best) - np.square(
                                self.now_iter_x_best)) * self.vMin
                # Calculate the denominator part
                denominator = 2 * ((self.now_iter_x_best - self.theBestVct) * self.pre_iter_y_best + (
                        self.theBestVct - self.pre_iter_x_best) * self.now_iter_y_best + (
                                           self.pre_iter_x_best - self.now_iter_x_best) * self.vMin)
                # Deal with the zero element in the denominator
                denominator = np.where(denominator == 0, 1e-6, denominator)
                # The final result is calculated, and the predicted position and its corresponding fitness value are quadratic interpolated
                self.qi_p_x_best = numerator / denominator
                self.qi_p_x_best = self.checkO(self.qi_p_x_best)
                self.qi_p_y_best = fit_fun(self.model_param, self.qi_p_x_best)

                # The predicted value of quadratic interpolation was compared with the best value of this iteration population,
                # and the best value was compared with the global best value to update the global best value
                if self.qi_p_y_best < self.now_iter_y_best:
                    self.now_iter_x_best = self.qi_p_x_best
                    self.now_iter_y_best = self.qi_p_y_best
                if self.now_iter_y_best <= Fitness[i]:
                    Positions[i, :] = self.now_iter_x_best
                    Fitness[i] = self.now_iter_y_best
                if self.now_iter_y_best <= self.vMin:
                    self.theBestVct = self.now_iter_x_best
                    self.vMin = self.now_iter_y_best

                self.pre_iter_x_best = self.now_iter_x_best
                self.pre_iter_y_best = self.now_iter_y_best

            maxIdx = np.argmax(Fitness)
            self.vMax = Fitness[maxIdx]
            alphaMelanophore = self.alpha_melanophore(Fitness, self.vMin, self.vMax)

            # a=b1*a1+b2*a2
            self.X_new = self.b1 * self.now_iter_xbest + self.b2 * self.now_iter_x_best
            self.X_new=self.checkO(self.X_new)
            self.Y_new = fit_fun(self.model_param, self.X_new)
            print("X_new: ", self.X_new)
            print("Y_new: ", self.Y_new)
            print("fbset: ", self.fbest)
            print("theBestVct: ", self.theBestVct)
            if self.Y_new < self.fbest:
                self.xbest, self.fbest = self.X_new, self.Y_new

            if self.Y_new < self.vMin:
                self.vMin, self.theBestVct = self.Y_new, self.X_new

            self.Y_best, self.X_best = (self.vMin, self.theBestVct) if self.vMin < self.fbest else (
            self.fbest, self.xbest)

            print("X_best", self.X_best)
            print("Y_best", self.Y_best)

        return self.Y_best, self.X_best