import numpy as np
from control.matlab import *
import matplotlib.pyplot as plt


class GradientIdentification():
    def __init__(self, filePath, initParams, KMAX = 100) -> None:
        # data load:
        self.filePath = filePath
        self.data = np.loadtxt(filePath, delimiter=',')
        self.t = self.data[:,0]
        self.h = self.data[:,1]
        self.size = len(self.h)
        self.state = round(np.mean(self.h[int(0.2*self.size): ]))
        # coeffs load:
        self.num = initParams[ :len(initParams)//2]
        self.den = initParams[len(initParams)//2: ]
        self.numDim = len(self.num)
        self.denDim = len(self.den)
        self.X = np.concatenate([self.num, self.den])
        self.J = np.zeros(len(self.X))
        # log data
        self.kmax = KMAX
        self.currentK = None
        self.currentEps = None

    def GraphOut(self,graphs,title='График',filename = 'График', name='untitled', folder = ''):
        plt.figure(1)
        plt.title(title)
        for graph in graphs:
            plt.plot(graph[0], graph[1],label=graph[2])
        plt.legend()
        plt.grid()
        plt.show()
        return
    
    def MakeCoefs(self, _X):
        n = np.zeros(self.numDim)
        d = np.zeros(self.denDim)
        for i in range(self.numDim):
            n[i] = _X[i]
        for i in range(self.denDim):
            d[i] = _X[self.numDim + i]
        return n, d

    def StepResponseData(self, _X):
        numm, denn = self.MakeCoefs(_X)
        sys_ = tf(numm, denn)
        t = np.linspace(0,self.t[-1],self.size)
        yM, t = step(sys_, T=t)
        return t, yM 

    def I(self,_X):
        res = 0.0
        t, yM = self.StepResponseData(_X)   
        for i in range(len(yM)):
            res += (yM[i] - self.h[i])**2
        return res

    def Gradient(self):
        eps = 1e-6
        Xi = np.copy(self.X)
        gr = np.zeros(len(Xi))
        for i in range(len(Xi)):
            if Xi[i] == 0:
                dx = eps
            else:
                dx = 0.001 * abs(Xi[i])
            Xi[i] += dx
            f1 = self.I(Xi)
            Xi[i] -= 2 * dx
            f2 = self.I(Xi)
            gr[i] = (f1 - f2) / (dx + dx)
            Xi[i] += dx
        return gr

    def f1(self, alpha):
        Xi = np.copy(self.X)
        for i in range(len(Xi)):
            Xi[i] = Xi[i] + alpha*(-self.J[i])
        return self.I(Xi)

    def min_fun_split(self, a, b):
        xi = 0
        k_max = 250
        
        ai = a
        bi = b
        l = (b - a)*0.01
        #print('L = ', l)
        while(abs(bi-ai)>=l and k_max > 0):
            dx = (bi-ai)*0.001
            
            pi = ((ai + bi) - dx) * 0.5
            qi = ((ai + bi) + dx) * 0.5
            
            fp = self.f1(pi)
            fq = self.f1(qi)
            
            if(fp < fq):
                bi = qi
            else:
                ai = pi
            k_max = k_max - 1
        xi = (ai+bi) / 2.0
        return xi

    def GetMinimization(self):
        I0 = self.I(self.X)
        eps = 0.001 * I0
        kmax = self.kmax
        k = 0
        epsCurrent = 1
        while k < kmax and epsCurrent > eps:
            a = -10
            b = 10
            self.J = self.Gradient()
            for i in range(len(self.X)):
                if self.J[i] != 0:
                    alf1 = -self.X[i] / self.J[i]
                    alf2 = 0.5 * self.X[i]/self.J[i]
                    a1 = min(alf1, alf2)
                    b1 = max(alf1, alf2)
                    a = max(a, a1)
                    b = min(b,b1)
            step = self.min_fun_split(a,b)
                
            for i in range(len(self.X)):
                self.X[i] = self.X[i] - step * self.J[i]

            epsCurrent = self.I(self.X)/I0
            k += 1
        
        self.currentEps = epsCurrent
        self.currentK = k

        return self.X
        
    def Info(self,tM=None, hM=None):
        self.GraphOut([[self.t,self.h,'Задание'],[tM,hM, 'Модель']])
        print('Вычисление завершено за ', self.currentK, ' иттераций')
        print('Ошибка относительно задания ', self.currentEps*100, ' %')
        print('Числитель ', self.MakeCoefs(self.X)[0])
        print('Знаменатель ',  self.MakeCoefs(self.X)[1])
        print('Значение функционала ', self.I(self.X))    

if __name__ == '__main__':
    print("ЗАПУСК НЕ КАК МОДУЛЯ")
    k = int(input("Введите максимальное кол-во итераций: "))
    Tparam = 0.05
    xi = 1.0
    # Последний коэффициент в числителе д.б. равен установившемуся значению ПХ  
    numerator = [0.0000001, 0.0000001, 2] # b1 b2 b3
    # Последний коэффициент в знаменателе д.б. равен одному
    denumerator = [0.000001, Tparam**2, 2*xi*Tparam, 1] # a0 a1 a2 a3
    m = GradientIdentification('test_data/h.txt', numerator+denumerator, KMAX = k)
    print(m.num, m.den)
    print("============Start Minim=============")
    coefs = m.GetMinimization()
    t,y = m.StepResponseData(coefs)
    m.Info(t, y)



