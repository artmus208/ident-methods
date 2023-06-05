


# TODO:
# [x]: Подготовить методы для тестирования
# [ ]: Протестировать методы


from enum import Enum
import control
import numpy as np
import scipy.linalg as linalg

class Methods(Enum):
    lsm = 1,
    vim = 2,
    grad = 3


class IdentifyIt:
    """IdentifyIt(x, y, degree, method)

    Класс для идентификации динамичесих систем. 

    Экспериментальные данные - переходная характеристика объекта.

    Параметры
    ---------
    x : array_like, or list of array_like
        Экспериментальные данные по времени переходной хар-ки
    y : array_like, or list of array_like
        Экспериментальные данные по значения переходной хар-ки
    degree : int
        Задает струткуру ПФ. 
        Максимальная степень числителя и знаменателя.
    method : int
        Классификатор метода. На момент 04.06.2023 доступно три метода:

        1 - МНК; 
        2 - ВИМ; 
        3 - Градиентный метод.

    """
    @property
    def method(self):
        return self._method
    
    @method.setter
    def method(self, value):
        if not (1 <= value <= 3):
            raise ValueError("Unavaible method!")
        self._method = value

    @property
    def num(self):
        return self._num
    
    @num.setter
    def num(self, value):
        self._num = value
        
    @property
    def den(self):
        return self._den
    
    @den.setter
    def den(self, value):
        self._den = value

    @property
    def dt(self):
        self._dt = self.x[2] - self.x[1]
        return self._dt

    @property
    def model(self):
        if self.iscont:
            self._model = control.tf(self.num, self.den)
        else:
            self._model = control.tf(self.num, self.den, self.dt)
        return self._model

    @property
    def y_m(self):
        self.x_m, self._y_m = control.step_response(self.model, np.linspace(self.x[0], self.x[-1], len(self.x)))
        return self._y_m

    @property
    def error(self):
        if isinstance(self.y, np.ndarray) and isinstance(self.y_m, np.ndarray):
            return np.sum((self.y_m - self.y)**2)
        raise TypeError("Type mismath")

    def __repr__(self) -> str:
        return f"num:{self.num}\ndenum:{self.den}\nerror:{self.error}\nIs cont.:{self.iscont}"

    def __init__(self, x:list, y:list, degree:int, method:int):
        self.x = x
        self.y = y
        self.degree = degree
        self.method = method
        self.iscont = True
        self.run_method()


    def run_method(self):
        print('Running method...')
        match self.method:
            case 1:
                print('LSM is runing...')
                self.lsm(self.x, self.y, self.degree)
            case 2:
                print('VIM is runing...')
                self.vim(self.x, self.y, self.degree)
            case 3:
                print('GRAD is runing...')
                self.grad(self.x, self.y, self.degree)
            case _:
                raise ValueError("Wrong Method Choosen")

    def load_xy(self, file_path):
        """Загрузка экспериментальных данных из файла"""
        self.x, self.y = np.loadtxt(file_path, delimiter=',', unpack=True)


    def validate_before_ident(self):
        res = False
        if self.x and self.y and self.degree:
            res = True
        return res
    

    def lsm(self, t, y, degree):
        N = len(t)
        phi = np.zeros((N-1,2*degree))
        for i in range(N-1):
            for j in range(degree):
                if i - j <= 0:
                    phi[i][j] = 0
                else:
                    phi[i][j] = -y[i-j]
        for i in range(N-1):
            for j in range(degree,2*degree):
                if i + degree - j <= 0:
                    phi[i][j] = 0
                else:
                    phi[i][j] = 1
        Y = y[1:N]
        B = np.dot(phi.T,Y)
        A = np.dot(phi.T,phi)
        x = np.linalg.solve(A,B)
        x = x.tolist()
        x.insert(0, 1)
        x.insert(degree+1, 0)
        self.num = x[degree+1:] 
        self.den = x[:degree+1]
        self.iscont = False
        return [self.num, self.den]


    def vim(self, t, y, degree=1):
        eps = 0.001
        d_min = -np.log(eps) / t[-1]
        n = len(y)
        m = degree * 2
        d = np.zeros(n)
        d_max = d_min * 10
        for i in range(len(d)):
            d[i] = d_min + i * (d_max - d_min)/len(d)
        Wd = np.zeros(len(d))
        for i in range(len(d)):
            s = 0
            for j in range(n):
                s += y[j] * np.exp(-d[i] * t[j]) * 0.01
            Wd[i] = d[i] * s

        f = np.zeros((n, m), dtype = 'double')
        for i in range(n):
            for j in range(1, degree+1):
                f[i][j-1] = d[i] ** (degree - j)
            for j in range(1, degree+1):
                f[i][degree + j - 1] = (-d[i] ** (degree - j + 1)) * Wd[i]
        
        f_t = np.transpose(f)
        left = np.dot(f_t, f)
        right = np.dot(f_t, Wd)
        A = linalg.solve(left, right)
        A = list(A)
        A.append(1)
        self.num = A[:degree]
        self.den = A[degree:]
        return [self.num, self.den]

    def grad(self, t, y, degree=1):
        X = [np.ones((1, degree + 1))[0], np.ones((1, degree + 1))[0]]
        X[0][0] = 1e-6

        def tfSysStep(_sys):
            return control.step_response(_sys, np.linspace(0, 10, 100))

        def error(_X):
            _sys = control.tf(_X[:len(X[0])], _X[len(X[0]):])
            t_pred, y_pred = control.step_response(_sys, np.linspace(t[0], t[-1], len(t)))

            if (len(y_pred) != len(y)):
                return -1

            er = 0.0
            for i in range(len(y)):
                er = er + (y[i] - y_pred[i])**2
            return er

        def grad(_X):
            G = np.zeros(len(_X))
            XX = np.copy(_X)
            eps = 0.0001
            for i, x in enumerate(XX):
                dx = 0.00001
                if x != 0.0:
                    dx = x * eps

                XX[i] = _X[i] + dx
                fb = error(XX)

                XX[i] = _X[i] - dx
                fa = error(XX)

                XX[i] = _X[i]
                G[i] = (fb-fa)/(2*dx)
            return G

        def min_fun_split(a, b, _X, _Gr):
            xi = 0
            k_max = 10

            ai = a
            bi = b
            l = (b - a)*0.01

            Xi = np.copy(_X)

            while (abs(bi-ai) >= l and k_max > 0):
                dx = (bi-ai)*0.001

                pi = ((ai + bi) - dx) * 0.5
                qi = ((ai + bi) + dx) * 0.5

                for i, x in enumerate(Xi):
                    Xi[i] = _X[i] - pi * Gr[i]
                fp = error(Xi)

                for i, x in enumerate(Xi):
                    Xi[i] = _X[i] - qi * Gr[i]
                fq = error(Xi)

                if (fp < fq):
                    bi = qi
                else:
                    ai = pi

                k_max = k_max - 1
            xi = (ai+bi) / 2.0
            return xi

        N = 0
        X_i = np.concatenate([X[0], X[1]])
        f_i = error(X_i)

        N_max = 300
        while (N < N_max):
            Gr = grad(X_i)
            a = -100.0
            b = 100.0
            for i in range(len(X_i)):
                if (Gr[i] != 0 and X_i[i] != 0):
                    alf1 = -X_i[i]/Gr[i]
                    alf2 = 0.5*X_i[i]/Gr[i]
                    a1 = min(alf1, alf2)
                    b1 = max(alf1, alf2)
                    a = max(a, a1)
                    b = min(b, b1)

            step = min_fun_split(a, b, X_i, Gr)

            for i, x in enumerate(X_i):
                X_i[i] = X_i[i] - step * Gr[i]

            f_i = error(X_i)
            N = N + 1
        self.num = X_i[:len(X[0])]
        self.den = X_i[len(X[0]):]
        return [self.num, self.den]