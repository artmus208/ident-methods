


# TODO:
# [x]: Подготовить методы для тестирования
# [x]: Протестировать методы
# [ ]: Подготовить JSON-графика для Plotly


from enum import Enum
import control
import numpy as np
import scipy.linalg as linalg

from grad import Grad

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
        g = Grad(t, y, degree)