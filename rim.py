import numpy as np
import control as co

def load(filepath):
    data = np.loadtxt(filepath)
    x = data[:, 0]
    y = data[:, 1]
    return x, y
# TODO:
# [x] проверить интеграл, положил туда значения из h.txt и посмотреть, что будет
# [x] сравнить то, что выдало ВИМ и то, что выдал метод в этом файле
#   интеграл считается верно
class rim:
    def __init__(self, x=0, y=0, degree=0, eps=1e-3, u=None) -> None:
        self.eps = eps # погрешность для ВИМ
        self.x = x # вектор времени из временного ряда
        self.y = y # вектор значений выхода объекта
        self.u = u # входной вектор
        self.N = len(self.y) - 1
        self.degree = degree
        self.d_min = - np.log(eps) / x[-1]
        self.d_max = 10 * self.d_min
        self.d = self.get_d(
            d_min=self.d_min,
            d_max=self.d_max,
            N=len(self.x)
        )
        if self.u is None:
            self.F = self.get_F_active()
        else:
            self.F = self.get_F_passive()
        self.n = 2*degree # кол-во коэффициентов (в 2 раза больше, чем степень)

        self.A = self.get_A()
        self.num = None
        self.den = None
        self.solve()
    
    def solve(self):
        At = np.transpose(self.A)
        left = At @ self.A
        right = At @ self.F
        X = np.linalg.solve(left, right)
        X = np.append(X, 1)
        self.num = X[:len(X)//2]
        self.den = X[len(X)//2:]
        
        
    @property
    def model(self):
        if not(self.num is None and self.den is None):
            return co.tf(self.num, self.den)
        raise ValueError("нет числителя или знаменателя")
    
    def get_A(self):
        A = np.zeros((len(self.x), self.degree), dtype="double")
        B = np.copy(A)
        
        # заполняем первую половину столбцов:
        for i, k in zip(range(self.degree-1, -1, -1), range(self.degree)):
            A[:, k] = self.d**i

        # заполняем вторую половину столбцов:
        for i, k in zip(range(self.degree, 0, -1), range(self.degree)):
            B[:, k] = -self.F * self.d**i
        
        return np.hstack([A, B])

    def get_F_active(self):
        F = np.zeros(len(self.d))
        for i in range(len(self.d)):
            F[i] = self.d[i] * self.integral(self.y, self.x, self.d[i])
        return F
    
    def get_F_passive(self):
        F = np.zeros(len(self.d))
        for i in range(len(self.d)):
            y = self.integral(self.y, self.x, self.d[i])
            u = self.integral(self.u, self.x, self.d[i])
            F[i] =  y / u
        return F  
    
    def integral(self, f, t, d):
        """Численное вычисление интеграла для ВИМ с использованием numpy
        
        f - входные или выходные значения
        t - их время
        d - вещественная переменная
        """
        return np.sum(f * np.exp(-d*t)) * (t[1] - t[0])
    
    
    def get_d(self, d_min, d_max, N):
        return np.arange(
            start=d_min,
            stop=d_max,
            step=(d_max-d_min)/N,
        )
        
if __name__ == "__main__":
    x, y = load('h.txt')
    rim1 = rim(x, y, 4, 1e-3)
    print(rim1.F[0], rim1.F[-1])
    print(rim1.d[0], rim1.d[-1])
    print(rim1.A[0,0], rim1.A[-1,-1])
    rim1.solve()
    print(rim1.num)
    print(rim1.den)