class BankAccount:
    # Магический метод для инициализации объекта
    def __init__(self, name, balance=0.0):
        self._name = name
        self._balance = balance  # Применение концепции инкапсуляции

    # Строковое представление объекта
    def __str__(self):
        return f"BankAccount({self._name}, {self._balance})"

    # Представление объекта для отладки
    def __repr__(self):
        return self.__str__()

    # Свойство для чтения баланса
    @property
    def balance(self):
        return self._balance

    # Методы класса
    def deposit(self, amount):
        self._balance += amount
        return self._balance

    def withdraw(self, amount):
        if amount > self._balance:
            raise ValueError("Insufficient balance!")
        self._balance -= amount
        return self._balance

    # Магический метод для сложения двух банковских счетов
    def __add__(self, other):
        if isinstance(other, BankAccount):
            return BankAccount(self._name + "&" + other._name, self._balance + other._balance)
        raise TypeError("Operands don't match!")

    # Статический метод
    @staticmethod
    def validate_amount(amount):
        if not isinstance(amount, (int, float)):
            raise ValueError("Invalid amount!")
        return amount

    # Классовый метод
    @classmethod
    def create_with_zero_balance(cls, name):
        return cls(name, 0.0)
