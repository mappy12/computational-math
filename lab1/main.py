import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate


def task_1():
    print("Матрица 10х10 из вещественных единиц:", "\n")
    a = np.ones((10, 10), dtype=float)
    print(a, "\n")

    print("Единичная матрица 10х10:", "\n")
    b = np.identity(10, dtype=int)
    print(b, "\n")


def task_2():
    mtrx = np.array([[2, 1, 3, 6],
                     [4, 1, 3, 3],
                     [5, 2, 4, 1],
                     [5, 1, 2, 2]])

    det = np.linalg.det(mtrx)

    print("Матрица:", "\n", "\n", mtrx, "\n")
    print("Определитель: ", det, "\n")
    print(f"Определитель (в целочисленном виде): {int(det)}")


def task_3():
    a = np.random.randint(-3, 6, (4, 4))

    print("Случайная матрица из целых чисел из отрезка [-3, 5] размера 4х4:", "\n")
    print(a, "\n")

    b = np.random.randint(-10, 11, (4, 1))

    print("Вектор-столбец B подходящего размера: ", "\n")
    print(b, "\n")

    try:
        x = np.linalg.solve(a, b)
        x_round = np.round(x, 4)

        print("Решение системы:", "\n\n", x_round, "\n")

        print("----Проверка----", "\n")
        check = np.dot(a, x).astype(int)
        print("(A * X): ", "\n\n" , check, "\n")
        print("Вектор B:", "\n")
        print(b)
    except np.linalg.LinAlgError:
        print("Система не имеет единственного решения!")


def task_4():
    try:
        result, error = integrate.quad(lambda x: 1 / (1 - x**2) ** 0.5, -0.5, 0.5)
        print(f"∫_{-1/2}^{1/2} dx/√(1-x^2) = {result}")

    except Exception as e:
        print(f"Интеграл расходится: {e}")


def task_5():
    try:
        result, error = integrate.quad(lambda x: np.cos(2*x), 0, np.inf)
        print(f"∫_0^(inf) cos(2x) dx = {result}")

    except Exception as e:
        print(f"Интеграл расходится: {e}")

def task_6():
    plt.figure(figsize=(8,5))
    a

def main():
    print("###########Задание 1##########", "\n")

    task_1()

    print("###########Задание 2##########", "\n")

    task_2()

    print("###########Задание 3##########", "\n")

    task_3()

    print("###########Задание 4##########", "\n")

    task_4()

    print("###########Задание 4##########", "\n")

    task_5()

    print("###########Задание 6##########", "\n")

    task_6()

if __name__ == "__main__":
    main()