import numpy as np


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



def main():
    print("###########Задание 1##########", "\n")

    task_1()

    print("###########Задание 2##########", "\n")

    task_2()

    print("###########Задание 3##########", "\n")

    task_3()


if __name__ == "__main__":
    main()