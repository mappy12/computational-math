import numpy as np


def main():
    print("###########Задание 1##########", "\n")

    print("Матрица 10х10 из вещественных единиц:", "\n")
    a = np.ones((10,10), dtype=float)
    print(a, "\n")

    print("Единичная матрица 10х10:", "\n")
    b = np.identity(10, dtype=int)
    print(b, "\n")


if __name__ == "__main__":
    main()