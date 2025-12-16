import numpy as np
import math


def givens_rotation(a, b):
    """Возвращает косинус и синус поворота Гивенса,
       который обращает элемент b в ноль."""
    if b == 0:
        return 1.0, 0.0
    r = math.hypot(a, b)
    c = a / r
    s = -b / r
    return c, s

def givens_qr(A):
    """QR-разложение методом Гивенса."""
    A = A.astype(float)
    m, n = A.shape
    Q = np.eye(m)
    R = A.copy()

    for j in range(n):
        for i in range(m - 1, j, -1):

            a = R[i - 1, j]
            b = R[i, j]

            if abs(b) < 1e-15:
                continue

            c, s = givens_rotation(a, b)

            G = np.eye(m)
            G[i - 1, i - 1] = c
            G[i - 1, i] = -s
            G[i, i - 1] = s
            G[i, i] = c

            R = G @ R
            Q = Q @ G.T

    return Q, R

def back_substitution(R, b):
    """Решение верхнетреугольной системы R x = b."""
    n = R.shape[0]
    x = np.zeros(n, dtype=float)
    for i in range(n-1, -1, -1):
        if abs(R[i,i]) < 1e-15:
            raise np.linalg.LinAlgError("Zero diagonal element in back substitution")
        x[i] = (b[i] - np.dot(R[i, i+1:], x[i+1:])) / R[i,i]
    return x

def back_substitution(R, b):
    """Обратный ход метода Гаусса."""
    n = R.shape[0]
    x = np.zeros(n, dtype=float)
    for i in range(n - 1, -1, -1):
        if abs(R[i, i]) < 1e-15:
            raise ValueError("Ошибка: диагональный элемент R равен нулю (невозможно выполнить обратный ход).")
        x[i] = (b[i] - np.dot(R[i, i + 1:], x[i + 1:])) / R[i, i]
    return x


# 1. QR-разложение 3×3 матрицы и обнуление элемента a32

np.random.seed(0)
A1 = np.random.randint(-8, 11, (3, 3)).astype(float)

print("\n=== ЗАДАНИЕ 1 ===\n")
print("Случайная матрица A:")
print(A1)

c, s = givens_rotation(A1[1, 1], A1[2, 1])

G = np.eye(3)
G[1, 1] = c
G[1, 2] = -s
G[2, 1] = s
G[2, 2] = c

A1_after = G @ A1

print("\nМатрица G, которая зануляет элемент a32:")
print(G)

print("\nРезультат G * A:")
print(A1_after)

Q1, R1 = givens_qr(A1)

print("\nQR-разложение методом Гивенса:")
print("Матрица Q:")
print(Q1)

print("\nМатрица R:")
print(R1)

print("\nПроверка A - Q*R (должно быть близко к нулю):")
print(np.max(np.abs(A1 - Q1 @ R1)))


# 2. Решение СЛАУ 4×4 методом QR Гивенса

print("\n=== ЗАДАНИЕ 2 ===\n")

A2 = np.array([
    [4.3, -12.1, 23.2, -14.1],
    [2.4, -4.4, 3.5, 5.5],
    [5.4, 8.3, -7.4, -12.7],
    [6.3, -7.6, 1.34, 3.7]
], dtype=float)

b2 = np.array([15.5, 2.5, 8.6, 12.1], dtype=float)

print("Матрица A:")
print(A2)
print("\nВектор b:")
print(b2)

Q2, R2 = givens_qr(A2)
y2 = Q2.T @ b2
x2 = back_substitution(R2, y2)

print("\nРешение x (метод Гивенса):")
print(x2)

x2_np = np.linalg.solve(A2, b2)
print("\nПроверка numpy.linalg.solve:")
print(x2_np)


# 3. Метод простых итераций (Якоби)

print("\n=== ЗАДАНИЕ 3 ===")

A3 = np.array([
    [3.2, -2.5, 3.7],
    [0.5, 0.34, 1.7],
    [1.6, 2.3, -1.5]
], dtype=float)

b3 = np.array([6.5, -0.24, 4.3], dtype=float)


def jacobi(A, b, tol=1e-3, maxiter=200):
    """Метод Якоби."""
    n = A.shape[0]
    x = np.zeros(n, dtype=float)
    D = np.diag(A)
    R = A - np.diagflat(D)

    if np.any(np.abs(D) < 1e-12):
        raise ValueError("Ошибка: нулевой диагональный элемент, метод Якоби невозможен.")

    print("\nТаблица итераций (каждая строка — новое приближение x):")

    for k in range(maxiter):
        x_new = (b - R @ x) / D
        print(f"Итерация {k+1}: {x_new}")

        if np.linalg.norm(x_new - x, np.inf) < tol:
            print("\nМетод Якоби сошёлся.")
            return x_new, k + 1

        x = x_new

    print("\nМетод Якоби достиг максимального числа итераций.")
    return x, maxiter


diag_dom = np.all(np.abs(np.diag(A3)) > np.sum(np.abs(A3), axis=1) - np.abs(np.diag(A3)))

print("\nПроверка диагонального преобладания:", diag_dom)

x3, it3 = jacobi(A3, b3)

print("\nРешение методом Якоби:")
print(x3)
print("Число итераций:", it3)


# 4. Нелинейное уравнение

print("\n=== ЗАДАНИЕ 4 ===\n")

def f4(x): return 2*x**3 + 3.41*x**2 - 23.74*x + 2.95
def df4(x): return 6*x**2 + 2*3.41*x - 23.74


def bisection(f, a, b, tol=1e-3):
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError("На отрезке нет смены знака — бисекция невозможна.")

    for _ in range(100):
        c = (a + b) / 2
        fc = f(c)
        if abs(fc) < tol or abs(b - a) < tol:
            return c
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
    return c


def newton(f, df, x0, tol=1e-4):
    x = x0
    for _ in range(100):
        dfx = df(x)
        if abs(dfx) < 1e-12:
            raise ValueError("Производная почти ноль — метод Ньютона невозможен.")
        x_new = x - f(x) / dfx
        if abs(x_new - x) < tol:
            return x_new
        x = x_new
    return x

def secant(f, x0, x1, tol=1e-5):
    for _ in range(200):
        f0, f1 = f(x0), f(x1)
        if abs(f1 - f0) < 1e-15:
            raise ValueError("Разность функций почти ноль — метод секущих не работает.")
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        if abs(x2 - x1) < tol:
            return x2
        x0, x1 = x1, x2
    return x1


print("Корни ищем на интервале [-5, 4]")
xs = np.linspace(-5, 4, 2000)
ys = f4(xs)

brackets = []
for i in range(len(xs) - 1):
    if ys[i] * ys[i + 1] < 0:
        brackets.append((float(xs[i]), float(xs[i + 1])))

print("Найденные отрезки со сменой знака:")
print(brackets)

roots = []
for (a, b) in brackets:
    rb = bisection(f4, a, b)
    rn = newton(f4, df4, (a + b) / 2)
    rs = secant(f4, a, b)
    roots.append((rb, rn, rs))

print("\nКорни уравнения (бисекция, Ньютон, секущие):")

for r_bis, r_newt, r_sec in roots:
    print(
        f"бисекция = {float(r_bis):.10f},  "
        f"Ньютон = {float(r_newt):.10f},  "
        f"секущие = {float(r_sec):.10f}"
    )

# 5. Система нелинейных уравнений

print("\n=== ЗАДАНИЕ 5 ===\n")

def F_sys(z):
    x, y = z
    return np.array([
        math.sin(y) - 2 * x - 1.6,
        math.cos(x + 0.5) + y - 0.8
    ])

def J_sys(z):
    x, y = z
    return np.array([
        [-2.0, math.cos(y)],
        [-math.sin(x + 0.5), 1.0]
    ])

def newton_system(F, J, x0, tol=1e-4):
    x = np.array(x0, float)

    for k in range(50):
        Fx = F(x)
        Jx = J(x)
        delta = np.linalg.solve(Jx, -Fx)
        x_new = x + delta

        print(f"Итерация {k+1}: x = {x_new}")

        if np.linalg.norm(delta, np.inf) < tol:
            print("\nМетод Ньютона сошёлся.")
            return x_new

        x = x_new

    print("\nМетод Ньютона достиг максимального числа итераций.")
    return x


sol = newton_system(F_sys, J_sys, [0.0, 0.0])

print("\nРешение системы нелинейных уравнений:")
print(sol)