"""
Лабораторная работа №4. Вариант 6.
"""

import numpy as np
import pandas as pd

# -------------------------------
# 1. ДАННЫЕ
# -------------------------------

x_nodes = np.array([
    1.340, 1.345, 1.350, 1.355, 1.360, 1.365,
    1.370, 1.375, 1.380, 1.385, 1.390, 1.395
])
y_values = np.array([
    4.25562, 4.35325, 4.45522, 4.56184, 4.67344, 4.79038,
    4.91306, 5.04192, 5.17744, 5.32016, 5.47069, 5.62968
])

h = x_nodes[1] - x_nodes[0]
print(f"Шаг сетки: h = {h:.3f}")

# -------------------------------
# 2. ТАБЛИЦА КОНЕЧНЫХ РАЗНОСТЕЙ
# -------------------------------

def build_finite_difference_table(y):
    n = len(y)
    T = np.zeros((n, n))
    T[:, 0] = y
    for j in range(1, n):
        for i in range(n - j):
            T[i, j] = T[i+1, j-1] - T[i, j-1]
    return T

diff_table = build_finite_difference_table(y_values)

df = pd.DataFrame(
    diff_table,
    columns=[f"Δ^{k}y" for k in range(len(y_values))],
    index=[f"x{i}" for i in range(len(x_nodes))]
)
print("\n" + "="*60)
print("ТАБЛИЦА КОНЕЧНЫХ РАЗНОСТЕЙ")
print("="*60)
print(df.round(6).fillna('').to_string())

# -------------------------------
# 3. ФОРМУЛЫ НЬЮТОНА
# -------------------------------

def newton_forward(x, x0, h, y0, diffs, n_terms=5):
    q = (x - x0) / h
    res = y0
    term = 1.0
    for i in range(1, n_terms):
        term *= (q - i + 1) / i
        res += term * diffs[i-1]
    return res

def newton_backward(x, xn, h, yn, diffs, n_terms=5):
    q = (x - xn) / h
    res = yn
    term = 1.0
    for i in range(1, n_terms):
        term *= (q + i - 1) / i
        res += term * diffs[i-1]
    return res

# -------------------------------
# 4. ВЫЧИСЛЕНИЯ (исходные значения из задания)
# -------------------------------

x_query = np.array([1.3617, 1.3921, 1.3359, 1.400])
results = []

print("\n" + "="*60)
print("ВЫЧИСЛЕНИЕ ЗНАЧЕНИЙ ФУНКЦИИ")
print("="*60)

for x in x_query:
    if x < x_nodes[0]:
        # Экстраполяция влево
        diffs = diff_table[0, 1:5]  # Δ¹y0, Δ²y0, Δ³y0, Δ⁴y0
        val = newton_forward(x, x_nodes[0], h, y_values[0], diffs, n_terms=5)
        method = "Первая формула (экстраполяция влево)"
    elif x > x_nodes[-1]:
        # Экстраполяция вправо
        diffs = np.array([
            diff_table[-2, 1],  # Δ¹y_{n-1}
            diff_table[-3, 2],  # Δ²y_{n-2}
            diff_table[-4, 3],  # Δ³y_{n-3}
            diff_table[-5, 4]   # Δ⁴y_{n-4}
        ])
        val = newton_backward(x, x_nodes[-1], h, y_values[-1], diffs, n_terms=5)
        method = "Вторая формула (экстраполяция вправо)"
    else:
        # Интерполяция внутри
        # Определяем, к какому концу ближе точка
        dist_to_start = abs(x - x_nodes[0])
        dist_to_end = abs(x - x_nodes[-1])
        if dist_to_start <= dist_to_end:
            diffs = diff_table[0, 1:5]
            val = newton_forward(x, x_nodes[0], h, y_values[0], diffs, n_terms=5)
            method = "Первая формула"
        else:
            diffs = np.array([
                diff_table[-2, 1],
                diff_table[-3, 2],
                diff_table[-4, 3],
                diff_table[-5, 4]
            ])
            val = newton_backward(x, x_nodes[-1], h, y_values[-1], diffs, n_terms=5)
            method = "Вторая формула"

    results.append(val)
    print(f"\nf({x:.4f}) = {val:.6f} ({method})")

# -------------------------------
# 5. ИТОГ
# -------------------------------

print("\n" + "="*60)
print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
print("="*60)
for x, y in zip(x_query, results):
    print(f"f({x:.4f}) = {y:.6f}")


print("\n")
print("="*80)
print("     ЧАСТЬ 2")
print("=" * 80)
print("\n\n")

import numpy as np
from scipy.integrate import quad

# -------------------------------
# 1. ОПРЕДЕЛЕНИЕ ФУНКЦИИ
# -------------------------------
def f(x):
    return np.sqrt(x) * np.log(x)

a, b = 1.0, 4.0
eps = 0.001

# -------------------------------
# 2. ВЫЧИСЛЕНИЕ M = max |f''(x)|
# -------------------------------
# Аналитическая вторая производная:
# f''(x) = (-1/4) * x^(-3/2) * ln(x)
def f_double_prime(x):
    return (-0.25) * x**(-1.5) * np.log(x)

# Численный поиск максимума |f''(x)| на [1, 4]
xs = np.linspace(1.0001, 4, 10000)
M = np.max(np.abs(f_double_prime(xs)))
print(f"M = max |f''(x)| на [{a}, {b}] = {M:.6f}")

# -------------------------------
# 3. ВЫЧИСЛЕНИЕ ШАГА h
# -------------------------------
h_max = np.sqrt(12 * eps / (M * (b - a)))
n_min = (b - a) / h_max

# Округляем вверх до целого, кратного 4
n = int(np.ceil(n_min))
if n % 4 != 0:
    n = n + (4 - n % 4)
h = (b - a) / n

print(f"Расчёт шага:")
print(f"h_max = {h_max:.6f} → n_min = {n_min:.2f}")
print(f"Выбрано n = {n} (кратно 4), h = {h:.6f}")

# -------------------------------
# 4. ФОРМУЛА ТРАПЕЦИЙ
# -------------------------------
def trapezoidal(func, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = func(x)
    return h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])

I_h = trapezoidal(f, a, b, n)
I_2h = trapezoidal(f, a, b, n // 2)

print(f"\nРезультаты интегрирования:")
print(f"I_h   (n={n:3d}) = {I_h:.6f}")
print(f"I_2h  (n={n//2}) = {I_2h:.6f}")

# -------------------------------
# 5. ПРАВИЛО РУНГЕ
# -------------------------------
error_runge = abs(I_h - I_2h) / 3
print(f"Оценка погрешности (Рунге): Δ ≈ {error_runge:.6f}")

# -------------------------------
# 6. ТОЧНОЕ ЗНАЧЕНИЕ (НЬЮТОН–ЛЕЙБНИЦ)
# -------------------------------
def antiderivative(x):
    return (2/3) * x**1.5 * (np.log(x) - 2/3)

I_exact = antiderivative(b) - antiderivative(a)
print(f"Точное значение (Ньютон–Лейбниц): {I_exact:.6f}")

# -------------------------------
# 7. ПРОВЕРКА: SCIPY
# -------------------------------
I_scipy, _ = quad(f, a, b)
print(f"Scipy.integrate.quad:             {I_scipy:.6f}")

# -------------------------------
# 8. СРАВНЕНИЕ
# -------------------------------
print("\n" + "="*50)
print("СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
print("="*50)
print(f"Формула трапеций (h)     : {I_h:.6f}")
print(f"Формула трапеций (2h)    : {I_2h:.6f}")
print(f"Точное значение          : {I_exact:.6f}")
print(f"Scipy                    : {I_scipy:.6f}")
print(f"Погрешность (h)          : {abs(I_h - I_exact):.6f}")
print(f"Погрешность (2h)         : {abs(I_2h - I_exact):.6f}")
print(f"Оценка погрешности (Рунге): {error_runge:.6f}")

if abs(I_h - I_exact) < abs(I_2h - I_exact):
    print("\nФормула трапеций с шагом h дала более точный результат.")
else:
    print("\nФормула трапеций с шагом 2h дала более точный результат.")

# Проверка выполнения точности
if abs(I_h - I_exact) < eps:
    print(f"\nТребуемая точность ε = {eps} достигнута")
else:
    print(f"\nТребуемая точность НЕ достигнута.")