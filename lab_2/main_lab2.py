import numpy as np

# 1. Создание квадратной матрицы и вычисление минора

print("----------1. Минор 4-го порядка----------", "\n")
np.random.seed(42)
A = np.random.randint(-7, -1, (9, 9))
print("Матрица A:\n", A, "\n")

minor = A[:4, [1,2,7,8]]
print("Минор (строки 1-4, столбцы 2,3,8,9):\n", minor, "\n")
det_min = np.linalg.det(minor)
print("Определитель минора:", np.round(det_min))

# 2. Умножение матриц тремя способами

print("\n----------2. Умножение матриц----------", "\n")
A = np.random.rand(3, 4)
B = np.random.rand(4, 5)
print("Матрица A (3x4):\n\n", A, "\n")
print("Матрица B (4x5):\n\n", B, "\n")

def vector(A, B):
    n, k = A.shape
    k, m = B.shape
    result = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            result[i,j] = np.dot(A[i,:], B[:,j])
    return result

C1 = vector(A, B)
print("Векторный алгоритм:\n\n", C1, "\n")

def matrix_mult(A, B):
    n, k = A.shape
    k, m = B.shape
    result = np.zeros((n, m))
    for j in range(m):
        result[:,j] = A @ B[:,j]
    return result

C2 = matrix_mult(A, B)
print("Матричный алгоритм (по столбцам):\n\n", C2, "\n")

C3 = np.dot(A, B)
print("DOT:\n", C3, "\n")
print("Совпадение результатов:", np.allclose(C1, C2) and np.allclose(C2, C3))

# 3. Норма вектора ‖x‖3

print("\n3. Норма вектора ‖x‖₃", "\n")
x = np.random.randint(-10, 10, 8)
print("Вектор x:", x)

def norm_3(vector):
    return np.sum(np.abs(vector)**3)**(1/3)

norm_custom = norm_3(x)
norm_numpy = np.linalg.norm(x, ord=3)
print("Собственная функция:", norm_custom)
print("np.linalg.norm:", norm_numpy)
print("Совпадение:", np.allclose(norm_custom, norm_numpy))

# 4. Спектральная норма матрицы
print("\n4. Спектральная норма матрицы\n")
M = np.random.randint(-10, 10, (5, 5))
print("Матрица M:\n\n", M, "\n")

def spectral_norm(matrix):
    eigenvals = np.linalg.eigvals(matrix.T @ matrix)
    return np.sqrt(np.max(eigenvals))

spec_norm_custom = spectral_norm(M)
spec_norm_numpy = np.linalg.norm(M, ord=2)
print("Собственная функция:", spec_norm_custom)
print("np.linalg.norm:", spec_norm_numpy)
print("Совпадение:", np.allclose(spec_norm_custom, spec_norm_numpy), "\n")

# 5. Отражение Хаусхолдера
print("\n5. Отражение Хаусхолдера")

def householder_reflection_for_zeroing_tail(x, start):
    x = x.astype(float).copy()
    n = x.size
    v_sub = x[start:].copy()
    if v_sub.size == 0:
        return np.eye(n), x, None
    norm_v = np.linalg.norm(v_sub)
    if norm_v == 0:
        return np.eye(n), x, None
    sigma = np.sign(v_sub[0]) if v_sub[0] != 0 else 1.0
    e1 = np.zeros_like(v_sub); e1[0] = 1.0
    u = v_sub - sigma * norm_v * e1
    u_norm = np.linalg.norm(u)
    if u_norm == 0:
        H_sub = np.eye(v_sub.size)
        H_sub[0,0] = -1.0
    else:
        v = u / u_norm
        H_sub = np.eye(v_sub.size) - 2.0 * np.outer(v, v)
    H = np.eye(n)
    H[start:, start:] = H_sub
    y = H @ x
    if 'v' in locals() and u_norm != 0:
        v_full = np.zeros(n); v_full[start:] = v
    else:
        v_full = None
    return H, y, v_full

H, y, v_full = householder_reflection_for_zeroing_tail(x, start=2)
print("Исходный вектор:", x)
print("После отражения:", y)
print("Обнулены элементы с 4 по 8:", np.allclose(y[3:8], 0, atol=1e-10))

# 6. LU-разложение (упрощенная версия)
print("\n----------6. LU-разложение----------\n")


def lu_decomposition(matrix):

    n = matrix.shape[0]
    L = np.eye(n)
    U = matrix.copy().astype(float)

    for i in range(n):
        for j in range(i + 1, n):
            L[j, i] = U[j, i] / U[i, i]
            U[j, i:] = U[j, i:] - L[j, i] * U[i, i:]

    return L, U

print("Матрица для LU-разложения:")
print(M)
print()

L, U = lu_decomposition(M)

print("Матрица L:")
print(np.round(L, 4))
print("\nМатрица U:")
print(np.round(U, 4))

L_times_U = L @ U
print("\nПроверка L * U = M:")
print("L * U:")
print(np.round(L_times_U, 4))
print("Исходная матрица M:")
print(M)
print("\nСовпадение:", np.allclose(L_times_U, M))

print("\n" + "=" * 50)
print("Демонстрация решения системы с помощью LU-разложения:")

b = np.random.randint(-5, 5, 5)
print(f"Правая часть b: {b}")

y = np.linalg.solve(L, b)
x_lu = np.linalg.solve(U, y)

x_direct = np.linalg.solve(M, b)

print(f"Решение через LU: {np.round(x_lu, 4)}")
print(f"Прямое решение:   {np.round(x_direct, 4)}")
print("Совпадение решений:", np.allclose(x_lu, x_direct))

# 7. QR-разложение
print("\n----------7. QR-разложение----------\n")

np.set_printoptions(precision=4, suppress=True)

print("Матрица M:\n", M, "\n")

# --- 1. Метод Грама–Шмидта ---
def gram_schmidt_qr(A):
    A = A.astype(float)
    Q = np.zeros_like(A)
    cnt = 0

    for a in A.T:
        u = np.copy(a)
        for i in range(cnt):
            u -= np.dot(np.dot(Q[:, i].T, a), Q[:, i])  # шаг Грама–Шмидта
        e = u / np.linalg.norm(u)
        Q[:, cnt] = e
        cnt += 1

    R = np.dot(Q.T, A)
    return Q, R

Q_gs, R_gs = gram_schmidt_qr(M)
print("1) Метод Грама–Шмидта")
print("Q:\n", np.round(Q_gs, 4))
print("\nR:\n", np.round(R_gs, 4))
print("\nПроверка M ≈ Q @ R:", np.allclose(M, Q_gs @ R_gs, atol=1e-8))
print("Ортогональность Q^T Q ≈ I:", np.allclose(Q_gs.T @ Q_gs, np.eye(Q_gs.shape[1]), atol=1e-8), "\n")


# --- 2. Метод отражений Хаусхолдера ---
def householder_qr(A):
    A = A.astype(float)
    n, m = A.shape
    R = A.copy()
    Q = np.eye(n)

    for k in range(m):
        x = R[k:, k]
        e1 = np.zeros_like(x)
        e1[0] = 1.0
        alpha = np.sign(x[0]) * np.linalg.norm(x)
        u = x + alpha * e1
        v = u / np.linalg.norm(u)
        H_k = np.eye(n)
        H_k[k:, k:] -= 2.0 * np.outer(v, v)
        R = H_k @ R
        Q = Q @ H_k.T
    return Q, R

Q_hh, R_hh = householder_qr(M)
print("2) Метод отражений Хаусхолдера")
print("Q:\n", np.round(Q_hh, 4))
print("\nR:\n", np.round(R_hh, 4))
print("\nПроверка M ≈ Q @ R:", np.allclose(M, Q_hh @ R_hh, atol=1e-8))
print("Ортогональность Q^T Q ≈ I:", np.allclose(Q_hh.T @ Q_hh, np.eye(Q_hh.shape[1]), atol=1e-8), "\n")


# --- 3. Проверка встроенной функцией numpy ---
Q_np, R_np = np.linalg.qr(M)
print("3) NumPy QR-разложение")
print("Q:\n", np.round(Q_np, 4))
print("\nR:\n", np.round(R_np, 4))
print("\nПроверка M ≈ Q @ R:", np.allclose(M, Q_np @ R_np, atol=1e-8))
print("Ортогональность Q^T Q ≈ I:", np.allclose(Q_np.T @ Q_np, np.eye(Q_np.shape[1]), atol=1e-8))
