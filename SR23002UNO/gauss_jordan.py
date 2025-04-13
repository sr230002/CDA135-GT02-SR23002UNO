import numpy as np

def gauss_jordan_solver(matrix, tol=1e-10):
    
    if matrix is None or not isinstance(matrix, np.ndarray):
        raise ValueError("El argumento 'matrix' no puede ser nulo.")
    
    m = matrix.astype(float)
    rows, cols = m.shape

    for i in range(rows):
        # Buscar el pivote mayor en columna i para evitar divisiones por cero
        max_row = i + np.argmax(np.abs(m[i:, i]))
        # Intercambiar filas
        m[[i, max_row]] = m[[max_row, i]] 
        # Si el pivote es cero o menor que la tolerancia, indicamos que no es unica la solucion
        if abs(m[i, i]) < tol:
            raise ValueError("El sistema no tiene solucion unica.")

        m[i] = m[i] / m[i, i]

        # Hacemos ceros en la columna i para todas las demás filas.
        for j in range(rows):
            if j != i:
                factor = m[j, i]
                m[j] = m[j] - factor * m[i]

    return m[:, -1]



# referencia de la imagen "gauss-example.png"
# A = np.array([
#     [1, 1, 1, 1],
#     [2, 3, -4, 9],
#     [1, -1, 1, -1]
# ], dtype=float)

# sol = gauss_jordan_solver(A)

# for i, x in enumerate(sol):
#     print(f"x{i+1} = {x:.6f}")


# Sistema:
# 3x + 2y + z = 0
# 5x + 3y + 4z = 2
# x + y - z = 1
A = np.array([
    [3, 2, 1, 1],
    [5, 3, 4, 2],
    [1, 1, -1, 1]
], dtype=float)

sol = gauss_jordan_solver(A)

for i, x in enumerate(sol):
    print(f"x{i+1} = {x:.6f}")