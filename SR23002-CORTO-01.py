# Deberá crear y publicar una librería en python. El estudiante debe hacer una investigación sobre como crear una librería en python y 
# publicarla de manera que se pueda instalar en otros equipos a través del comando pip install.

# Los criterios de la evaluación son:

# 35% Se incluye el código de todos los métodos solicitados. (el código de todas las funciones solicitadas es funcional).
# 10% Documentación completa.
# 35% Creacion de la librería (el proyecto/carpeta tiene la estructura y configuración adecuada).
# 20% Publicación de la librería (publicación exitosa en Pypi).
# La librería debe ser capaz de resolver sistemas de ecuaciones lineales y no lineales a través de los siguientes métodos:

# Eliminación de Gauss
# Gauss-Jordan
# Crammer
# Descomposición LU
# Jacobi
# Gauss-Seidel
# Bisección


# Entregables:

# Carpeta con el código de la librería.
# Archivo .py con la implementación de la librería (colocar por cada método un ejemplo de su uso).


# La librería debe llamarse CARNETUNO, por ejemplo mi carnet era LJ09002, por lo que mi librería debería llamarse LJ09002UNO.





import numpy as np
from SR23002UNO import biseccion, crammer, descomposicion_lu as dl, eliminacion_gauss as eg, gauss_jordan as gj, gauss_seidel as gs, jacobi as jc

def print_separator():
    print( "═" * 60)

def print_sol_ecu(i, valor):
    print(f"\t → x{i+1} = {round(valor, 6)}")

print("🧑‍💻".center(60, "*"))
print(" IMPLEMENTACIÓN DE LIBRERIA SR23002UNO ".center(60))
print("🧑‍💻".center(60, "*"))

# ------------------------ Bisección ------------------------
print_separator()
print("  📌 Metodo de Bisección 📌".center(60))
print_separator()
print("""
  Basado en la imagen 'biseccion-example.png' 
  🧮 Función: 
  f(x) = x³ + 2x² + 10x - 20
  Intervalo: [-7, 16]
""".strip())

sol_bs = biseccion.biseccion_solver(lambda x: x**3 + 2*x**2 + 10*x - 20, -7, 16)

print_separator()
print(f"""
  Resultado de la raiz aproximada:  → x = {round(sol_bs, 6)}
""".strip())
print("\n")

# ------------------------ Crammer ------------------------
print_separator()
print("  📌 Metodo de Crammer 📌".center(60))
print_separator()
print("""
  Basado en la imagen 'cramer-example.png' 
  🧮 Ecuaciones:
  {
    2x - y + z = 3
    2y - z     = 1
    x + y      = 1
  }
""".strip())
print_separator()

A_crammer = np.array([[2, -1, 1], [0, 2, -1], [1, -1, 0]])
b_crammer = np.array([3, 1, 1])
sol_crammer = crammer.cramer_solver(A_crammer, b_crammer)

print("✅ Solución:")
for i, value in enumerate(sol_crammer):
    print_sol_ecu(i, value)

# ------------------------ Descomposición LU ------------------------
print_separator()
print("  📌 Metodo de Descomposición LU 📌".center(60))
print_separator()
print("""
  Basado en la imagen 'descomposicion-lu-example.png' 
  🧮 Ecuaciones:
  {
    25x  + 5y  + z  = 106.8
    64x  + 8y  + z  = 177.2
    144x + 12y + z  = 279.2
  }
""".strip())

A_lu = np.array([
    [25, 5, 1],
    [64, 8, 1],
    [144, 12, 1]
], dtype=float)

b_lu = np.array([106.8, 177.2, 279.2], dtype=float)
x_lu = dl.dcp_lu_solver(A_lu, b_lu)

print_separator()
print("✅ Solución:")
for i, value in enumerate(x_lu):
    print_sol_ecu(i, value)
    
# ------------------------ Eliminación de Gauss ------------------------
print_separator()
print("  📌 Metodo de Eliminación de Gauss 📌".center(60))
print_separator()
print("""
  Basado en la imagen 'gauss-example.png' 
  🧮 Ecuaciones:
  {
    x  + y  + z   = 1
    2x + 3y - 4z  = 9
    x  - y  + z   = 1
  }
""".strip())

A_gauss = np.array([
    [1, 1, 1],
    [2, 3, -4],
    [1, -1, 1]
], dtype=float)

b_gauss = np.array([1,9,-1], dtype=float)
sol_gauss = eg.eliminacion_gauss_solver(A_gauss, b_gauss)

print_separator()
print("✅ Solución:")
for i, value in enumerate(sol_gauss):
    print_sol_ecu(i, value)

# ------------------------ Gauss-Jordan ------------------------
print_separator()
print("  📌 Metodo de Gauss-Jordan 📌".center(60))
print_separator()
print("""
  Basado en la imagen 'gauss-example.png' 
  🧮 Ecuaciones:
  {
    x  + y  + z   = 1
    2x + 3y - 4z  = 9
    x  - y  + z   = 1
  }
""".strip())

A_gj = np.array([
    [1, 1, 1, 1],
    [2, 3, -4, 9],
    [1, -1, 1, -1]
], dtype=float)

sol_gj = gj.gauss_jordan_solver(A_gj)

print_separator()
print("✅ Solución:")
for i, value in enumerate(sol_gj):
    print_sol_ecu(i, value)

# ------------------------ Gauss-Seidel ------------------------
print_separator()
print("  📌 Metodo de Gauss-Seidel 📌".center(60))
print_separator()
print("""
  Basado en la imagen 'gauss-seidel-jacobi-example.png' 
  🧮 Ecuaciones:
  {
    10x + 3y  + z    = 14
    5x  - 10y + 3z   = -5
    x   + 3y  - 10z  = 1
  }
""".strip())

A_gs = np.array([
    [10, 3, 1],
    [5, -10, 3],
    [1, 3, 10]
])

b_gs = np.array([14, -5, 14])

x_aprox, iteraciones_js = gs.gauss_seidel_solver(A_gs, b_gs)

print_separator()
print("✅ Solución:")
for i, value in enumerate(x_aprox):
    print_sol_ecu(i, value)

print(f"""
  🔄️ Número de iteraciones:
  \t → {iteraciones_js}
""".strip())

# ------------------------ Jacobi ------------------------
print_separator()
print("  📌 Metodo de Jacobi 📌".center(60))
print_separator()
print("""
  Basado en la imagen 'gauss-seidel-jacobi-example.png' 
  🧮 Ecuaciones:
  {
    10x + 3y  + z    = 14
    5x  - 10y + 3z   = -5
    x   + 3y  - 10z  = 1
  }
""".strip())

A_jc = np.array([
    [10, 3, 1],
    [5, -10, 3],
    [1, 3, 10]
])

b_jc = np.array([14, -5, 14])

x_aprox_jc, iteraciones_jc = jc.jacobi_solver(A_jc, b_jc)

print_separator()
print("✅ Solución:")
for i, value in enumerate(x_aprox_jc):
    print_sol_ecu(i, value)

print_separator()
print(f"""
  🔄️ Número de iteraciones:
      \t → {iteraciones_jc}
""".strip())