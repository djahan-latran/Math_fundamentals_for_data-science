import numpy as np
from numpy.linalg import det, inv, eig


# VECTORS

# Create a 3-dimensional vector
v = np.array([3, 4, 1])
print(f"A 3d-vektor: {v}")

# Vector addition
v1 = np.array([2, 4])
v2 = np.array([3, 1])

v_new = v1 + v2
print(f"{v1} + {v2} = {v_new}")

# Scale a vector by a scalar
v3 = np.array([3, 1])
scalar = 2
v3_new = scalar * v3

print(f"{scalar} * {v3} = {v3_new}")

# Vector-matrix-multiplication
basis = np.array(
    [[3, 0],
     [0, 2]]
)

v4 = np.array([1, 1])

v4_new = basis.dot(v4)
print(f"{basis} * {v4} = {v4_new}")

# Disassemble matrix into basis-vectors and the use as transformation
i_hat = np.array([2, 4])
j_hat = np.array([6, 3])

transform_matrix = np.array([i_hat, j_hat]).transpose()
v5 = np.array([1, 1])

v5_new = transform_matrix.dot(v5)
print(f"{transform_matrix} * {v5} = {v5_new}")

# MATRICES

# Matrix multiplication

# create first transformation matrix
i_hat1 = np.array([0, 1])
j_hat1 = np.array([-1, 0])

transform1 = np.array([i_hat1, j_hat1]).transpose()

# create second transformation matrix
i_hat2 = np.array([1, 0])
j_hat2 = np.array([1, 1])

transform2 = np.array([i_hat2, j_hat2]).transpose()

# multiple both matrices. First transformation sheering and then transformation rotation
combined = transform1 @ transform2

v6 = np.array([1, 2])

# create dot-product of combined-matrix and vector
solution = combined.dot(v6)
print(solution)

# Determinants
A = np.array([i_hat2, j_hat2]).transpose()
determinant = det(A)
print(determinant)

# LINEAR EQUATIONS

# solving a linear equation with numpy
# 4x + 2y + 4z = 44
# 5x + 3y + 7z = 56
# 9x + 3y + 6z = 72

A1 = np.array([[4, 2, 4],
               [5, 3, 7],
               [9, 3, 6]]
              )
B1 = np.array([
    44,
    56,
    72
])

X = inv(A1).dot(B1)
print(f"X = {int(X[0])}, Y = {int(X[1])}, Z = {int(X[2])}")

# EIGENVECTORS

# Get eigenvectors and eigenvalues from a matrix

matrix_A = np.array([
    [1, 2],
    [4, 5]
])

eigenvals, eigenvecs = eig(matrix_A)

print(f"These are the eigenvectors of {matrix_A} : {eigenvecs}")
print(f"These are the eigenvalues of {matrix_A} : {eigenvals}")

# disassembly of Matrix_B and reassembly

matrix_B = np.array([
    [1, 2],
    [4, 5]
])

eigenvals_B, eigenvecs_B = eig(matrix_B)

Q = eigenvecs_B
# diagonalized eigenvalues
L = np.diag(eigenvals_B)

# inverted eigenvectors
R = inv(Q)

matrix_B_reconstructed = Q @ L @ R

print(matrix_B_reconstructed)