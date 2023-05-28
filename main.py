import numpy as np
from itertools import product, permutations
from math import factorial, sqrt


def basisChange():
    n = int(input("введите кол-во векторов базисов: "))
    a = []
    b = []
    print("Введите матрицу векторов откуда производится переход:")
    for i in range(n):
        a.append(list(map(float, input().split())))
    print("Введите матрицу векторов куда производится переход:")
    for i in range(n):
        b.append(list(map(float, input().split())))
    a = np.array(a)
    b = np.array(b)
    E = np.eye(n)
    while (a != E).any():
        for i in range(n):
            print(a[i], b[i])

        step = input().split()
        result_a = (a[int(step[0]) - 1]).copy()
        result_b = (b[int(step[0]) - 1]).copy()
        for i in range(2, len(step), 2):
            if step[i - 1] == '+':
                result_a += a[int(step[i]) - 1]
                result_b += b[int(step[i]) - 1]
            else:
                result_a -= a[int(step[i]) - 1]
                result_b -= b[int(step[i]) - 1]
        a[int(step[0]) - 1] = result_a.copy()
        b[int(step[0]) - 1] = result_b.copy()

    print("T = ")
    print(*b, sep='\n')
    print("\nS = T^-1 = ")
    print(*np.around(np.linalg.inv(b), 5), sep='\n')


def printTensor(tensor, size, isInInt=False):
    print('[', end="")
    for j in range(size):
        for i in range(size):
            for k in range(size):
                if isInInt:
                    print(int(tensor[i][j][k]), end="")
                else:
                    print(tensor[i][j][k], end="")
                if k != size - 1 or (k == size - 1 and i != size - 1):
                    print("", end=", ")
        if j != size - 1:
            print(';')
    print(']')


def permutation_inversions_number(int_permutation):
    ans = 0
    for i in range(len(int_permutation)):
        for j in range(i + 1, len(int_permutation)):
            if int_permutation[i] > int_permutation[j]:
                ans += 1
    return ans


def permutation_parity(int_permutation):
    return 1 if ((permutation_inversions_number(int_permutation) % 2) == 0) else -1


def get_element(tensor, indx):
    ans = tensor
    for i in indx:
        ans = ans[i]
    return ans


def set_element(tensor, indx, val, i=0):
    if i == len(indx) - 1:
        tensor[indx[i]] = val
    else:
        tensor[indx[i]] = set_element(tensor[indx[i]], indx, val, i + 1)
    return tensor


def valence(tensor):
    p = 0
    size = len(tensor)
    while type(tensor) == type(np.array([])):
        p += 1
        assert (size == len(tensor))
        tensor = tensor[0]
    return p


def Sym(tensor):
    p = valence(tensor)
    size = len(tensor)

    ans = np.zeros_like(tensor, dtype=float)
    for indexs in product(range(size), repeat=p):
        new_val = 0
        for indexs_permutaton in permutations(range(p), p):
            indexs_permutaton_value = list(map(lambda x: indexs[x], indexs_permutaton))
            new_val += get_element(tensor, indexs_permutaton_value)
        ans = set_element(ans, indexs, (1 / factorial(p)) * new_val)
    return ans


def Asym(tensor, indexs=[]):
    p = valence(tensor)
    size = len(tensor)
    ans = np.zeros_like(tensor, dtype=float)

    if len(indexs) == 0:
        for indexs in product(range(size), repeat=p):
            new_val = 0
            for indexs_permutaton in permutations(range(p), p):
                indexs_permutaton_value = list(map(lambda x: indexs[x], indexs_permutaton))
                new_val += permutation_parity(indexs_permutaton) * get_element(tensor, indexs_permutaton_value)
            ans = set_element(ans, indexs, (1 / factorial(p)) * new_val)
    else:
        ...
    return ans


def external_work_2(tensor_a, tensor_b):  # c = a ^ b
    p = valence(tensor_a)
    r = valence(tensor_b)
    return (factorial(p + r) / (factorial(p) * factorial(r))) * Asym(np.tensordot(tensor_a, tensor_b, axes=0))


def external_work(tensor_a, tensor_b, *tensors):
    ans = external_work_2(tensor_a, tensor_b)
    for tensor in tensors:
        ans = external_work_2(ans, tensor)
    return ans


def euclide_scalar_multiply(vec_a, G, vec_b):
    # return np.transpose(vec_a).dot(G).dot(vec_b)
    ans = 0.0
    for i in range(len(vec_a)):
        for j in range(len(vec_b)):
            ans += G[i][j] * vec_a[i] * vec_b[j]
    return ans


def euclide_normalize_vector(vect, G):
    return np.around(vect / sqrt(euclide_scalar_multiply(vect, G, vect)), 5)


def euclide_ortonormalized_vectors(vectors, G):
    ans = np.array([[0.0 for j in range(len(vectors[0]))] for i in range(len(vectors))], dtype=float)
    for i in range(len(vectors)):
        ans[i] = vectors[i]
        for j in range(i):
            ans[i] -= ans[j] * (
                        euclide_scalar_multiply(vectors[i], G, ans[j]) / euclide_scalar_multiply(ans[j], G, ans[j]))
        ans[i] = euclide_normalize_vector(ans[i], G)
    return ans


l1 = np.array([-2, 4, -10, 40], dtype=float)
l2 = np.array([-2, 6, -16, 62], dtype=float)
l3 = np.array([2, -6, 14, -56], dtype=float)
G = np.eye(4)

print(euclide_ortonormalized_vectors([l1, l2, l3], G))

'''
[-0.04822,  0.09645, -0.24112, 0.96449; 0.90065, -0.16731, -0.39925, -0.03805; -0.34591, -0.83436, -0.42725, -0.04067]

'''


# input data
# also edit indices in the last cell

# A = np.array([[[1, 7, -2],
#                [-2, -8, -3],
#                [-7, 5, 6]],
#
#               [[7, -4, 6],
#                [-3, -4, 7],
#                [-5, -5, 7]],
#
#               [[-2, 1, 0],
#                [6, -6, 4],
#                [3, -6, 1]]])
#
#
#
# T = np.array([[-2, 3, 4],
#               [1, 0, -1],
#               [-1, 1, 2]])
# S = np.array([[-1, 2, 3],
#               [1, 0, -2],
#               [-1, 1, 3]])
#
# new_A = np.zeros_like(A)
# for t, p, r in product(range(3), repeat=3):
#     for m, n, l in product(range(3), repeat=3):
#         # здесь тензор имел 3 верхних индекса
#         new_A[t][p][r] += A[l][m][n] * T[n][r] * T[l][t] * S[p][m]
#
#
# print('[', end="")
# for j in range(3):
#     for i in range(3):
#         for k in range(3):
#             print(new_A[i][j][k], end=", ")
#
#     print(';')
# print(']')
# print(new_A)
