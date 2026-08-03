#!/usr/bin/env python3

add_matrices2D = __import__('5-across_the_planes').add_matrices2D

mat1 = [[1, 2], [3, 4]]
mat2 = [[5, 6], [7, 8]]
print(add_matrices2D(mat1, mat2))
print(mat1)
print(mat2)
print(add_matrices2D(mat1, [[1, 2, 3], [4, 5, 6]]))

# mat1 = [[1, 2, 4], [3, 4, 7], [3, 4, 7], [1, 2, 4]]
# mat2 = [[5, 6, 6], [7, 8, 9], [3, 4, 7], [1, 2, 4]]

# result = []
# for x in range(len(mat1)):
#     print("x: ")
#     print(x)
#     new_line = []
#     for y in range(len(mat1[0])):
#         print("y:")
#         print(y)
#         new_line.append(mat1[x][y] + mat2[x][y])
#         print(new_line)
#     result.append(new_line)
# return result
# print(result)


# mat3 = [[5, 6, 6, 7], [7, 8, 8, 9]]
# print(len(mat3))
# for i in range(len(mat3)):
#     print(i)
#     print(mat3[i])
#     print(range(len(mat3[i])))


# print(add_matrices2D(mat1, mat2))
# print(mat1)
# print(mat2)
# print(add_matrices2D(mat1, [[1, 2, 3], [4, 5, 6]]))
