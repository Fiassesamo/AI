k = [1, 5, 2]
x = [1, 2, 3]
b = [5, 6 ,7]

zip_list = [(k*x+b) for k, x, b in zip(k, x, b)]
print(zip_list)