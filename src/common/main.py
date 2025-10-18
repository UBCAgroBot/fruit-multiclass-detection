import numpy as np
from autograd import Value

if __name__ == "__main__":
    a = Value(5)  # single value working
    print(a.data)
    b = Value(10)  # single value working
    print(b.data)
    c = a * b
    c.backward()  # this is the key to make sure that the gradients are working
    print(b.grad)  # Working now !!!!!
    ### NEW STUFF
    print("-----------------------------------------")
    # ex 1
    print("ex1")
    x = Value(np.array([1, 2, 3]))
    y = Value(np.array([4, 5, 6]))
    z = x * y
    z.backward()
    print(x.grad)  # dz/dx (works)
    print(y.grad)  # dz/dy (works)
    print(z.grad)  # dz/dz which is 1
    print("-----------------------------------------")
    # ex 2
    print("ex2")
    x_array = np.array([1, 2, 0])
    x = Value(x_array)
    z = x.exp()
    z.backward()
    print(x.grad)  # dz/dx
    print(z.grad)  # dz/dx
    print("-----------------------------------------")
    # ex 2
    print("ex3")
    a = Value(np.array([[1, 2], [3, 4]]))
    b = Value(np.array([[5, 6], [7, 8]]))
    c = a @ b
    c.backward()  # key
    print(a.grad)  # dc/da
    print(b.grad)  # dc/db
