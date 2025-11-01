import numpy as np

from common.autograd import Value


def issueSixTest() -> None:
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


def issueSevenTest() -> None:
    x = Value(
        np.array(
            [
                [[1, 2], [1, 2], [1, 2], [1, 2]],
                [[3, 4], [1, 2], [1, 2], [1, 2]],
                [[5, 6], [1, 2], [1, 2], [1, 2]],
            ]
        )
    )  # shape (3,4,2)
    y = Value(np.array([[10], [20], [30], [40]]))  # shape (4,1)

    z = x * y  # (3, 4, 2) according to neil :c
    print(z.data.shape)
    z.backward()
    print(x.grad.shape)  # should be (3,2,2)
    print(y.grad.shape)  # should be (4,1)
    print("-------------------------")
    a = z + x  # (3,4,2)
    a.backward()
    print(x.grad.shape)  # should be (3,4,2)
    print(z.grad.shape)  # should be (3,4,2)
    print("-------------------------")
    c = z - y  # (3,4,2)
    c.backward()
    print(z.grad.shape)  # should be (3,4,2)
    print(y.grad.shape)  # should be (4,1)
    print("-------------------------")
    d = z / y  # (3,4,2)
    d.backward()
    print(z.grad.shape)  # should be (3,4,2)
    print(y.grad.shape)  # should be (4,1)
    print("-------------------------")
    e = d.exp()  # (3,4,2)
    print(e.grad.shape)  # should be (3,4,2)
    print(d.grad.shape)  # should be (3,4,2)
    print("-------------------------")
    f = e.relu()  # (3,4,2)
    print(f.grad.shape)  # should be (3,4,2)
    print(e.grad.shape)  # should be (3,4,2)


if __name__ == "__main__":
    # issueSixTest()
    issueSevenTest()
