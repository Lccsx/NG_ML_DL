import numpy as np
num = np.matrix([[1, 9], [2, 3]])
#print(num)
print(np.array(num[:, 1].A1))

def normalEqu(X, y):
    theta = np.linalg.inv(X.T@X)@X.T@y
    return theta