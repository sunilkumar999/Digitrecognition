#!/usr/bin/env python3


import numpy as np
from random import randrange

if __name__ == "__main__":

    a = np.load('digits.npy')                   
    b = a.reshape(10,500,400)                      # reshape to [digit, picture number, pixels]
    c  = b[0:10,np.random.permutation(500),0:400]  

    train = c[0:10,0:400,0:400]                    
    test  = c[0:10,400:500,0:400]                  

    alpha = 0.1
    w = np.random.randn(4000).reshape(10,400)

    totalerr = 0
    for i in range(1,1000001):
        dig = randrange(10)             
        pix = randrange(400)             
        x = train[dig,pix]               
        y = np.zeros(10)                
        y[dig] = 1.0
        z = np.matmul(w,x)               
        a = 1.0 / (1.0 + np.exp(-z))     
        predict = np.argmax(a)       
        delta = a - y                    
        err = np.sum(delta**2)           
        totalerr += err
#         print(z)
#         print(a)
#         print(dig,predict,err)
        dw = np.outer(delta,x)
        w -= alpha * dw 
        if i % 100000 == 0:
            print(totalerr / i)

    np.save('weights', w)
    good = 0
    total = 0
    right = [0]*10
    for i in range(10):
        for j in range(100):
            x = test[i,j]
            z = np.matmul(w,x)
            a = 1.0/(1.0 + np.exp(-z))
            predict = np.argmax(a)
            if predict == i:
                good += 1
                right[i] += 1
            total += 1

    print(good/total)
    print(right)
