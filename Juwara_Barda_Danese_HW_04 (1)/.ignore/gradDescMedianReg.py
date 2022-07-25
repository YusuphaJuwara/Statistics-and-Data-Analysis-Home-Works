import pandas as pd
import numpy as np

with open(r"C:\Users\devin\Desktop\Università\Statistica\HW4\Cook.txt", 'r') as data :

    # Skip the first line as it is the header | we keep the np array as it might be usefull later
    data = np.asarray([x.split() for x in data.readlines()[1:]], dtype=np.float32)

x = data[:,0]
y = data[:,1]

def grad_beta_one(b0,b1):
    return np.sum( (-x*y + np.power(x,2)*b1 + x*b0)/(abs(y-x*b1-b0)) )

def grad_beta_zero(b0,b1):
    return np.sum( (-y +x*b1 + b0 )/(abs(y-x*b1-b0)) )


current_b0 = 0.1
current_b1 = 1
max_iterations = 1000
learning_Rate = 0.01
current_iteration = 0
previous_step_size_b0 = 0.5
previous_step_size_b1 = 0.5
precision = 0.01

#print('Loss value = ', np.sum(abs(y - current_b1*x - current_b0)))

while (previous_step_size_b0 > precision or previous_step_size_b1 > precision) and current_iteration < max_iterations:
    previous_b0 = current_b0
    current_b0 = current_b0 - learning_Rate*grad_beta_zero(current_b0,current_b1)
    previous_step_size_b0 = abs(current_b0 - previous_b0)
    
    previous_b1 = current_b1
    current_b1 = current_b1 -learning_Rate*grad_beta_one(current_b0,current_b1)
    previous_step_size_b1 = abs(current_b1 - previous_b1)
    current_iteration += 1
    #print('Iteration count ',current_iteration,'Values of b0,b1 are ',current_b0,',',current_b1)
    print('Loss value = ', np.sum(abs(y - current_b1*x - current_b0)))

#print('Loss value = ', np.sum(abs(y - current_b1*x - current_b0)))

print('Local minima found at b0,b1 =', current_b0,',',current_b1)

