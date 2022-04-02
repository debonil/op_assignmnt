from numpy import *
from numpy import array
from scipy.optimize import *

"""
Write a MATLAB/Python code to solve the problem

min(x1 − R) ^ 2 + (x1 − 2*x2) ^ 2
s. t. x1 ^ 2 − x2 = 0.

(R is the last two digits of your roll number) using penalty function method. Consider μ0 = 0.1, β = 10,
and stopping criteria μk*α*(x^k) < 10−5 or maximum 200 iterations.
 """

R = 25


x0, eps, tol, mu, beta, iter1 = array([R, R/2]), pow(
    10, -5), 1, 0.1, 10, 0  # initialization


def f(x):
    # define f(x)
    return pow((x[0] - R), 2) + pow((x[0] - 2*x[1]), 2)


def hx(x):
    # define h(x)
    return pow(x[0], 2)-x[1]


def pen(x):
    # define objective function of unconstrained problem
    # min f(x)+mu h(x)^2
    return f(x)+mu*pow(hx(x), 2)


def jac_pen(x):
    # gradient of above function
    return array([2*(x[0]-R)+2*(x[0]-2*x[1]+2*mu*(pow(x[0], 2)-x[1])*2*x[0]),
                  -4*(x[0]-2*x[1])-2*mu*(pow(x[0], 2)-x[1])])


while (tol > eps) and (iter1 < 200):
    # solve unconstrained problem
    res = minimize(pen, x0, method='BFGS', jac=jac_pen,
                   options={'disp': False})
    x0, h = res.x, hx(x0)  # updates
    print('h=', h)
    tol, mu, iter1 = mu*pow(h, 2), mu*beta, iter1+1
    print('tol=', tol)
print('*************************************')
print('optimal solution=', x0)
print('Total no of iteration', iter1)
