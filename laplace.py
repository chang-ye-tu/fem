#
# Reddy J. N. An Introduction to the Finite Element Methods 3ed p.510 Exercise 8.19
#
from dolfin import *
from numpy import sin, sinh, pi
import matplotlib.pyplot as plt

# mesh and function space
mesh = UnitSquareMesh(4, 2)
#mesh = RectangleMesh(Point(0.5, 0), Point(1, 1), 2, 2)
mesh = RectangleMesh(Point(0, 0), Point(2, 1), 20, 10)
V = FunctionSpace(mesh, 'Lagrange', 1)

# variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(0.0)
a = inner(grad(u), grad(v)) * dx
L = f * v * dx

# boundary conditions
def bd0(x):
    return x[0] < DOLFIN_EPS or x[0] > 1. - DOLFIN_EPS or x[1] < DOLFIN_EPS
bc0 = DirichletBC(V, Constant(0.0), bd0)

def bd1(x):
    return x[1] > 1. - DOLFIN_EPS
bc1 = DirichletBC(V, Constant(1.0), bd1)

# solve
u = Function(V)
solve(a == L, u, [bc0, bc1])

plot(mesh)
plt.savefig('laplace_mesh.pdf')

print(u(0.5, 0.5), u(0.75, 0.5))

def anly(xx, yy):
    aa = 0
    for i in range(100):
        aa += sin((2*i + 1) * pi * xx) * sinh((2*i + 1) * pi * yy) / ((2*i + 1) * sinh((2*i + 1) * pi))
    return pi / 4 * aa

#for l in range(5):
#    for k in range(5):
#        xx, yy = 0.25 * k, 0.25 * l
#        print('true value: {:10.15f} | fem value: {:10.15f}'.format(anly(xx, yy), u(xx, yy)))
#        print('point: ({:1.2f}, {:1.2f}) | fem value: {:10.15f}'.format(xx, yy, u(xx, yy)))
