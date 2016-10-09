# Required imports
from pyomo.environ import *
from pyomo.dae import *
from pyomo.dae.plugins.finitedifference import Finite_Difference_Transformation

m = ConcreteModel()

m.pi = Param(initialize=3.1416)
L = 1.0
T = 5.0

#variables del problema
m.t = ContinuousSet(bounds=(0, T))
m.x = ContinuousSet(bounds=(0, L))
m.u = Var(m.x, m.t)

#variables derivadas respecto a las variables independientes x t
m.dudt2 = DerivativeVar(m.u, wrt=(m.t,m.t))
m.dudx2 = DerivativeVar(m.u, wrt=(m.x,m.x))

#restricciones que se deben cumplir
def _pde(m, i, j):
    if i == 0 or i == L or j == 0:
        return Constraint.Skip
    return 2**2*m.dudx2[i,j] == m.dudt2[i,j]
m.pde = Constraint(m.x, m.t, rule=_pde)

def _initcon(m, i):
    if i == 0 or i == L:
        return Constraint.Skip
    return m.u[i,0] == sin(m.pi*i)+sin(2*m.pi*i)
m.initcon = Constraint(m.x,  rule=_initcon)

def _lowerbound(m, j):
    return m.u[0,j] == 0
m.lowerbound = Constraint(m.t, rule=_lowerbound)

def _upperbound(m, j):
    return m.u[1,j] == 0
m.upperbound = Constraint(m.t, rule=_upperbound)

#funcion objetivo "dummy"
m.obj = Objective(expr=1)

#numero de puntos
Nx = 60 #puntos de la coordenada espacial x
Nt = 180 #puntos de la coordenada temporal t

#discretizar puntos
discretize = Finite_Difference_Transformation()
disc = discretize.apply(m,nfe=Nx,wrt=m.x,scheme='BACKWARD')
disc = discretize.apply(m,nfe=Nt,wrt=m.t,scheme='BACKWARD',clonemodel=False)

#resolver usando el solver ipopt
solver = SolverFactory('ipopt')
results = solver.solve(m, tee=True)

#obtencion de la solcion en x, t y U
import numpy as np

x = sorted(disc.x)
t = sorted(disc.t)
u = sorted(disc.u)

U = np.zeros((Nx+1, Nt+1))
for i in range(Nx+1):
    ix = x[i]
    for j in range(Nt+1):
        it = t[j]
        U[i,j] = value(disc.u[ix,it])

np.save("x", x)
np.save("t", t)
np.save("U", U)

"""
aniacion de la solucion obtenida
"""
import matplotlib.animation as animation
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

line, = ax.plot(x, U[:,0])

def animate(j):
    line.set_ydata(U[:,j])  # update the data
    return line,


# Init only required for blitting to give a clean slate.
def init():
    line.set_ydata(np.ma.array(U[:,0], mask=True))
    return line,

ani = animation.FuncAnimation(fig, animate, np.arange(1, Nt+1), init_func=init,
                              interval=40, blit=True)
ax.set_xlim(0, x[-1])
ax.set_ylim(-max(U[:,0]), max(U[:,0]))
ax.set_xlabel('x')
ax.set_ylabel('u')
plt.show()
