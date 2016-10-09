# Required imports
from pyomo.environ import *
from pyomo.dae import *
from pyomo.dae.plugins.finitedifference import Finite_Difference_Transformation

m = ConcreteModel()

m.pi = Param(initialize=3.1416)

#variables del problema
m.t = ContinuousSet(bounds=(0, 2))
m.x = ContinuousSet(bounds=(0, 1))
m.u = Var(m.x, m.t)

#variables derivadas respecto a las variables independientes x t
m.dudx  = DerivativeVar(m.u, wrt=m.x)
m.dudx2 = DerivativeVar(m.u, wrt=(m.x,m.x))
m.dudt  = DerivativeVar(m.u, wrt=m.t)

#restricciones que se deben cumplir
def _pde(m, i, j):
    if i == 0 or i == 1 or j == 0:
        return Constraint.Skip
    return m.pi**2*m.dudt[i,j] == m.dudx2[i,j]
m.pde = Constraint(m.x, m.t, rule=_pde)

def _initcon(m, i):
    if i == 0 or i == 1:
        return Constraint.Skip
    return m.u[i,0] == sin(m.pi*i)
m.initcon = Constraint(m.x,  rule=_initcon)

def _lowerbound(m, j):
    return m.u[0,j] == 0
m.lowerbound = Constraint(m.t, rule=_lowerbound)

def _upperbound(m, j):
    return m.pi*exp(-j)+m.dudx[1,j] == 0
m.upperbound = Constraint(m.t, rule=_upperbound)

#funcion objetivo "dummy"
m.obj = Objective(expr=1)

#numero de puntos
Nx = 60 #puntos de la coordenada espacial x
Nt = 25 #puntos de la coordenada temporal t

#discretizar puntos
discretize = Finite_Difference_Transformation()
disc = discretize.apply(m,nfe=Nx,wrt=m.x,scheme='BACKWARD')
disc = discretize.apply(m,nfe=Nt,wrt=m.t,scheme='BACKWARD')

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
#representacion grafica
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as plt

from matplotlib import cm

fig = plt.figure(figsize=plt.figaspect(0.5))

ax = fig.add_subplot(1, 2, 1, projection='3d')
tt, xx = np.meshgrid(t, x)
#figura 1
surf = ax.plot_surface(xx, tt, U, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.set_zlim3d(0, 1.01)
fig.colorbar(surf, shrink=0.5, aspect=5)
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('u')
# figura 2
ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.plot_wireframe(xx, tt, U)
ax.set_zlim3d(0, 1.01)
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('u')
#guardar figura
#plt.savefig('heat2.png')
plt.show()
