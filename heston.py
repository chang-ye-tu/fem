from numpy import real, sqrt, log, exp, pi, arange, zeros
from scipy.integrate import trapz

def heston_anly(S, y0, K, r, q, kappa, theta, sigma, rho, T, lmbd=0., cp=1, L=1.e-6, U=50, dp=1.e-2):
    def heston_prob(phi, pnum):
        x = log(S)
        a = kappa * theta

        if pnum == 1:
            u = 0.5
            b = kappa + lmbd - rho * sigma
        else:
            u = -0.5
            b = kappa + lmbd
        d = sqrt((rho * sigma * 1.j * phi - b)**2 - sigma**2 * (2 * u * 1.j * phi - phi**2))
        g = (b - rho * sigma * 1.j * phi + d) / (b - rho * sigma * 1.j * phi - d)
        
        c = 1 / g
        D = (b - rho * sigma * 1.j * phi - d) / sigma**2 * ((1 - exp(-d * T)) / (1 - c * exp(-d * T)))
        G = (1 - c * exp(-d * T)) / (1 - c)
        C = (r - q) * 1.j * phi * T + a / sigma**2 * ((b - rho * sigma * 1.j * phi - d) * T - 2 * log(G))

        f = exp(C + D * y0 + 1.j * phi * x)

        return real(exp(-1.j * phi * log(K)) * f / 1.j / phi)
    
    phi = arange(L, U, dp)
    N = len(phi)
    
    int1 = zeros(N)
    int2 = zeros(N)
    for k in range(N):
        int1[k] = heston_prob(phi[k], 1)
        int2[k] = heston_prob(phi[k], 2)

    I1 = trapz(int1) * dp
    I2 = trapz(int2) * dp
    P1 = 1. / 2 + 1. / pi * I1
    P2 = 1. / 2 + 1. / pi * I2

    call = S * exp(-q * T) * P1 - K * exp(-r * T) * P2 
    put = call - S * exp(-q * T) + K * exp(-r * T)

    return (call if cp == 1 else put)

print(heston_anly(S=100, y0=0.25, K=110, r=0.05, q=0.01, kappa=1., theta=0.09, sigma=0.4, rho=-0.7, T=1.))

from dolfin import *
set_log_active(False)

#def heston_fem(s0=100., y0=0.0175, K=100., r=0.025, q=0.0, 
#               kappa=1.5768, theta=0.0398, xi=0.5751, rho=-0.5711, cp=1,
#               T=3., dt=1./100):
#def heston_fem(s0=100., y0=0.04, K=100., r=0.025, q=0.0, 
#               kappa=1.5, theta=0.04, xi=0.3, rho=-0.9, cp=1,
#               T=3., dt=1./100):
#def heston_fem(s0=100., y0=0.04, K=100., r=0.025, q=0.0, 
#               kappa=1.5, theta=0.04, xi=0.3, rho=0.0, cp=1,
#               T=0.5, dt=1./100):
def heston_fem(s0=100., y0=0.25, K=110., r=0.05, q=0.01, kappa=1., theta=0.09, xi=0.4, rho=-0.7, cp=1, T=1., dt=1./100):
    s0 = log(s0 / K)
    mesh_size = (100, 100) 
    domain = ((log(5. / K), log(400. / K)), (0., 3)) 
    s_min, s_max = domain[0]
    y_min, y_max = domain[1]
    
    mesh = RectangleMesh(Point(s_min, y_min), Point(s_max, y_max), mesh_size[0], mesh_size[1], 'right/left')
    V = FunctionSpace(mesh, 'Lagrange', 2)
    n = FacetNormal(mesh)
    
    #################################################################
    ## boundary conditions
    #################################################################
    
    # bc as s -> s_min
    def boundary_s_min(x, on_boundary):
        return on_boundary and near(x[0], s_min, 1e-14)
    
    class BoundaryValues_s_min(UserExpression):
        
        def set_t(self, t):
            self.t = t
        
        def value_shape(self):
            return ()

        def eval(self, values, x):
            values[0] = (1 - cp) / 2 * max(cp * (K * exp(x[0]) * exp(-q * self.t) - K * exp(-r * self.t)), 0)
        
    u_s_min = BoundaryValues_s_min(degree=2)
    bc_s_min = DirichletBC(V, u_s_min, boundary_s_min)
    
    # bc as s -> s_max
    def boundary_s_max(x, on_boundary):
        return on_boundary and near(x[0], s_max, 1e-14)

    class BoundaryValues_s_max(UserExpression):
        
        def set_t(self, t):
            self.t = t
        
        def value_shape(self):
            return ()

        def eval(self, values, x):
            values[0] = (1 + cp) / 2 * max(cp * (K * exp(x[0]) * exp(-q * self.t) - K * exp(-r * self.t)), 0)
        
    u_s_max = BoundaryValues_s_max(degree=2)
    bc_s_max = DirichletBC(V, u_s_max, boundary_s_max)

    # bc as y -> y_min 
    def boundary_y_min(x, on_boundary):
        return on_boundary and near(x[1], y_min, 1e-14)
    
    class BoundaryValues_y_min(UserExpression):
        
        def set_t(self, t):
            self.t = t
        
        def value_shape(self):
            return ()

        def eval(self, values, x):
            values[0] = max(cp * (K * exp(x[0]) * exp(-q * self.t) - K * exp(-r * self.t)), 0) 
        
    u_y_min = BoundaryValues_y_min(degree=2)
    bc_y_min = DirichletBC(V, u_y_min, boundary_y_min)

    # bc as y -> y_max
    def boundary_y_max(x, on_boundary):
        return on_boundary and near(x[1], y_max, 1e-14)

    class BoundaryValues_y_max(UserExpression):
        
        def set_t(self, t):
            self.t = t
        
        def value_shape(self):
            return ()

        def eval(self, values, x):
            values[0] = (1 + cp) / 2 * K * exp(x[0]) * exp(-q * self.t) + (1 - cp) / 2 * K * exp(-r * self.t) 
        
    u_y_max = BoundaryValues_y_max(degree=2)
    bc_y_max = DirichletBC(V, u_y_max, boundary_y_max)

    #bcs = [bc_s_min, bc_s_max, bc_y_min, bc_y_max]
    bcs = [bc_s_min, bc_s_max, bc_y_max]
    
    #################################################################
    ##
    #################################################################

    u0 = interpolate(Expression('fmax(cp * (K * exp(x[0]) - K), 0)', degree=2, cp=cp, K=K), V)

    s, y = SpatialCoordinate(mesh)
    u = TrialFunction(V)
    v = TestFunction(V)
    A = as_matrix(((y / 2, rho * xi * y / 2), (rho * xi * y / 2, xi ** 2 * y / 2))) 
    b = as_vector(((y + rho * xi) / 2 - (r - q), xi ** 2 / 2 - kappa * (theta - y))) 
    a = u * v * dx - dot(A * grad(u), n) * v * dt * ds + dot(A * grad(u), grad(v)) * dt * dx + dot(b, grad(u)) * v * dt * dx + r * u * v * dt * dx
    L = u0 * v * dx

    A_, b_ = None, None
    u = Function(V)

    for i in range(1, int(T / dt) + 1):
        t = i * dt 

        # update boundary condition
        u_s_min.set_t(t)
        u_s_max.set_t(t)
        u_y_min.set_t(t)
        u_y_max.set_t(t)

        A_, b_ = assemble_system(a, L, bcs)
        solve(A_, u.vector(), b_)

        # update previous solution
        u0.assign(u)

    return u(s0, y0)

print(heston_fem())

#Vv = VectorFunctionSpace(mesh, 'P', 1)
#gradu = project(grad(u), Vv)
#print('grad u(0.5, 0.5)', gradu((0.5, 0.5)))
