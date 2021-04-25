from numpy import exp, log, sin, sqrt, pi 
from dolfin import *
set_log_active(False)

# Lipton formula is valid only when rho = 0 !!!
def dbo_heston_anly(s0=100., y0=0.12, K=85., r=0.03, kappa=1.5, theta=0.1, xi=0.5, cp=1, T=1., L=65., U=135., n_terms=30000):
    ans = 0.
    for n in range(1, n_terms + 1):
        kn = pi * n / log(U / L)
        mu = - 1. / 2. * kappa
        zeta = 1. / 2. * sqrt(kn**2 * xi**2 + kappa**2 + xi**2 / 4.)
        AA = -1. * kappa * theta * (mu + zeta) * T - kappa * theta * log((-mu + zeta + (mu + zeta) * exp(-2. * zeta * T)) / (2. * zeta)) 
        BB = (xi ** 2 * (kn ** 2 + 1. / 4.) * (1 - exp(-2. * zeta * T))) / (4. * (-mu + zeta + (mu + zeta) * exp(-2. * zeta * T)))
        if cp == 1:
            phin = 2. * ((-1)**(n+1) * kn * (sqrt(U / K) - sqrt(K / U)) + sin(kn * log(L / K))) / ((kn ** 2 + 1. / 4.) * log(U / L))
        else:
            phin = 2. * (kn * (sqrt(K / L) - sqrt(L / K)) + sin(kn * log(L / K))) /  ((kn ** 2 + 1. / 4.) * log(U / L))
        ans += exp(2. * (AA - BB * y0) / (xi**2)) * phin * sin(kn * log(s0 / L))
    
    return exp(- r * T) * sqrt(s0 * K) * ans

def dbo_heston_fem(s0=100., y0=0.12, K=85., r=0.03, q=0.03, kappa=1.5, theta=0.1, xi=0.5, rho=0.0, cp=1, T=1., L=65., U=135., dt=1./100):
    s0 = log(s0 / K)
    mesh_size = (100, 100) 
    domain = ((log(L / K), log(U / K)), (0., 3)) 
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
            values[0] = 0
        
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
            values[0] = 0
        
    u_s_max = BoundaryValues_s_max(degree=2)
    bc_s_max = DirichletBC(V, u_s_max, boundary_s_max)

    bcs = [bc_s_min, bc_s_max,]

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
    l = u0 * v * dx

    A_, b_ = None, None
    u = Function(V)

    for i in range(1, int(T / dt) + 1):
        t = i * dt 
        u_s_min.set_t(t)
        u_s_max.set_t(t)
        
        A_, b_ = assemble_system(a, l, bcs)
        solve(A_, u.vector(), b_)

        u0.assign(u)

    return u(s0, y0)
