from fenics import *
from numpy import exp, log, sqrt 
from numpy.linalg import cholesky
import scipy.stats as st

N = st.norm.cdf

def bs(cp, s0, K, t, sigma, r, q):
    d1 = (log(s0 / K) + (r - q + 0.5 * sigma ** 2) * t) / (sigma * sqrt(t))
    d2 = d1 - sigma * sqrt(t)
    return cp * s0 * exp(-q * t) * N(cp * d1) - cp * K * exp(-r * t) * N(cp * d2)

def two_assets_corr_call(domain1, domain2, K, r, mesh_size, s0, sigma, T, d_t):
    
    sigma = cholesky(sigma)

    a_11 = sigma[0, 0] * sigma[0, 0] + sigma[0, 1] * sigma[0, 1]
    a_22 = sigma[1, 0] * sigma[1, 0] + sigma[1, 1] * sigma[1, 1]
    a_12 = sigma[0, 0] * sigma[1, 0] + sigma[0, 1] * sigma[1, 1]
    a_21 = sigma[1, 0] * sigma[0, 0] + sigma[1, 1] * sigma[0, 1]
    rho = 0.5 * (a_12 + a_21) / sqrt(a_11 * a_22)

    s_1_min, s_1_max = domain1
    s_2_min, s_2_max = domain2

    mesh = RectangleMesh(Point(s_1_min, s_2_min), Point(s_1_max, s_2_max), mesh_size[0], mesh_size[1], 'right/left')
    V = FunctionSpace(mesh, 'Lagrange', 2)
    n = FacetNormal(mesh)

    #################################################################
    ##
    #################################################################
    
    u_s_1_min = Constant(0)
    def boundary_s_1_min(x, on_boundary):
        return on_boundary and near(x[0], s_1_min, 1e-10)
    bc_s_1_min = DirichletBC(V, u_s_1_min, boundary_s_1_min)
    
    u_s_2_min = Constant(0)
    def boundary_s_2_min(x, on_boundary):
        return on_boundary and near(x[1], s_2_min, 1e-10)
    bc_s_2_min = DirichletBC(V, u_s_2_min, boundary_s_2_min)

    def boundary_s_1_max(x, on_boundary):
        return on_boundary and near(x[0], s_1_max, 1e-10)

    class BoundaryValues_s_1_max(UserExpression):

        def set_t(self, t):
            self.t = t
        
        def eval(self, values, x):
            values[0] = bs(1, x[1], K[1], self.t, sqrt(a_22), r, 0)

        def value_shape(self):
            return ()

    u_s_1_max = BoundaryValues_s_1_max(degree=2)
    bc_s_1_max = DirichletBC(V, u_s_1_max, boundary_s_1_max)

    def boundary_s_2_max(x, on_boundary):
        return on_boundary and near(x[1], s_2_max, 1e-10)

    class BoundaryValues_s_2_max(UserExpression):

        def set_t(self, t):
            self.t = t
        
        def eval(self, values, x):
            y_1 = (log(x[0] / K[0]) + (r - a_11 / 2) * self.t) / sqrt(a_11 * self.t)
            values[0] = x[1] * N(y_1 + rho * sqrt(a_22 * self.t)) - K[1] * exp(-r * self.t) * N(y_1) 

        def value_shape(self):
            return ()

    u_s_2_max = BoundaryValues_s_2_max(degree=2)
    bc_s_2_max = DirichletBC(V, u_s_2_max, boundary_s_2_max)

    bcs = [bc_s_1_min, bc_s_2_min, bc_s_1_max, bc_s_2_max]

    #################################################################
    ##
    #################################################################
    
    u0 = interpolate(Expression('x[0] > K_0 ? fmax(x[1] - K_1, 0) : 0', degree=2, K_0=K[0], K_1=K[1], r=r, t=0), V)

    x1, x2 = SpatialCoordinate(mesh)
    u = TrialFunction(V)
    v = TestFunction(V)
    A = as_matrix(((a_11 * x1 ** 2 / 2, a_12 * x1 * x2 / 2), (a_21 * x2 * x1 / 2, a_22 * x2 ** 2 / 2))) 
    b = as_vector((-(r - a_11 - a_21 / 2) * x1, -(r - a_12 / 2 - a_22) * x2)) 
    a = u * v * dx - dot(A * grad(u), n) * v * d_t * ds + dot(A * grad(u), grad(v)) * d_t * dx + dot(b, grad(u)) * v * d_t * dx + r * u * v * d_t * dx

    L = u0 * v * dx

    A_, b_ = None, None
    u = Function(V)

    for i in range(1, int(T / d_t) + 1):
        t = i * d_t
        u_s_1_max.set_t(t)
        u_s_2_max.set_t(t)

        A_, b_ = assemble_system(a, L, bcs)
        solve(A_, u.vector(), b_)

        u0.assign(u)

    return u(s0[0], s0[1])
