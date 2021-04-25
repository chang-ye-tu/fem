from dolfin import *
from numpy import sqrt, exp, log 
import scipy.stats as st

N = st.norm.cdf

def bs(s0, K, sigma, r, T, q=0, cp=1):
    # cp=1: call, cp=-1: put
    d1 = (log(s0 / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return cp * s0 * exp(-q * T) * N(cp * d1) - cp * K * exp(-r * T) * N(cp * d2)

# c.f. Haug E. G. The Complete Guide to Option Pricing Formulas 2ed pp.157--pp.158 
def dbo_bs_anly(sigma, r, T, L, U, S, X, cp=1, q=0, delta1=0, delta2=0, n_terms=10):
    b = r if q == 0 else r - q

    def mu1(n):
        return 2 * (b - delta2 - n * (delta1 - delta2)) / (sigma**2) + 1 
    def mu2(n):
        return 2 * n * (delta1 - delta2) / (sigma**2)  
    def mu3(n):
        return 2 * (b - delta2 + n * (delta1 - delta2)) / (sigma**2) + 1 
    
    F = U * exp(delta1 * T)
    E = L * exp(delta2 * T)

    def d1(n):
        return (log((S * U**(2*n)) / (X * L**(2*n))) + (b + sigma**2 / 2) * T) / (sigma * sqrt(T))
    def d2(n):
        return (log((S * U**(2*n)) / (F * L**(2*n))) + (b + sigma**2 / 2) * T) / (sigma * sqrt(T))
    def d3(n):
        return (log((L**(2*n + 2)) / (X * S * U**(2*n))) + (b + sigma**2 / 2) * T) / (sigma * sqrt(T))
    def d4(n):
        return (log((L**(2*n + 2)) / (F * S * U**(2*n))) + (b + sigma**2 / 2) * T) / (sigma * sqrt(T))
    def y1(n):
        return (log((S * U**(2*n)) / (E * L**(2*n))) + (b + sigma**2 / 2) * T) / (sigma * sqrt(T))
    def y2(n):
        return (log((S * U**(2*n)) / (X * L**(2*n))) + (b + sigma**2 / 2) * T) / (sigma * sqrt(T))
    def y3(n):
        return (log((L**(2*n + 2)) / (E * S * U**(2*n))) + (b + sigma**2 / 2) * T) / (sigma * sqrt(T))
    def y4(n):
        return (log((L**(2*n + 2)) / (X * S * U**(2*n))) + (b + sigma**2 / 2) * T) / (sigma * sqrt(T))

    if cp == 1: # call
        return S * exp((b - r) * T) * sum([((U**n) / (L**n))**mu1(n) * (L / S)**mu2(n) * (N(d1(n)) - N(d2(n))) - ((L**(n+1)) / (U**n * S))**mu3(n) * (N(d3(n)) - N(d4(n))) for n in range(-n_terms, n_terms + 1)]) - X * exp(-r * T) * sum([((U**n)/(L**n))**(mu1(n) - 2) * (L/S)**mu2(n) * (N(d1(n) - sigma * sqrt(T)) - N(d2(n) - sigma * sqrt(T))) - ((L**(n+1)) / (U**n * S))**(mu3(n) - 2) * (N(d3(n) - sigma * sqrt(T)) - N(d4(n) - sigma * sqrt(T))) for n in range(-n_terms, n_terms + 1)])

    else: # put
        return X * exp(-r * T) * sum([((U**n)/(L**n))**(mu1(n) - 2) * (L/S)**mu2(n) * (N(y1(n) - sigma * sqrt(T)) - N(y2(n) - sigma * sqrt(T))) - ((L**(n+1)) / (U**n * S))**(mu3(n) - 2) * (N(y3(n) - sigma * sqrt(T)) - N(y4(n) - sigma * sqrt(T))) for n in range(-n_terms, n_terms + 1)]) - S * exp((b - r) * T) * sum([((U**n) / (L**n))**mu1(n) * (L / S)**mu2(n) * (N(y1(n)) - N(y2(n))) - ((L**(n+1)) / (U**n * S))**mu3(n) * (N(y3(n)) - N(y4(n))) for n in range(-n_terms, n_terms + 1)]) 

def dbo_bs_fem(s0, K, sigma, r, T, dt, lb, ub, cp=1):
    n_el = 1000  # number of elements
    mesh = IntervalMesh(n_el, lb, ub)
    V = FunctionSpace(mesh, 'Lagrange', 2)

    # bc as s -> lb
    def boundary_lb(x, on_boundary):
        return on_boundary and near(x[0], lb, 1e-10)
    bc_lb = DirichletBC(V, Constant(0), boundary_lb)

    # bc as s -> ub
    def boundary_ub(x, on_boundary):
        return on_boundary and near(x[0], ub, 1e-10)
    bc_ub = DirichletBC(V, Constant(0), boundary_ub)

    bcs = [bc_lb, bc_ub]
    
    u0 = interpolate(Expression('fmax(cp * (x[0] - K), 0)', degree = 2, K = K, cp = cp), V)
    u = TrialFunction(V)
    v = TestFunction(V)
    el = V.ufl_element()
    exp1 = Expression('0.5 * pow(sigma, 2) * pow(x[0], 2) * dt', sigma=sigma, dt=dt, element=el)
    exp2 = Expression('(-r + pow(sigma, 2)) * x[0] * dt', r=r, sigma=sigma, dt=dt, element=el)
    exp3 = Expression('r * dt', r=r, dt=dt, element=el)
    a = u * v * dx + exp1 * u.dx(0) * v.dx(0) * dx + exp2 * u.dx(0) * v * dx + exp3 * u * v * dx
    L = u0 * v * dx

    A, b = None, None
    _ = Function(V)
    for i in range(1, int(T / dt) + 1):
        A, b = assemble_system(a, L, bcs)
        solve(A, _.vector(), b)
        u0.assign(_) 
    return _(s0)
