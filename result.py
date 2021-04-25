import os
os.chdir(os.path.dirname(__file__))
from timeit import default_timer as timer
from datetime import timedelta
from numpy import array
from tabulate import tabulate

import dbo_bs, dbo_heston, two_assets_corr_call, two_assets_max_call

# Run with
#   docker run --rm -v $(pwd):/home/fenics/shared -w /home/fenics/shared reg "python3 ./result.py"
#
#####################################################################
#
#
#
#####################################################################

def create_dbo_bs():
    start = timer()
    s0 = 100
    K = 100
    r = .1
    dt = 1e-4

    llu = [[50, 150], [60, 140], [70, 130], [80, 120], [90, 110]]
    list_sigma = [.15, .25, .35]
    list_T = [.25, .5]
    list_delta = [[0, 0],]
    cp = [1,]

    for cp_ in cp:
        l = []
        for T in list_T:
            for sigma in list_sigma:
                for lb, ub in llu:
                    for delta1, delta2 in list_delta:
                        u_a = dbo_bs.dbo_bs_anly(sigma=sigma, r=r, T=T, L=lb, U=ub, S=s0, X=K, delta1=delta1, delta2=delta2, cp=cp_)
                        u_n = dbo_bs.dbo_bs_fem(s0=s0, K=K, sigma=sigma, r=r, T=T, dt=dt, lb=lb, ub=ub, cp=cp_)
                        l.append([T, lb, ub, sigma, u_n, u_a, abs((u_n - u_a) / u_a)])
        open('dbo_bs_%s.txt' % ('call' if cp_ == 1 else 'put',), 'w').write(tabulate(l, headers=['T', 'lower', 'upper', r'$\sigma$', 'FEM', 'analytic', 'error'], floatfmt=('.2f', '.0f', '.0f', '.2f', '.6f', '.6f', '.6f'), tablefmt='latex_raw')) 
    print(timedelta(seconds=timer()-start))

create_dbo_bs()

#####################################################################
#
#
#
#####################################################################

def create_dbo_heston():
    start = timer()
    s0 = 100.
    y0 = 0.12
    K = 100.
    r = 0.03
    kappa = 1.5
    theta = 0.1
    xi = 0.5
    dt = 1./1000

    llu = [[50, 150], [60, 140], [70, 130], [80, 120], [90, 110]]
    list_T = [0.25, 0.5, 1.]
    cp = [1, -1]
    for cp_ in cp:
        l = []
        for T in list_T:
            for lb, ub in llu:
                u_a = dbo_heston.dbo_heston_anly(s0=s0, y0=y0, K=K, r=r, kappa=kappa, theta=theta, xi=xi, T=T, L=lb, U=ub, cp=cp_)
                u_n = dbo_heston.dbo_heston_fem(s0=s0, y0=y0, K=K, r=r, q=r, kappa=kappa, theta=theta, xi=xi, T=T, L=lb, U=ub, dt=dt, cp=cp_)
                l.append([T, lb, ub, u_n, u_a, abs((u_n - u_a) / u_a)])
        open('dbo_heston_%s.txt' % ('call' if cp_ == 1 else 'put',), 'w').write(tabulate(l, headers=['T', 'lower', 'upper', 'FEM', 'analytic', 'error'], floatfmt=('.2f', '.0f', '.0f', '.6f', '.6f', '.4f'), tablefmt='latex_raw'))

    print(timedelta(seconds=timer()-start))

create_dbo_heston()

#####################################################################
#
#
#
#####################################################################

def create_two_assets_txt(f, l):
    open(f, 'w').write(tabulate(l, headers=['corr', 'sigma1', 'sigma2', 'FEM'], floatfmt=('.2f', '.2f', '.2f', '.6f'), tablefmt='latex')) 

def create_two_assets_corr_call():
    s01 = 52 
    s02 = 65
    size1 = 100
    size2 = 100
    K1 = 50
    K2 = 70
    d_t = 0.01
    T = 0.5
    r = 0.1

    l = [] 
    for corr in [-0.75, -0.5, -0.25, 0.25, 0.5, 0.75]:
        for sigma1, sigma2 in [(0.1, 0.15), (0.2, 0.3), (0.4, 0.6), (0.8, 1.2),  (1.6, 2.4), (3.2, 4.8), (6.4, 9.6)]:
            u = two_assets_corr_call.two_assets_corr_call([0.1, 200], [0.1, 200], [K1, K2], r, [size1, size2], [s01, s02], array([[sigma1**2, sigma1 * sigma2 * corr], [sigma1 * sigma2 * corr, sigma2**2]]), T, d_t)
            l.append((corr, sigma1, sigma2, u))        
    create_two_assets_txt('two_assets_corr_call.txt', l)    

create_two_assets_corr_call()

def create_two_assets_max_call():
    s01 = 52
    s02 = 65
    size1 = 100
    size2 = 100
    K1 = 50
    K2 = 70 
    d_t = 0.01
    T = 0.5
    r = 0.1

    l = []
    for corr in [-0.75, -0.5, -0.25, 0.25, 0.5, 0.75]:
        for sigma1, sigma2 in [(0.1, 0.15), (0.2, 0.3), (0.4, 0.6), (0.6, 0.5), (0.8, 1.2),  (1.6, 2.4), (3.2, 4.8), (6.4, 9.6)]:
            u = two_assets_max_call.two_assets_max_call([0.1, 200], [K1, K2], r, [size1, size2], [s01, s02], array([[sigma1**2, sigma1 * sigma2 * corr], [sigma1 * sigma2 * corr, sigma2**2]]), T, d_t)
            l.append((corr, sigma1, sigma2, u))        
    create_two_assets_txt('two_assets_max_call.txt', l)

create_two_assets_max_call()
