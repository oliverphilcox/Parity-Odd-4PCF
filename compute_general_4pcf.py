### Compute the Odd-Parity 4PCF arising from general scale-invariant parity-odd trispectrum as in Philcox 2021
# The output contains the 4PCF multiplets, but without the prefactors of d_0^odd and (-d_1^odd/3)

import numpy as np, time, tqdm, multiprocessing as mp, itertools
from classy import Class
from scipy.interpolate import interp1d
from scipy.integrate import simps
from scipy.special import spherical_jn, sici
from sympy.physics.wigner import wigner_3j, wigner_6j, wigner_9j

init_time = time.time()

########################### PARAMETERS #########################################

# Cosmology
OmegaM = 0.307115
OmegaL = 0.692885
Omegab = 0.048
sigma8 = 0.8288
h = 0.6777

# Sample Parameters
b = 2.0
z = 0.57

# Binning
R_min = 20
R_max = 160
n_r = 10
LMAX_data = 4

# Internal L
LMAX = 8

# CPU-cores
cores = 24

print("Radial binning: %d bins in [%d,%d]"%(n_r,R_min,R_max))
print("ell_max for data: %d"%LMAX_data)
print("LMAX for internal sums: %d"%LMAX)

# k, x, x' arrays
dk = 0.003
kmax = 3
dx = 0.5
xmax = 2*R_max

### Load matter power spectrum and transfer function
cosmo = Class()
cosmo.set({'output':'mPk,mTk,tCl','z_pk':z,'h':h,'Omega_b':Omegab,
           'Omega_cdm':(OmegaM-Omegab),'sigma8':sigma8,'P_k_max_1/Mpc':100.0})
cosmo.compute()
fz = cosmo.scale_independent_growth_factor_f(z)

# Compute number of multiplets
n_l = 0
l1s,l2s,l3s = [],[],[]
for l1 in range(LMAX_data+1):
    for l2 in range(LMAX_data+1):
        for l3 in range(abs(l1-l2),min([l1+l2,LMAX_data])+1):
            if (-1.)**(l1+l2+l3)==1: continue
            n_l+=1
            l1s.append(l1)
            l2s.append(l2)
            l3s.append(l3)
print("Using %d multiplets in data\n"%n_l)

# Load transfer functions
k_arr = np.logspace(-4,2,10000)
pk_lin_arr = np.asarray([cosmo.pk_lin(kk*h,z)*h**3. for kk in k_arr])
T_all = cosmo.get_transfer(z,'class')
kTk = T_all['k (h/Mpc)']
Tk = T_all['d_tot']

# Compute interpolator for curvature power spectrum and transfer function
Tk_arr = interp1d(kTk,Tk)(k_arr)
p_zeta = interp1d(k_arr,pk_lin_arr/Tk_arr**2.)
M = interp1d(k_arr,Tk_arr)

# Compute x,x',k arrays
x_arr = np.arange(0,xmax,dx)
xp_arr = np.arange(0,xmax,dx)
k_arr = np.arange(dk,kmax,dk)[:,None]

def integ_bessel(ell,bin1,k):
    """Return Bessel functions integrated over finite bins. Only ell<=5 are currently implemented"""
    r_min = bin1*(R_max-R_min)/n_r+R_min
    r_max = (bin1+1.)*(R_max-R_min)/n_r+R_min
    if ell==0:
        tmp1 = (-k*r_max*np.cos(k*r_max)+np.sin(k*r_max))/k**3.
        tmp2 = (-k*r_min*np.cos(k*r_min)+np.sin(k*r_min))/k**3.
    elif ell==1:
        tmp1 = -2.*np.cos(k*r_max)/k**3. - r_max*np.sin(k*r_max)/k**2.
        tmp2 = -2.*np.cos(k*r_min)/k**3. - r_min*np.sin(k*r_min)/k**2.
    elif ell==2:
        tmp1 = (r_max*np.cos(k*r_max))/k**2 - (4*np.sin(k*r_max))/k**3 + (3*sici(k*r_max)[0])/k**3
        tmp2 = (r_min*np.cos(k*r_min))/k**2 - (4*np.sin(k*r_min))/k**3 + (3*sici(k*r_min)[0])/k**3
    elif ell==3:
        tmp1 = (7.*np.cos(k*r_max))/k**3 - (15.*np.sin(k*r_max))/(k**4*r_max) + (r_max*np.sin(k*r_max))/k**2
        # avoid zero errors!
        if r_min==0:
            tmp2 = -(8./k**3.)
        else:
            tmp2 = (7.*np.cos(k*r_min))/k**3 - (15.*np.sin(k*r_min))/(k**4*r_min) + (r_min*np.sin(k*r_min))/k**2
    elif ell==4:
        tmp1 = (105.*np.cos(k*r_max))/(2.*k**4*r_max) - (r_max*np.cos(k*r_max))/k**2 + (11*np.sin(k*r_max))/k**3 -(105*np.sin(k*r_max))/(2.*k**5*r_max**2) + (15*sici(k*r_max)[0])/(2.*k**3)
        if r_min==0:
            tmp2 = 0.
        else:
            tmp2 = (105.*np.cos(k*r_min))/(2.*k**4*r_min) - (r_min*np.cos(k*r_min))/k**2 + (11*np.sin(k*r_min))/k**3 -(105*np.sin(k*r_min))/(2.*k**5*r_min**2) + (15*sici(k*r_min)[0])/(2.*k**3)
    elif ell==5:
        tmp1 = ((315*k*r_max - 16*k**3*r_max**3)*np.cos(k*r_max) - (315 - 105*k**2*r_max**2 + k**4*r_max**4)*np.sin(k*r_max))/(k**6*r_max**3)
        if r_min==0:
            tmp2 = -16./k**3.
        else:
            tmp2 = ((315*k*r_min - 16*k**3*r_min**3)*np.cos(k*r_min) - (315 - 105*k**2*r_min**2 + k**4*r_min**4)*np.sin(k*r_min))/(k**6*r_min**3)
    else:
        raise Exception("not implemented yet!")
    return (tmp1-tmp2)/((r_max**3.-r_min**3.)/3.)


########################### I, J, K INTEGRALS ##################################

# Compute j_l(kx) possibilities
jl_kx = []
print("Computing j_ell(kx)")
for l in range(LMAX+1):
    jl_kx.append(spherical_jn(l,k_arr*x_arr))

# Compute j_l(kx) possibilities
jl_kxp = []
print("Computing j_ell(kx')")
for l in range(LMAX+1):
    jl_kxp.append(spherical_jn(l,k_arr*xp_arr))

# Compute j_ell(kr) possibilities
# add one extra row for the r = 0 case!
jell_kr = []
print("Computing j_ell(kr)")
for ell in range(LMAX_data+1):
    tmp_jell_kr = []
    for rbin in range(n_r):
        tmp_jell_kr.append(integ_bessel(ell,rbin,k_arr))
    tmp_jell_kr.append(np.ones_like(integ_bessel(ell,rbin,k_arr)))
    jell_kr.append(tmp_jell_kr)

### Compute the k1 - k4 integrals

Mk = M(k_arr)
pk_zeta = p_zeta(k_arr)
k2_norm = k_arr**2./(2.*np.pi**2.)
x2 = (x_arr**2.)[:,None]
x_s = x_arr[:,None]
x2p = xp_arr**2.

def Pk1(L, lH, bH):
    return simps(k2_norm*Mk*pk_zeta*jell_kr[lH][bH]*jl_kx[L],k_arr,axis=0)

def Pk2(L, lH, bH):
    return simps(k2_norm*Mk*jell_kr[lH][bH]*jl_kx[L],k_arr,axis=0)

def Pk3(L, lH, bH):
    return simps(k2_norm*Mk*pk_zeta*jell_kr[lH][bH]*jl_kxp[L],k_arr,axis=0)

def Pk4(L, lH, bH):
    return simps(k2_norm*Mk*jell_kr[lH][bH]*jl_kxp[L],k_arr,axis=0)

def Ps(L1, L2):
    return simps(k2_norm[:,None]*pk_zeta[:,None]*jl_kx[L1][:,:,None]*jl_kxp[L2][:,None,:],k_arr[:,None],axis=0)

# Compute K(x,x') arrays
print("Computing K arrays")
Ps_arr = []
for L1 in range(LMAX+1):
    Ps1 = []
    for L2 in range(LMAX+1):
        Ps1.append(Ps(L1,L2))
    Ps_arr.append(Ps1)

# Compute I(x) and J(x) arrays
Pk1_arr = []
Pk2_arr = []
Pk3_arr = []
Pk4_arr = []
print("Computing I and J arrays\n")
for L in range(LMAX+1):
    Pk1a,Pk2a,Pk3a,Pk4a = [],[],[],[]
    for l in range(LMAX_data+1):
        Pk1b,Pk2b,Pk3b,Pk4b = [],[],[],[]
        for b in range(n_r+1):
            Pk1b.append(Pk1(L,l,b)[:,None])
            Pk2b.append(Pk2(L,l,b)[:,None])
            Pk3b.append(Pk3(L,l,b)[None,:])
            Pk4b.append(Pk4(L,l,b)[None,:])
        Pk1a.append(Pk1b)
        Pk2a.append(Pk2b)
        Pk3a.append(Pk3b)
        Pk4a.append(Pk4b)
    Pk1_arr.append(Pk1a)
    Pk2_arr.append(Pk2a)
    Pk3_arr.append(Pk3a)
    Pk4_arr.append(Pk4a)

######################## FUNCTION DEFINITIONS ##################################

# Compute Wigner 3j matrix
tj0 = lambda l1,l2,l3: np.float64(wigner_3j(l1,l2,l3,0,0,0))

def ninej(l1,l2,l3,lp1,lp2,lp3,lpp1,lpp2,lpp3):
    """Wigner 9j symbol, checking if two rows are equal (in which case result = 0 if odd sum of elements)"""
    if (-1.)**(l1+l2+l3+lp1+lp2+lp3+lpp1+lpp2+lpp3)==-1:
        if [l1,l2,l3]==[lp1,lp2,lp3] or [l1,l2,l3]==[lpp1,lpp2,lpp3] or [lp1,lp2,lp3]==[lpp1,lpp2,lpp3]:
            return 0.
        elif [l1,lp1,lpp1]==[l2,lp2,lpp2] or [l1,lp1,lpp1]==[l3,lp3,lpp3] or [l2,lp2,lpp2]==[l3,lp3,lpp3]:
            return 0.
        else:
            return np.float64(wigner_9j(l1,l2,l3,lp1,lp2,lp3,lpp1,lpp2,lpp3))
    else:
        return np.float64(wigner_9j(l1,l2,l3,lp1,lp2,lp3,lpp1,lpp2,lpp3))

C_ell = lambda Ls: np.sqrt(np.product(2.*np.asarray(Ls)+1.))

def Z_ell(ell):
    """Compute the Z_ell coefficients"""
    if ell==0:
        return (b+fz/3.)
    elif ell==2:
        return (2./15.*fz)
    else:
        raise Exception('Wrong ell!')

def Gaunt4(L,Lp,Lpp,wigs=[]):
    """Return G^{L,L',L''} for N=4.
    Indexing is L = {l1,l2,l12,l3,l4}.
    We skip computing Wigner 3j symbols if they are supplied."""
    if len(wigs)!=0:
        wig3s = wigs[0]*wigs[1]*wigs[2]*wigs[3]
    else:
        wig3s = tj0(L[0],Lp[0],Lpp[0]) # l1
        wig3s *= tj0(L[1],Lp[1],Lpp[1]) # l2
        wig3s *= tj0(L[3],Lp[3],Lpp[3]) # l3
        wig3s *= tj0(L[4],Lp[4],Lpp[4]) # l4
    wig9s = ninej(L[0],L[1],L[2],Lp[0],Lp[1],Lp[2],Lpp[0],Lpp[1],Lpp[2])
    if wig9s==0: return 0.
    wig9s *= ninej(L[2],L[3],L[4],Lp[2],Lp[3],Lp[4],Lpp[2],Lpp[3],Lpp[4])
    if wig9s==0: return 0.
    pref = C_ell(L)*C_ell(Lp)*C_ell(Lpp)/(4.*np.pi)**2.
    return pref*wig3s*wig9s

def Gaunt5(L,Lp,Lpp,wigs=[]):
    """Return G^{L,L',L''} for N=5.
    Indexing is L = {l1,l2,l12,l3,l123,l4,l5}.
    We skip computing Wigner 3j symbols if they are supplied."""
    if len(wigs)!=0:
        wig3s = wigs[0]*wigs[1]*wigs[2]*wigs[3]*wigs[4]
    else:
        wig3s = tj0(L[0],Lp[0],Lpp[0]) # l1
        wig3s *= tj0(L[1],Lp[1],Lpp[1]) # l2
        wig3s *= tj0(L[3],Lp[3],Lpp[3]) # l3
        wig3s *= tj0(L[5],Lp[5],Lpp[5]) # l4
        wig3s *= tj0(L[6],Lp[6],Lpp[6]) # l5
    wig9s = ninej(L[0],L[1],L[2],Lp[0],Lp[1],Lp[2],Lpp[0],Lpp[1],Lpp[2])
    if wig9s==0: return 0.
    wig9s *= ninej(L[2],L[3],L[4],Lp[2],Lp[3],Lp[4],Lpp[2],Lpp[3],Lpp[4])
    if wig9s==0: return 0.
    wig9s *= ninej(L[4],L[5],L[6],Lp[4],Lp[5],Lp[6],Lpp[4],Lpp[5],Lpp[6])
    if wig9s==0: return 0.
    pref = C_ell(L)*C_ell(Lp)*C_ell(Lpp)/(4.*np.pi)**2.5
    return pref*wig3s*wig9s

def contract5(L,Lp,L5zero=False):
    """Contract P_Lambda and P_Lambda'. Returns a list of Lambda' and the weights.
    Note that we compute the Wigner-3j symbols as early as possible to skip computing zeros.
    We also note that 9j symbols are zero unless all {L1, L2, L3} and {L1, L1', L1''} triplets obey triangle conditions.

    If L5zero = True, we skip any multiplets with L5'' \neq 0, since these are not needed later.
    """
    ells = []
    weights = []
    for Lpp1 in range(abs(L[0]-Lp[0]),min([LMAX,L[0]+Lp[0]])+1):
        if (-1.)**(L[0]+Lp[0]+Lpp1)==-1: continue
        wig1 = tj0(L[0],Lp[0],Lpp1)
        for Lpp2 in range(abs(L[1]-Lp[1]),min([LMAX,L[1]+Lp[1]])+1):
            wig2 = tj0(L[1],Lp[1],Lpp2)
            if (-1.)**(L[1]+Lp[1]+Lpp2)==-1: continue
            for Lpp12 in range(abs(Lpp1-Lpp2),Lpp1+Lpp2+1):
                if Lpp12<abs(L[2]-Lp[2]) or Lpp12>(L[2]+Lp[2]): continue
                for Lpp3 in range(abs(L[3]-Lp[3]),min([LMAX,L[3]+Lp[3]])+1):
                    if (-1.)**(L[3]+Lp[3]+Lpp3)==-1: continue
                    wig3 = tj0(L[3],Lp[3],Lpp3)
                    for Lpp123 in range(abs(Lpp12-Lpp3),Lpp12+Lpp3+1):
                        if Lpp123<abs(L[4]-Lp[4]) or Lpp123>(L[4]+Lp[4]): continue
                        for Lpp4 in range(abs(L[5]-Lp[5]),min([LMAX,L[5]+Lp[5]])+1):
                            if (-1.)**(L[5]+Lp[5]+Lpp4)==-1: continue
                            wig4 = tj0(L[5],Lp[5],Lpp4)
                            for Lpp5 in range(abs(L[6]-Lp[6]),min([LMAX,L[6]+Lp[6]])+1):
                                if L5zero==True and Lpp5>0: continue
                                if (-1.)**(L[6]+Lp[6]+Lpp5)==-1: continue
                                wig5 = tj0(L[6],Lp[6],Lpp5)

                                # Compute Gaunt coefficient
                                gaunt = (-1.)**(Lpp1+Lpp2+Lpp3+Lpp4+Lpp5)*Gaunt5(L,Lp,[Lpp1,Lpp2,Lpp12,Lpp3,Lpp123,Lpp4,Lpp5],wigs=[wig1,wig2,wig3,wig4,wig5])
                                if gaunt==0: continue

                                ells.append([Lpp1,Lpp2,Lpp12,Lpp3,Lpp123,Lpp4,Lpp5])
                                weights.append(gaunt)

        return np.asarray(ells),np.asarray(weights)

def load_integrals(lH1,lH2,lH3,lH4,bH1,bH2,bH3,bH4,L1,L2,L3,L4,L5,L5p,phiH,tjs=np.inf):
    """Load parts involving radial bins, from I*I*J*J*K integrals."""
    if tjs==np.inf:
        pref = tj0(L1,L2,L5)*tj0(L3,L4,L5p)
        if pref==0: return 0.
    else:
        pref = tjs
    pref *= (4.*np.pi)**11.*phiH*np.sqrt(2.)*(1.0j)**(L1+L2+L3+L4-L5+L5p) # adding (4pi)^4 prefactor from angular coupling
    pref *= C_ell([L1,L2,L3,L4,L5,L5p])
    pref *= (-1.0j)**(lH1+lH2+lH3+lH4)

    integ = Pk1_arr[L1][lH1][bH1]*Pk2_arr[L2][lH2][bH2]*Pk3_arr[L3][lH3][bH3]*Pk4_arr[L4][lH4][bH4]*Ps_arr[L5][L5p]
    out = pref*simps(simps(integ*x2,x_s,axis=0)*x2p,xp_arr)
    return out

def kernel1(L1,L2,L3,L4,L5,L5p):
    """Load part of coupling matrix which is ell-independent. This computes two terms; for simplicity we add a factor of 1.0j to the first to allow for later separation."""

    # First contract {L1,L2,(L5),0,(L5),0,L5} and {0,0,(0),L3,(L3),L4,L5p} contributions
    first_contract = contract5([L1,L2,L5,0,L5,0,L5],[0,0,0,L3,L3,L4,L5p])
    if len(first_contract[0])==0: return []

    # Now contract these outputs with {lambda1,0,lambda1,lambda2,lambda3,0,lambda3} parts
    lam1s = [1,2,2,1]
    lam2s = [1,2,1,2]
    lam3s = [1,1,2,2]
    lamWs = [1.0j,1./np.sqrt(5.),1./np.sqrt(5.),-1./np.sqrt(5.)] # weights

    # NB: we only allow for contributions from bins with L''_5 = 0, since this is required to couple with next contractions
    s_ell, s_weight = [],[]
    for j in range(len(first_contract[0])):
        first_ells = first_contract[0][j]
        first_w = first_contract[1][j]
        for i in range(len(lamWs)):
            tmp_ells,tmp_weights = contract5([lam1s[i],0,lam1s[i],lam2s[i],lam3s[i],0,lam3s[i]],first_ells,L5zero=True)
            if len(tmp_ells)==0: continue
            tmp_weights = np.asarray(tmp_weights,dtype=np.complex128)
            tmp_weights *= lamWs[i]*first_w

            s_ell += list(tmp_ells)
            s_weight += list(tmp_weights)
    if len(s_ell)==0: return []
    
    # Combine identical elements
    s_ell = np.ascontiguousarray(s_ell)
    order = np.lexsort(s_ell.T)
    diff = np.diff(s_ell[order], axis=0)
    uniq_mask = np.append(True, (diff != 0).any(axis=1))
    
    uniq_inds = order[uniq_mask]
    inv_idx = np.zeros_like(order)
    inv_idx[order] = np.cumsum(uniq_mask) - 1
    
    s_weight = np.bincount(inv_idx, weights=np.asarray(s_weight).real) + 1.0j*np.bincount(inv_idx, weights=np.asarray(s_weight).imag)
    s_ell = s_ell[uniq_inds]
    
    second_contract = [np.asarray(s_ell),np.asarray(s_weight)]
    return second_contract

def kernel2(L1,L2,L3,L4,L5,L5p,second_contract,lH1,lH2,lH12,lH3,lH4):
    """Load ell-dependent part of coupling matrix."""
    # Now perform Gaunt integral over the new P_Lambda functions with the P_j and P_ell functions and sum the weights
    sum_contrib = 0.
    for k in range(len(second_contract[0])):
        LL1,LL2,LL12,LL3,_,LL4,LL5 = second_contract[0][k]
        assert LL5==0
        second_w = second_contract[1][k]

        ### Iterate over j, ensuring triangle conditions are respected
        for j1 in [0,2]:
            if j1<abs(LL1-lH1) or j1>LL1+lH1: continue
            if (-1.)**(j1+LL1+lH1)==-1: continue
            wig1 = tj0(LL1,j1,lH1)
            for j2 in [0,2]:
                if j2<abs(LL2-lH2) or j2>LL2+lH2: continue
                if (-1.)**(LL2+j2+lH2)==-1: continue
                wig2 = tj0(LL2,j2,lH2)
                for j12 in [0,2,4]:
                    if j12<abs(j1-j2) or j12>j1+j2: continue
                    if j12<abs(LL12-lH12) or j12>LL12+lH12: continue
                    tja = tj0(j1,j2,j12)
                    for j3 in [0,2]:
                        if j3<abs(LL3-lH3) or j3>LL3+lH3: continue
                        if (-1.)**(LL3+j3+lH3)==-1: continue
                        wig3 = tj0(LL3,j3,lH3)
                        for j4 in [0,2]:
                            if j4<abs(j12-j3) or j4>j12+j3: continue
                            if j4<abs(LL4-lH4) or j4>LL4+lH4: continue
                            if (-1.)**(LL4+j4+lH4)==-1: continue
                            tjb = tj0(j12,j3,j4)
                            wig4 = tj0(LL4,j4,lH4)

                            contrib = tja*tjb*Z_ell(j1)*Z_ell(j2)*Z_ell(j3)*Z_ell(j4)*C_ell([j1,j2,j12,j3,j4])

                            # Compute coupling using N=4 symbol for speed
                            coupling = Gaunt4([LL1,LL2,LL12,LL3,LL4],[j1,j2,j12,j3,j4],[lH1,lH2,lH12,lH3,lH4],wigs=[wig1,wig2,wig3,wig4])/(4.*np.pi)**1.5

                            if coupling*contrib==0: continue
                            sum_contrib += coupling*contrib*second_w

    return sum_contrib

######################## COMPUTE L COUPLINGS ###################################

# Compute all possible L-dependent coupling-matrix contributions
print("Computing L couplings")
t_L_couplings = time.time()

def load_L_couplings(L1):
    """Load couplings for given L1"""
    this_L_couplings = []
    for L2 in range(LMAX+1):
        for L5 in range(abs(L1-L2),min([LMAX,L1+L2])+1):
            tj125 = tj0(L1,L2,L5)
            if tj125==0: continue
            for L3 in range(LMAX+1):
                for L4 in range(LMAX+1):
                    for L5p in range(abs(L3-L4),min([LMAX,L3+L4])+1):
                        tj345p = tj0(L3,L4,L5p)
                        if tj345p==0: continue

                        second_contract = kernel1(L1,L2,L3,L4,L5,L5p)
                        if len(second_contract)==0: continue
                        this_L_couplings.append([L1,L2,L3,L4,L5,L5p,second_contract,tj125*tj345p])
    return this_L_couplings

p = mp.Pool(cores)

outs = list(tqdm.tqdm(p.imap_unordered(load_L_couplings,np.arange(LMAX+1)),total=LMAX+1))
p.close()
p.join()

L_couplings = []
for i in range(len(outs)):
    L_couplings += outs[i]

t_L_couplings = time.time()-t_L_couplings
print("Loaded %d L couplings in %.2f s on %d cores"%(len(L_couplings),t_L_couplings,cores))

###################### COMPUTE PERMUTATIONS ####################################

indices=set(itertools.permutations([0,1,2,3]))
sum_output = np.zeros((n_l,int(n_r*(n_r-1)*(n_r-2)/6)),dtype=np.complex64)

def load_perm(index_id):
    """Load 4PCF contributions from a single permutation."""

    sum_output = np.zeros((n_l,int(n_r*(n_r-1)*(n_r-2)/6)),dtype=np.complex64)
    t_coupling,t_matrix = 0.,0.

    # Sum over all Ls
    coupling_kernel = 0.
    for L_i in range(len(L_couplings)):
        if L_i%10==0: print("Permutation %d: L index %d of %d"%(index_id,L_i,len(L_couplings)))
        L1,L2,L3,L4,L5,L5p,second_contract,tjprod = L_couplings[L_i]

        # Iterate over odd-parity {l1,l2,l3}
        l_index = 0
        for l1 in range(LMAX_data+1):
            for l2 in range(LMAX_data+1):
                for l3 in range(abs(l1-l2),min([l1+l2,LMAX_data])+1):
                    if (-1.)**(l1+l2+l3)==1: continue

                    # Define permutated ells
                    ells = np.asarray([l1,l2,l3,0])
                    ells_perm = ells[list(list(indices)[index_id])]
                    lH1,lH2,lH3,lH4 = ells_perm

                    # Determine intermediate ell
                    if lH1==0: lH12=lH2
                    elif lH2==0: lH12 = lH1
                    elif lH3==0: lH12 = lH4
                    elif lH4==0: lH12 = lH3
                    else: raise Exception("wrong ells!")

                    # Determine permutation factor
                    cnt = 0
                    ells_perm = np.delete(ells_perm,np.where(ells_perm==0)[0][0])
                    for i in range(3):
                        for j in range(i+1,3):
                            if (ells_perm[i]>ells_perm[j]):
                                cnt+=1
                    if cnt%2==0: phiH = 1
                    else: phiH = -1

                    ta = time.time()
                    coupling_kernel = kernel2(L1,L2,L3,L4,L5,L5p,second_contract,lH1,lH2,lH12,lH3,lH4)
                    t_coupling += time.time()-ta
                    if coupling_kernel==0:
                        l_index += 1
                        continue

                    # Load in the integrals
                    tb = time.time()
                    bin_index = 0
                    for b1 in range(n_r):
                        for b2 in range(b1+1,n_r):
                            for b3 in range(b2+1,n_r):
                                # Define bin quadruplet
                                bH1, bH2, bH3, bH4 = np.asarray([b1,b2,b3,-1])[list(list(indices)[index_id])]

                                integ = load_integrals(lH1,lH2,lH3,lH4,bH1,bH2,bH3,bH4,L1,L2,L3,L4,L5,L5p,phiH,tjs=tjprod)
                                if integ==0: continue

                                sum_output[l_index,bin_index] += integ*coupling_kernel

                                bin_index += 1
                    t_matrix += time.time()-tb
                    l_index += 1

    print("Coupling time: %.2f s"%t_coupling)
    print("Matrix time: %.2f s"%t_matrix)
    return sum_output

t_perms = time.time()
p = mp.Pool(cores)
out = list(tqdm.tqdm(p.imap_unordered(load_perm,np.arange(len(indices))),total=len(indices)))
p.close()
p.join()

for i in range(len(indices)):
    sum_output += out[i]
t_perms = time.time()-t_perms

print("Computed %d permutations in %.2f s on %d cores"%(len(indices),t_perms,cores))

# Separate into first and second term, accounting for earlier factor of 1.0j
first_term = 1.0j*np.imag(-1.0j*sum_output)
second_term = 1.0j*np.real(-1.0j*sum_output)

############################ SAVE & EXIT #######################################

outfile1 = '/home/ophilcox/Parity-Odd-4PCF/general_4pcf_L%d_term1.txt'%LMAX
np.savetxt(outfile1,first_term)
outfile2 = '/home/ophilcox/Parity-Odd-4PCF/general_4pcf_L%d_term2.txt'%LMAX
np.savetxt(outfile2,second_term)
print("Saved output to %s / %s after %.1f s; exiting."%(outfile1, outfile2, time.time()-init_time))