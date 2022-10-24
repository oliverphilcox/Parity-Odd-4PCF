### Compute the Odd-Parity 4PCF arising from ghost inflation as in Cabass+22
# The output contains the 4PCF multiplets, but without a prefactor of (Delta^2_zeta)^3*Lambda^5(H tilde-Lambda)^{3/2}/[Lambda^2_PV tilde-Lambda^6 Gamma(3/4)^2]

import sys
sys.path.append('/home/ophilcox/wigxjpf-1.11/')
import numpy as np, time, tqdm, multiprocessing as mp, itertools
from classy import Class
from scipy.interpolate import interp1d
from scipy.integrate import simps
from scipy.special import spherical_jn, sici, hankel1
import pywigxjpf as wig

if __name__=='__main__':
    init_time = time.time()

    if len(sys.argv)!=2: 
        raise Exception("LMAX not specified!")

    # Internal L (sets computation accuracy, LMAX ~ 8 is probably good, but LMAX ~ 4 may be okay)
    LMAX = int(sys.argv[1])

    # Cosmology
    OmegaM = 0.307115
    OmegaL = 0.692885
    Omegab = 0.048
    sigma8 = 0.8288
    h = 0.6777
    ns = 1

    # Galaxy sample Parameters
    b = 2.0
    z = 0.57

    # Binning (from measured 4PCF, not to be changed)
    R_min = 20
    R_max = 160
    n_r = 10
    LMAX_data = 4

    # CPU-cores
    cores = 24

    print("Radial binning: %d bins in [%d,%d]"%(n_r,R_min,R_max))
    print("ell_max for data: %d"%LMAX_data)
    print("LMAX for internal sums: %d"%LMAX)

    # q, y, lambda arrays
    # note that we use a discrete array for y
    dq = 0.025 #0.005
    qmax = 5
    dy = 0.2 #0.01
    ymax = 100
    dlambda = 0.5 #0.1
    lambdamax = 250.

    ### Load matter power spectrum and transfer function
    cosmo = Class()
    cosmo.set({'output':'mPk,mTk,tCl','z_pk':z,'h':h,'Omega_b':Omegab,'n_s':ns,
            'Omega_cdm':(OmegaM-Omegab),'sigma8':sigma8,'P_k_max_1/Mpc':100.0})
    cosmo.compute()
    fz = cosmo.scale_independent_growth_factor_f(z)

    if ns!=1:
        raise Exception("Only flat templates computed!")

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
    T_all = cosmo.get_transfer(z,'class')
    kTk = T_all['k (h/Mpc)']
    Tk = T_all['d_tot']

    # Compute interpolator for curvature power spectrum and transfer function (with D(z) factor)
    Tk_arr = interp1d(kTk,Tk)(k_arr)*cosmo.scale_independent_growth_factor(z)
    M_tmp = interp1d(np.log(k_arr),np.log(-Tk_arr),fill_value='extrapolate',bounds_error=False)
    M = lambda k: -np.exp(M_tmp(np.log(k)))

    # 3j/9j set-up
    wig.wig_table_init(2*(2*LMAX),9)
    wig.wig_temp_init(2*(2*LMAX))

    # Wigner 3j + 9j
    tj0 = lambda l1, l2, l3: wig.wig3jj(2*l1 , 2*l2, 2*l3, 0, 0, 0)
    ninej = lambda l1,l2,l3,lp1,lp2,lp3,lpp1,lpp2,lpp3: wig.wig9jj(2*l1,2*l2,2*l3,2*lp1,2*lp2,2*lp3,2*lpp1,2*lpp2,2*lpp3)

    # Compute x,lambda,k arrays
    q_arr = np.arange(dq/100,qmax,dq)[:,None,None]
    y_arr = np.arange(dy/100,ymax,dy)[None,:,None]
    lambda_arr = np.arange(dlambda/100,lambdamax,dlambda)[None,None,:]
    print("N_q: %d"%len(q_arr.ravel()))
    print("N_y: %d"%len(y_arr.ravel()))
    print("N_lambda: %d"%len(lambda_arr.ravel()))

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

    ########################### J INTEGRALS ##################################

    integ_time = time.time()

    # Compute j_l(q*y) possibilities
    jl_kx = []
    print("Computing j_L(kx)")
    for l in range(LMAX+1):
        jl_kx.append(spherical_jn(l,q_arr*y_arr))

    # Compute j_ell(kr) possibilities
    # add one extra row for the r = 0 case!
    jell_kr = []
    print("Computing bin-averaged j_ell(kr)")
    for ell in range(LMAX_data+1):
        tmp_jell_kr = []
        for rbin in range(n_r):
            tmp_jell_kr.append(integ_bessel(ell,rbin,q_arr/lambda_arr))
        tmp_jell_kr.append(np.ones_like(integ_bessel(ell,rbin,q_arr/lambda_arr)))
        jell_kr.append(tmp_jell_kr)

    # Compute H_{alpha}(2iq^2) possibilities
    print("Computing Hankel functions")
    Hankel34=hankel1(3./4.,2.*1.0j*q_arr**2.)
    Hankelm14=hankel1(-1./4.,2.*1.0j*q_arr**2.)

    ### Compute the various I integrals
    Mk = M(q_arr/lambda_arr)
    q2_norm = q_arr**2./(2.*np.pi**2.)
    qrav = q_arr.ravel()

    J_pref = q2_norm*Mk*dq

    def J_integrals(lH, L, rBin):
        """Compute all necessary J integrals for a given l, L and bin"""
        master_integrand = J_pref*jell_kr[lH][rBin]*jl_kx[L]
        J_34_12 = np.sum(master_integrand*q_arr**(1./2.)*Hankel34,axis=0)
        J_34_52 = np.sum(master_integrand*q_arr**(5./2.)*Hankel34,axis=0)
        J_34_32 = np.sum(master_integrand*q_arr**(3./2.)*Hankel34,axis=0)
        J_m14_12 = np.sum(master_integrand*q_arr**(1./2.)*Hankelm14,axis=0)
        return J_34_12, J_34_52, J_34_32, J_m14_12

    # Compute J(x,lambda) arrays for the necessary alpha / beta
    print("Computing J arrays")
    J_34_12_arr = np.zeros((LMAX+1,LMAX_data+1,n_r+1,len(y_arr.ravel()),len(lambda_arr.ravel())),dtype='complex')
    J_34_52_arr = np.zeros((LMAX+1,LMAX_data+1,n_r+1,len(y_arr.ravel()),len(lambda_arr.ravel())),dtype='complex')
    J_34_32_arr = np.zeros((LMAX+1,LMAX_data+1,n_r+1,len(y_arr.ravel()),len(lambda_arr.ravel())),dtype='complex')
    J_m14_12_arr = np.zeros((LMAX+1,LMAX_data+1,n_r+1,len(y_arr.ravel()),len(lambda_arr.ravel())),dtype='complex')

    def compute_J_integral(L):
        all_ints = np.zeros((4,LMAX_data+1,n_r+1,len(y_arr.ravel()),len(lambda_arr.ravel())),dtype='complex')
        for l in range(LMAX_data+1):
            for b in range(n_r+1):
                all_ints[:,l,b] = J_integrals(l,L,b)
        return all_ints

    p = mp.Pool(cores)
    out = list(tqdm.tqdm(p.imap(compute_J_integral,np.arange(LMAX+1)),total=LMAX+1))
    p.close()
    p.join()

    for i in range(LMAX+1):
        J_34_12_arr[i] = out[i][0]
        J_34_52_arr[i] = out[i][1]
        J_34_32_arr[i] = out[i][2]
        J_m14_12_arr[i] = out[i][3]
        
    integ_time = time.time()-integ_time
    print("J integrals took %.3f seconds"%integ_time)

    ########################### COMPUTE RADIAL INTEGRALS ###########################

    C_ell = lambda Ls: np.sqrt(np.product(2.*np.asarray(Ls)+1.))

    def Z_ell(ell):
        """Compute the Z_ell coefficients, which are the spherical harmonic expansion of the Kaiser kernel"""
        if ell==0:
            return (b+fz/3.)
        elif ell==2:
            return (2./15.*fz)
        else:
            raise Exception('Wrong ell!')

    y2lam2 = (y_arr**2.*lambda_arr**-1.*dlambda*dy)[0]

    def load_integrals(lH1,lH2,lH3,lH4,bH1,bH2,bH3,bH4,L1,L2,L3,L4,Lp,phiH,tjprod=np.inf):
        """Load contribution to 4PCF involving radial bins (but not the coupling matrix), and perform the y and lambda integrals.
        
        NB: the x integral is done via summation over Bessel functions, rather than integration, using the trick of Philcox & Slepian 2021.
        
        We keep only the imaginary part (the real part is ~ 0)"""
        
        # Compute 3j symbols if needed
        if tjprod!=np.inf:
            pref = tjprod
        else:
            pref = tj0(L1,L2,Lp)*tj0(Lp,L3,L4)
            if pref==0: return 0.
        
        pref *= (8.*np.sqrt(2.))/(3.*np.sqrt(5.))*(4.*np.pi)**(11./2.)*(-1.0j)**(lH1+lH2+lH3+lH4)*phiH*(1.0j)**(L1+L2+L3+L4)
        pref *= C_ell([L1,L2,L3,L4,Lp])
        
        integ = J_34_12_arr[L1][lH1][bH1]*J_34_52_arr[L2][lH2][bH2]*J_34_32_arr[L3][lH3][bH3]*J_m14_12_arr[L4][lH4][bH4]
        out = pref*np.sum(integ*y2lam2)
        return out.imag

    ########################### COUPLING MATRICES ###########################

    def check(l1,l2,l3,even=True):
        """Check triangle conditions for a triplet of momenta"""
        if l1<abs(l2-l3): return True
        if l1>l2+l3: return True
        if even:
            if (-1)**(l1+l2+l3)==-1: return True
        return False

    class wigner_acc():
        def __init__(self,nmax):
            """Accelerator for 3j and 9j computation. This only computes a symbol explicitly if it has not been computed before."""
            self.recent_inputs9 = []
            self.recent_outputs9 = []
            self.recent_inputs3 = []
            self.recent_outputs3 = []
            self.nmax = nmax
            self.calls9 = 0
            self.computes9 = 0
            self.calls3 = 0
            self.computes3 = 0
        
        def ninej(self,l1,l2,l3,lp1,lp2,lp3,lpp1,lpp2,lpp3):
            """9j accelerator"""
            self.calls9 += 1
            if [l1,l2,l3,lp1,lp2,lp3,lpp1,lpp2,lpp3] in self.recent_inputs9:
                return self.recent_outputs9[self.recent_inputs9.index([l1,l2,l3,lp1,lp2,lp3,lpp1,lpp2,lpp3])]
            else:
                val = ninej(l1,l2,l3,lp1,lp2,lp3,lpp1,lpp2,lpp3)
                self.computes9+=1
                self.recent_inputs9.append([l1,l2,l3,lp1,lp2,lp3,lpp1,lpp2,lpp3])
                self.recent_outputs9.append(val)
            if len(self.recent_inputs9)>self.nmax:
                self.recent_inputs9 = self.recent_inputs9[1:]
                self.recent_outputs9 = self.recent_outputs9[1:]
            return val
        
        def tj0(self,l1,l2,l3):
            """3j (m1=m2=m3=0) accelerator"""
            self.calls3 += 1
            if [l1,l2,l3] in self.recent_inputs3:
                return self.recent_outputs3[self.recent_inputs3.index([l1,l2,l3])]
            else:
                val = tj0(l1,l2,l3)
                self.computes3+=1
                self.recent_inputs3.append([l1,l2,l3])
                self.recent_outputs3.append(val)
            if len(self.recent_inputs3)>self.nmax:
                self.recent_inputs3 = self.recent_inputs3[1:]
                self.recent_outputs3 = self.recent_outputs3[1:]
            return val

    def coupling_matrix(lH1,lH2,lH12,lH3,lH4,L1,L2,L3,L4,Lp,acc=None):
        """Compute the M coupling matrix"""
        
        # Assemble matrix prefactor
        pref = 15.*C_ell([lH1,lH2,lH12,lH3,lH4])*C_ell([L1,L2,Lp,L3,L4,L4])
        output = 0.
        
        # Sum over lambda
        for lam1 in range(abs(L1-2),L1+3,2):
            tj1 = acc.tj0(2,L1,lam1)
            
            for lam2 in range(abs(L2-2),L2+3,2):
                tj12 = tj1*acc.tj0(2,L2,lam2)
                        
                for lam12 in range(abs(lam1-lam2),lam1+lam2+1):
                    if check(1,Lp,lam12,False): continue
                    
                    for lam3 in range(abs(L3-1),L3+2,2):
                        if check(lam12,lam3,L4,False): continue
                        
                        tj123 = tj12*acc.tj0(1,L3,lam3)
                        
                        # lambda factor
                        lam_piece = (-1)**(lam1+lam2+lam3)*C_ell([lam1,lam2,lam12,lam3])**2.

                        # nine-js (j-independent)
                        nj1 = acc.ninej(2,2,1,L1,L2,Lp,lam1,lam2,lam12)
                        if nj1==0: continue
                        nj1 *= acc.ninej(1,1,0,Lp,L3,L4,lam12,lam3,L4)
                        if nj1==0: continue
                        
                        # Sum over j
                        for j1 in [0,2]:
                            if check(lH1,lam1,j1): continue
                            ttj1 = acc.tj0(j1,lH1,lam1)
                            
                            for j2 in [0,2]:
                                if check(lH2,lam2,j2): continue
                                ttj12 = ttj1*acc.tj0(j2,lH2,lam2)
                                
                                for j12 in [0,2,4]:
                                    if check(j1,j2,j12): continue
                                    if check(lH12,lam12,j12,0): continue
                                    
                                    # Assemble j pieces
                                    j_piece1 = acc.tj0(j1,j2,j12)
                                    
                                    for j3 in [0,2]:
                                        if check(lH3,lam3,j3): continue
                                        ttj123 = ttj12*acc.tj0(j3,lH3,lam3)
                                    
                                        for j4 in range(max([0,abs(lH4-L4)]),min([2,lH4+L4])+1,2):
                                            if check(j12,j3,j4): continue
                                            if check(j4,lH4,L4): continue
                                                
                                            ttj1234 = ttj123*acc.tj0(j4,lH4,L4)
                                    
                                            j_piece12 = j_piece1*acc.tj0(j12,j3,j4)
                                            j_piece12 *= Z_ell(j1)*Z_ell(j2)*Z_ell(j3)*Z_ell(j4)*C_ell([j1,j2,j12,j3,j4])**2.

                                            # nine-js (j-dependent)
                                            nj2 = acc.ninej(j1,j2,j12,lH1,lH2,lH12,lam1,lam2,lam12)
                                            if nj2==0: continue
                                            nj2 *= acc.ninej(j12,j3,j4,lH12,lH3,lH4,lam12,lam3,L4)
                                            if nj2==0: continue
                                                
                                            output += pref*j_piece12*lam_piece*tj123*ttj1234*nj1*nj2
                                            
        return output

    ########################### COMPUTE ZETA FOR SINGLE PERMUTATION ###########################

    # Permutations of [0,1,2,3]
    indices=set(itertools.permutations([0,1,2,3]))

    def load_perm(index_id):
        """Load 4PCF contributions from a single permutation."""

        sum_output = np.zeros((n_l,int(n_r*(n_r-1)*(n_r-2)/6)))
        t_coupling,t_matrix = 0.,0.

        # Create 3j/9j accelerator (to be threadsafe)
        acc = wigner_acc(100000)

        # Sum over Ls
        for L1 in range(LMAX+1):
            for L2 in range(LMAX+1):
                for Lp in range(abs(L1-L2),L1+L2+1,2):
                    # Compute first 3j symbol
                    tj1 = acc.tj0(L1,L2,Lp)
                    if tj1==0: continue
                    
                    for L3 in range(LMAX+1):
                        for L4 in range(LMAX+1):
                            if (-1)**(L1+L2+L3+L4)==-1: continue
                            if Lp<abs(L3-L4): continue
                            if Lp>L3+L4: continue
                            
                            # Compute second 3j symbol
                            tj2 = acc.tj0(Lp,L3,L4)
                            if tj2==0: continue
                                
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
                                        if (-1.)**(L4+lH4)==-1: 
                                            l_index += 1
                                            continue # from j4-lH4-L4 3j, given that j4 is even
            
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

                                        # Load coupling
                                        ta = time.time()
                                        coupling_kernel = coupling_matrix(lH1,lH2,lH12,lH3,lH4,L1,L2,L3,L4,Lp,acc=acc)
                                        
                                        t_coupling += time.time()-ta
                                        if coupling_kernel==0:
                                            l_index += 1
                                            continue
                                        
                                        # Load integrals    
                                        tb = time.time()
                                        bin_index = 0
                                        for b1 in range(n_r):
                                            for b2 in range(b1+1,n_r):
                                                for b3 in range(b2+1,n_r):
                                                    # Define bin quadruplet
                                                    bH1, bH2, bH3, bH4 = np.asarray([b1,b2,b3,-1])[list(list(indices)[index_id])]
                                                    
                                                    # compute integrals
                                                    integ = load_integrals(lH1,lH2,lH3,lH4,bH1,bH2,bH3,bH4,L1,L2,L3,L4,Lp,phiH,tj1*tj2)
                                                    if integ!=0: 
                                                        sum_output[l_index,bin_index] += integ*coupling_kernel
                                                    
                                                    bin_index += 1
                                        t_matrix += time.time()-tb
                                        l_index += 1
        if index_id==0:
            print("Coupling time: %.2f s"%t_coupling)
            print("Matrix time: %.2f s"%t_matrix)
            print("3j calls: %d, 3j computations: %d"%(acc.calls3,acc.computes3))
            print("9j calls: %d, 9j computations: %d"%(acc.calls9,acc.computes9))
            
        return sum_output


    ########################### COMPUTE ALL PERMUTATIONS ###########################

    all_output = np.zeros((n_l,int(n_r*(n_r-1)*(n_r-2)/6)))
    t_perms = time.time()
    p = mp.Pool(cores)
    out = list(tqdm.tqdm(p.imap_unordered(load_perm,np.arange(len(indices))),total=len(indices)))
    p.close()
    p.join()

    for i in range(len(indices)):
        all_output += out[i]
    t_perms = time.time()-t_perms

    print("Computed %d permutations in %.2f s on %d cores"%(len(indices),t_perms,cores))

    outfile = 'ghost_4pcf_L%d.txt'%LMAX
    np.savetxt(outfile,all_output)

    print("Saved output to %s after %.1f s; exiting."%(outfile,time.time()-init_time))
