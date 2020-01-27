from __future__ import print_function, division
import sys,os

from quspin.basis import spin_basis_1d,boson_basis_1d, photon_basis, tensor_basis # Hilbert space bases
from quspin.operators import hamiltonian, quantum_operator # Hamiltonian and observables
from quspin.tools.measurements import obs_vs_time # t_dep measurements
from quspin.tools.Floquet import Floquet,Floquet_t_vec # Floquet Hamiltonian
from quspin.basis.photon import coherent_state # HO coherent state
import numpy as np # generic math functions
import scipy as sp
import pandas as pd

from quspin.operators import exp_op # operators
from quspin.basis import spin_basis_general # spin basis constructor
from quspin.tools.measurements import ent_entropy # Entanglement Entropy

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from qutip import *
from qutip.piqs import *

from scipy.sparse import load_npz, save_npz


class spin_cavity:

    def __init__(self, num_spins, construction='Chain', spin_interactions=None):
        if num_spins > 24:
            ValueError("Good luck with that system size")

        self.S=num_spins;
        self.construction=construction;
        self.interaction_dict=spin_interactions;
        self.neighbor=None;


    def set_spin_interactions(self, spin_interactions):
        '''Ensure that the new spin_interactions follows the appropriate format for identifying spin_interactions

        Parameters
        -----------
        spin_interactions : dict
                Dictionary that defines the type of keywords and corresponding adjacency matrix for each.
                Currently accepts {'Heisenberg', 'Kitaev', 'Field'}.
        '''

        assert type(spin_interactions) == dict, "Spin interactions must be stored in dictionary" ;

        keywords = ('Field', 'Heisenberg', 'Kitaev');
        for keyword in keywords:
            assert keyword in spin_interactions.keys(), keyword + "term is missing";
            adj_shape = spin_interaction(keyword).shape();
            assert (self.S, 3) < adj_shape, "Adjacency Matrix for " + keyword + " is too small.";

        self.interaction_dict = spin_interactions;

    def _generate_spin_Hamiltonian(self):

        if self.construction == "Chain":
            Ham,static = _chain_Hamiltonian(self.S, spin_interaction)

        elif self.construction == "Ladder":
            Ham,static = _ladder_Hamiltionian(self.S, spin_interaction)

    def _chain_Hamiltonian(self):

        heis_ray = self.interaction_dict['Heisenberg'];
        kit_ray = self.interaction_dict['Kitaev'];
        field_ray = self.interaction_dict['Field'];

        assert heis_ray.shape[0] == self.S;
        assert kit_ray.shape[0]==self.S//2;
        assert field_ray.shape[0]==self.S

        Jxx_list = [[heis_ray[i][0],i,(i+1)%L] for i in range(self.S)];
        Jyy_list = [[heis_ray[i][1],i,(i+1)%L] for i in range(self.S)];
        Jzz_list = [[heis_ray[i][2],i,(i+1)%L] for i in range(self.S)];

        Kxx_list = [[kit_ray[i][0] * (1-i//2),i,(i+1)%L] for i in range(self.S)];
        Kyy_list = [[kit_ray[i][1] * (i//2),i,(i+1)%L] for i in range(self.S)];

        Hx_list = [[field_ray[i][0],i] for i in range(L)];
        Hy_list = [[field_ray[i][1],i] for i in range(L)];
        Hz_list = [[field_ray[i][2],i] for i in range(L)];

        static= [["xx", Jxx_list],["yy", Jyy_list],["zz",J_z_list],["xx", Kxx_list],["yy", Kyy_list], ["x", Hx_list],["y", Hy_list],["z", Hz_list]]
        dynamic=[]
        Hamiltion = hamiltonian(static,dynamic,dtype=np.float64,basis=basis);

        return Hamiltion,static;

    #def spin_cavity_coupling(self, cavity, coupled_sites=self.S, spin_excitation='x',cavity_excitation='+'):



    #    return -1;

    def _product_state(self, align='ferro', H_vector='z', Sz_sector=None):
        '''Generate a product state along a particular vector direction on Bloch
        sphere.

        Parameters
        --------------
        align : string
                determines whether the initial state is ferromagnetic,
                antiferromagnetic, or random

        H_vector : string
                vector on the bloch sphere along which the spins are (anti) aligned
                ***(Currently only allows {x,y, or z})***

        Sz_sector : int
                Argument for QuSpin Spin Basis Constructor that defines the limited
                spin space for the basis states. Condition that N<L.

        Returns
        --------------
        ps_state : numpy.ndarray
                Product state that is (anti)feromagnetically aligned.

        basis : quspin.basis.basis_1d.spin.spin_basis_1d
                Basis in which the state is represented.
        '''

        basis = spin_basis1d(L=self.S, Nup=Sz_sector);
        pi_control = (-1) ** (align=='antiferro'); #(Anti) align neighboring spins

        if align != 'random':

            H_field = [[1.0 * pi_control**i, i] for i in range(self.S)]; #Magnetic field energies
            static=[[H_vector,H_field]] #Assign direction to magnetic field
            dynamic=[]
            basis=spin_basis_1d(L=self.S);
            H=hamiltonian(static,dynamic,dtype=np.float64,basis=basis)

            E_min,psi_0 = H.eigsh(k=self.S,which="SA"); #Get ground state (A)FM product state
            ps_state = psi_0.T[0].reshape((-1,));

        elif align == 'random':
            ps_state = rand_ket(2**self.S);

        return ps_state, basis;

    def _n_state(self, initial_Hamiltonian, n=0, Sz_sector=None):
        '''Generate a eigenstate state for a particular Hamiltonian. Particularly useful
        for performing quench dynamics. n=0 is the ground state, n=1 is first excited state etc.

        Parameters
        --------------

        n : int
                Index for which energy eigenstate to return.

        initial_Hamiltonian : list
                list contains basis over which components of the Hamiltonian are acting in line
                with Static lists used in QuSpin: [["operator string", ["Energy", "adjacency indices for interactions"]]];
                Example operator strings found at http://weinbe58.github.io/QuSpin/basis.html

        Sz_sector : int
                Argument for QuSpin Spin Basis Constructor that defines the limited
                spin space for the basis states. Condition that N<L.

        Returns
        --------------
        ps_state : numpy.ndarray
                Ground state of provided Hamiltonian.

        basis : quspin.basis.basis_1d.spin.spin_basis_1d
                Basis in which the state is represented.
        '''
        max_index = max(self.S, n)
        basis = spin_basis1d(L=self.S, Nup=Sz_sector);
        H = hamiltonian(initial_Hamiltonian, [], dtype=np.float64, basis=basis);
        E_min,psi = H.eigsh(k=2*max_index,which="SA"); #Get ground state (A)FM product state
        ps_state = psi.T[n].reshape((-1,));

        return ps_state, basis;

    def _photon_state(cavity, expt_N=1, state="fock"):
        '''Define the cavity state for a provided cavity class instance

        Parameters
        -------------
        cavity : cavity.cavity
                Instance of cavity class that is initialized with number of sites and modes and the type (Bosonic, Fermionic)

        expt_N : float
                Expectation value for the number of active modes on each site in the cavity.

        state : string
                State in which to prepare the cavity: Fock, Coherent, or Thermal
        '''
        cavity.cavity_state(expt_N, state);

    def time_evolve_static(initial_spin_state, initial_photon_state, tE_Hamiltonian, t_f = 10.0, d_t=1000):
        ''' Time evolve an initial spin and cavity state with a provided hamiltonian.

        Parameters
        --------------
        initial_spin_state : numpy.ndarray
                Initial spin state of the system, can be an entangled state or product state.

        initial_photon_state : numpy.ndarray
                Initial state of the uniform cavity, individual ones, etc.

        tE_Hamiltonian : quspin.operators.hamiltonian_core.hamiltonian
                Hamiltion that describes how the initial state will time evolve.

        t_f : float
                Final time the system is evolved to.

        d_t : float
                Time step interval.

        Returns
        --------------
        tE_state : numpy.ndarray
                Array of time evolved states between {0, t_f} in steps of d_t.

        t : numpy.ndarray
                Array of times where the state has been time evolved to.
        '''

        initial_state = np.kron(initial_spin_state, initial_photon_state); #Kronecker product spin+cavity states
        t = np.linspace(0, t_f, d_t);
        tE_state = tE_Hamiltonian.evolve(initial_state, 0.0, t);

        return tE_state, t;


def spin_photon_Nsite_DM(N, coupling, Nph_tot=10, state='ferro', decouple=0, photon_state=0,coherent=False, t_max=10.00,t_steps=100, obs_photon=5, vect='x', omega=0.5, J_zz=0.0, J_xx=0.0, J_yy=0, J_xy=0.0, J_z=0.0, J_x=0.0, return_state_DM=False, periodic=True, init_state=None, Dynamical_Spins=False):

    ##### define Boson model parameters #####
    Nph_tot=Nph_tot # maximum photon occupation
    Nph=1.0# mean number of photons in initial state
    L=N;
    Omega=omega # drive frequency
    A=coupling # spin-photon coupling strength (drive amplitude)
    Delta=0.0 # difference between atom energy levels

    ph_energy=[[Omega]] # photon energy
    loss=[[0.2]] # emission term
    at_energy=[[Delta,i] for i in range(L)] # atom energy

    if type(decouple) == int:

        absorb=[[A / np.sqrt(N-decouple),i] for i in range(0, L-decouple)] # absorption term
        emit=[[A/ np.sqrt(N-decouple),i] for i in range(0, L-decouple)] # emission term


    elif type(decouple) == tuple:
        absorb=[[A / np.sqrt(len(decouple)),i] for i in decouple] # absorption term
        emit=[[A/ np.sqrt(len(decouple)),i] for i in decouple] # emission term


    else:
        ValueError('Improper input for decouple variable');
        return -1;

    ##### define Boson model parameters #####

    if periodic==True:
        boundary = L;
    else:
        boundary = L-1;

    H_zz = [[J_zz,i,(i+1)%L] for i in range(boundary)] # PBC
    H_xx = [[J_xx,i,(i+1)%L] for i in range(boundary)] # PBC
    H_xy = [[J_xy,i,(i+1)%L] for i in range(boundary)] # PBC
    H_yy = [[J_yy,i,(i+1)%L] for i in range(boundary)] # PBC
    H_z = [[J_z,i] for i in range(L)] # PBC
    H_x = [[J_x,i] for i in range(L)] # PBC

    psi_atom = get_product_state(N,state=state,vect=vect);


    # define static and dynamics lists
    static=[["|n",ph_energy],["x|-",absorb],["x|+",emit],["z|",at_energy], ["+-|",H_xy],["-+|",H_xy],["zz|",H_zz], ["xx|",H_xx], ["yy|",H_yy], ["z|",H_z],["x|",H_x]]
    dynamic=[]

    # compute atom-photon basis
    basis=photon_basis(spin_basis_1d,L=N,Nph=Nph_tot)
    atom_basis=spin_basis_1d(L=N)
    H=hamiltonian(static,dynamic,dtype=np.float64,basis=basis,check_herm=False)
    # compute atom-photon Hamiltonian H
    if type(init_state) == tuple:
        H_zz = [[init_state[0],i,(i+1)%L] for i in range(boundary)] # PBC
        H_xy = [[0,i,(i+1)%L] for i in range(boundary)] # PBC
        H_z = [[0,i] for i in range(L)] # PBC
        H_x = [[init_state[1],i] for i in range(L)] # PBC
        static=[["+-",H_xy],["-+",H_xy],["zz",H_zz], ["z",H_z],["x",H_x]];
        init_Spin_Hamiltonian = hamiltonian(static,dynamic,dtype=np.float64,basis=spin_basis_1d(N),check_herm=False)
        psi_atom = get_ground_state(N,init_Spin_Hamiltonian);
    if coherent==False:
        psi_boson=np.zeros([Nph_tot+1]);
        psi_boson[photon_state]=1
    else:
        psi_boson = coherent_state(np.sqrt(photon_state), Nph_tot+1)
    psi=np.kron(psi_atom,psi_boson)
    ##### calculate time evolution #####
    obs_args={"basis":basis,"check_herm":False,"check_symm":False}
    t = np.linspace(0, t_max, t_steps);
    psi_t = H.evolve(psi, 0.0, t);


    a = hamiltonian([["|-", [[1.0  ]] ]],[],dtype=np.float64,**obs_args);
    n=hamiltonian([["|n", [[1.0  ]] ]],[],dtype=np.float64,**obs_args);
    a_psi = a.dot(psi);
    a_psi_t = H.evolve(a_psi, 0.0, t);
    g2_0 = n.expt_value(a_psi_t);
    n_t = n.expt_value(psi_t);
    g2_0 = g2_0 / (n_t[0] * n_t);
    g2_0 = np.real(g2_0);


    ##### define observables #####
    # define GLOBAL observables parameters
    n_t=hamiltonian([["|n", [[1.0  ]] ]],[],dtype=np.float64,**obs_args)
    nn_t=hamiltonian([["|nn", [[1.0  ]] ]],[],dtype=np.float64,**obs_args)

    z_tot_t = hamiltonian([["z|", [[1.0,i] for i in range(L)] ]],[],dtype=np.float64,**obs_args);
    x_tot_t = hamiltonian([["x|", [[1.0,i] for i in range(L)] ]],[],dtype=np.float64,**obs_args);

    zz_tot_t = hamiltonian([["zz|", [[1.0,i,(i+1)%L] for i in range(boundary)] ]],[],dtype=np.float64,**obs_args);
    xx_tot_t = hamiltonian([["xx|", [[1.0,i,(i+1)%L] for i in range(boundary)] ]],[],dtype=np.float64,**obs_args);


    ising_static=[["|n",0],["x|-",0],["x|+",0],["z|",0], ["+-|",H_xy],["-+|",H_xy],["zz|",H_zz], ["xx|",H_xx], ["yy|",H_yy], ["z|",H_z],["x|",H_x]]
    Ising_E_t = hamiltonian(static,dynamic,dtype=np.float64,basis=basis,check_herm=False)

    Obs_dict = {"n":n_t,"nn":nn_t,"z_tot":z_tot_t,
                "x_tot":x_tot_t, "zz_tot":zz_tot_t, "xx_tot":xx_tot_t, "Ising_t":Ising_E_t};

    zz_dynamical = np.zeros([L, len(t)]);
    xx_dynamical = np.zeros([L, len(t)]);



    for i in range(N):
        for j in range(i+1, N):
            stringz = "z%1dz%1d" % (i, j);
            stringx = "x%1dx%1d" % (i, j);

            zz_ham = hamiltonian([["zz|", [[1.0,i,j]] ]],[],dtype=np.float64,**obs_args);
            xx_ham = hamiltonian([["xx|", [[1.0,i,j]] ]],[],dtype=np.float64,**obs_args);

            Obs_dict.update({stringz:zz_ham, stringx: xx_ham});

        stringz = "z%1d" % (i);
        stringx = "x%1d" % (i);

        z_ham = hamiltonian([["z|", [[1.0,i]] ]],[],dtype=np.float64,**obs_args);
        x_ham = hamiltonian([["x|", [[1.0,i]] ]],[],dtype=np.float64,**obs_args);

        Obs_dict.update({stringz:z_ham, stringx: x_ham});

        if Dynamical_Spins == True:
            x0 = hamiltonian([["x|", [[1.0, 0]] ]],[],dtype=np.float64,**obs_args);
            xl=hamiltonian([["x|", [[1.0, i]] ]],[],dtype=np.float64,**obs_args);
            x0_psi = x0.dot(psi);
            x0_psi_t = H.evolve(x0_psi, 0.0, t);
            xl = xl.expt_value(x0_psi_t);

            xx_dynamical[i] = xl;

            z0 = hamiltonian([["z|", [[1.0, 0]] ]],[],dtype=np.float64,**obs_args);
            zl=hamiltonian([["z|", [[1.0, i]] ]],[],dtype=np.float64,**obs_args);
            z0_psi = z0.dot(psi);
            z0_psi_t = H.evolve(z0_psi, 0.0, t);
            zl = zl.expt_value(z0_psi_t);

            zz_dynamical[i] = zl;

    Obs_t = obs_vs_time(psi_t,t,Obs_dict);

    ####### Number of times to sample the cavity state, could sample at every time so num = 1 ###############
    Sent = np.zeros_like(t);
    AC_ent = np.zeros_like(t);
    num = t_steps//obs_photon
    print(num)
    obs_pht_t = t[::num];
    obs_pht_ray = np.zeros([len(obs_pht_t), Nph_tot+1, Nph_tot+1]);
    spin_subsys_dm = np.zeros([len(t), 2**(N//2), 2**(N//2)])
    pairwise_concurrence = np.zeros([len(t), N, N]);

    for i in range(len(t)):
        dictionary = basis.ent_entropy(psi_t.T[i],sub_sys_A='particles',return_rdm='both');

        AC_ent[i]=dictionary["Sent_A"];
        spin_dm = dictionary["rdm_A"];
        photon_dm = dictionary["rdm_B"];

        dict_atom_subsys = atom_basis.ent_entropy(spin_dm,sub_sys_A=range(N//2),return_rdm='both');

        Sent[i]=N//2 * dict_atom_subsys['Sent_A'];
        spin_subsys_dm[i]=dict_atom_subsys['rdm_A'];

        for j in range(N):
            for k in range(j+1, N):
                two_spin_dict = atom_basis.ent_entropy(spin_dm,sub_sys_A=(j,k),return_rdm='both');
                two_spin_dm = two_spin_dict['rdm_A'];
                two_spin_conc = concurrence(two_spin_dm);
                pairwise_concurrence[i,j,k] = two_spin_conc;

        if t[i] in obs_pht_t:
            obs_pht_ray[i//num] = photon_dm

    if return_state_DM==False:
        return t, AC_ent, Sent, Obs_t, obs_pht_ray, g2_0, pairwise_concurrence, zz_dynamical, xx_dynamical;
    else:
        return t, AC_ent, Sent, Obs_t, obs_pht_ray, g2_0, pairwise_concurrence, zz_dynamical, xx_dynamical, spin_subsys_dm;


def concurrence(rho, subsys_len=2,thresh_error=1e-6):
    """Evaluate the concurrance for a two particle system
    Accepts either a two-particle density matrix, row vector, or just simple array
    """

    sigma_y_array = np.array(sigmay())
    concurrence_array = np.kron(sigma_y_array, sigma_y_array)

    dm = check_particle_state(rho,subsys_len);

    if np.sum(dm) == None:
        raise ValueError("Enter proper two-particle state.")

    if subsys_len==2:
        val = concur_2q(dm,thresh_error);

    else:
        val = concur_nq(dm, thresh_error);
    return val;

def concur_2q(rho,thresh_error):
    sigma_y_array = np.array(sigmay())
    concurrence_array = np.kron(sigma_y_array, sigma_y_array)
    #print(rho)
    super_threshold_indices = np.abs(rho) < thresh_error;
    rho[super_threshold_indices] = 0;
    psi_dm = rho;
    psi_dm_tilde = concurrence_array * psi_dm.conjugate() * concurrence_array;
    R = sp.linalg.sqrtm(sp.linalg.sqrtm(psi_dm) * psi_dm_tilde * sp.linalg.sqrtm(psi_dm))
    eig_vals = sp.linalg.eigvalsh(R);
    eig_vals = np.sort(eig_vals);
    eig_vals = eig_vals[::-1]
    eig_vals_sum = eig_vals[0] - eig_vals[1] - eig_vals[2] - eig_vals[3];
    #print(np.max([0, eig_vals_sum]))
    return np.max([0, eig_vals_sum]);

def concur_nq(rho,thresh_error):
    sigma_y_array = np.array(sigmay())
    concurrence_array = np.kron(sigma_y_array, sigma_y_array)
    #print(rho)
    super_threshold_indices = np.abs(rho) < thresh_error;
    rho[super_threshold_indices] = 0;
    psi_dm = rho;
    psi_dm_tilde = concurrence_array * psi_dm.conjugate() * concurrence_array;
    R = sp.linalg.sqrtm(sp.linalg.sqrtm(psi_dm) * psi_dm_tilde * sp.linalg.sqrtm(psi_dm))
    eig_vals = sp.linalg.eigvalsh(R);
    eig_vals = np.sort(eig_vals);
    eig_vals = eig_vals[::-1];
    eig_vals_sum = eig_vals[0] - eig_vals[1] - eig_vals[2] - eig_vals[3];
    print(np.max([0, eig_vals_sum]))
    return np.max([0, eig_vals_sum])

def check_particle_state(rho, subsys_len):
    dim = rho.shape;

    if dim[0] != 2**subsys_len:
        print('***Error: Need a proper two particle state representation!***');
        return

    if len(dim) < 2:
            psi_vector=np.reshape(rho, (2**subsys_len,1));
            dm=psi_vector * psi_vector.T;
    else:
        if dim[1] == 1:
            dm=rho * rho.T;
        if dim[1] == 2**subsys_len:
            dm = rho;
        else:
            print('***Error: Need a proper two particle state representation!***')
            return
    return dm;

def get_product_state(N,state='ferro',vect='z'):

    if state=='ferro':
        fm = 1.0;

    elif state=='random':
        psi_0 = random
    else:
        fm=-1.0
    H_field = [[1.0 * fm**i, i] for i in range(N)] # PBC
    static=[[vect,H_field]]
    dynamic=[]
    # compute atom-photon basis
    basis=spin_basis_1d(L=N);
    H=hamiltonian(static,dynamic,dtype=np.float64,basis=basis)

    E_min,psi_0 = H.eigsh(k=5,which="SA");
    psi_0 = psi_0.T[0].reshape((-1,));
    #print(psi_0)
    return psi_0;

def get_ground_state(N, Hamiltonian):
    basis=spin_basis_1d(L=N);
    E_min,psi_0 = Hamiltonian.eigsh(k=5,which="SA");
    psi_0 = psi_0.T[0].reshape((-1,));
    #print(psi_0)
    return psi_0;
