import sys,os
from __future__ import print_function,division
from time import clock

import numpy as np
import scipy as sp
from sp.sparse import load_npz, save_npz

from quspin.basis import spin_basis1d,photon_basis,boson_basis1d
from quspin.operators import hamiltonian, quantum_operator
from quspin.tools.measurements import obs_vs_time
from quspin.tools.Floquet import Floquet, Flotquet_t_vec
from quspin.basis.photon import coherent_state

from qutip import *
from qutip.piqs import *


class spin_cavity(num_spins, cavity_modes, spin_interaction):

    def __init__(num_spins, cavity_modes):
        if num_spins > 24:
            ValueError("Good luck with that system size")

        self.S = num_spins;
        self.cavity_modes = cavity_modes;


    def _generate_spin_Hamiltonian(spin_interaction, connectivity='nn'):

        if self.construction == "Chain":
            Ham = _chain_Hamiltoian(self.S, spin_interaction)

        elif self.construction == "Ladder":
            Ham = _ladder_Hamiltionian(self.S, spin_interaction)



    def _product_state(align='ferro', H_vector='z', Sz_sector=None):
        '''Generate a product state along a particular vector direction on Bloch
        sphere.

        Parameters
        --------------
        align : string
                determines whether the initial state is ferromagnetic or
                antiferromagnetic

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

        H_field = [[1.0 * pi_control**i, i] for i in range(self.S)]; #Magnetic field energies
        static=[[H_vector,H_field]] #Assign direction to magnetic field
        dynamic=[]
        basis=spin_basis_1d(L=self.S);
        H=hamiltonian(static,dynamic,dtype=np.float64,basis=basis)

        E_min,psi_0 = H.eigsh(k=self.S,which="SA"); #Get ground state (A)FM product state
        ps_state = psi_0.T[0].reshape((-1,));

        return ps_state, basis;

    def _ground_state(initial_Hamiltonian, Sz_sector=None):
        '''Generate a ground state for a particular Hamiltonian. Particularly useful
        for performing quench dynamics.

        Parameters
        --------------
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
        basis = spin_basis1d(L=self.S, Nup=Sz_sector);
        H = hamiltonian(initial_Hamiltonian, [], dtype=np.float64, basis=basis);
        E_min,psi_0 = H.eigsh(k=self.S,which="SA"); #Get ground state (A)FM product state
        ps_state = psi_0.T[0].reshape((-1,));

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

    def time_evolve(initial_spin_state, initial_photon_state, tE_Hamiltonian, t_f = 10.0, d_t=1000):
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
