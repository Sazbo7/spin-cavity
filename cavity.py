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

class boson_cavity(type='Uniform', Ntot, sites=1, species=1):

    def __init__(type, Ntot):

        self.type=type;
        self.Ntot=Ntot;
        self.sites=sites;
        self.species=species;
        self.basis=None;

        _get_cavity_basis();


    def _get_cavity_basis():
        '''Get the basis states for the external coupled system in which to work in.

        Parameters
        --------------
        type : string
                Defines the type of cavity. Possible options include:

                Uniform -- Boson cavity coupled to each site. Can have multiple modes
                Local -- Individual boson coupled to each "SITES" number of sites. Can have multiple modes
                fLocal -- Fermionic system coupled to each "SITES" number of sites. 1 mode.
        '''

        if self.type == 'Uniform':
            basis=photon_basis(Nph = self.Ntot);
            basis_ms = basis;
            for i in range(self.species-1):
                basis_ms = tensor_basis(basis_ms, basis);
            self.basis=basis_ms;

        if self.type == 'Local':
            basis = boson_basis1d(L=self.sites,sps=Ntot);
            basis_ms = basis;
            for i in range(self.species-1):
                basis_ms = tensor_basis(basis_ms, basis);
            self.basis=basis_ms;

        if self.type == 'fLocal':
            self.basis=spinless_fermion_basis_1d(L=self.sites, Nf=None, nf=None);
