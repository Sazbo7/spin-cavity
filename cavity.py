import sys,os

import numpy as np
import scipy as sp

from quspin.basis import spin_basis_1d,boson_basis_1d, photon_basis, tensor_basis # Hilbert space bases
from quspin.operators import hamiltonian, quantum_operator
from quspin.basis.photon import coherent_state

from qutip import *
from qutip.piqs import *

class cavity:

    def __init__(self, Ntot, types='Uniform', sites=1, species=1):
        '''Initialize a cavity state. Can be made of up bosons or fermions and can be a single uniform cavity
         or a cavity coupled to individuals sites.

        Parameters
        --------------
        types : string
                Defines the types of cavity. Possible options include:

                Uniform -- Boson cavity coupled to each site. Can have multiple modes
                Local -- Individual boson coupled to each "SITES" number of sites. Can have multiple modes
                fLocal -- Fermionic system coupled to each "SITES" number of sites. 1 mode.

        Ntot : int
            Maximum number occupation of individual cavity sites.

        sites : int
            Number of individual cavities to consider. Default is a single cavity.

        species : int
            Number of different bosonic modes that exist on each site (species $\geq$1). In the thermal limit this is a continuum.
        '''

        self.types=types;
        self.Ntot=Ntot;
        self.sites=sites;
        self.species=species;
        self.basis=None;

        self._set_cavity_basis();

    def cavity_state(self, expt_N=0, state='fock'):
        '''Sets an initial cavity wave function provided the basis created at initialization.

        Parameters
        ------------
        expt_N : float
            Expectation value for the number occupation on each site

        state : string
            Description of type of state the cavity mode is in. Option are:
            'fock': Pure number state
            'coherent': Minimum uncertainty state
            'thermal': Thermal density matrix description
        '''
        self.state=state;
        self._set_cavity_state(expt_N);

    def _set_cavity_basis(self):
        '''Get the basis states for the external coupled system in which to work in.

        Parameters
        --------------
        types : string
                Defines the types of cavity. Possible options include:

                Uniform -- Boson cavity coupled to each site. Can have multiple modes
                Local -- Individual boson coupled to each "SITES" number of sites. Can have multiple modes
                fLocal -- Fermionic system coupled to each "SITES" number of sites. 1 mode.
        '''
        if self.types == 'Uniform':
            basis = boson_basis_1d(L=1,sps=self.Ntot);
            basis_ms = basis;
            for i in range(self.species-1):
                basis_ms = tensor_basis(basis_ms, basis);
            self.basis=basis_ms;

        if self.types == 'Local':
            basis = boson_basis_1d(L=self.sites,sps=self.Ntot);
            basis_ms = basis;
            for i in range(self.species-1):
                basis_ms = tensor_basis(basis_ms, basis);
            self.basis=basis_ms;

        if self.types == 'fLocal':
            self.basis=spinless_fermion_basis_1d(L=self.sites, Nf=None, nf=None);

    def get_cavity_basis(self):
        '''Return the basis states that make up the cavity Hilber space.
        '''
        return self.basis;

    def _set_cavity_state(self, expt_N):
        '''Sets an initial cavity wave function provided the basis created at initialization.

        Parameters
        ------------
        expt_N : float, list or numpy.array
            Expectation value for the number occupation on each site ***Need to include functionality to accept array of size SITES***
        '''

        if self.type=='Uniform':
            try:
                if self.state=='coherent':
                    psi = coherent_state(expt_N, self.Ntot);
                    psi = psi.reshape(Ntot,1)
                    self.psi = psi;
                    self.dm = psi.T * psi;

                if self.state=='thermal':
                    dm = thermal(self.Ntot, expt_N);
                    dm = dm.get_data().toarray();
                    self.dm = dm;
                    self.psi=None;

                if self.state=='fock':
                    psi=np.zeros([self.Ntot]);
                    psi[expt_N] = 1.0;
                    psi = psi.reshape(Ntot,1)
                    self.psi = psi;
                    self.dm = psi.T * psi;
            except TypeError:
                print("Using a uniform environment: occupation must be a float not list")

        if self.type=='Local':
            if isinstance(expt_N, float):
                if self.state=='coherent':
                    psi_s = coherent_state(expt_N, self.Ntot);
                    psi_s = psi_s.reshape(Ntot,1);
                    psi = psi_s;
                    for i in range(self.sites):
                        psi = np.kron(psi, psi_s);
                    self.psi=psi;
                    self.dm = psi.T*psi;

                if self.state=='thermal':
                    dm_s = thermal(self.Ntot, expt_N);
                    dm_s = dm.get_data().toarray();
                    dm = dm_s;
                    for i in range(self.sites):
                        dm = np.kron(dm, dm_s);
                    self.dm=dm;
                    self.psi=None;

                if self.state=='fock':
                    psi_s=np.zeros([self.Ntot]);
                    psi_s[expt_N] = 1.0;
                    psi_s = psi_s.reshape(Ntot,1)
                    psi = psi_s;
                    for i in range(self.sites):
                        psi = np.kron(psi, psi_s);
                    self.psi = psi;
                    self.dm = psi.T * psi;
            else:
                try:
                    if self.state=='coherent':
                        psi_s = coherent_state(expt_N[0], self.Ntot);
                        psi_s = psi_s.reshape(Ntot,1);
                        psi = psi_s;
                        for i in range(1, self.sites):
                            psi_s = coherent_state(expt_N[i], self.Ntot);
                            psi_s = psi_s.reshape(Ntot,1);
                            psi = np.kron(psi, psi_s);
                        self.psi=psi;
                        self.dm = psi.T*psi;

                    if self.state=='thermal':
                        dm_s = thermal(self.Ntot, expt_N);
                        dm_s = dm.get_data().toarray();
                        dm = dm_s;
                        for i in range(1, self.sites):
                            dm = np.kron(dm, dm_s);
                        self.dm=dm;
                        self.psi=None;

                    if self.state=='fock':
                        psi_s=np.zeros([self.Ntot]);
                        psi_s[expt_N] = 1.0;
                        psi_s = psi_s.reshape(Ntot,1)
                        psi = psi_s;
                        for i in range(1, self.sites):
                            psi_s=np.zeros([self.Ntot]);
                            psi_s[expt_N[i]] = 1.0;
                            psi_s = psi_s.reshape(Ntot,1)
                            psi = np.kron(psi, psi_s);
                        self.psi = psi;
                        self.dm = psi.T * psi;
                except TypeError:
                    print('Invalid entry for initialinzing the cavity state');

    def get_cavity_state(self, rep='psi'):
        '''Return the cavity wavefunction or density matrix.

        Parameters:
        --------------

        rep : string
            Allows for returning the vector description of the wave function or 'rep = "dm"' for returning
            the density matrix.
        '''
        if rep == "dm":
            return self.dm;
        else:
            return self.psi;
