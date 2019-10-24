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


    def _generate_spin_Hamiltonian(spin_interaction, ):

        if self.construction == "Chain":
            Ham = _chain_Hamiltoian(self.S, spin_interaction)

        elif self.construction == "Ladder":
            Ham = _ladder_Hamiltionian(self.S, spin_interaction)
