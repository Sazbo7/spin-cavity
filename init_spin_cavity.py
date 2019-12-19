def spin_photon_Nsite_DM(N, coupling, Nph_tot=10, state='ferro', decouple=0, photon_state=0,coherent=False, t_max=10.00,t_steps=100, obs_photon=5, vect='x', omega=0.5, J_zz=0.0, J_xx=0.0, J_xy=0.0, J_z=0.0, J_x=0.0, return_state_DM=False, periodic=True, init_state=None):

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
    H_z = [[J_z,i] for i in range(L)] # PBC
    H_x = [[J_x,i] for i in range(L)] # PBC

    psi_atom = get_product_state(N,state=state,vect=vect);


    # define static and dynamics lists
    static=[["|n",ph_energy],["x|-",absorb],["x|+",emit],["z|",at_energy], ["+-|",H_xy],["-+|",H_xy],["zz|",H_zz], ["z|",H_z],["x|",H_x]]
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
    # define observables parameters
    n_t=hamiltonian([["|n", [[1.0  ]] ]],[],dtype=np.float64,**obs_args)
    nn_t=hamiltonian([["|++--", [[1.0  ]] ]],[],dtype=np.float64,**obs_args)

    x0zL_t= hamiltonian([["xz|", [[1.0,0,L-1]] ]],[],dtype=np.float64,**obs_args)
    x0xL_t= hamiltonian([["xx|", [[1.0,0,L-1]] ]],[],dtype=np.float64,**obs_args)
    z0zL_t= hamiltonian([["zz|", [[1.0,0,L-1]] ]],[],dtype=np.float64,**obs_args)

    z_tot_t = hamiltonian([["z|", [[1.0,i] for i in range(L)] ]],[],dtype=np.float64,**obs_args);
    x_tot_t = hamiltonian([["x|", [[1.0,i] for i in range(L)] ]],[],dtype=np.float64,**obs_args);

    zz_tot_t = hamiltonian([["zz|", [[1.0,i,(i+1)%L] for i in range(boundary)] ]],[],dtype=np.float64,**obs_args);
    xx_tot_t = hamiltonian([["xx|", [[1.0,i,(i+1)%L] for i in range(boundary)] ]],[],dtype=np.float64,**obs_args);


    ####### Number times to sample the cavity state, could sample at every time so num = 1 ###############
    Obs_t = obs_vs_time(psi_t,t,{"n":n_t,"nn":nn_t,"x0zL": x0zL_t, "x0xL":x0xL_t, "z0zL":x0zL_t,"z_tot":z_tot_t,
                                 "x_tot":x_tot_t, "zz_tot":zz_tot_t, "xx_tot":xx_tot_t})
    O_n = Obs_t["n"]
    Sent = np.zeros_like(t);
    AC_ent = np.zeros_like(t);

    num = t_steps//obs_photon

    print(num)

    obs_pht_t = t[::num];

    obs_pht_ray = np.zeros([len(obs_pht_t), Nph_tot+1, Nph_tot+1]);
    spin_subsys_dm = np.zeros([len(t), 2**(N//2), 2**(N//2)])

    for i in range(len(t)):
        dictionary = basis.ent_entropy(psi_t.T[i],sub_sys_A='particles',return_rdm='both');

        AC_ent[i]=dictionary["Sent_A"];
        spin_dm = dictionary["rdm_A"];
        photon_dm = dictionary["rdm_B"];

        dict_atom_subsys = atom_basis.ent_entropy(spin_dm,sub_sys_A=range(N//2),return_rdm='both');

        Sent[i]=N//2 * dict_atom_subsys['Sent_A'];
        spin_subsys_dm[i]=dict_atom_subsys['rdm_A'];

        if t[i] in obs_pht_t:
            obs_pht_ray[i//num] = photon_dm

    if return_state_DM==False:
        return t, AC_ent, Sent, Obs_t, obs_pht_ray, g2_0;
    else:
        return t, AC_ent, Sent, Obs_t, obs_pht_ray, g2_0, spin_subsys_dm;


def spin_photon_Nsite_JC(N, coupling, Nph_tot=10, state='ferro', decouple=0, photon_state=0,coherent=False, t_max=10.00,t_steps=100, obs_photon=5, vect='x', omega=0.5, J_zz=0.0, J_xy=0.0, J_z=0.0, J_x=0.0, return_state_DM=False, periodic=True, init_state=None):

    ##### define Boson model parameters #####
    Nph_tot=Nph_tot # maximum photon occupation
    Nph=1.0# mean number of photons in initial coherent state
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
    H_xy = [[J_xy,i,(i+1)%L] for i in range(boundary)] # PBC
    H_z = [[J_z,i] for i in range(L)] # PBC
    H_x = [[J_x,i] for i in range(L)] # PBC

    psi_atom = get_product_state(N,state=state,vect=vect);


    # define static and dynamics lists
    static=[["|n",ph_energy],["+|-",absorb],["-|+",emit],["z|",at_energy], ["+-|",H_xy],["-+|",H_xy],["zz|",H_zz], ["z|",H_z],["x|",H_x]]
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
    # define observables parameters
    n_t=hamiltonian([["|n", [[1.0  ]] ]],[],dtype=np.float64,**obs_args)
    nn_t=hamiltonian([["|nn", [[1.0  ]] ]],[],dtype=np.float64,**obs_args)

    x0zL_t= hamiltonian([["xz|", [[1.0,0,L-1]] ]],[],dtype=np.float64,**obs_args)
    x0xL_t= hamiltonian([["xx|", [[1.0,0,L-1]] ]],[],dtype=np.float64,**obs_args)
    z0zL_t= hamiltonian([["zz|", [[1.0,0,L-1]] ]],[],dtype=np.float64,**obs_args)

    z_tot_t = hamiltonian([["z|", [[1.0,i] for i in range(L)] ]],[],dtype=np.float64,**obs_args);
    x_tot_t = hamiltonian([["x|", [[1.0,i] for i in range(L)] ]],[],dtype=np.float64,**obs_args);

    zz_tot_t = hamiltonian([["zz|", [[1.0,i,(i+1)%L] for i in range(boundary)] ]],[],dtype=np.float64,**obs_args);
    xx_tot_t = hamiltonian([["xx|", [[1.0,i,(i+1)%L] for i in range(boundary)] ]],[],dtype=np.float64,**obs_args);


    ####### Number times to sample the cavity state, could sample at every time so num = 1 ###############
    Obs_t = obs_vs_time(psi_t,t,{"n":n_t,"nn":nn_t,"x0zL": x0zL_t, "x0xL":x0xL_t, "z0zL":x0zL_t,"z_tot":z_tot_t,
                                 "x_tot":x_tot_t, "zz_tot":zz_tot_t, "xx_tot":xx_tot_t})
    O_n = Obs_t["n"]
    Sent = np.zeros_like(t);
    AC_ent = np.zeros_like(t);

    num = t_steps//obs_photon

    print(num)

    obs_pht_t = t[::num];

    obs_pht_ray = np.zeros([len(obs_pht_t), Nph_tot+1, Nph_tot+1]);
    spin_subsys_dm = np.zeros([len(t), 2**(N//2), 2**(N//2)])

    for i in range(len(t)):
        dictionary = basis.ent_entropy(psi_t.T[i],sub_sys_A='particles',return_rdm='both');

        AC_ent[i]=dictionary["Sent_A"];
        spin_dm = dictionary["rdm_A"];
        photon_dm = dictionary["rdm_B"];

        dict_atom_subsys = atom_basis.ent_entropy(spin_dm,sub_sys_A=range(N//2),return_rdm='both');

        Sent[i]=N//2 * dict_atom_subsys['Sent_A'];
        spin_subsys_dm[i]=dict_atom_subsys['rdm_A'];

        if t[i] in obs_pht_t:
            obs_pht_ray[i//num] = photon_dm

    if return_state_DM==False:
        return t, AC_ent, Sent, Obs_t, obs_pht_ray, g2_0;
    else:
        return t, AC_ent, Sent, Obs_t, obs_pht_ray, g2_0, spin_subsys_dm;


def concurrence(rho, subsys_len=2,thresh_error=1e-5):
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
