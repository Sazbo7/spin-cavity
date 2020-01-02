import sys
from spin_cavity import *
import pandas as pd

NAME=sys.argv[1];
init_spin=sys.argv[2];
init_cavity=sys.argv[3];
boundary=sys.argv[4];


df_columns = ["L", "coupling", "omega", "Jzz", "Jx", "Boundary", "Times",
                "P-S Ent", "S-S Ent", "Observables","P_DM", "G2_0_T", "PW_CONC"];
df = pd.DataFrame(columns=df_columns);


field_vals = np.linspace(0, 3, 40);
coupling_vals = np.linspace(0, 1.2, 10);
omega_vals = (0.5, 0.8, 1.2, 5.0);

for omega in omega_vals:
    for field in field_vals:
        for coupling in coupling_vals:

            t, AC_ent, Sent, Obs_t, obs_pht_ray, g2_0, pairwise_concurrence = spin_photon_Nsite_DM(8, omega * coupling,
                                            decouple=0,Omega=omega,Nph_tot=50,photon_state=0,t_steps=2000,
                                            obs_photon=2000,t_max=400.0,state='ferro',vect='z',
                                            J_zz=1.0,J_x=field,return_state_DM=False,periodic=True, init_state=init_state);

            df = df.append({"L": 8, "coupling": omega * coupling,"omega": omega,"Jzz":1.0,
                            "Jx":x_field,"Boundary":boundary,"Times":t,"P-S Ent":AC_ent,
                            "S-S Ent":Sent,"Observables":Obs,"P_DM":obs_pht_ray,
                            "G2_0_T":g2_0,"PW_CONC":pairwise_concurrence}, ignore_index=True);


df.to_pickle("/Users/szabo.48/Desktop/"+ NAME +".pickle");
