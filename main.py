import sys
from spin_cavity import *
import pandas as pd

### Load txt file that contains initial parameters #######

SAVE_LOC = "Users/user/Desktop/folder"
file_name=sys.argv[1];
initialize = np.loadtxt(fname = file_name, dtype=str);
NAME = initialize[0][1];
Length = int(initialize[1][1]);
init_spin=initialize[2][1];
init_cavity=initialize[3][1];
init_cavity_num=int(initialize[4][1]);
boundary=initialize[5][1];
decouple=int(initialize[6][1:]);


print(initialize);
#quit()

if boundary == "O":
    BC = False;
elif boundary == "P":
    BC = True;
else:
    BC = True;

if init_cavity == "Fock":
    CH = False;
elif boundary == "CH":
    CH = True;
else:
    CH = False;

### Print the initial parameters #####
for rows in range(initialize.shape[0]):
    print(initialize[rows]);

### Data saved using Pandas DataFrame to .pickle files ####
df_columns = ["L", "coupling", "omega", "Jzz", "Jx", "Boundary", "Times",
                "P-S Ent", "S-S Ent", "Observables","P_DM", "G2_0_T", "PW_CONC", "ZZ_Dyna", "XX_Dyna"];
df = pd.DataFrame(columns=df_columns);


field_vals = np.linspace(0, 3, 40);
coupling_vals = np.linspace(0, 1.2, 21);
omega_vals = (0.5, 0.8, 1.2, 5.0);

count = 0;
for omega in omega_vals:
    for field in field_vals:
        for coupling in coupling_vals:

            if init_spin == "UNIFORM":
                init_state=None;
            elif init_spin=="EQ":
                init_state=(1.0, field);
            elif init_spin=="Quench":
                init_state=(1.0, 5.0);

            t, AC_ent, Sent, Obs_t, obs_pht_ray, g2_0, pairwise_concurrence, zz_dyna, xx_dyna = spin_photon_Nsite_DM(Length, omega * coupling,
                                            decouple=decouple,omega=omega,Nph_tot=40,coherent=CH,photon_state=init_cavity_num,t_steps=1250,
                                            obs_photon=1250,t_max=200.0,state='ferro',vect='z',
                                            J_zz=1.0,J_x=field,return_state_DM=False,periodic=BC, init_state=init_state, Dynamical_Spins=True);

            df = df.append({"L": Length, "coupling": omega * coupling,"omega": omega,"Jzz":1.0,
                            "Jx":field,"Boundary":boundary,"Times":t,"P-S Ent":AC_ent,
                            "S-S Ent":Sent,"Observables":Obs_t,"P_DM":obs_pht_ray,
                            "G2_0_T":g2_0,"PW_CONC":pairwise_concurrence, "ZZ_Dyna":zz_dyna, "XX_Dyna":xx_dyna}, ignore_index=True);
            print(count)
            count+=1;

df.to_pickle(SAVE_LOC+ NAME + "_" + str(Length) +".pickle");
print('Completed');
