import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

Npoints = 5000

Remax = 5e6
Remin = 3e6
Re = np.abs(np.random.random((Npoints,)))*(Remax-Remin)+Remin

Prmax = 2000
Prmin = 0.5
Pr = np.abs(np.random.random((Npoints,)))*(Prmax-Prmin)+Prmin

f = (0.79*np.log(Re)-1.64)**-2

Nu = f/8*(Re-1000)*Pr/(1+12.7*(f/8)**0.5*(Pr**(2/3)-1))


Umax = 1000
Umin = 100
U = np.abs(np.random.random((Npoints,)))*(Umax-Umin)+Umin

rhomax = 1000
rhomin = 100
rho = np.abs(np.random.random((Npoints,)))*(rhomax-rhomin)+rhomin

Dmax = 5
Dmin = 0.5
D = np.abs(np.random.random((Npoints,)))*(Dmax-Dmin)+Dmin
 
mu = U*rho*D/Re
alpha = mu/(rho*Pr)

cpmax = 1001
cpmin = 1000
cp = np.abs(np.random.random((Npoints,)))*(cpmax-cpmin)+cpmin

Tmax = 350
Tmin = 250

T = np.abs(np.random.random((Npoints,)))*(Tmax-Tmin)+Tmin
columns_names= ['rho','U','D','mu','cp','alpha', 'T','Re','Pr','Nu']

db = np.vstack((rho,U,D,mu,cp,alpha,T,Re,Pr,Nu)).T

db = pd.DataFrame(db, columns=columns_names)

db.to_excel('db_Nusselt_mio.xlsx', index=False)


