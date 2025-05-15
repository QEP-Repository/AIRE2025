%%
clear; clc;

%% Gift and Gauvin equation

Re_min = 3000;
Re_max = 5e6;

Pr_min = 0.5;
Pr_max = 2000;

Ndata = 5000;

Re = rand(Ndata,1)*(log10(Re_max)-log10(Re_min)) + log10(Re_min);
Re = 10.^Re;

Ndata = 5000;

Pr = rand(Ndata,1)*(log10(Pr_max)-log10(Pr_min)) + log10(Pr_min);
Pr = 10.^Pr;

f = (0.79*log(Re)-1.64).^(-2);

Nu = f/8.*(Re-1000).*Pr./(1+12.7.*(f/8).^(0.5).*(Pr.^(2/3)-1));

%%

U = rand(Ndata,1)*30 + 0.0001;
rho = rand(Ndata,1)*1000 + 0.1;
D = rand(Ndata,1)*1 + 0.0001;

mu = (U.*rho.*D)./Re;

nu = mu./rho;

alpha = nu./Pr;

T = rand(Ndata,1)*200 + 273.15;
cp = rand(Ndata,1)*1000 + 0.1;


%%

 Data = [rho U D mu alpha T cp Re Pr Nu];




