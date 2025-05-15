

function [gradients,Loss,Loss_MSE,Loss_Physics] = ModelGradient_PINN(parameters,Layer,dlXm,dlYm,dlXp,alpha)

% data
dlYm_pred = Network_PINN(dlXm,parameters,1,Layer);

Loss_MSE = mean((dlYm_pred-dlYm).^2);

% physics
dlYp = Network_PINN(dlXp,parameters,1,Layer);

dYdX = dlgradient(sum(dlYp,'all'),dlXp,EnableHigherDerivatives=true);
d2YdX2 = dlgradient(sum(dYdX,'all'),dlXp);

% known parameters
% gamma = 0.5;
% beta = 490.5;

% % unknown parameters
gamma = parameters.param.gamma;
beta = parameters.param.beta*100;

f = d2YdX2 + gamma.*dYdX + beta.*sin(dlYp);

Loss_Physics = mean(f.^2);

%% Losses

Loss = (Loss_MSE + alpha*Loss_Physics);

gradients = dlgradient(Loss,parameters);

end
