
% we clear workspace, command window and GPU

if canUseGPU()
    gpu = gpuDevice();
    reset(gpu)

    gpu = gpuDevice();
    disp(gpu)
    wait(gpu)

    clear all; clc;
else
    clear; clc;
end

%% Configuration

% Epoch and Iteration
Config.Epoch_Max = 1e4;
Config.Iteration_per_Epoch = 100;
Config.Physics_MiniBatch = 1000;
Config.CollocationPoints = Config.Iteration_per_Epoch.*Config.Physics_MiniBatch;

% Learning rate
Config.Learning_Rate = 5e-3;
Config.Decay_Rate = 1e-4;

% physics weight
Config.alpha = 1e-5;

%%

load("Dataset.mat")

%% measurements to dlarray

dlXm = dlarray(Data.t,'CB');
dlYm = dlarray(Data.theta,'CB');

if canUseGPU
    dlXm = gpuArray(dlXm);
    dlYm = gpuArray(dlYm);
end

%% Physics grid 

points = sobolset(1);
points = points(1:Config.CollocationPoints);

tp = points*(max(Data.t_domain)-min(Data.t_domain))+min(Data.t_domain);
dlXp = dlarray(tp','CB'); 

if canUseGPU
    dlXp = gpuArray(dlXp);
end

clear points tp

%% points for plot

dlXplot = dlarray(Data.t_domain,'CB');

%%

% Define neural network architecture (fully connect)
Layer = [20 20 20 20 20 20 20 20 20];

% initalise the neural network
[~,parameters] = Network_PINN(dlXp(:,1:10),[],0,Layer);

% predict (just for tests that everything is ok)
dlYplot = Network_PINN(dlXplot,parameters,1,Layer);

% Plot

figure(2)
clf
plot(PDE.t,PDE.theta,'-k')
hold on
plot(dlXm,dlYm,'.r','markersize',16)
plot(dlXplot,dlYplot,'-b')

%% We need a function where we calculate the loss and the compute the gradients

AcceleratedFunction = dlaccelerate(@ModelGradient_PINN);

%% Initialise Some Variables

% These are variables for ADAM algorithm (advanced descent gradient
% algorithm)

averageGrad = [];
averageSqGrad = [];

iteration = 0;

figure(3)
clf

%%

for epoch = 1 : Config.Epoch_Max

    for i = 1 : Config.Iteration_per_Epoch

        iteration = iteration + 1;

        % minibatch for the physics
        ind_physics = ((i-1)*Config.Physics_MiniBatch+1):(i*Config.Physics_MiniBatch);
        dlXp_now = dlXp(1,ind_physics);

        % Model Gradient
        [gradient,Loss,Loss_Data,Loss_physics] = dlfeval(AcceleratedFunction,...
            parameters,Layer,dlXm,dlYm,dlXp_now,Config.alpha);

        % Learning rate update
        LearningRate = Config.Learning_Rate./(1+Config.Decay_Rate*iteration);

        % ADAM update (update the parameters)
        [parameters,averageGrad,averageSqGrad] = adamupdate(parameters,gradient,averageGrad, ...
            averageSqGrad,iteration,Config.Learning_Rate);


    end

    %% Plot Results

    figure(3)
    subplot(2,1,1)
    plot(epoch,Loss,'.k','Markersize',16)
    hold on
    plot(epoch,Loss_Data,'.b','Markersize',16)
    plot(epoch,Config.alpha*Loss_physics,'.r','Markersize',16)
    grid on
    grid minor
    xlabel("epoch")
    ylabel("loss")
    set(gca,'YScale','log')
    legend("Total Loss","MSE","Physics")
    
    % predict (just for tests that everything is ok)
    dlYplot = Network_PINN(dlXplot,parameters,1,Layer);
    
    % Plot

    subplot(2,1,2)
    hold off
    plot(PDE.t,PDE.theta,'-k')
    hold on
    plot(dlXm,dlYm,'.r','markersize',16)
    plot(dlXplot,dlYplot,'-b')
    legend("ideal","data","prediction")

    disp(parameters.param.gamma)
    disp(parameters.param.beta*100)


end


