
function [X,parameters] = Network_PINN(X,parameters,Predict,Layer)

if Predict == 0 % it means that we want to initialise the neural network

    parameters = [];

    for i = 1 : length(Layer)

        parameters.("p"+i).weights = dlarray(randn([Layer(i) size(X,1)])*sqrt(2/Layer(i)));
        parameters.("p"+i).bias = dlarray(zeros([Layer(i) 1]));

        X = fullyconnect(X,parameters.("p"+i).weights,parameters.("p"+i).bias);
        X = tanh(X);

    end

    parameters.output.weights = dlarray(randn([1 size(X,1)]))/3;
    parameters.output.bias = dlarray(zeros([1 1]));

    X = fullyconnect(X,parameters.output.weights,parameters.output.bias);
    
    parameters.param.gamma = dlarray(1,'CB');
    parameters.param.beta = dlarray(1,'CB');

elseif Predict == 1 % it means that we want to predict Y given X and parameters

    for i = 1 : length(Layer)
        X = fullyconnect(X,parameters.("p"+i).weights,parameters.("p"+i).bias);
        X = tanh(X);
    end

    X = fullyconnect(X,parameters.output.weights,parameters.output.bias);

end

end


