function [gp]=gp_conf_file_crossSectional(gp,Database)
%This file is used in order to configure the symbolic regression run

% Main run control parameters
% ---------------------------
gp.runcontrol.pop_size=200;			% Population size
gp.runcontrol.num_gen=500;          % Number of generations to run for including generation zero  
                                    % (i.e. if set to 100, it'll finish after generation 99).
                                    
gp.runcontrol.CrossValPartition = false;    %set always to FALSE, its just an experiment
gp.runcontrol.CrossValPartition_times = 5; 

gp.runcontrol.runs = 1;      %number of runs to perform and merge together at the end                                   
gp.runcontrol.parallel.auto = false; %set to true if you want to enable parallel computing

gp.runcontrol.lambda = 10^-5;
gp.runcontrol.arg_power_constant = true; %set to true if you want only constants as argument of the power function
                                         %added because I needed to check
                                         %some functions with riccardo that
                                         %were x2^x3+x2^x3...
gp.runcontrol.onlyIntegerPowerExp = true; %set to true if you want only ( 0.5, 1,1.5,2..)
                                            %as arguments of powers.
                                            %highly recommended when
                                            %dimensionality check is true

gp.runcontrol.dim_check =false;        %set to true if you want to have a dimensionality check of the models
                                      %A dim_score is assosiaceted to every
                                      %model. It is a measure of how the
                                      %vector of units of the model is
                                      %different from the one of the real
                                      %output. This dim_score is evaluated
                                      %and used inside the fitness function
                                      
gp.runcontrol.dim_check_boundary = 1; % maturity boundary after which starting dimensionality check
gp.runcontrol.dim_score2beachieved = 0.1; % min. acceptable dim_score to be achieved 

gp.runcontrol.timeSeries = false;       %set true if input variables are time-series
gp.runcontrol.partial_derivatives = false;
gp.nodes.inputs.maxShiftTime = 100;          %maximum number of time-shifts

gp.nodes.inputs.shiftWidthTime = 50;        %shifts only multiples of 10
gp.nodes.inputs.maxShiftSpace = 2;          %maximum number of time-shifts
gp.runcontrol.verbose=1;              % Set to n to display run information to screen every n generations
gp.runcontrol.usecache=false;          %set always to FALSE, needs fix

% Selection method options
% -------------------------
gp.selection.tournament.size=8;
gp.selection.tournament.lex_pressure=true;  % True to use Luke & Panait's plain lexicographic tournament selection
gp.selection.elite_fraction=0.02;%0.02      

%probability to use tournament and not the new rules
gp.selection.prob_selection=0.5;
%probability for the tournament to be a pareto one
gp.pselection.tournament.p_pareto = 0.5;

% Fitness function and optimisation specification 
% ------------------------------------------------
%using AIC  (default having RMSE/n instead of RSS)
%gp.fitness.criterion='akaike_num_nodes';
%using AIC _median
% gp.fitness.criterion='AIC_median';
%using BIC
gp.fitness.criterion='BIC_num_nodes';
%using TIC
%gp.fitness.criterion='TIC_num_nodes';
%using aic with KLD
% gp.fitness.criterion='akaike_KLD';
%using bic with KLD
% gp.fitness.criterion='BIC_KLD';
%using NL e LL
% gp.fitness.criterion='akaike_NL_LL';
%using the GD distance
% gp.fitness.criterion='GD_distance';
% using the RMSE distance
% gp.fitness.criterion='RMSE';
%using the Median as distance
% gp.fitness.criterion='AIC_median';
% using the Robust Statistic 
% gp.fitness.criterion='RobustStat';
%Using the partial derivatives
%gp.fitness.criterion='partial_derivatives';
%Using the entropy criteria
%gp.fitness.criterion='Entropy';
%Using the mutual information bic criteria
% gp.fitness.criterion='BIC_Mi';
%Using the z score bic criteria. WARNINGS! it has not a general
%implememation yet
%gp.fitness.criterion='BIC_Zscore'; 
%Using the MFC
% gp.fitness.criterion='MFC';

gp.fitness.fitfun=@FF;
gp.fitness.minimisation=true;                % True if to minimise the fitness function (if false it is maximised).
gp.fitness.terminate=false;                   % True to terminate run early if fitness threshold met


% Parameters about the FF
if strcmp(gp.fitness.criterion,'GD_distance')
    gp.fitness.s_data = 0.1;
    gp.fitness.s_model = 0.001;
elseif strcmp(gp.fitness.criterion,'RobustStat')% 
    gp.fitness.robust_name='GD_winsor';         % 'normal' 'median' 'trimmed' 'winsor' 'GD_normal' 'GD_median' 'GD_trimmed' 'GD_winsor' <-POSSIBLE FLAGS
    gp.fitness.robust_GDpnoise = 20;            %  LNT   LNT   LNT    LNT    LN     LN      LN       LN     <-LNT: Line NOT Needed, please comment it
    gp.fitness.robust_trimmed_per = 20;         %  LNT   LNT   LN    LN    LNT     LNT      LN       LN     <-LN: Line Needed, please comment it
elseif strcmp(gp.fitness.criterion,'Entropy') %NB: condition stands for Conditional Entropy.
                                              %    matlab: built in binning
                                              %    shannon: KLD binning
    gp.fitness.entropy_name = 'condition';        %  'condition'    'matlab'   'shannon'
    gp.fitness.boundary_flag = 1;                 %    1, 2, 3       LNT      1, 2      <- LNT: Line NOT Needed;
                                                  %                                        1<-best binwidth for each variable
                                                  %                                        2<-same binning for both variable
                                                  %                                        3<-"Thumb Rule": sqrt(n)~#of bins
end


%termination criteria
%Residuals or fitness crtiteria
if strcmp(gp.fitness.criterion,'akaike_num_nodes') || ...
        strcmp(gp.fitness.criterion,'akaike_KLD') || ...
        strcmp(gp.fitness.criterion,'akaike_NL_LL') || ...
        strcmp(gp.fitness.criterion,'BIC_num_nodes') || ...
        strcmp(gp.fitness.criterion,'TIC_num_nodes') || ...
        strcmp(gp.fitness.criterion,'AIC_median') || ...
        strcmp(gp.fitness.criterion,'MFC')
    gp.fitness.terminate_value=-2*1e8;
    gp.fitness.terminate_value_RMSE=2*1e-5;
elseif strcmp(gp.fitness.criterion,'RMSE')
    gp.fitness.terminate_value=1e-5;
elseif strcmp(gp.fitness.criterion,'GD_distance')
    gp.fitness.terminate_value=1e-9;
elseif strcmp(gp.fitness.criterion,'partial_derivatives')
    gp.fitness.terminate_value=-1e8;
    gp.fitness.terminate_value_RMSE=1e-4;
elseif strcmp(gp.fitness.criterion,'Entropy')
    gp.fitness.terminate_value=-1e8;
    gp.fitness.terminate_value_RMSE =1e-5; 
elseif strcmp(gp.fitness.criterion,'RobustStat')
    gp.fitness.terminate_value=-1e8;
    gp.fitness.terminate_value_RMSE =1e-5; 
end

% maximum evolution time in hours
gp.runcontrol.maxtime=24;
% percentage of maturity to be achieved.
% IDEAL 90%, real data 50% .... I will set (75%)
% If this is not set, It will be used 90%
gp.runcontrol.maturity2beachieved=10;
gp.runcontrol.maturity_step_perc=0.01;
%set for unloaded data
gp.info.loaded=false;


% User data specification  
% ---------------------------------------------------------------
    if nargin>1
        load(Database);
        
    elseif ~exist('gp.userdata.xtrain','var') && ~exist('gp.userdata.ytrain','var')
        name=input('I need a DB, please\n','s');
        load(['.\Databases\',name]);
        
        %check if variables are cell-arrays;
        if ~iscell(y) || ~iscell(x) || ~exist('y','var') || ~exist('x','var')
            error(['y(dep. var.) and x(ind. var.) should be cell arrays'...
            'Each cell containing a single repetition of the Database. If'...
            'database is made of just one repetition, cell arrays should have'...
            'just one cell']);
        end
    end
    
        Nreps = numel(x);
        if Nreps == 1 && gp.runcontrol.CrossValPartition
            gp.userdata.x = x;
            gp.userdata.y = y;
        end
        %train set
        gp.userdata.xtrain = x;
        %test set 
%         gp.userdata.xtest = xtest;
        if gp.runcontrol.dim_check
            gp.userdata.xtrain_unit = x_unit;
            gp.userdata.xtrain_const = x_const;
            num_units = numel(gp.userdata.xtrain_unit{1});
            gp.userdata.null_units = ['[', num2str(zeros(1,num_units)),']'];
        end
        for j = 1 : Nreps %number of database repetitions
        [~,num_var]=size(x{j}); 
            for cc=1:num_var
                if any(isnan(x{j}(:,cc)))
                    error(['You have NaNs in the column ', num2str(cc), ', check and try again. Thanks.']);
                end
            end
        gp.userdata.ytrain=y;
%         gp.userdata.ytest = ytest;
            if gp.runcontrol.dim_check
                gp.userdata.ytrain_unit = y_unit;
            end
        end
    

%Weighted data.
%You must specify if you want to weight and using what, for residuals or
%LSW
%Actually weight is a vector, used for residuals, while W is a matrix
%having o its diag the weight vector. In the LSW case,  provide a matrix,
%even if it will be reported in a vector level because it is more accurate
%formally
%__________________________________________
gp.runcontrol.ResWeight=false;
gp.runcontrol.LSW=false;

if gp.runcontrol.ResWeight
    gp.userdata.ResWeight=weight;
else
    gp.userdata.ResWeight=NaN;
end
if gp.runcontrol.LSW
    gp.userdata.LSW=diag(W);
else
    gp.userdata.LSW=NaN;
end

%Fixed function if given must have the name specified here
%__________________________________________
gp.runcontrol.fixed_variable.bool=false;

if gp.runcontrol.fixed_variable.bool
    fprintf(['\nRemember that you are fixing a variable!\n' ...
        'It will NOT be shown at the end\nFuthermore in the regression_'...
        'WLSorLS_suingSVD\nfile you HAD TO SELECT how to use it\n']);
    gp.nodes.inputs.fixed_variable.name= ...
        input('Name of the fixed variable\n','s');
    gp.userdata.fixed(:,1)=eval(gp.nodes.inputs.fixed_variable.name);
end

% Input configuration
% --------------------
% This sets the number of inputs 
gp.nodes.inputs.num_inp=size(gp.userdata.xtrain{1},2); 		         


if strcmp(gp.fitness.criterion,'akaike_KLD')||strcmp(gp.fitness.criterion,'BIC_KLD')
    gp.fitness.crit.dx=75; % <- number of points for computing the KDEs
    [gp.fitness.crit.fdata,gp.fitness.crit.xdata]= ...
        KDE2KLDiv(gp.userdata.ytrain{1}(gp.nodes.inputs.maxShift+1:end),gp.fitness.crit.dx);
end

gp.fitness.entropy_name = 'condition';
% Constants
% ---------
% When building a tree this is
% the probability with which a constant will
% be selected instead of a terminal.
% [1=all ERCs, 0.5=1/2 ERCs 1/2 inputs, 0=no ERCs]
gp.nodes.const.p_ERC=0.4;
%constant range
gp.nodes.const.range=[-10,10];


% Tree build options
% -------------------
%genetic operators probabilities
gp.operators.mutation.p_mutate=0.45;
gp.operators.crossover.p_cross=0.45;
gp.operators.directrepro.p_direct=0.10;

%different mutation probabilities
if gp.runcontrol.timeSeries
    if gp.runcontrol.dim_check
        gp.operators.mutation.mutate_par=[0.50 0.10 0.10 0 0 0.10 0.10 0.10];
    else
        gp.operators.mutation.mutate_par=[0.60 0.20 0.10 0 0.05 0.05 0 0];
    end
else
    if gp.runcontrol.dim_check
       gp.operators.mutation.mutate_par=[0.5 0.15 0.15 0 0 0.1 0.1 0];
    else
        gp.operators.mutation.mutate_par=[0.80 0.05 0.05 0 0 0.10 0 0];
    end
end

%different probabilities of using functions with arity>0
gp.nodes.functions.squash_yn_parameters=[0.90,0.10];
%probabilities of using basic functions, trigonometric ones, exponential
%and logarithm or squashing terms
gp.nodes.functions.bte=[0.35,0.35,0.30];
gp.nodes.functions.squash_cumsum=cumsum(gp.nodes.functions.squash_yn_parameters);
gp.nodes.functions.bte_cumsum=cumsum(gp.nodes.functions.bte);

% Maximum depth of trees 
gp.treedef.max_depth=7;

% Maximum depth of sub-trees created by mutation operator
gp.treedef.max_mutate_depth=7;%5;

gp.treedef.max_nodes = inf;
% Multiple gene settings
% ----------------------

gp.genes.multigene=true;    % Set to true to use multigene individuals and false to use ordinary single gene individuals.
gp.genes.max_genes=4;       % The absolute maximum number of genes allowed in an individual.
gp.genes.operators.p_cross_hi=0.2; %0.2
gp.runcontrol.savefreq=0;

% Define function nodes
% ---------------------
gp.nodes.functions.name{1}='times';
gp.nodes.functions.name{2}='minus';
gp.nodes.functions.name{3}='plus';
gp.nodes.functions.name{4}='rdivide';
gp.nodes.functions.name{5}='power';
gp.nodes.functions.name{6}='sigmf1p';
gp.nodes.functions.name{7}='sin';
gp.nodes.functions.name{8}='cos';
gp.nodes.functions.name{9}='tan';
gp.nodes.functions.name{10}='log';
gp.nodes.functions.name{11}='exp';
gp.nodes.functions.name{12}='sinh';
gp.nodes.functions.name{13}='cosh';
gp.nodes.functions.name{14}='sqrt';
gp.nodes.functions.name{15}='add3';
gp.nodes.functions.name{16}='mult3'; 
gp.nodes.functions.name{17}='negexp'; %negative exponents. Needs fix!
gp.nodes.functions.name{18} ='hypboneplusx';
gp.nodes.functions.name{19}='sigmnf1p';



% Active functions
% ----------------
%
% Manually setting a function node to inactive allows you to exclude a function node in a 
% particular run.
gp.nodes.functions.active(1)=1;%1;                          
gp.nodes.functions.active(2)=1;%1;                          
gp.nodes.functions.active(3)=1;%1;                          
gp.nodes.functions.active(4)=1;%1;                          
gp.nodes.functions.active(5)=1;                          
gp.nodes.functions.active(6)=1;                          
gp.nodes.functions.active(7)=0;                           
gp.nodes.functions.active(8)=0;
gp.nodes.functions.active(9)=0;
gp.nodes.functions.active(10)=0;
gp.nodes.functions.active(11)=1;
gp.nodes.functions.active(12)=0;
gp.nodes.functions.active(13)=0;
gp.nodes.functions.active(14)=0;
gp.nodes.functions.active(15)=0;
gp.nodes.functions.active(16)=0;
gp.nodes.functions.active(17)=0;
gp.nodes.functions.active(18)=0;
gp.nodes.functions.active(19)=0;

% Check power function selected
%Make a part of the struct for the special functions starting from false,
%so there are not, then check
gp.nodes.functions.power.bool=false;
gp.nodes.functions.sigmf1p.bool=false;
gp.nodes.functions.sigmnf1p.bool=false;


for m=1:length(gp.nodes.functions.name)
    y_np=strcmp(gp.nodes.functions.name{m},'power');
    y_ns=strcmp(gp.nodes.functions.name{m},'sigmf1p');
    y_nns=strcmp(gp.nodes.functions.name{m},'sigmnf1p');
    
    if y_np && gp.nodes.functions.active(m)==1
        gp.nodes.functions.power.bool=true;
        gp.nodes.functions.power.location=m;
    elseif y_ns && gp.nodes.functions.active(m)==1
        gp.nodes.functions.sigmf1p.bool=true;
        gp.nodes.functions.sigmf1p.location=m;
    elseif y_nns && gp.nodes.functions.active(m)==1
        gp.nodes.functions.sigmnf1p.bool=true;
        gp.nodes.functions.sigmnf1p.location=m;
    end
end
