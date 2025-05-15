clc;
clear
close all
disp('GPSRG_upgrade program using the configuration in gp_configuration_runs'); 
dbstop if error
in = input('cross-sectional or time series?(c/t) ','s');

if strcmp(in,'c')
    gp=rungp('gp_conf_file_crossSectional');
elseif strcmp(in,'t')
    gp=rungp('gp_conf_file_timeSeries');
else 
    disp('error: answer with c or t')
    return;
end

%If Symbolic Math toolbox is present
if license('test','symbolic_toolbox')

    disp(' ');
    disp('Using the symbolic math toolbox simplified versions of this');
    disp('expression can be found: ')
    disp('E.g. using the the GPPRETTY command on the best individual: ');
   
    gppretty(gp,'best');
end

in=input('file name for saving?\n ','s');
if ~strcmp(in,'n')
    path = ['.\workspaces\',in];
    save(path,'gp');
end
