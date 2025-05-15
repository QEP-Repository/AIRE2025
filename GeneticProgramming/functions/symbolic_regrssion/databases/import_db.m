%% Import data from spreadsheet
%% Set up the Import Options and import the data
opts = spreadsheetImportOptions("NumVariables", 10);

% Specify sheet and range
opts.Sheet = "Sheet1";
opts.DataRange = "A2:J5001";

% Specify column names and types
opts.VariableNames = ["rho", "U", "D", "mu", "F_D", "cp", "alpha", "T", "Re", "Cd"];
opts.VariableTypes = ["double", "double", "double", "double", "double", "double", "double", "double", "double", "double"];

% Import the data
ReynoldsCd = readtable("C:\Users\Luca\Documents\Python\DimensionalAnalysisCornell\dataset\Reynolds Cd.xlsx", opts, "UseExcel", false);
%% Clear temporary variables
clear opts

y = {ReynoldsCd.Cd};
x = {ReynoldsCd.Re};

save('db2run',"x","y")