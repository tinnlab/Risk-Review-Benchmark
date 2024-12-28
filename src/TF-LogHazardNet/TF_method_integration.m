%%
%clean the environment
clear all;

%%
datapath = '/nfs/blanche/share/daotran/SurvivalPrediction/AllData/TF-ProcessData/';

% Get all items in the folder
contents = dir(datapath);
subFolders = contents([contents.isdir]); % Get only directories

% Remove '.' and '..' directories
subFolders = subFolders(~ismember({subFolders.name}, {'.', '..'}));

% Get just the names in a cell array
folderNames = {subFolders.name};
%%
for i = 1:length(folderNames)
    fprintf('%s\n', folderNames{i});  % Use {} for cell arrays and add \n for newline
    newpath = fullfile(datapath, folderNames{i}); 
    for j = 1:10
        fprintf('seed %d\n', j);
        % get the train data
        GE = csvread(fullfile(newpath, sprintf('mRNA_train_%d.csv', j)), 1, 1);
        ME = csvread(fullfile(newpath, sprintf('meth_train_%d.csv', j)), 1, 1);
        CN = csvread(fullfile(newpath, sprintf('cnv_train_%d.csv', j)), 1, 1);
        surv = readtable(fullfile(newpath, sprintf('survival_train_%d.csv', j)), ReadRowNames=true, VariableNamingRule="preserve");
        
        
        Temp(:,:,1) = CN;
        Temp(:,:,2) = GE;
        Temp(:,:,3) = ME;
        Tensor_ob = tensor(Temp);
        clear Temp;
        
        % cpals
        rank = 17; % this is the default rank number
        trainfac = cp_als(Tensor_ob, rank);
        
        A = trainfac.u{1};
        B = trainfac.u{2};
        C = trainfac.u{3};
        lam = trainfac.lambda.';
        
        A = array2table(A);
        A.Properties.RowNames = surv.Properties.RowNames;
        
        to_write_name = fullfile(newpath, sprintf('omicsinter_train_%d.csv', j));
        writetable(A, char(to_write_name), WriteRowNames=true);

        % load the val data
        GE_val = csvread(fullfile(newpath, sprintf('mRNA_val_%d.csv', j)), 1, 1);
        ME_val = csvread(fullfile(newpath, sprintf('meth_val_%d.csv', j)), 1, 1);
        CN_val = csvread(fullfile(newpath, sprintf('cnv_val_%d.csv', j)), 1, 1);
        surv_val = readtable(fullfile(newpath, sprintf('survival_val_%d.csv', j)), ReadRowNames=true, VariableNamingRule="preserve");


        Tempval(:,:,1) = CN_val;
        Tempval(:,:,2) = GE_val;
        Tempval(:,:,3) = ME_val;
        tensor_val = tensor(Tempval);
        nval = height(CN_val);
        
        replam = repmat(lam,nval,1);
        BC = pinv(kr(C,B).');
        [nbc ncol2] = size(BC);
        X1 = reshape(Tempval, nval, nbc);
        A_hat = X1 * BC;
        A_hat_norm = A_hat ./ replam;
        clear Tempval;

        A_hat_norm = array2table(A_hat_norm);
        A_hat_norm.Properties.RowNames = surv_val.Properties.RowNames;
        
        to_write_name = fullfile(newpath, sprintf('omicsinter_val_%d.csv', j));
        writetable(A_hat_norm, char(to_write_name), WriteRowNames=true);

    end
end

