function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%       

% 特征数
feature_num = size(X, 2);
% 样本数
example_num = size(X, 1);
% 循环每个特征
for i = 1: feature_num
    mu(i) = mean(X(:, i));
    sigma(i) = std(X(:, i));
endfor

% 打印中间结果
% fprintf('mu:\n');
% fprintf(' %f \n', mu);
% fprintf('sigma:\n');
% fprintf(' %f \n', sigma);

% 循环样本
for i = 1:example_num
    % 循环特征
    for j = 1:feature_num
        # 样本标准化公式：(样本值 - 平均值) / 标准差
        X_norm(i, j) = (X_norm(i, j) - mu(j)) / sigma(j);
    endfor
endfor

% 打印结果
% fprintf('X_norm:\n');
% fprintf(' %f \n', X_norm);





% ============================================================

end
