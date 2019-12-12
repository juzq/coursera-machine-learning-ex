function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% 待选列表
test_set = [0.01 0.03 0.1 0.3 1 3 10 30];

% 当前错误值
error = 9999999999;

for test_c = test_set
    for test_sigma = test_set
        % 使用C与sigma训练模型
        model= svmTrain(X, y, test_c, @(x1, x2) gaussianKernel(x1, x2, test_sigma)); 
        % 使用交叉验证集在模型上预测结果
        predictions = svmPredict(model, Xval);
        % 计算预测结果误差的平均值
        predict = mean(double(predictions ~= yval));
        % 小于当前错误值，则选为最优参数
        if predict < error
            error = predict;
            C = test_c;
            sigma = test_sigma;
        end
    endfor
endfor


% =========================================================================

end
