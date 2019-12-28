function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)
    
    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the F1 score of choosing epsilon as the
    %               threshold and place the value in F1. The code at the
    %               end of the loop will compare the F1 score for this
    %               choice of epsilon and set it to be the best epsilon if
    %               it is better than the current choice of epsilon.
    %               
    % Note: You can use predictions = (pval < epsilon) to get a binary vector
    %       of 0's and 1's of the outlier predictions

    % 预测值
    predictions = (pval < epsilon);

    % yval:1->异常，0->非异常
    % 真阳性：样本是阳性（True），预测结果也是阳性（True）
    tp = sum((predictions == 1) & (yval == 1));
    % 假阳性：样本是阴性（Flase），预测结果是阳性（True）
    fp = sum((predictions == 1) & (yval == 0));
    % 假阴性：样本是阳性（True），预测结果是阴性（Flase）
    fn = sum((predictions == 0) & (yval == 1));

    % 查准率：预测结果为真的真样本在所有预测结果为真样本中所占的比例。
    prec = tp / (tp + fp);
    % 召回率：预测结果为真的真样本在所有真样本中所占的比例。
    rec = tp / (tp + fn);

    F1 = (2 * prec * rec) / (prec + rec);

    % =============================================================

    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
end

end
