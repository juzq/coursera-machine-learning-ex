function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

% 遍历所有样本
for i = 1:size(X, 1)
    distance = 9999999999;
    for j = 1:K
        % 距离公式：A(a1, a2)与B(b1, b2)之间的距离为根号下(a1-b1)^2 + (a2 - b2)^2
        dist = sum((X(i, :) - centroids(j, :)) .^ 2);
        if dist < distance
            idx(i) = j;
            distance = dist;
        end
    endfor
endfor





% =============================================================

end

