function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

row = size(z, 1);
column = size(z, 2);
% 循环行
for i = 1:row
    % 循环列
    for j = 1:column
        % sigmoid公式：g(z) = 1 / (1 + e ^ (-z))
        g(i, j) = 1 / (1 + e ^ (-z(i, j)));
    endfor
endfor

% =============================================================

end
