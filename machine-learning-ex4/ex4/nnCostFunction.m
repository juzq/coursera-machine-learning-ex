function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% 转化y
y2 = zeros(m, num_labels);
for i = 1 : m
    y2(i, y(i)) = 1;
endfor

% 计算h(x)
a1 = [ones(m, 1) X]';
z2 = Theta1 * a1;
a2 = [ones(1, m); sigmoid(z2)];
z3 = Theta2 * a2;
a3 = sigmoid(z3);
h = a3';

% 计算J(theta)

% 循环方式
%for i = 1 : m
%    for k = 1: num_labels
%        J = J + (-1 * y2(i, k) * log(h(i, k)) - (1 - y2(i, k)) * log(1 - h(i, k)));
%    endfor
%endfor
%J = J / m;

% 向量方式
J = - 1 / m * (sum(sum(y2 .* log(h))) + sum(sum((1 - y2) .* log(1-h))));

% 计算正则化后的J
% 去掉theta的第一列，即去掉theta0
Theta1_new = Theta1(:, 2:end);
Theta2_new = Theta2(:, 2:end);
J = J + (sum(sum(Theta1_new .^ 2)) + sum(sum(Theta2_new .^ 2))) * lambda / (2 * m);

% 梯度计算——循环方式
%for i = 1 : m
%    % 将y2转化为每一列为一个样本
%    delta3 = a3(:, i) - y2(i, :)';
%    Theta2_grad = Theta2_grad + delta3 * a2(:, i)';
%    % Theta2需要使用去掉了第一列针对偏置的权重
%    delta2 = Theta2_new' * delta3 .* sigmoidGradient(z2(:, i));
%    Theta1_grad = Theta1_grad + delta2 * a1(:, i)';
%endfor
%Theta2_grad = Theta2_grad / m;
%Theta1_grad = Theta1_grad / m;

% 梯度计算——矩阵方式
% 将y2转化为每一列为一个样本
delta3 = a3 - y2';
Theta2_grad = delta3 * a2' / m;
% Theta2需要使用去掉了第一列针对偏置的权重
delta2 = Theta2_new' * delta3 .* sigmoidGradient(z2);
Theta1_grad = delta2 * a1' / m;

% 正则化，需要将theta的第一列设置为0
Theta1(:, 1) = 0;
Theta2(:, 1) = 0;
Theta2_grad = Theta2_grad + lambda / m * Theta2;
Theta1_grad = Theta1_grad + lambda / m * Theta1;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
