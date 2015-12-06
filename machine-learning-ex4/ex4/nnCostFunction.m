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
allThetas = {Theta1, Theta2};
numberOfLayers = length(allThetas) + 1;
labels = eye(num_labels);

a = cell(numberOfLayers, 1);
z = cell(numberOfLayers, 1);
z{1} = X;
a{1} = z{1};
for layer = 2:numberOfLayers
    a{layer-1} = [ones(m, 1) a{layer-1}];
    z{layer} = a{layer-1} * allThetas{layer-1}';
    a{layer} = 1 ./ (1 + exp(-z{layer}));
end;

for i = 1:m
    currentY = labels(y(i),:);
    currentX = a{3}(i, :);
    J = J + sum(-currentY.*log(currentX) - (1 - currentY).*log(1 - currentX));
end;
J = J/m;

delta = cell(3, 1);
Delta = cell(2, 1);

delta{3} = zeros(m, size(z{3}, 2));
for i = 1:m
   delta{3}(i, :) = a{3}(i, :) - labels(y(i), :); 
end

delta{2} = zeros(m, size(z{2}, 2));
delta{2} = delta{3} * Theta2(:, 2:end) .* sigmoidGradient(z{2});

Delta{1} = delta{2}' * a{1};
Delta{2} = delta{3}' * a{2};

RegTerms = cell(2, 1);
RegTerms{1} = [zeros(size(Theta1) .^ [1 0]), ones(size(Theta1)-[0 1])] .* Theta1 * lambda;
RegTerms{2} = [zeros(size(Theta2) .^ [1 0]), ones(size(Theta2)-[0 1])] .* Theta2 * lambda;

Theta1_grad = (Delta{1} + RegTerms{1})/m;
Theta2_grad = (Delta{2} + RegTerms{2})/m;

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
regularization = 0;
for layer = 1:numberOfLayers-1
    thetaSq = allThetas{layer}(:, 2:end).^2;
    regularization = regularization + sum(thetaSq(:));
end;
J = J + regularization*lambda / (2*m); 


















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
