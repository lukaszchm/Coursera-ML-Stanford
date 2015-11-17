function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%
x1 = X(:, size(X, 2) - 1);
x2 = X(:, size(X, 2));
plot(x1(y == 1), x2(y == 1), 'k+', 'MarkerSize', 7, 'LineWidth', 2);
plot(x1(y == 0), x2(y == 0), 'ro', 'MarkerSize', 7, 'MarkerFaceColor', 'y');
% =========================================================================



hold off;

end
