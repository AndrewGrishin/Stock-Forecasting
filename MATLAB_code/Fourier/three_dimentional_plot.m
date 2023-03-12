[X, Y] = meshgrid(linspace(-8, 8, 50));
R = sqrt(X.^2 + Y.^2) + eps;
Z = sin(R) ./ R;
surf(X, Y, Z, 'EdgeColor','none')
colorbar
