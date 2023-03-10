[X, Y] = meshgrid(linspace(eps, 1, 50));
Z = 1 - X.^Y;
surf(X, Y, Z, 'EdgeColor','none')
colorbar
