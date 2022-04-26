x = -5:0.25:5;
r = 5;
y = sqrt(r^2-x.^2);
circle_coords = [x; y; zeros(1,length(x))].';
writematrix(circle_coords, 'sin_source_circular_xy.csv');