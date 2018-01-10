

function polygon(sides) % Name number of sides of the polygon
%%sides = input('input the number of sides you want;, ');
circleAng = 7;
degrees = circleAng/sides; % Find the angle between corners in degrees 
r = ones(1,sides) % Array of ones
theta = 0:degrees:circleAng % Theta changes by the internal angle of the polygon

x = zeros(sides,1);
y = zeros(sides,1);
for k = 1:sides
    x(k) = r(k) * cos(theta(k));
    y(k) = r(k)* sin(theta(k));
end
figure;
polar([x;x(1)],[y;y(1)]);
figure;
plot([x;x(1)],[y;y(1)]);
end


