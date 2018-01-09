

% preallocate x and y 
x = zeros(numVert,1);
y = zeros(numVert,1);

% angle of the unit circle in radians
circleAng = 2*pi;
% the average angular separation between points in a unit circle
angleSeparation = circleAng/numVert;
% create the matrix of angles for equal separation of points
angleMatrix = 0: angleSeparation: circleAng;
% drop the final angle since 2Pi = 0
angleMatrix(end) = [];

% Specify polygon variables
numVert = 3;
radius = 10;
radVar = 1.5; % variance in the spikiness of vertices
angVar = 1; % variance in spacing of vertices around the unit circle

% generate the points x and y
for k = 1:numVert
    x(k) = 50 +abs(((radius + radius*rand(1)*radVar) * cos(angleMatrix(k) + angleSeparation*rand(1)*angVar)));
    y(k) = 45 +abs(((radius + radius*rand(1)*radVar) * sin(angleMatrix(k) + angleSeparation*rand(1)*angVar)));
end

% Graph the polygon and connect the final point to the first point
plot([x; x(1)],[y; y(1)],'ro-')
hold on
plot([-2 2],[0 0],'k')
plot([0 0],[-2 2],'k')