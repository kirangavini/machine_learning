
% Specify polygon variables
numVert = 20;
radius = 1;

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
radVar = 1; % variance in the spikiness of vertices
angVar = 1;

% generate the points x and y
for k = 1:numVert
    x(k) = (radius + radius*rand(1)*radVar) * cos(angleMatrix(k) + angleSeparation*rand(1)*angVar);
    y(k) = (radius + radius*rand(1)*radVar) * sin(angleMatrix(k) + angleSeparation*rand(1)*angVar);
end

% Graph the polygon and connect the final point to the first point
plot([x; x(1)],[y; y(1)],'ro-')
%hold on
%plot([-2 2],[0 0],'k')
%plot([0 0],[-2 2],'k')
