function createpolygon(count)
doc open
doc close
doc fprintf
filename = 'data.txt';
txt = fopen(filename, 'w');
for k =1 : count  
points = floor(3+ 5*rand(1));
radius = 1;
x = zeros(points,1);
y = zeros(points,1);
angle = 2*pi;
divider = angle/points;
seperation= 0: divider: angle;
seperation(end) = [];

%xvar =[];
%yvar =[];
fprintf(txt, '%6s %6s %6s %6s\n', 'x(j)', 'y(j)', 'count', 'points');
for j = 1:points
    x(j) = (radius*rand(1)) * cos(seperation(j) + divider*rand(1));
    y(j) = (radius*rand(1)) * sin(seperation(j) + divider*rand(1));
   % fprintf(txt, '%6s %6s\n', 'x(j)', 'y(j)');
    fprintf(txt, '%6.2f %6.2f %6.2f %6.2f\n', x(j), y(j), k, points);
end

figure(1);
plot([x; x(1)],[y; y(1)],'ro-')
hold on 
end  
fclose(txt);
end 