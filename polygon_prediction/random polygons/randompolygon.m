function randompolygon(count)
if count >= 12 
for k = 1: count/12 
points = 3:12 ;
radius = 1;
for i = 3:12 
x = zeros(points(i),1);
y = zeros(points(i),1);
angle = 2*pi;
divider = angle/points(i);
seperation= 0: divider: angle;
seperation(end) = [];
for j = 1:points(i)
    x(j) = (radius*rand(1)) * cos(seperation(j) + divider*rand(1));
    y(j) = (radius*rand(1)) * sin(seperation(j) + divider*rand(1));
end
plot([x; x(1)],[y; y(1)],'ro-')
hold on 
end 
end
else
 for k =1 : count  
points = floor(3+ 10*rand(1));
radius = 1;
x = zeros(points,1);
y = zeros(points,1);
angle = 2*pi;
divider = angle/points;
seperation= 0: divider: angle;
seperation(end) = [];
for j = 1:points
    x(j) = (radius*rand(1)) * cos(seperation(j) + divider*rand(1));
    y(j) = (radius*rand(1)) * sin(seperation(j) + divider*rand(1));
end
plot([x; x(1)],[y; y(1)],'ro-')
hold on 
 end     
end