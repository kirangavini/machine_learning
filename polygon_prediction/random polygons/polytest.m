count =200
doc open
doc close
doc fprintf
filename = 'data10.txt';
txt = fopen(filename, 'w');
fprintf(txt, '%6s %6s %6s %6s\n', 'x(j)', 'y(j)', 'count', 'points');
for k =1 : count  
points = 10;
radius = 1;
x = zeros(points,1);
y = zeros(points,1);
angle = 2*pi;
divider = angle/points;
seperation= 0: divider: angle;
seperation(end) = [];

%xvar =[];
%yvar =[];

for j = 1:points
    x(j) = (5*radius*rand(1)) * cos(seperation(j) + divider*rand(1));
    y(j) = (5*radius*rand(1)) * sin(seperation(j) + divider*rand(1));
   % fprintf(txt, '%6s %6s\n', 'x(j)', 'y(j)');
    fprintf(txt, '%6.2f %6.2f %6.2f %6.2f\n', 55+x(j), 45+y(j), k, points);
    if(j==points)
    fprintf(txt, '%6.2f %6.2f %6.2f %6.2f\n', 55+x(1), 45+y(1), k, points);
    end    
end
figure(k);
plot([55+x; 55+x(1)],[45+y; 45+y(1)],'ro-')
%rectangle('Position',[-2 -1 5 4])
%axis([-4 8 -2 7])
end  
fclose(txt);
