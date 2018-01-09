side1 = linspace(3,10000,199941)
side2 = linspace(3,10000,199941)
side3 = linspace(2,10000,199961)
side3 = side3(1:199941)
point = 100 * rand(1,2,199941)


doc open
doc close
doc fprintf
filename = 'yaxis.txt';
txt = fopen(filename, 'w');
%fprintf(txt, '\t%6s\t %6s\t %6s\t %6s\t %6s\t %6s\t %6s\t %6s\t %6s\t %6s\t %6s\t %6s\t %6s\t\n', 'x1', 'y1','x2', 'y2','x3', 'y3','base','side','height','area','perimeter','angle1','angle2');
for i = 1:10000
[a,b,c,d,e,f,base,side,height,area,perimeter, angle1, angle2] =generatepoints(point(1,1,i),point(1,2,i),side2(1,i),side3(1,i));
%fprintf(txt, '\t%6.2f \t%6.2f \t%6.2f \t%6.2f \t%6.2f \t%6.2f \t%6.2f \t%6.2f \t%6.2f \t%6.2f \t%6.2f \t%6.2f \t%6.2f\t\n', a, b, c, d, e, f, base, side, height, area, perimeter, angle1, angle2);
fprintf(txt, '%6.2f\t %6.2f\t %6.2f\t %6.2f\t', b, d, f, b)
end









   
