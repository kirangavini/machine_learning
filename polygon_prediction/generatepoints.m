function [X1,Y1,X2,Y2,X3,Y3,base,side,height,area,perimeter,angle1,angle2] = generatepoints(x,y,s,b)

X1 = x - b/2;
Y1 = y;
base = b;
side = s;
height = sqrt(s^2 - (b/2)^2);
area = 1/2*base*height;
perimeter = (2*s)+b;
angle1 = acosd(0.5*b/s);
angle2 = 180- (2*angle1);
X2 = x + b/2;
Y2 = y;

X3 = x;
Y3 = y + sqrt(s^2 - (b/2)^2);

X = [X1 X2 X3 X1];
Y = [Y1 Y2 Y3 Y1];

plot(X,Y)

end