function [hf,hp]=plotsem_alpha(x,y,sem,color1,color2, a, lineStyle)

x=x(:)';
y=y(:)';
sem=sem(:)';

hf=fill([x,x(end:-1:1)],[y+sem,y(end:-1:1)-sem(end:-1:1)]',color2, 'FaceAlpha', a);
set(hf,'edgec',color2);
hold on
hp=plot(x,y,'color',color1,'linew',2,'LineStyle',lineStyle);


