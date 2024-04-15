filePath_A = '02/A.csv';
filePath_b = '02/b.csv';
filePath_p = '02/gf_pivot.csv';
filePath_np = '02/gf_nopivot.csv';
filePath_solution = '02/np_solution.csv';


A = csvread(filePath_A);
b = csvread(filePath_b);
p = csvread(filePath_p);
np = csvread(filePath_np);
solution = csvread(filePath_solution);

x = linsolve(A,b);

tolerance = 1e-10;
disp(isequal(abs(x-p) < tolerance, ones(size(x))));
disp(isequal(abs(x-np) < tolerance, ones(size(x))));
disp(isequal(abs(x-solution) < tolerance, ones(size(x))));
