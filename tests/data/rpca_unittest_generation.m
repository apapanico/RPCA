warning('off','PROPACK:NotUsingMex')

%% Small Matrix
n=50; r=2;
rng(123)
base=100+cumsum(randn(n,r)); 
scales=abs(randn(n,r));
L=base*scales';
S=round(0.25*randn(n)); 
M=L+S;

fileout = 'test_matrix';
dlmwrite([fileout,'.csv'],M,'delimiter',',','precision','%.12f');

tic
[Lhat, Shat, niter]  = RPCA_ALM(M);
toc

dlmwrite([fileout,'_L.csv'],Lhat,'delimiter',',','precision','%.12f');
dlmwrite([fileout,'_S.csv'],Shat,'delimiter',',','precision','%.12f');

%% Larger Matrix
n = 500;
r = 3;

rng(123)
base=100+cumsum(randn(n,r)); 
scales=abs(randn(n,r));
L=base*scales';
S=round(0.25*randn(n)); 
M=L+S;

fileout = 'larger_test_matrix';
dlmwrite([fileout,'.csv'],M,'delimiter',',','precision','%.12f');

tic
[Lhat, Shat, niter]  = RPCA_ALM(M);
toc

dlmwrite([fileout,'_L.csv'],Lhat,'delimiter',',','precision','%.12f');
dlmwrite([fileout,'_S.csv'],Shat,'delimiter',',','precision','%.12f');