function A=Cholesky(A)
% Cholesky Factorization for symmetric positive definite matrix
% Algorithm 2.7 Heath, p.86
% Factorize A such that A = L*L',
% where L is a lower triangular matrix whose diagonal entries are not
% necessarily unity
% In the output the lower triangular part of A is over-written by L
% The upper triangular part of A is left unchanged and
% has no significance
% K. Ming Leung, 01/25/03

[n ~]=size(A);
for k=1:n
    A(k,k)=sqrt(A(k,k));
    A(k+1:n,k)=A(k+1:n,k)/A(k,k);
    for j=k+1:n
        A(j:n,j)=A(j:n,j)-A(j,k)*A(j:n,k);
    end
A = real(A);    
end
