function A=Cholesky(A)
% Cholesky Factorization for symmetric positive definite matrix
% Factorize A such that A = L*L',

[n ~]=size(A);
for k=1:n
    A(k,k)=sqrt(A(k,k));
    A(k+1:n,k)=A(k+1:n,k)/A(k,k);
    for j=k+1:n
        A(j:n,j)=A(j:n,j)-A(j,k)*A(j:n,k);
    end
A = real(A);    
end
