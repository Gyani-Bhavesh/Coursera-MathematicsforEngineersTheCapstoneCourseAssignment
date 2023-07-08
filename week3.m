function omega = flow_around_cylinder_unsteady
Re=60;
%%%%% define the grid %%%%%
n=101; m=202; % number of grid points
N=n-1; M=m-2; % number of grid intervals: 2 ghost points, theta=-h,2*pi
h=2*pi/M; % grid spacing based on theta variable
xi=(0:N)*h; theta=(-1:M)*h; % xi and theta variables on the grid
%%%%% time-stepping parameters %%%%%
t_start=0; t_end=0.5; %vortex street starts at around t=1000
tspan=[t_start t_end]; 
%%%%% Initialize vorticity field %%%%%
omega=zeros(n,m);
%%%%% Construct the matrix A for psi equation %%%%%
%%%%% Define the indices associated with the boundaries %%%%%%%%%%%%%%%%%%%
% l_boundary_index = [left]
l_boundary_index=1:n:1+(m-1)*n;
% r_boundary_index = [right]
r_boundary_index=n:n:m*n;
% b_boundary_index = [bottom]
b_boundary_index=1:n;
% t_boundary_index = [top]
t_boundary_index=1+(m-1)*n:m*n;
%%%%% Construct the matrix A for psi equation %%%%%
diagonals = [4*ones(m*n,1), -ones(m*n,4)];
A=spdiags(diagonals,[0 -1 1 -n n], m*n, m*n); %use sparse matrices
I=speye(m*n);
A(l_boundary_index,:)=I(l_boundary_index,:);
A(r_boundary_index,:)=I(r_boundary_index,:);
A(b_boundary_index,:)=I(b_boundary_index,:)-I(b_boundary_index+(m-2)*n,:);
A(t_boundary_index,:)=I(t_boundary_index,:)-I(t_boundary_index-m*n+2*n,:);
%%%%% Find the LU decomposition %%%%%
[L,U]=lu(A); clear A;
%%%%% Compute any time-independent constants %%%%%
hsqexp=h^2*exp(2*xi(:));
coeff1=2/Re./hsqexp;
coeff2=1/4./hsqexp;
psi_free_stream=exp(xi(n))*sin(theta(:));
omega_cyl_denom=2*h^2;
%%%%% advance solution using ode23 %%%%%
options=odeset('RelTol', 1.e-03);
omega=omega(2:n-1,2:m-1); % strip boundary values for ode23
omega=omega(:); % make a column vector
[t,omega]=ode23...
  (@(t,omega)omega_eq(omega,L,U,hsqexp,coeff1,coeff2,psi_free_stream,n,m,omega_cyl_denom,...
  l_boundary_index,r_boundary_index,b_boundary_index,t_boundary_index),...
                                                    tspan, omega, options);
%%%%% expand omega to include boundaries %%%%%
temp=zeros(n,m);
temp(2:n-1,2:m-1)=reshape(omega(end,:),n-2,m-2);
omega=temp; clear temp;
%%%%% compute stream function (needed for omega boundary values) %%%%%
omega_tilde=zeros(n,m);
for j=1:m
    for i=1:n
        omega_tilde(i,j)=hsqexp(i)*omega(i,j);
    end
end
omega_tilde = omega_tilde(:);
b=zeros(m*n,1);
b(:)=omega_tilde(:);
b(l_boundary_index)=0;
b(r_boundary_index)=psi_free_stream;
b(b_boundary_index)=0;
b(t_boundary_index)=0;
psi = U\(L\b);
psi=reshape(psi,n,m);
%%%%% set omega boundary conditions %%%%%
omega(1,:)=(psi(3,:)-8*psi(2,:))/omega_cyl_denom;
omega(n,:)=0;
omega(:,1)=omega(:,m-1);
omega(:,m)=omega(:,2);
%%%%% plot scalar vorticity field %%%%%
plot_Re60(omega,t_end);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function d_omega_dt=omega_eq(omega,L,U,hsqexp,coeff1,coeff2,psi_free_stream,n,m,omega_cyl_denom,l_boundary_index,r_boundary_index,b_boundary_index,t_boundary_index)%%%%% expand omega to include boundary points %%%%%
temp=zeros(n,m);
index1=2:n-1; index2=2:m-1;
temp(index1,index2)=reshape(omega,n-2,m-2);
omega=temp; clear temp;
%%%%% compute stream function %%%%%
omega_tilde=zeros(n,m);
for j=1:m
    for i=1:n
        omega_tilde(i,j)=hsqexp(i)*omega(i,j);
    end
end
omega_tilde = omega_tilde(:);
b=zeros(m*n,1);
b(:)=omega_tilde(:);
b(l_boundary_index)=0;
b(r_boundary_index)=psi_free_stream;
b(b_boundary_index)=0;
b(t_boundary_index)=0;
psi = U\(L\b);
psi=reshape(psi,n,m);
%%%%% set omega boundary conditions %%%%%
omega(1,:)=(psi(3,:)-8*psi(2,:))/omega_cyl_denom;
omega(n,:)=0;
omega(:,1)=omega(:,m-1);
omega(:,m)=omega(:,2);
%%%%% compute derivatives of omega %%%%%
deriv=zeros(n,m);
for j=2:m-1
    for i=2:n-1
        deriv(i,j)=coeff1(i)*(omega(i+1,j)+omega(i-1,j)+omega(i,j+1)+omega(i,j-1)-4*omega(i,j))...
                +coeff2(i)*((psi(i+1,j)-psi(i-1,j))*(omega(i,j+1)-omega(i,j-1))...
                -(psi(i,j+1)-psi(i,j-1))*(omega(i+1,j)-omega(i-1,j)));
    end
end
deriv=deriv(2:n-1,2:m-1); % strip boundaries
deriv=deriv(:);
d_omega_dt=deriv;
clear deriv;
end
