%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Goal:
%     This Program aims to solve the deterministic benchmark optimal 
%     saving problem with Dynamic Programming
%
% Methodology:
%    The methodology used here is value function iteration and the implicit 
%    finite difference method
%
% The PDE Equations to solve:
%    1. rho*V(a) = max{u(c) + (ra + w - c)*(dV/da)}
%    2. u'(c) = dV/da
%    here, u(c) = (c^{1-sigma}-1)/(1-sigma)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all; clc;

tic;

%%% Set all parameters
rho = 0.1; % subjective discount factor
r = 0.01;  % interest rate
w = 1;  % wage
sig = 0.6; % CRRA utility with parameter sigma

%%% Set the asset grid
I=200000;
amin = 0.01; %borrowing constraint
amax = 20;
a = linspace(amin,amax,I)';
da = (amax-amin)/(I-1);

%%% Initialize the value function
v0 = (w + r.*a).^(1-sig)/(1-sig)/rho;

%%% Value Function Iteration

maxit = 1000;
crit = 10^(-12);
V = v0;

for i=1:maxit

    [V_change, V_new, c] = updating(a, amax, amin, V, I, da, w, r, sig, rho);

    dist(i) = max(abs(V_change));
    if dist(i)<crit
        disp('Value Function Converged, Iteration = ')
        disp(i)
        break
    end

    V = V_new;

end

toc

%%% Graphs

% Distance
set(gca,'FontSize',14)
plot(dist,'LineWidth',2)
grid
xlabel('Iteration')
ylabel('||V^{n+1} - V^n||')

% Value Function
adot = w + r.*a - c;

set(gca,'FontSize',12)
plot(a,V,'LineWidth',2)
grid
xlabel('a')
ylabel('V_i(a)')
xlim([amin amax])

% Policy Function
set(gca,'FontSize',14)
plot(a,c,'LineWidth',2)
grid
xlabel('a')
ylabel('c_i(a)')
xlim([amin amax])

% Savings
set(gca,'FontSize',14)
plot(a,adot,a,zeros(1,I),'-','LineWidth',2)
grid
xlabel('a')
ylabel('s_i(a)')
xlim([amin amax])

%%% Compare the analytical solution with numerical solution by the time
%%% path

dt = 10^(-5);
T = 2;
t = linspace(0,T,T/dt);

c0 = 1.5;

% Analytical solution
c_ana = (c0 .* exp((r-rho).*t./sig))';

% Numerical solution
c_num = zeros(length(t),1);
err = zeros(length(t),1);
c_num(1) = c0;

for i=2:length(t)

    c_num(i) = num_path(c,c_num(i-1),a,w,r,dt,da);
    err(i) = abs(c_num(i)-c_ana(i));
    %disp(c_num(i))

end

Err = max(err)

set(gca,'FontSize',14)
plot(t,c_ana,t,c_num,'--','LineWidth',2)
grid
xlabel('Time')
ylabel('Optimal Consumption')
xlim([0, t(end)+0.5])

%%% ========================================================================

%%% Value Function Updating 
% Use Finite Difference Method to approximate value function's derivative
% One-Step updating

function [V_change, V_new, c] = updating(a,amax,amin,V,I,da,w,r,sig, rho)
    
    % initialize forward, backward derivative and consumption 
    dVf = zeros(I,1);
    dVb = zeros(I,1);
    c = zeros(I,1);

    % forward difference
    dVf(1:I-1) = (V(2:I)-V(1:I-1))/da;
    dVf(I) = (w + r.*amax).^(-sig);
    % backward difference
    dVb(2:I) = (V(2:I)-V(1:I-1))/da;
    dVb(1) = (w + r.*amin).^(-sig);

    I_concave = dVb > dVf; %indicator whether value function is concave 
                           %(problems arise if this is not the case)
    
    %consumption and savings with forward difference
    cf = dVf.^(-1/sig);
    ssf = w + r.*a - cf;
    %consumption and savings with backward difference
    cb = dVb.^(-1/sig);
    ssb = w + r.*a - cb;
    %consumption and derivative of value function at steady state
    c0 = w + r.*a;
    dV0 = c0.^(-sig);

    % dV_upwind makes a choice of forward or backward differences based on
    % the sign of the drift    
    If = ssf > 0; %positive drift --> forward difference
    Ib = ssb < 0; %negative drift --> backward difference
    I0 = (1-If-Ib); %at steady state
    
    dv_upwind = dVf.*If + dVb.*Ib + dV0.*I0;
    c = dv_upwind.^(-1/sig);
    u = c.^(1-sig)/(1-sig);

    % construct the A matrix
    X = - min(ssb,0)/da;
    Y = - max(ssf,0)/da + min(ssb,0)/da;
    Z = max(ssf,0)/da;
    
    A=spdiags(Y,0,I,I)+spdiags(X(2:I),-1,I,I)+spdiags([0;Z(1:I-1)],1,I,I);
    
    if max(abs(sum(A,2)))>10^(-12)
        disp('Improper Transition Matrix')
    end

    % sove the function to get the updated value function
    Delta = 1000;
    B = (rho + 1/Delta)*speye(I) - A;

    b = u + V/Delta;
    V_new = B\b;

    V_change = V_new - V;

end

%%% generate the numerical path one-step further

function c_new = num_path(c,c_old,a,w,r,dt,da)

    index_c = c_old > c;
    index = find(index_c > 0);

    prop = (c_old - c(index(end)))/(c(index(end)+1)-c(index(end)));
    a_old = a(index(end)) * prop + a(index(end)+1) * (1-prop);
    adot = w + r*a_old - c_old;
    a_new = a_old + adot*dt;

    index_a = a_new > a;
    index = find(index_a > 0);
    prop = (a_new - a(index(end)))/da;
    c_new = c(index(end)) * prop + c(index(end)+1) * (1-prop);

end


