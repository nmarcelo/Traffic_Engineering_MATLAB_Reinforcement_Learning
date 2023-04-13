function [E_N_t, Pb_t] = M_M_1_k_t(K,Lambda_t,mu)
% Model M/M/1/K for SWn at time t
% N_t: Number of flows in the SWn: Nt can be a pdf
% Lambda_t: Input rate to SWn, variable
% mu: Output rate of SWn, constant
% K: Flows capacity of SWn

% ratio input to output: To avoid overflow, rho_t<1
rho_t = Lambda_t/mu;

% Loss probability that a flow is not admited at time t
if rho_t==1
    Pb_t = (rho_t^K);
else
    Pb_t = ((1-rho_t)*(rho_t^K))/(1-(rho_t^(K+1)));
end



% Expected queue occupation
if rho_t==1
    E_N_t = K/2;
elseif rho_t<1
    E_N_t = rho_t/(1-rho_t)-((K+1)*(rho_t^(K+1)))/((K^(1-(rho_t^(K+1))))/2);
end
end