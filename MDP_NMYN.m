%% Project Security aware routing in SDN networks
% Starts: 25 march 2022

% Papers used: 
% Kim et al. Deep Reinforcement learning-based Routing on Software defined 
% networks.<<Model of the network based on M/M/1/k>>
% Rezapour et al. RL-Shield: Mitigating Target Link-Flooding Attacks Using SDN and Deep
% Reinforcement Learning Routing algorithm. << MDP >>
% 
%% Clean slate
clear
close all

%% Model of the SWs
%  SW model: M/M/1/k
% 
% <<StateTransitionMM1k.PNG>>
%
% Parameters
% M: Arrival process have Poisson distribution,
% M: Service process have Poisson distribution,
% 1: One server,
% k: Length of the server queue: Capacity
clc
lambda_n_t = 10; % rate of arrivals for SWn at t=t
mu_n     = 15; % rate of service for SWn : constant
rho_n_t    =  lambda_n_t/mu_n; % ratio mandatory 0<rho_n_t<=1
Kn = 10; % capacity of SWn: constant
%
% Expected delay E[dn(t)] for SWn at t=t
E_d_n_t = 0; % initial value
% Probability of loss a flow due to overflow for SWn at t=t
P_b_n_t = 0; 

% M/M/1/k model for SW nx at time t
[E_d_n_t, P_b_n_t] = M_M_1_k_t(Kn,lambda_n_t,mu_n);
%% Analysis for a single SWn with different parameters
close all
clc
lambda_n_t = sort(randi([0 10],1,10)); % U(0,10)
mu_n = 20;
Kn = 2;
rho_n_t    =  lambda_n_t/mu_n; % mandatory 0<rho_n_t<=1
mmk1 = zeros(length(rho_n_t),2);
for i=1:length(rho_n_t)
    [E_d_n_t, P_b_n_t] =  M_M_1_k_t(Kn,lambda_n_t(i),mu_n);
    mmk1(i,:) = [E_d_n_t, P_b_n_t];
end
plot(rho_n_t,mmk1(:,1),'b--o','LineWidth',2); hold on
plot(rho_n_t,mmk1(:,2),'r--x','LineWidth',2);
xlabel('\rho=\lambda/\mu')
legend('Expected delay','Loss Probability')
grid('minor')
%% Network graph
% $$ E = \{e_1, e_2, ..., e_{|E|}\} $$ : Directional Link 
%
% $$ V = \{v_1, v_2, ..., v_{|N|}\} $$ : Node
% 
% $$ G = (V, E) $$  : Graph
%
close all
clc
N = 5;  % number of edges
A = ones(N); % Adjacency matrix (edges or links)
% W = -0 + (0+1)*rand(N); % Weight: initial values Uniform distribution(0,1)
W = [0.2625    0.4886    0.5468    0.6791    0.8852
    0.8010    0.5785    0.5211    0.3955    0.9133
    0.0292    0.2373    0.2316    0.3674    0.7962
    0.9289    0.4588    0.4889    0.9880    0.0987
    0.7303    0.9631    0.6241    0.0377    0.2619];
G = digraph(W, 'omitSelfLoops');
g = plot(G,'Layout','force','EdgeLabel',G.Edges.Weight);
%% Shortest path 
close all
i = 3;   % node i src
j = 5;   % node j dst
[pij,cij] = shortestpath(G,i,j);
g = plot(G,'Layout','force','EdgeLabel',G.Edges.Weight);
highlight(g,pij,'EdgeColor','g');
%% Flows in the network
% 
% $$F_t = \{f_t^{1}, f_t^{2}, ..., f_t^{k}, ..., f_t^{M_t}\} $$ : Flow in
% the network
% 
% $$ p_{i,j} = \{v_1^{i,j}, v_2^{i,j}, ..., v_{|p|}^{i,j}\} $$: Shortest path for a pair of nodes i,j

% Create 100 flows arbitrary
clc
Mt = 10;           
F_t=randi([1 N],Mt,2); 
F_t(F_t(:,1)==F_t(:,2),:)=[]; % avoid internal flows
Mt = length(F_t);
Lambda_k_t = -0 + (0+1)*rand(Mt); % Poisson arrivals of each flow fk

%% Network of M/M/1/k SWs
% 
clc
close
g = plot(G,'Layout','force','EdgeLabel',G.Edges.Weight);
% 
% Parameters of SWs
lambda_n_t = randi([10 15],N,1); % rate of arrivals for SWn at t=t
mu_n     = randi([15 20],N,1); % rate of service for SWn : constant
rho_n_t    =  lambda_n_t./mu_n; % ratio mandatory 0<rho_n_t<=1
Kn = randi([12 20],N,1); % capacity of SWn: constant
% Shortest paths for each fk in F
pij_k = {};
for k=1:Mt
    i = F_t(k,1);
    j = F_t(k,2);
    [pij,cij] = shortestpath(G,i,j);
    pij_k{k} = pij;
    highlight(g,pij,'EdgeColor',rand(1,3));
end
%% Expected Total  end to end flow of each flow
De2ek = zeros(1,Mt);
D_e2e_avg_t = 0;
for k=1:Mt  % for each flow
    pij = pij_k{k};
    D_e2e_k_t = 0;
    for n=1:length(pij) % for each SWn in path for fk
        [E_d_n_t, P_b_n_t] =  M_M_1_k_t(Kn(pij(n)),lambda_n_t(pij(n)),mu_n(pij(n))); % delay at SWn
        D_e2e_k_t = D_e2e_k_t + E_d_n_t;
    end
    D_e2e_avg_t = D_e2e_avg_t + D_e2e_k_t/length(pij);
end
D_e2e_avg_t = D_e2e_avg_t/Mt
%% Expected Total  loss traffic in the network
E_L_total_n_t = 0;
for n=1:N
    [E_d_n_t, P_b_n_t] =  M_M_1_k_t(Kn(n),lambda_n_t(n),mu_n(n)); % delay at SWn
    E_L_total_n_t = E_L_total_n_t + P_b_n_t*lambda_n_t(n);
end
E_L_total_n_t
%% MDP model
% state representation
% Observations from the SWn
% {Kn, Nn_t, lambda_n_t, L_n_t, rho_n_t, d_n_t}
% time: time step
% Kn: Capacity of SWn
% Nn_t: Quee ocupation at t
% lambda_n_t: Aggregate arrival rate at t
% L_n_t: Espected loss traffic at t; is it calculated? Using M/M/1/k model?
% rho_n_t: Utilization of SWn at t
% d_n_t: Expected delay at t; is it calculated? Using M/M/1/k model?
%
% Get state st
% s_t = [f1,f2,f3,f4]'
% f_1 = [0, 1, 0, 0, ... , 0_|N|]'; Src node
% f_2 = [0, 0, 1, 0, ... , 0_|N|]'; Dst node
% f_3 = [0, 1, 0, 1, ... , 1_|E|]'; current path
% f_4 = [0, 1, 0, 0, ... , 0_|N|]'; current node

%
% Actions
% A = [a_1, a_2 ..., a_|E|]'     ; a_i: next hop to node i. If current  
% node=i then a_i means blocking

%
% Evaluating in this example a random policy
%
%
clc;
close all
G; % directed graph
Edges = G.Edges; % undirected graph
Links = Edges.EndNodes;
L = length(Links);
% Action set
A = 1:N; % hop to SWn. a_i: next hop to node i. If current  
% node=i then a_i means blocking
t = 1;
T = 10; % max_steps per episode


% EPISODE
e  = 1;
reward = 0;
% initial state
src_dst = randperm(N,2)  %Src-Dst nodes
f_1 = zeros(N,1); f_1(src_dst(1))=1; %Src node
f_2 = zeros(N,1); f_2(src_dst(2))=1; %Dst node
f_3 = false(L,1); % current path
f_4 = zeros(N,1); f_4(src_dst(1))=1; % current node is src node
s_t = [f_1;f_2;f_3;f_4];

for step = 1:T % steps
    a = randi(length(A));  % random policy next node?
    % execute action
    f_1 = s_t(1:N); %Src node
    f_2 = s_t(N+1:2*N); %Dst node
    f_3 = s_t(2*N+1:2*N+L); % current path
    f_4 = s_t(2*N+L+1:end); % current node
    current_node=find(f_4==1);  % current node
    neighb = successors(G,current_node);    % neighbors
    a_posible = find(neighb==a);   % is it a valid action?
    if isempty(a_posible) % is an invalid action
        reward = -0.02;   % if a_i is not possible
        break;            % END OF EPISODE
    else
        % next state
        f_1_ = f_1; % src_dst doesnot change inside the epoch
        f_2_ = f_2; % src_dst doesnot change inside the epoch
        [tf_1, index_1]=ismember(Links,[current_node a],'rows');  % which link?
        f_3_ = f_3 | tf_1;  %  current path
        f_4_ = zeros(N,1); f_4_(a)=1;   % current node
        s_t_ = [f_1_;f_2_;f_3_;f_4_];

        dst = find(f_2_==1);
        if (a == dst) % dst node reached
            reward = 1;  % maximum reward, successfully routing
            break; % END OF EPISODE
        else
            if step == T   % maximum steps reached and dst was not found
                reward = 0.01;
                break; % END OF EPISODE
            else
                reward = 0;
            end
        end
    end
    s_t = s_t_; % next state
end
s_t = s_t_;  % needed for final states
close
f_3 = logical(s_t(2*N+1:2*N+L)); % current path
edges = find(f_3==1);
g = plot(G,'Layout','force','EdgeLabel',G.Edges.Weight);
highlight(g,'Edges',edges,'EdgeColor','red','LineWidth',2);
reward
step
%







