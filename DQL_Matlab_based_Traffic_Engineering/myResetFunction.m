function [InitialObservation, LoggedSignal] = myResetFunction(EnvConstants)
% Reset function to place custom network environment into a random
% initial state.
% Define the environment constants.
N = EnvConstants.N; % # of nodes
E = EnvConstants.E; % # of edges
T = EnvConstants.T; % # of steps per episode

src_dst = randperm(N,2);  %Src-Dst nodes
f_1 = zeros(N,1); f_1(src_dst(1))=1; %Src node
f_2 = zeros(N,1); f_2(src_dst(2))=1; %Dst node
f_3 = false(E,1); % current path
f_4 = zeros(N,1); f_4(src_dst(1))=1; % current node is src node

% Flows in the network
Mt = 1000;  % number of flows in the network             
F_k=randi([1 N],Mt,2); 
F_k(F_k(:,1)==F_k(:,2),:)=[]; % avoid internal flows
Mt = length(F_k);
Lambda_k = -0 + (0+1)*rand(Mt); % Poisson arrivals of each flow fk

% Return initial environment state variables as logged signals.
LoggedSignal.State = [f_1;f_2;f_3;f_4]; % state
LoggedSignal.Step  = 0;  % time step
LoggedSignal.MaxStepsPerEpisode  =  T; % # of steps per episode 
InitialObservation = LoggedSignal.State; % initial state

end