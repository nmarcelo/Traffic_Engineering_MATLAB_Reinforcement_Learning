function [NextObs,Reward,IsDone,LoggedSignals] = myStepFunction(Action,LoggedSignals, EnvConstants)
% Custom step function to construct network environment for the function
% name case.
%
% This function applies the given action to the environment and evaluates
% the system dynamics for one simulation step.

% Define the environment constants.
N = EnvConstants.N; % # of nodes
E = EnvConstants.E; % # of edges
G = EnvConstants.G; % # graph
Links = EnvConstants.Links; % # links 


%step

Action;

% Unpack the state vector from the logged signals.
step = LoggedSignals.Step; % step 
T = LoggedSignals.MaxStepsPerEpisode; % # of steps per episode
State = LoggedSignals.State; % state

f_1 = State(1:N); %Src node
f_2 = State(N+1:2*N); %Dst node
f_3 = State(2*N+1:2*N+E); % current path
f_4 = State(2*N+E+1:end); % current node

%
% Apply action

current_node=find(f_4==1);  % current node
neighb = successors(G,current_node);    % neighbors
a_posible = find(neighb==Action);   % is it a valid action?

IsDone = false; 

% Check if the given action is valid.
if isempty(a_posible) % is an invalid action
    Reward = -0.02;   % if a_i is not possible
    IsDone = true;            % % Check terminal condition.
    s_t_ = [f_1;f_2;f_3;f_4];
else
    % next state
    f_1_ = f_1; % src_dst doesnot change inside the epoch
    f_2_ = f_2; % src_dst doesnot change inside the epoch
    [tf_1, index_1]=ismember(Links,[current_node Action],'rows');  % which link?
    f_3_ = f_3 | tf_1;  %  current path
    f_4_ = zeros(N,1); f_4_(Action)=1;   % current node
    s_t_ = [f_1_;f_2_;f_3_;f_4_];

    dst = find(f_2_==1);
    if (Action == dst) % dst node reached
        Reward = 1;  % maximum reward, successfully routing
        IsDone = true;   % % Check terminal condition.
    else
        if step == T-1   % maximum steps reached and dst was not found
            Reward = 0.01; 
            IsDone = true;   % % Check terminal condition.
            warning('maximum steps reached') 
        else
            Reward = 0;
        end
    end
end

step = step + 1;
LoggedSignals.State = s_t_; % new state
LoggedSignals.MaxStepsPerEpisode = T; % same
LoggedSignals.Step = step; % step 
step 
% Transform state to observation.

NextObs = LoggedSignals.State;

end