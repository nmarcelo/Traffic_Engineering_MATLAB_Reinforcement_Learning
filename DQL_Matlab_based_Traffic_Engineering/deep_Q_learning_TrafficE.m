%% 
%% Project Traffic engineering in SDN networks
% Starts: 29 march 2022

% Papers used: 
% Kim et al. Deep Reinforcement learning-based Routing on Software defined 
% networks.<<Model of the network based on M/M/1/k>>
% Rezapour et al. RL-Shield: Mitigating Target Link-Flooding Attacks Using SDN and Deep
% Reinforcement Learning Routing algorithm. << MDP >>
% Library of Matlab: Reinforcement Learning


%% Observation and Action Specifications

% Network graph
% $$ E = \{e_1, e_2, ..., e_{|E|}\} $$ : Directional Link 
%
% $$ V = \{v_1, v_2, ..., v_{|N|}\} $$ : Node
% 
% $$ G = (V, E) $$  : Graph
%
close all
clc
clear
N = 5;  % number of nodes
A = ones(N); % Adjacency matrix (edges or links)
% W = -0 + (0+1)*rand(N); % Weight: initial values Uniform distribution(0,1)
W = [0.2625    0.4886    0.5468    0.6791    0.8852
    0.8010    0.5785    0.5211    0.3955    0.9133
    0.0292    0.2373    0.2316    0.3674    0.7962
    0.9289    0.4588    0.4889    0.9880    0.0987
    0.7303    0.9631    0.6241    0.0377    0.2619];
G = digraph(W, 'omitSelfLoops');

Edges = G.Edges; % undirected graph
Links = Edges.EndNodes;
E = length(Links);

% State st
% s_t = [f1,f2,f3,f4]'
% f_1 = [0, 1, 0, 0, ... , 0_|N|]'; Src node    N 
% f_2 = [0, 0, 1, 0, ... , 0_|N|]'; Dst node    N 
% f_3 = [0, 1, 0, 1, ... , 1_|E|]'; current path E
% f_4 = [0, 1, 0, 0, ... , 0_|N|]'; current node N

%
% Observations from the environment

ObservationInfo = rlNumericSpec([3*N+E 1]); %  s is R^3N+L
ObservationInfo.Name = 'Network States';
f1 = '';f2 = ''; f3 = '';f4 = '';
for i = 1:N
    f1 = strcat(f1,'f1_',num2str(i),', ');
    f2 = strcat(f2,'f2_',num2str(i),', ');
    f4 = strcat(f4,'f4_',num2str(i),', ');
end
index = strfind(f4,','); % delete last comma
f4(index(end))=[]; 

for i = 1:E
    f3 = strcat(f3,'f3_',num2str(i),', ');
end

ObservationInfo.Description = strcat(f1, f2, f3,f4);


%
% Action space where
ActionInfo = rlFiniteSetSpec(1:N); % hop to SWn. a_i: next hop to node i. If current  
% node=i then a_i means blocking
ActionInfo.Name = 'Routing Action';


% Create the structure that contains the environment constants.
% Define the environment constants.

envConstants.N = N; % # of nodes
envConstants.E = E; % # of edges
envConstants.G = G; % # graph
envConstants.Links = Links; % # links 
MaxStepsPerEpisode= 10;
envConstants.T = MaxStepsPerEpisode;


%
% Create an anonymous function handle to the                                             custom step function, 
% passing envConstants as an additional input argument.

ResetHandle = @()myResetFunction(envConstants);
StepHandle = @(Action,LoggedSignals) myStepFunction(Action,LoggedSignals,envConstants);


%
% Create the environment using the custom function handles.

env = rlFunctionEnv(ObservationInfo,ActionInfo,StepHandle,ResetHandle);

%
% Create DQN Agent from Observation and Action Specifications

% obtain observation and action specifications
obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);

% Create a deep neural network to be used as approximation model 
% within the critic. 
% The network for this critic must take only the observation as input and 
% return a vector of values for each action. Therefore, it must have an 
% input layer with as many elements as the dimension of the observation 
% space and an output layer having as many elements as the number of
% possible discrete actions. Each output element represents the expected
% cumulative long-term reward following from the observation given as 
% input, when the corresponding action is taken.

dnn = [
    featureInputLayer(prod(obsInfo.Dimension), ...
        'Normalization', 'none', 'Name', 'state')
    fullyConnectedLayer(24, 'Name', 'CriticStateFC1')
    reluLayer('Name', 'CriticRelu1')
    fullyConnectedLayer(24, 'Name', 'CriticStateFC2')
    reluLayer('Name','CriticCommonRelu')
    fullyConnectedLayer(length(actInfo.Elements), ...
        'Name', 'output')];

% Create the critic using rlVectorQValueFunction, the network dnn as well 
% as the observation and action specifications.

critic = rlVectorQValueFunction(dnn,obsInfo,actInfo);

% Define some training options for the critic.

criticOpts = rlOptimizerOptions( ...
    'LearnRate',1e-2,'GradientThreshold',1);

% Specify agent options, and create a DQN agent using the critic.
agentOpts = rlDQNAgentOptions(...
    'UseDoubleDQN',false, ...    
    'TargetUpdateMethod',"periodic", ...
    'TargetUpdateFrequency',4, ...   
    'ExperienceBufferLength',100000, ...
    'DiscountFactor',0.99, ...
    'MiniBatchSize',256, ...
    'CriticOptimizerOptions',criticOpts);
agentOpts.EpsilonGreedyExploration.Epsilon = 0.9;
agent = rlDQNAgent(critic,agentOpts);

%
% Train Agent
trainOpts = rlTrainingOptions(...
    'MaxEpisodes',500, ...
    'MaxStepsPerEpisode',MaxStepsPerEpisode, ...
    'Verbose',false, ...
    'Plots','training-progress',...
    'StopTrainingCriteria','AverageReward',...
    'StopTrainingValue',480); 

% 
doTraining = true;
if doTraining
    % Train the agent.
    trainingStats = train(agent,env,trainOpts);
else
    % Load the pretrained agent for the example.
    load('MATLABCartpoleDQNMulti.mat','agent')
end

%% 
% testing
close all
clc

simOpts = rlSimulationOptions('MaxSteps',500,...
    'NumSimulations',100);
experience = sim(env,agent,simOpts);
   
% simulate network
clc
NumSimulations = 100 ;
for i = 1:NumSimulations
    close
    clc
    % get states of the episode
    s_t = experience(i).Observation.NetworkStates.Data;
    % get final state
    s_t_final = s_t(:,:,end);
    % get src dst
    f_1 = s_t(1:N); %Src node
    f_2 = s_t(N+1:2*N); %Dst node
    src = find(f_1==1)
    dst = find(f_2==1)
    % get path
    f_3 = logical(s_t_final(2*N+1:2*N+E)); % final path
    edges = find(f_3==1);
    g = plot(G,'Layout','force','EdgeLabel',G.Edges.Weight);
    highlight(g,'Edges',edges,'EdgeColor','red','LineWidth',2);
    pause;
end

%%
figure
clc
experience(1).Action.RoutingAction.Time
experience(1).Action.RoutingAction.Data
experience(1).Observation.NetworkStates.Data
experience(1).Reward.Data

subplot(2,1,1); plot(experience(1).Action.RoutingAction.Data)


