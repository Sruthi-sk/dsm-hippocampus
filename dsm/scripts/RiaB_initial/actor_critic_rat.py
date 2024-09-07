import numpy as np
import ratinabox
from ratinabox.contribs.NeuralNetworkNeurons import NeuralNetworkNeurons #for the Actor and Critic

T_TIMEOUT = 15 # Time out
TAU = 5 # Discount time horizon
TAU_E = 5 # Eligibility trace time horizon
ETA = 0.01 # Learning rate 
L2 = 0.000 # L2 regularization

class BaseActorCritic(NeuralNetworkNeurons):
    """Since actors and critics have similar learning rules and trace updates we share some logic here. This is a RatInABox Neurons subclass so you can query rate maps with `.plot_rate_map()` and see history in `.history`"""
    
    default_params = {
        "tau": TAU, #The time horizon of the value function 
        "tau_z": TAU_E, #The time horizon of the eligibility trace
        "input_layers": [],  # a list of input layers, each must be a ratinabox.Neurons class
        "NeuralNetworkModule": None, #Any torch nn.Sequential or nn.Module with a .forward() method
        # "optimizer": lambda params: torch.optim.AdamW(params),
        "optimizer": lambda params: torch.optim.SGD(params, lr=ETA,  maximize=True, weight_decay=L2), #The optimizer to use (in practise I've tried Adam but it aint great). Also, remember this must maximize not minimize. 
        }

    def __init__(self, Agent, params={}):
        """Initialise the actor or critic neurons. Provide the Agent and any parameters which must include the pytorch nn.Module to use as the neural network and a list of Neurons which act as the input layers."""
        self.params = __class__.default_params.copy()
        self.params.update(params)
        super().__init__(Agent, self.params)
        self.initialise_traces()
        self.firingrate = self.get_state(save_torch=True) 
        self.firingrate_last = self.firingrate
        if self.params["optimizer"] is not None:
            self.optimizer = self.params["optimizer"](self.NeuralNetworkModule.parameters())
        return  
    
    def initialise_traces(self):
        """We maintain a trace of the gradients for all parameters in the network. This function initialises the traces to zero."""
        self.traces = []
        for (i,param) in enumerate(self.NeuralNetworkModule.parameters()):
            shape = param.detach().numpy().shape #one trace in total 
            self.traces.append(np.zeros(shape))
        return   

    def _train_step(self, L, td_error):
        """Implements a full training step: calculates the gradients, updates the traces then steps the optimizer"""
        self._calculate_gradients(L = L)
        self._update_traces(td_error = td_error)
        self.optimizer.step()
        return
    
    def _calculate_gradients(self, L):
        """Calculate the gradients of L with respect to the weights. This is generic, for the critic L = V(S) (the value of the state) and for the actor L = log_prob(A | S) (the log probability of the action just taken)
    
        Args:   
            L (torch.Tensor): What to take the gradient of (must be differentiable)"""
        self.NeuralNetworkModule.zero_grad()
        L.backward(retain_graph=True)
        return

    def _update_traces(self, td_error):
        """Update the gradient traces for each output. These eligibility traces are just the gradients smoothed with an exponential kernel of timescale tau_z. We also then loop back through and """
        for (j,param) in enumerate(self.NeuralNetworkModule.parameters()):
            #trace update : z(t+dt) = (1-dt/tau_z) * z(t) + dt/tau_z * x(t) where z is the trace and x is whats being traced
            dt = self.Agent.dt; tau_z = self.tau_z
            x = param.grad.detach().numpy() #the update
            e = self.traces[j] #the trace for this output
            self.traces[j] = (1-dt/tau_z) * e + (dt/tau_z) * x
            #then set the grad to be the trace times the td error so the optimizer can access it
            param.grad = torch.tensor(self.traces[j] * td_error * self.Agent.dt, dtype=torch.float) # dt makes this update timestep invariant


# The actor and critic only different slightly in their .update() functions so we can inherit from the same base class

class Critic(BaseActorCritic):    
    default_params = {} # see BaseActorCritic for the default params
    firingrate = None
    def update(self, reward, train=True):
        """Accepts the reward just observed, calcuates the TD error, then updates the weights based on the gradient of its firing rate and the TD error. Finally, it updates the firing rate to reflect to new position of the Agent."""
        self._update_td_error(reward) 
        if train: super()._train_step(L = self.firingrate_torch, td_error = self.td_error)#does learning on the weights
        self.firingrate_last = self.firingrate
        super().update()  # FeedForwardLayer builtin function. 
        return

    def _update_td_error(self, reward):
        """Update the temporal difference error using the current firing rate, temporal derivative of the firing rate and the reward."""
        self.dfiringrate_dt = (self.firingrate - self.firingrate_last) / self.Agent.dt
        if reward is None: reward = 0
        if self.dfiringrate_dt is None: self.dfiringrate_dt = 0
        if self.firingrate is None: self.firingrate = 0
        self.td_error = (reward + self.dfiringrate_dt - self.firingrate / self.tau).item()  # this is the continuous analog of the TD error (a scalar) 
        return

class Actor(BaseActorCritic):
    default_params = {} # see BaseActorCritic for the default params
    def update(self, log_prob=None, td_error=None, train=True):
        """Accepts the (differentiable) log probability of the action just taken and the critic's latest TD error them updates the weights based on the gradient of the log probability and the TD error. Finally, it updates the firing rate to reflect to new position of the Agent."""
        if train: super()._train_step(L = log_prob, td_error = td_error)#does learning on the weights #does learning on the weights
        super().update()
        return
    
# for the Actor and Critic neural networks
import torch 
import torch.nn as nn

#=================================== a basic MLP for the CRITIC =======================================
class MultiLayerPerceptron(nn.Module):
    """A generic ReLU neural network class, default used for the core function in NeuralNetworkNeurons. 
    Specify input size, output size and hidden layer sizes (a list). Biases are used by default.

    Args:
        n_in (int, optional): The number of input neurons. Defaults to 20.
        n_out (int, optional): The number of output neurons. Defaults to 1.
        n_hidden (list, optional): A list of integers specifying the number of neurons in each hidden layer. Defaults to [20,20]."""

    def __init__(self, n_in=20, n_out=1, n_hidden=[20,20]):
        nn.Module.__init__(self)
        n = [n_in] + n_hidden + [n_out]
        layers = nn.ModuleList()
        for i in range(len(n)-1):
            layers.append(nn.Linear(n[i],n[i+1]))
            if i < len(n)-2: layers.append(nn.ReLU()) #add a ReLU after each hidden layer (but not the last)
        self.net = nn.Sequential(*layers)

    def forward(self, X):
        """Forward pass, X must be a torch tensor. Returns an (attached) torch tensor through which you can take gradients. """
        return self.net(X)
    

#=================================== a little more wrapping for the ACTOR ===========================================
class VxVyGaussianMLP(MultiLayerPerceptron):
    """In this instance, the output of the actor is a 2 dimensional vector representing the mean of v_x and 
    the mean of v_y (each will then be sampled from a gaussian with the same variance)."""
    def __init__(self,n_in,
                 n_hidden = [50,], 
                 max_speed=0.5,):
        self.n = 2
        self.max_speed = max_speed
        super().__init__(n_in = n_in, n_hidden=n_hidden, n_out=self.n)

    def forward(self, X): # tanh activation to bound the output between -max_speed and max_speed
        return self.max_speed*torch.tanh(super().forward(X))

    def sample_action(self, firingrate : torch.tensor):
        """Samples the 2D Gaussian distribution and returns a 2D action vector corresponding to the sampled velocity and the log prob of that action."""
        std = torch.sqrt(torch.sum(firingrate.detach()**2))/2 #the standard deviation is half the magnitude of the mean
        std = 0.1
        vx_dist = torch.distributions.Normal(firingrate[:,0],scale=std)
        vy_dist = torch.distributions.Normal(firingrate[:,1],scale=std)
        vx = vx_dist.sample()
        vy = vy_dist.sample()
        action = np.array([vx.item(), vy.item()])
        log_prob = vx_dist.log_prob(torch.tensor([vx])) + vy_dist.log_prob(torch.tensor([vy]))
        return action, log_prob


#you can also try a discrete action space if you like (it works well)
class NESWCategoricalMLP(MultiLayerPerceptron):
    """In this instance, the output of the network is a 4 dimensional vector representing the probability (hence softmax) of moving in each direction, NESW."""
    def __init__(self,n_in, 
                 n_hidden = [50,],
                 speed = 0.2):
        self.n = 4
        self.speed = speed
        super().__init__(n_in = n_in, n_hidden=n_hidden, n_out=self.n)
    
    def forward(self, X): #extra softmax layer for probabilities
        return torch.softmax(super().forward(X),dim=1) 
    
    def sample_action(self, firingrate : torch.tensor):
        """Samples the categorical distribution and returns a 2D action vector corresponding to the selected action (NESW) and the log prob of that action."""
        dist = torch.distributions.Categorical(firingrate)
        firingrate = firingrate.detach()
        choice = dist.sample()
        if   choice.item() == 0: action = self.speed*np.array([0,1])
        elif choice.item() == 1: action = self.speed*np.array([1,0])
        elif choice.item() == 2: action = self.speed*np.array([0,-1])
        elif choice.item() == 3: action = self.speed*np.array([-1,0])
        log_prob = dist.log_prob(torch.tensor([choice]))
        return action, log_prob


def ego_to_allo(v_ego, head_direction):
    """Converts an egocentric velocity vector to an allocentric one by by rotating it by the bearing of the agents current head direction"""
    bearing = ratinabox.utils.get_bearing(head_direction)
    v_allo = ratinabox.utils.rotate(v_ego, -bearing) #bearing measured clockwise from north, so we rotate anticlockwise by -bearing
    return v_allo


def run_episode(env, 
                ag, 
                actor, 
                critic,
                state_cells = [],
                egocentric_actions = False,):
    """Run an episode of the agent in the environment.
    Returns 1 if the episode timed out, 0 otherwise.
    """
    # reset the actor and the critic
    critic.initialise_traces()
    actor.initialise_traces()

    while True:
        # SAMPLE ACTION AND ITS LOG PROB
        action, log_prob = actor.NeuralNetworkModule.sample_action(actor.firingrate_torch)
        if egocentric_actions: 
            action = ego_to_allo(action, ag.head_direction) #convert action in in [V_leftright, V_forwardbackward] (ego) to [V_x, V_y] (allo)
        
        # STEP THE ENVIRONMENT AND OBSERVE THE REWARD
        _, reward, terminate_episode, _ , _ =  env.step(action=action,
                                                         drift_to_random_strength_ratio=1,)
        # _, reward, terminate_episode, _ , _ =  env.step1(action=action,
        #                                                  drift_to_random_strength_ratio=1,)
        # wall_penalty = WALL_PENALTY * (ag.distance_to_closest_wall < 0.1)

        # UPDATE THE STATE CELLS
        for cell in state_cells:
            cell.update()

        # UPDATE THE CRITIC AND ACTOR (INCLUDING LEARNING)
        critic.update(reward=reward)
        actor.update(log_prob=log_prob, td_error=critic.td_error)

        # CHECK IF THE EPISODE IS OVER
        if env.t - env.episodes['start'][-1] > T_TIMEOUT: 
            env.reset(episode_meta_info="timeout")
            return actor, critic
        elif terminate_episode:
            env.reset(episode_meta_info="completed")
            return actor, critic
        

def eval_episode(env, 
                ag, 
                actor, 
                critic,
                state_cells = [],
                egocentric_actions = False,):
    """Run an episode of the agent in the environment.
    Returns 1 if the episode timed out, 0 otherwise.
    """
    # reset the actor and the critic
    critic.initialise_traces()
    actor.initialise_traces()

    while True:
        # SAMPLE ACTION AND ITS LOG PROB
        action, log_prob = actor.NeuralNetworkModule.sample_action(actor.firingrate_torch)
        if egocentric_actions: 
            action = ego_to_allo(action, ag.head_direction) #convert action in in [V_leftright, V_forwardbackward] (ego) to [V_x, V_y] (allo)
        
        # STEP THE ENVIRONMENT AND OBSERVE THE REWARD
        _, reward, terminate_episode, _ , _ =  env.step(action=action,
                                                         drift_to_random_strength_ratio=1,)
        # _, reward, terminate_episode, _ , _ =  env.step1(action=action,
        #                                                  drift_to_random_strength_ratio=1,)
        # wall_penalty = WALL_PENALTY * (ag.distance_to_closest_wall < 0.1)

        # UPDATE THE STATE CELLS
        for cell in state_cells:
            cell.update()

        # UPDATE THE CRITIC AND ACTOR (INCLUDING LEARNING)
        critic.update(reward=reward)
        actor.update(log_prob=log_prob, td_error=critic.td_error)

        # CHECK IF THE EPISODE IS OVER
        if env.t - env.episodes['start'][-1] > T_TIMEOUT: 
            env.reset(episode_meta_info="timeout")
            return actor, critic
        elif terminate_episode:
            env.reset(episode_meta_info="completed")
            return actor, critic
        