import copy
import gym
import numpy as np
import torch as th
import torch.nn as nn
import random 
from torch.autograd import Variable
import matplotlib.pyplot as plt

def update_epsilon(epsilon, decay):
    """
    Arguments:
        epsilon {float} -- inverse exploration probability 
        decay {float} -- decay factor

    Returns:
        float -- new exploration probability
    """
    epsilon = epsilon * decay
    return epsilon

def rollout(e, q, eps=0, T=200):
    """
    Generates trajectories

    Arguments:
        e  -- environment
        q  -- Q function

    Keyword Arguments:
        eps {float} -- inverse exploration probability (default: {0})
        T {int} -- trajectory steps (default: {200})

    Returns:
        list -- list of current state, new state, reward, action, done for each step
        float -- reward
    """
    traj = []
    rr = 0
    x = e.reset()
    
    for t in range(T):
        u = q.control(th.from_numpy(x).float().unsqueeze(0),
                      eps=eps)
        u = u.int().numpy().squeeze()

        xp,r,d,info = e.step(u)
        rr += r
        t = [x, xp, r, u, d]
        x = xp
        traj.append(t)
        if d:
            break
    return traj, rr

class q_t(nn.Module):
    """
    Creates a q function using a two layered neural network
    """
    def __init__(s, xdim, udim, hdim=16):
        super().__init__()
        s.xdim, s.udim = xdim, udim
        s.m = nn.Sequential(
                            nn.Linear(xdim, hdim),
                            nn.ReLU(True),
                            nn.Linear(hdim, udim),
                            )
    def forward(s, x):
        return s.m(x)

    def control(s, x, eps=0):
        """
        Generates controls based on epsilon greedy strategy

        Arguments:
            s {q_t} -- Q function
            x {list} -- state

        Keyword Arguments:
            eps {float} -- inverse exploration probability (default: {0})

        Returns:
            [int] -- action
        """
        q = s.m(x)
        u = 0
        if(1-eps <= np.random.random()):
            u=th.tensor(np.random.choice([0,1], p =[0.5, 0.5]))
        else:    
            values, indices = th.max(q, -1)
            u = th.tensor(indices.item())
        return u
        
def loss(q, qtarget, ds, itr):
    """
    Returns loss using double-q trick for minibatch of steps in relay buffer

    Arguments:
        q {q_t} -- Learnt Q function
        qtarget {q_t} -- Target Q function
        ds {list} -- list of trajectory step information
        itr {int} -- training step

    Returns:
        [float] -- average loss
    """
    global e, frames
    
    batch_size = 500# * (int(itr/1000) + 1)
    idx = random.sample(range(0, len(ds)-1), batch_size)
    tr = np.array(ds)[idx] #numpy.ndarray

    x, xp, r, u, d = zip(*tr) 
    x, xp, r, u, d = np.asarray(x), np.asarray(xp), np.asarray(r), np.asarray(u), np.asarray(d)
    d = 1 - d
    
    Q0 = q.forward(Variable(th.from_numpy(x).float()))
    
    mask = th.zeros(batch_size, 2, dtype=th.bool)
    mask[np.arange(batch_size), u] = True
    Q0 = th.masked_select(Q0, mask)
    
    Q1 = qtarget.forward(Variable(th.from_numpy(xp).float()))
    maxQ1, maxIdx = th.max(Q1, -1)
    
    Q2 = q.forward(Variable(th.from_numpy(xp).float()))
    mask1 = th.zeros(batch_size, 2, dtype=th.bool)
    mask1[np.arange(batch_size), maxIdx.numpy()] = True
    Q2 = th.masked_select(Q2, mask1)
    
    Q_target = th.tensor(r) + th.mul(th.Tensor(d), th.mul(0.9, Q2.detach()))
    f = th.sum((Q0 - Q_target).pow(2))

    return f/batch_size

def evaluate(q):
    """
    Evaluates the Learnt Q network

    Arguments:
        q {q_t} -- Learnt Q network

    Returns:
        [float] -- average reward over 100 trajectories
    """
    e = gym.make('CartPole-v0')
    s = 0
    for i in range(100):
        t, rr = rollout(e, q, 0)
        s += rr
    return s/100

def plot(loss_ls, avgtest_return, avgtrain_return):
    plt.figure()
    plt.ylim(0, 3)
    ax1 = plt.subplot(3,1,1)
    plt.plot(range(len(loss_ls)), loss_ls, linewidth = 1.0, color='green', label='train loss')
    
    ax2 = plt.subplot(3,1,2) 
    plt.plot(range(len(avgtest_return)), avgtest_return, linewidth = 1.0, color='blue', label='test return')
    
    ax3 = plt.subplot(3,1,3) 
    plt.plot(range(len(avgtrain_return)), avgtrain_return, linewidth = 1.0, color='black', label='train return')
    ax1.legend()
    ax2.legend()
    ax3.legend()
    plt.legend(loc="upper right")
    plt.savefig('loss.png')

if __name__=='__main__':
    e = gym.make('CartPole-v0')

    xdim, udim =    e.observation_space.shape[0], \
                    e.action_space.n

    # Initializations        
    q = q_t(xdim, udim, 8)
    optim = th.optim.Adam(q.parameters(), lr=0.05, weight_decay=1e-4)
    ds = []
    loss_ls = []
    frames = []
    avgtrain_return, avgtest_return = [], []
    qtarget = [None]*100

    # Hyperparameters
    episodes = 5001
    epsilon = 0.5
    decay=0.95
    
    sumrew = 0
    idx = 0

    # Collect few random trajectories with
    for i in range(1000):
        tr, rr = rollout(e, q, eps=1, T=200)
        ds = ds + tr

    # Training   
    for i in range(episodes):
        print('Episode %d'%i)
        q.train()
        t, rr = rollout(e, q, epsilon)
        epsilon = update_epsilon(epsilon, decay)
        ds = ds + t
        sumrew += rr
        
        q.zero_grad()
        if i < 100:
            f = loss(q, q, ds, i)            
            
        else:
            f = loss(q, qtarget[idx], ds, i)
            
        qtarget[idx] = copy.deepcopy(q)
            
        idx += 1
        if idx == 100:
            idx = 0
            
        f.backward()
        optim.step()
        
        if i%100 == 0 or i == episodes-1:
            print('Logging data to plot %d'%i)
            print('reward ',rr)
            testr = evaluate(q)
            
            avgtest_return.append(testr)
            avgtrain_return.append(sumrew/(100))   
            sumrew = 0
            
        loss_ls.append(f)
        
    plot(loss_ls, avgtest_return, avgtrain_return)