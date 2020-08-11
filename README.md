# qlearning_ddqn_cartpole

This project implements Q-learning with the DDQN trick. This problem is solved for a simple environment called the CartPole https://stanford.edu/ jeffjar/cartpole.html. Implementation is in PyTorch. Epsilon-greedy exploration and a neural network for Q-learning have been used.

OpenAI Gym’s version of CartPole, has been used - https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py.

The code plots average returns over 100 parameter updates. The maximum average return is 200. The problem is solved for an average return above 195.
DDQN trick was used to compute Bellman Loss. The targetQ taken was learningQ function 100 train steps ago.
