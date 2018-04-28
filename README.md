## Meaningful

This code is an implementation of the paper "Emergence of Grounded Compositional Language in Multi-Agent Populations" by Igor Mordatch and Pieter Abbeel - https://arxiv.org/pdf/1703.04908.pdf

This paper is about the emergence of language between agents of reinforcement learning, in a cooperative setting.
The work is currently in progress: a "compositional" language emerges in a simple case - when the agents only need to exchange the goal locations, with only one goal type.
In the general case - several actions with several landmarks - the agents are able to cooperate and exchange informations to achieve their goals while using only few vocabulary tokens, but the language is obviously not compositional. 
The next task is to understand why and correct the code.


Techno:
- Python
- Tensorflow
    
