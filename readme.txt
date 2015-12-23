------------------------------------
    Requirement
------------------------------------
This software is based on the following packages/softwares. Please install them before running the code.

* Python 2.7+
* Numpy
* Scipy
* Chainer: https://github.com/pfnet/chainer
* RL-glue core: https://sites.google.com/a/rl-community.org/rl-glue/Home/rl-glue
* RL-glue Python codec: https://sites.google.com/a/rl-community.org/rl-glue/Home/Extensions/python-codec
* Arcade Learning Environment (version ALE 0.4.4): http://www.arcadelearningenvironment.org/
* NVIDIA GPU (This code was tested on Geforce GTX 660 with Ubuntu 14.04 LTS)

Also you may need the binary rom of the ATARI games.
I reccomend you to run examples in RL-glue python codec and ALE before testing DQN.

------------------------------------
    How to run
------------------------------------
To run a DQN, we just follow the standard RL-glue experiment. 
Concretely, we will need to start the following processes.

* rl_glue
* RLGlueAgent (dqn_agent_*.py)
* RLGlueExperiment (experiment_ale.py)
* ale (ALE 0.4.4)
(So, you may need four terminal windows!)

The actual process will be:
(first window: rlglue)
rl_glue
(second window: RLGlueAgent)
python dqn_agent_nature.py
(third window: RLGlueExperiment)
python experiment_ale.py
(fourth window: ALE)
./ale -game_controller rlglue -use_starting_actions true -random_seed time -display_screen true -frame_skip 4 path_to_roms/pong.bin 

In the above example, we are assuming that the binary file of the roms ("Pong" in this case)
are in path_to_roms directory. 

------------------------------------
    Playing other games
------------------------------------
The default setting of the code is for playing "Pong". 
To run with other games, you need to modify a line in "agent_start" function in "dqn_agent" class.

To make DQN play "Breakout", we may set as

(before modification) self.DQN = DQN_class()
( after modification) self.DQN = DQN_class(enable_controller=[0, 1, 3, 4])

"enable_controller" is the list of available actions of the agents. 
The minimum set of the actions required for each game rom are described
in ale_0_4/src/games/supported/name_of_game.cpp,

and you can check the corrensponding integer numbers in the section 8.1 of the technical manual of ALE:

Technical Manual (you have same manual in your ale directory!): https://github.com/mgbellemare/Arcade-Learning-Environment/tree/master/doc/manual

------------------------------------
Modification of the hyper-parameters
------------------------------------

If your machine does not have enough memory to run the full-version DQN, 
try setting "data_size" variable much smaller value like 2*10**4.
This setting may reduce the final performance, but still works well at least in "Pong" domain.

------------------------------------
Copyright (c) 2015 Naoto Yoshida All Right Reserved.
