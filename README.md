<h1 align="center">
  <a href=>
    <img src="docs/logo_2.png" width="215" /></a><br>
  <b>The AI Scientist meets🤝 The AI Economist</b><br>
  
  <b>Open-Ended Economic Discovery</b><br>
</h1>

## General Idea
The idea involves mixing, combining and matching several open-source projects together. Combining the idea behind the AI Scientist and the AI Economist, the goal is to autonomously search a multi-agent environment (Agent Based model) with the aim of discovering emergent properties. 

This method, with the target set at finding emergent properties, prevents the creep of human biases in the design of the ABM experiment (we do not know beforehand if emergent properties will be discovered) and allows for faster than human iteration over the environment. Which has often been mentioned as a problem in using ABM's. 

The full execution loop is outlined below. Every agentic step goes through multiple iterations for refinement. 
1) Generate idea -> obvious first step, also dependent on the user input prompt
2) Idea novelty -> here it is checked how novel the idea is, looking at previously done research. This is currently done by calling the Semantic API. Instead we will be searching in the embedding space to check for novelty. 
   
3) Generate Parameters -> upon the given idea, decide which reinforcement learning agents (^*see note 1*) are necessary in our environment and what kind of properties and hyperparameters they need to have. 
	a) open source Deep Research (Smolagents Huggingface) -> searches the web for important experimental parameters (e.g Australian tax rates) The The parameters are cross-validated with real world data. 
	
4) Generate agents -> LLM agent creates(codes) the necessary RL agents for our experiment and initializes them with the required parameter information
	a) Agent archive -> creates an archive of the previously generated agents with feedback from their success rate within the ABM environment
	
5) Perform experiments -> performs multiple runs in our ABM (AI economist) environment with our economic agents to check for interesting behaviors/emergent properties. A very recent rewrite of it in Jax (econojax) allows for a larger number of agents and GPU parallelization of the environment and thus the experiment. 
   
6) Review -> reviews the final results and creates the necessary graphs and plots
	a) feeback -> provides feedback to iterations higher up in the execution loop for refinement 
   
7) Write up -> writes up the final report in latex and according to the design principles of a, to be specified, economic journal
	a) Experiment archive -> stores previous succesfull ideas and results for further work (e.g if interesting emergent properties discovered for Australia, try for Japan)

## How is this new? 
Mixes ideas of Machine Learning, Complex Systems, Economics into a single agentic loop. 
The idea of a Scientist supervising a subject(economist) has been explored in a couple of other recent papers (*^see note 3*), yet as far I am aware, hasn't been done in a way to examine emergent economic/policy phenomena. It also combines several different techniques that have only been very recently released such as the Meta Agent, Automatic Search for Life and Econojax paper. Exploration will also involve the best way to define/search for emergent properties through either Novelty search or Genetic algorithms. 
The final framework hopefully brings more relevant real-world data to the original AI Economist, has an automated hyperparameter search over interesting and emergent properties, and potentially writes a real-world relevant computational economic research paper. 

^Notes: 
1)  there are two different sets of agents. The overall framework agents that search the web, search for ideas and novelty, perform the experiments etc. On the other hand we also have the Reinforcement Learning agents which interact in the specific ABM environment (government agents, population agents)
2) Sakana yesterday released a v2 of their AI scientist (no OS code release so far). The loop above still relies on parts & ideas of their v1 system
3)  Benjamin Mannings paper explores this for social human settings 


## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)




## Main Sources ----------------------------------------------------------------

Applicable sources with the ideas that are combined in the full merge:
AI scientist: https://github.com/SakanaAI/AI-Scientist
```
@article{lu2024aiscientist,
  title={The {AI} {S}cientist: Towards Fully Automated Open-Ended Scientific Discovery},
  author={Lu, Chris and Lu, Cong and Lange, Robert Tjarko and Foerster, Jakob and Clune, Jeff and Ha, David},
  journal={arXiv preprint arXiv:2408.06292},
  year={2024}
}
```
AI economist: https://github.com/salesforce/ai-economist
```
@misc{2004.13332,
 Author = {Stephan Zheng, Alexander Trott, Sunil Srinivasa, Nikhil Naik, Melvin Gruesbeck, David C. Parkes, Richard Socher},
 Title = {The AI Economist: Improving Equality and Productivity with AI-Driven Tax Policies},
 Year = {2020},
 Eprint = {arXiv:2004.13332},
}
```

Econojax (Jax version of the AI economist): https://arxiv.org/html/2410.22165v1
```
@article{ponse2024econojax,
  title={EconoJax: A Fast \& Scalable Economic Simulation in Jax},
  author={Ponse, Koen and Plaat, Aske and van Stein, Niki and Moerland, Thomas M},
  journal={arXiv preprint arXiv:2410.22165},
  year={2024}
}
```

ADAS: https://www.shengranhu.com/ADAS/
```
@article{hu2024ADAS,
title={Automated Design of Agentic Systems},
author={Hu, Shengran and Lu, Cong and Clune, Jeff},
journal={arXiv preprint arXiv:2408.08435},
year={2024}
}
```
ASAL: https://pub.sakana.ai/asal/
```
@article{kumar2024asal,
  title = {Automating the Search for Artificial Life with Foundation Models},
  author = {Akarsh Kumar and Chris Lu and Louis Kirsch and Yujin Tang and Kenneth O. Stanley and Phillip Isola and David Ha},
  year = {2024},
  url = {https://asal.sakana.ai/}
}
```
LLM hyperparameter search : https://arxiv.org/abs/2312.04528
open source deep research: https://huggingface.co/blog/open-deep-research