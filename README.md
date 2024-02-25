# recagent-music
This repository is inspired by Lei Wang et al. (2023) ["When Large Language Model based Agent Meets User Behavior Analysis: A Novel User Simulation Paradigm](https://arxiv.org/abs/2306.02552) with accompanying [repository](https://github.com/RUC-GSAI/YuLan-Rec). 

We aim to simulate the interaction between a user and a music-based recommender system (e.g. Spotify or Deezer). We do so by extracting metadata about a user (e.g. their demographic, favourite artists, etc.), sampling a sequence of their listening history, and then prompting ChatGPT to predict whether the user will play or skip the next song (the "recommended song") in the sequence. In this way, ChatGPT acts as the "mind" of a user. 

For full details of setup and examples of running the code, please see RecAgent-Music_Tutorial.ipynb. 