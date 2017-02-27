# Optimized Frame Selection

Today, the replay memory of RL agents keeps a long buffer of experiences, but this method has big drawback. Anything that falls outside of the buffer will be forgotten by the RL agent, limiting the agent to tasks with short episodes. Frustratingly, while useful transitions are being forgotten, redundant information is stored in the buffer.

With this project we attempt to solve catastrophic forgetting in deep RL while also reducing the memory footprint and increasing the efficiency of the replay memory.

---

This code is based on [DRL](https://github.com/cgel/DRL)
