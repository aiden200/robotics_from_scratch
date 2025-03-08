# 100 Days of Robotics Learning Challenge: Teaching a Robot to Walk from scratch

## 🚀 Overview

This project is a self-imposed challenge to learn **robotics, reinforcement learning (RL), and NVIDIA Omniverse** to develop a simulated robot that learns to walk and then transfer that knowledge to a real, bittle robot.

This challenge was inspired by [Umar Jamil's](https://www.linkedin.com/in/ujamil/) 100 days of CUDA challenge. Although this isn't CUDA, or focused soley on RL, the mindset of **consistent learning and experimentation** is the key, and I will be tracking my progress with the 100 days challenge discord.

## Prerequisits

### Robotics

Nothing. I’m a computer scientist with no prior robotics experience. ~~I'm going to start from the Introduction to robotics by John J. Craig and build up from there.~~ I am switching to Probabilistic Robotics (Intelligent Robotics and Autonomous Agents series) by Sebastian Thrun. It is more of a Computer Science Robotics textbook.

### Computer Science / RL

I am well versed in Computer Vision and know some basic RL algorithms like Q-Learning and MCTS, but I plan to start with the fundamental papers and build up from there. Thank you [David Abel](https://david-abel.github.io/) for the paper recommendations!

## 📈 Progress Tracking

Check out the [`Daily Progress/`](daily_progress) folder for daily updates. My goal is to maintain **consistent progress**, even on slower days.

As a **full-time researcher and part-time student**, my progress will be steady but may vary until I complete my degree and my current research.

## 🎯 Objectives

- **Make the robot walk** – Develop and train a reinforcement learning policy to teach a simulated robot how to walk.
- **Transfer learning to a real robot** – Adapt the trained policy to a physical robot with real-world constraints. I will be using a [Bittle Robot Dog](https://www.amazon.com/Petoi-Pre-Assembled-Quadruped-Programmer-Developers/dp/B09GB7YNQ1?th=1)
- **Enhance autonomy** – Implement reasoning capabilities to allow the robot to navigate and interact with the world.
- **Optimize learning efficiency** – Explore different RL methods and physics simulations for faster, more robust learning.
- **Integrate multimodal inputs** – Utilize additional sensory inputs (e.g., vision, IMU, force sensors) to improve decision-making.

---

### Project Progress by Day

| Day   | Notes & Summaries                                                                                                                                                                                                                                                                                                                                                                                    |
| ----- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Day 1 | **Intro to Robotics**: Basic robotic definitions.<br>[Full Notes](daily_progress/day1.md)                                                                                                                                                                                                                                                                                                            |
| Day 2 | **Intro to Robotics**: Basic robotic definitions and common notations.<br>[Full Notes](daily_progress/day2.md)                                                                                                                                                                                                                                                                                       |
| Day 3 | **Intro to Robotics**: Compound Transformations.<br>[Full Notes](daily_progress/day3.md)                                                                                                                                                                                                                                                                                                             |
| Day 4 | **Intro to Robotics**: Z-Y-X Euler Angles & Different ways to represent rotation Matrices<br>[Full Notes](daily_progress/day4.md)                                                                                                                                                                                                                                                                    |
| Day5  | **PAPER:** Continuous Control with Deep Reinforcement Learning<br>**Intro to Robotics**: Rotation Matrices Continued, Notation, Computational Constraints<br>[Full Notes](daily_progress/day5.md)                                                                                                                                                                                                    |
| Day6  | **PAPER:** Proximal Policy Opyimization Algorithms - Part 1<br>**Intro to Robotics**: Relating frames to each other pt 1<br>[Full Notes](daily_progress/day6.md)                                                                                                                                                                                                                                     |
| Day7  | **PAPER:** Proximal Policy Opyimization Algorithms - Part 2<br>**Code Implementation**: [PPO, part 1](code/models/ppo.py) -> Actor & Config<br>**Intro to Robotics**: Relating frames to each other pt 2<br>[Full Notes](daily_progress/day7.md)                                                                                                                                                     |
| Day8  | **PAPER:** Human-level control through deep reinforcement learning<br>**Code Implementation**: [PPO, part 2](code/models/ppo.py) -> Critic, Clippled Loss, KL divergence coefficient, Action value <br>[Full Notes](daily_progress/day8.md)                                                                                                                                                          |
| Day9  | **PAPER:** PAPER: Trust Region Policy Optimization (TRPO)<br>**Code Implementation**: [PPO, part 3](code/models/ppo.py) -> Trajectory generation <br>[Full Notes](daily_progress/day9.md)                                                                                                                                                                                                            |
| Day10 | **PAPER:** PAPER: Trust Region Policy Optimization (TRPO) pt 2<br>**Intro to Robotics**: Examples of mapping between Kinematic descriptions <br>[Full Notes](daily_progress/day10.md)                                                                                                                                                                                                                |
| Day11 | **PAPER:** PAPER: Trust Region Policy Optimization (TRPO) pt 3<br>**TEXTBOOK UPDATE**: Switching textbooks to [Probabilistic Robotics (Intelligent Robotics and Autonomous Agents series)](https://books.google.com/books/about/Probabilistic_Robotics.html?id=2Zn6AQAAQBAJ)<br>**Code Implementation**: [PPO, part 4](code/models/ppo.py) -> Loss updates <br>[Full Notes](daily_progress/day11.md) |
| Day12 | **PAPER:** Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor pt 1<br>**Code Implementation**: [PPO, part 5](code/environment/cartPole.py) -> Started Omniverse development<br>[Full Notes](daily_progress/day12.md)                                                                                                                                  |
| Day13 | **Code Implementation**: [PPO, part 5](code/environment/cartPole.py) -> Cartpole simulation up, need to conifgure manager environment<br>[Full Notes](daily_progress/day13.md)                                                                                                                                                                                                                       |
| Day14 | **PAPER:** Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor pt 2<br>**Code Implementation**: [PPO, part 6](code/environment/cartPole.py) -> Started minibatching and supporting multi actors<br>[Full Notes](daily_progress/day14.md)                                                                                                               |
| Day15 | **PAPER:** The Difficulty of Passive Learning in Deep Reinforcement Learning<br>**BOOK:** Probabilistic Robotics -> New Book! Introduction and definitions<br>[Full Notes](daily_progress/day15.md)                                                                                                                                                                                                  |
| Day16 | **PAPER:** The Difficulty of Passive Learning in Deep Reinforcement Learning pt 2<br>**BOOK:** Probabilistic Robotics ->Probability<br>[Full Notes](daily_progress/day16.md)                                                                                                                                                                                                                         |
| Day17 | **PAPER:** The Difficulty of Passive Learning in Deep Reinforcement Learning pt 3<br>**BOOK:** Probabilistic Robotics ->Robot environment interaction definitions<br>[Full Notes](daily_progress/day17.md)                                                                                                                                                                                           |
| Day18 | **PAPER:** The Difficulty of Passive Learning in Deep Reinforcement Learning pt 4<br>**BOOK:** Probabilistic Robotics ->Bayes filtering<br>[Full Notes](daily_progress/day18.md)                                                                                                                                                                                                                     |

## 📂 Project Structure

```
📂 robotics-learning-challenge
│── 📜 README.md        # Project overview and documentation
│── 📂 progress         # Logs and updates on milestones
│   │── 📝 day1.md
│   │── 📝 day2.md
│   │── 📝 day3.md
│   └── ...
│── 📂 code             # Scripts, simulations, and training code
│   │── 🏗️ simulation   # Omniverse-based simulations
│   │── 🤖 real-robot   # Deployment & transfer learning
│   │── 🧠 models       # RL models and training scripts
│── 📂 logs             # Training and debugging logs
│── 📂 docs             # Additional documentation
└── 📂 experiments      # Experimental setups and results
```

## Requirements

Python >= 3.10

## 📚 Resources & Learning Materials

### Papers:

- [x] [Continous Control with Deep Reinforcement Learning](https://arxiv.org/pdf/1509.02971)
- [x] [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [x] [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)
- [x] [ConceptGraphs: Open-Vocabulary 3D Scene Graphs for Perception and Planning](https://concept-graphs.github.io/assets/pdf/2023-ConceptGraphs.pdf)
- [x] [sentdex series](https://www.youtube.com/watch?v=phTnbmXM06g&list=PLQVvvaa0QuDenVbxP4LXYZoGbjfgP-Y5i&index=1)
- [x] [Trust Region Policy Optimization](https://arxiv.org/pdf/1502.05477)
- [x] [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/pdf/1801.01290)
- [x] [Introduction to Robotics by John J. Craig](https://marsuniversity.github.io/ece387/Introduction-to-Robotics-Craig.pdf)
- [x] [The Difficulty of Passive Learning in Deep Reinforcement Learning](https://arxiv.org/pdf/2110.14020)
- [ ] [DEMO^3 Multi-Stage Manipulation with Demonstration-Augmented Reward, Policy, and World Model Learning](https://arxiv.org/pdf/2503.01837)
- [ ] [Deep Reinforcement Learning with Double Q-Learning](https://users.soe.ucsc.edu/~pang/200/f18/papers/2018/12389-55999-1-PB.pdf)
- [ ] [DREAM TO CONTROL: LEARNING BEHAVIORS BY LATENT IMAGINATION](https://arxiv.org/pdf/1912.01603)
- [ ] [Mastering the game of Go with deep neural networks and tree search](https://papoef.wordpress.com/wp-content/uploads/2016/05/nature16961-copy.pdf)
- [ ] [Mastering Diverse Domains through World Models](https://arxiv.org/pdf/2301.04104)
- [ ] [Mastering the game of Go without human knowledge](https://www.nature.com/articles/nature24270)
- [ ] [A Distributional Perspective on Reinforcement Learning](https://proceedings.mlr.press/v70/bellemare17a/bellemare17a.pdf)
- [ ] [Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning](https://arxiv.org/pdf/2109.11978)
- [ ] [Sim2Real Transfer in Robotics](https://arxiv.org/abs/1806.06752)

### Tutorials:

- [ ] [NVIDIA ISSAC SIM tutorial](https://docs.omniverse.nvidia.com/isaacsim/latest/core_api_tutorials/tutorial_core_hello_robot.html)
- [ ] [Probabilistic Robotics (Intelligent Robotics and Autonomous Agents series)](https://books.google.com/books/about/Probabilistic_Robotics.html?id=2Zn6AQAAQBAJ)
- [ ] [Visual and LIDAR based SLAM with ROS using Bittle and Raspberry Pi](https://www.youtube.com/watch?v=uXpQUIF_Jyk&list=PL5efXgSvwk9X8wQuiI_fomlSznZc-jShC)
- [ ] [Berkeley Humanoid Traning Code](https://github.com/HybridRobotics/isaac_berkeley_humanoid)
- [ ] [Transferring Robot Learning Policies From Simulation to Reality](https://learn.nvidia.com/courses/course-detail?course_id=course-v1:DLI+S-OV-28+V1)
- [ ] [Introduction to Robotic Simulations in Isaac Sim (Not available yet)](https://learn.nvidia.com/courses/course-detail?course_id=course-v1:DLI+S-OV-03+V1)
- [ ] [Huggingface RL course](https://huggingface.co/learn/deep-rl-course/)
- [ ] [Robotics 101](https://www.ubicoders.com/courses/robotics101?kcid=olcewrxmvgmunfiyvlwobogxxuximlvj?utm_soure=1)
- [ ] [Robot dog simulation -> NVIDIA tech. blog](https://developer.nvidia.com/blog/closing-the-sim-to-real-gap-training-spot-quadruped-locomotion-with-nvidia-isaac-lab/)

## 🏆 Challenges & Next Steps

- **Fine-tuning RL policies** for stable and energy-efficient walking.
- **Sim2Real transfer** to transfer learned motions to a real world robot.
- **Open-world reasoning** integrating work like [concept graphs](https://github.com/concept-graphs/concept-graphs/tree/ali-dev?tab=readme-ov-file)
- **GTC Insights & Updates** – Implementing new ideas and techniques learned from **NVIDIA GTC in March**.

---

Stay tuned for updates, especially after GTC!
