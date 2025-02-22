# 100 Days of Robotics Learning Challenge: Teaching a Robot to Walk from scratch

## 🚀 Overview

This project is a self-imposed challenge to learn **robotics, reinforcement learning (RL), and NVIDIA Omniverse** to develop a simulated robot that learns to walk and then transfer that knowledge to a real, bittle robot.

This challenge was inspired by [Umar Jamil's](https://www.linkedin.com/in/ujamil/) 100 days of CUDA challenge. Although this isn't CUDA, or focused soley on RL, the mindset of **consistent learning and experimentation** is the key, and I will be tracking my progress with the 100 days challenge discord.

## Prerequisits

### Robotics

Nothing. I’m a computer scientist with no prior robotics experience. I'm going to start from the Introduction to robotics by John J. Craig and build up from there.

### Computer Science / RL

I am well versed in Computer Vision and know some basic RL algorithms like Q-Learning and MCTS, but I plan to start with the fundamental papers and build up from there. Thank you [David Abel](https://david-abel.github.io/) for the paper recommendations!

## 📈 Progress Tracking

Check out the [`Daily Progress/`](daily_progress) folder for daily updates. The goal is consistent progress, even if some days are slow.

## 🎯 Objectives

- **Make the robot walk** – Develop and train a reinforcement learning policy to teach a simulated robot how to walk.
- **Transfer learning to a real robot** – Adapt the trained policy to a physical robot with real-world constraints. I will be using a [Bittle Robot Dog](https://www.amazon.com/Petoi-Pre-Assembled-Quadruped-Programmer-Developers/dp/B09GB7YNQ1?th=1)
- **Enhance autonomy** – Implement reasoning capabilities to allow the robot to navigate and interact with the world.
- **Optimize learning efficiency** – Explore different RL methods and physics simulations for faster, more robust learning.
- **Integrate multimodal inputs** – Utilize additional sensory inputs (e.g., vision, IMU, force sensors) to improve decision-making.

---

### Project Progress by Day

| Day   | Notes & Summaries                                                                                                                                                                                                                                |
| ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Day 1 | **Intro to Robotics**: Basic robotic definitions.<br>[Full Notes](daily_progress/day1.md)                                                                                                                                                        |
| Day 2 | **Intro to Robotics**: Basic robotic definitions and common notations.<br>[Full Notes](daily_progress/day2.md)                                                                                                                                   |
| Day 3 | **Intro to Robotics**: Compound Transformations.<br>[Full Notes](daily_progress/day3.md)                                                                                                                                                         |
| Day 4 | **Intro to Robotics**: Z-Y-X Euler Angles & Different ways to represent rotation Matrices<br>[Full Notes](daily_progress/day4.md)                                                                                                                |
| Day5  | **PAPER:** Continuous Control with Deep Reinforcement Learning<br>**Intro to Robotics**: Rotation Matrices Continued, Notation, Computational Constraints<br>[Full Notes](daily_progress/day5.md)                                                |
| Day6  | **PAPER:** Proximal Policy Opyimization Algorithms - Part 1<br>**Intro to Robotics**: Relating frames to each other pt 1<br>[Full Notes](daily_progress/day6.md)                                                                                 |
| Day7  | **PAPER:** Proximal Policy Opyimization Algorithms - Part 2<br>**Code Implementation**: [PPO, part 1](code/models/ppo.py) -> Actor & Config<br>**Intro to Robotics**: Relating frames to each other pt 2<br>[Full Notes](daily_progress/day7.md) |
| Day8  | **PAPER:** Human-level control through deep reinforcement learning<br>**Code Implementation**: [PPO, part 2](code/models/ppo.py) -> Critic, Clippled Loss, KL divergence coefficient, Action value <br>[Full Notes](daily_progress/day8.md)       |

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

## 📚 Resources & Learning Materials

### Tutorials I want to go through

- [ ] [DeepMind’s RL for Robotics](https://deepmind.com/research/highlighted-research)
- [ ] [Sim2Real Transfer in Robotics](https://arxiv.org/abs/1806.06752)
- [ ] [NVIDIA’s Issac Lab](https://developer.nvidia.com/omniverse)
- [ ] [Visual and LIDAR based SLAM with ROS using Bittle and Raspberry Pi](https://www.youtube.com/watch?v=uXpQUIF_Jyk&list=PL5efXgSvwk9X8wQuiI_fomlSznZc-jShC)
- [ ] [Arduino Machine Learning Tutorial: Introduction to TinyML with Wio Terminal](https://www.youtube.com/watch?v=iCmlKyAp8eQ&list=PL5efXgSvwk9UCtJ6JKTyWAccSVfTXSlA3)
- [x] [sentdex series](https://www.youtube.com/watch?v=phTnbmXM06g&list=PLQVvvaa0QuDenVbxP4LXYZoGbjfgP-Y5i&index=1)
- [ ] [NVIDIA Isaac Gym & RL](https://developer.nvidia.com/isaac-gym)
- [ ] [Berkeley Humanoid Traning Code](https://github.com/HybridRobotics/isaac_berkeley_humanoid)
- [ ] [Eurekaverse](https://eureka-research.github.io/eurekaverse/)
- [ ] [Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning](https://arxiv.org/pdf/2109.11978)
- [ ] [Giving continous values in deep learning](https://arxiv.org/pdf/1509.02971)
- [ ] [An Introduction to Robot Learning and Isaac Lab](https://learn.nvidia.com/courses/course-detail?course_id=course-v1:DLI+S-OV-36+V1)
- [ ] [Transferring Robot Learning Policies From Simulation to Reality](https://learn.nvidia.com/courses/course-detail?course_id=course-v1:DLI+S-OV-28+V1)
- [ ] [Introduction to Robotic Simulations in Isaac Sim (Not available yet)](https://learn.nvidia.com/courses/course-detail?course_id=course-v1:DLI+S-OV-03+V1)
- [ ] [Huggingface RL course](https://huggingface.co/learn/deep-rl-course/)
- [ ] [Robotics 101](https://www.ubicoders.com/courses/robotics101?kcid=olcewrxmvgmunfiyvlwobogxxuximlvj?utm_soure=1)
- [ ] [Introduction to Robotics by John J. Craig](https://marsuniversity.github.io/ece387/Introduction-to-Robotics-Craig.pdf)
- [x] [CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING](https://arxiv.org/pdf/1509.02971)
- [x] [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [x] [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)

## 🏆 Challenges & Next Steps

- **Fine-tuning RL policies** for stable and energy-efficient walking.
- **Sim2Real transfer** to transfer learned motions to a real world robot.
- **Open-world reasoning** integrating work like [concept graphs](https://github.com/concept-graphs/concept-graphs/tree/ali-dev?tab=readme-ov-file)
- **GTC Insights & Updates** – Implementing new ideas and techniques learned from **NVIDIA GTC in March**.

---

Stay tuned for updates, especially after GTC!
