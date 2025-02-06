# Robotics Learning Challenge: Teaching a Robot to Walk

## 🚀 Overview
This project is a self-imposed challenge to learn **robotics, reinforcement learning (RL), and NVIDIA Omniverse** to develop a simulated robot that learns to walk and then transfer that knowledge to a real, brittle robot.

## 🎯 Objectives
- **Make the robot walk** – Develop and train a reinforcement learning policy to teach a simulated robot how to walk.
- **Transfer learning to a real robot** – Adapt the trained policy to a physical robot with real-world constraints.
- **Enhance autonomy** – Implement reasoning capabilities to allow the robot to navigate and interact with the world.
- **Optimize learning efficiency** – Explore different RL methods and physics simulations for faster, more robust learning.
- **Integrate multimodal inputs** – Utilize additional sensory inputs (e.g., vision, IMU, force sensors) to improve decision-making.

## 📂 Project Structure
```
📂 robotics-learning-challenge
│── 📜 README.md        # Project overview and documentation
│── 📂 progress         # Logs and updates on milestones
│   │── 📝 week1.md     # Learning basics of Omniverse & RL
│   │── 📝 week2.md     # Implementing a simple walking policy
│   │── 📝 week3.md     # Improving stability and balance
│   └── ...           
│── 📂 resources        # Essential reading, courses, and tools
│   │── 📄 papers.md    # Relevant research papers on locomotion
│   │── 📄 tutorials.md # Tutorials on RL, robotics, and Omniverse
│   │── 📄 tools.md     # Software & hardware stack used
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
- [ ] [DeepMind’s RL for Robotics](https://deepmind.com/research/highlighted-r[ ] esearch)
- [ ] [Sim2Real Transfer in Robotics](https://arxiv.org/abs/1806.0675[ ] 2)
- [ ] [NVIDIA’s Issac Lab]([https://developer.nvidia.com/omniverse]([ ] https://developer.nvidia.com/isaac/lab))
- [ ] [Visual and LIDAR based SLAM with ROS using Bittle and Raspberry Pi[ ] ](https://www.youtube.com/watch?v=uXpQUIF_Jyk&list=PL5efXgSvwk9X8wQuiI_fomlSznZc-jShC)
- [ ] [Arduino Machine Learning Tutorial: Introduction to TinyML with [ ] Wio Terminal](https://www.youtube.com/watch?v=iCmlKyAp8eQ&list=PL5efXgSvwk9UCtJ6JKTyWAccSVfTXSlA3)
- [x] [sentdex series](https://www.youtube.com/watch?v=phTnbmXM06g&list[ ] =PLQVvvaa0QuDenVbxP4LXYZoGbjfgP-Y5i&index=1)
- [ ] [NVIDIA Isaac Gym & RL](https://developer.nvidia.com/isaac-[ ] gym)
- [ ] [Berkeley Humanoid Traning Code](https://github.com/[ ] HybridRobotics/isaac_berkeley_humanoid)
- [ ] [Eurekaverse](https://eureka-research.github.io/[ ] eurekaverse/)
- [ ] [Learning to Walk in Minutes Using Massively [ ] Parallel Deep Reinforcement Learning](https://arxiv.org/pdf/2109.11978)
- [ ] [Giving continous values in deep learning](https[ ] ://arxiv.org/pdf/1509.02971)
- [ ] [An Introduction to Robot Learning and Isaac Lab]([ ] https://learn.nvidia.com/courses/course-detail?course_id=course-v1:DLI+S-OV-36+V1)
- [ ] [Transferring Robot Learning Policies From Simulation [ ] to Reality](https://learn.nvidia.com/courses/course-detail?course_id=course-v1:DLI+S-OV-28+V1)
- [ ] [Introduction to Robotic Simulations in Isaac Sim ([ ] Not available yet)](https://learn.nvidia.com/courses/course-detail?course_id=course-v1:DLI+S-OV-03+V1)
- [ ] [Huggingface RL course](https://huggingface.co/learn/deep-rl-course/)



### 🛠️ Tools & Software
- **NVIDIA Omniverse** – High-fidelity simulation and RL training
- **Isaac Lab** – GPU-accelerated RL physics simulation
- **PyBullet / MuJoCo** – Alternative physics engines
- **ROS2** – Communication for real-world deployment
- **Gymnasium (OpenAI Gym)** – RL environment framework

## 📈 Progress Tracking
Check out the [`progress/`](progress) folder for weekly updates.

## 🏆 Challenges & Next Steps
- **Fine-tuning RL policies** for stable and energy-efficient walking.
- **Sim2Real transfer** to transfer learned motions to a real world robot.
- **Open-world reasoning** integrating work like [concept graphs](https://github.com/concept-graphs/concept-graphs/tree/ali-dev?tab=readme-ov-file)
- **GTC Insights & Updates** – Implementing new ideas and techniques learned from **NVIDIA GTC in March**.

---

🚀 Stay tuned for updates, especially after GTC!
