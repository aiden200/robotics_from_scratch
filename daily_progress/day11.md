### **Day 11**

- **PAPER: Trust Region Policy Optimization (TRPO) Pt 3**
  - Vine Sampling
  - TRPO estimate of the Hessian Matrix (2nd order derivative)
  - Finished TRPO, Rough paper!
- **TEXTBOOK UPDATE**
  - Introduction to robotics by John J. Craig is too ME focused, I want a CS focused textbook
  - I talked to a professor and she said that [Probabilistic Robotics (Intelligent Robotics and Autonomous Agents series)](https://books.google.com/books/about/Probabilistic_Robotics.html?id=2Zn6AQAAQBAJ) might be more appropriate
  - I am switching to this book. The purpose is to train a robot how to walk, not to build one.
- **Paper Implementation: PPO pt 4**
  - Implemented update_policy function that updates the actor and critic based on the KL divergence loss and clipped loss
  - Realized that I'm missing multiple Actor trajectory generation and minibatch updates
  - Code located in [here](../code/models/ppo.py)
  - Things left: Environment configuration, Actor Parallel, minibatch updates

### **Notes**

<div style="display: flex; justify-content: space-between;">
  <img src="../assets/day_11_paper_1.jpg" alt="Paper notes 1" width="45%">
</div>
<br>
