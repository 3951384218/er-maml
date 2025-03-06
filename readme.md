# ER-MAML: Meta-Reinforcement Learning with Evolving Gradient Regularization
**ER-MAML** is an improved version of the **Model-Agnostic Meta-Learning (MAML)** algorithm, designed to enhance **generalization** in meta-reinforcement learning tasks. This repository provides the core implementation of ER-MAML, hyperparameter configurations, and auxiliary code based on the original MAML framework.

---

## 📦 Key hyperparameters in *hyperparameters setting.py*

| \# Inner loop parameters |           |                                             |
| :----------------------- | :-------- | :------------------------------------------ |
| 'inner\_lr'              | 0\.01     |                                             |
| 'max\_path\_length'      | 150       |                                             |
| 'adapt\_steps'           | 3         |                                             |
| 'adapt\_batch\_size'     | 10        |                                             |
| 'ppo\_epochs'            | 3         |                                             |
| 'ppo\_clip\_ratio'       | 0\.2      |                                             |
| \# Outer loop parameters |           |                                             |
| 'meta\_batch\_size'      | 20        |                                             |
| 'outer\_lr'              | 0\.1      |                                             |
| 'backtrack\_factor'      | 0\.5      |                                             |
| 'ls\_max\_steps'         | 15        |                                             |
| 'max\_kl'                | 0\.01     |                                             |
| \# Common parameters     |           |                                             |
| 'activation'             | 'relu'    |                                             |
| 'tau'                    | 0\.95     |                                             |
| 'gamma'                  | 0\.99     |                                             |
| 'fc\_neurons'            | 100       |                                             |
| \# Other parameters      |           |                                             |
| 'algo\_name'             | 'ER-MAML' |                                             |
| 'adapt\_steps'           | 3         | \# Number of steps to adapt to a new task   |
| 'adapt\_batch\_size'     | 20        | \# Number of shots per task                 |
| 'n\_tasks'               | 10        | \# Number of different tasks to evaluate on |