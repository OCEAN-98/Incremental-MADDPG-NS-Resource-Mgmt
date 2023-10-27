# Inremental_MADDPG_Network_Slicing

**A Brief description of this repository:** "Multi-access edge computing provides local resources in mobile networks as the essential means for meeting the demands of emerging ultra-reliable low-latency communications. At the edge, dynamic computing requests require advanced resource management for adaptive network slicing, including resource allocations, function scaling and load balancing to utilize only the necessary resources in resource-constraint networks. 

Recent solutions are designed for a static number of slices. Therefore, the painful process of optimization is required again with any update on the number of slices. In addition, these solutions intend to maximize instant rewards, neglecting long-term resource scheduling. Unlike these efforts, we propose an algorithmic approach based on multi-agent deep deterministic policy gradient (MADDPG) for optimizing resource management for edge network slicing. Our objective is two-fold: (i) maximizing long-term network slicing benefits in terms of delay and energy consumption, and (ii) adapting to slice number changes. "

**Manuscript** 

As you have seen, there are files titled in the format "XtoY_slices" and "X_slices". These are the application scenarios where "X_slices" denotes the base scenario, training MADDPG from scratch; and on the other hand, "XtoY_slices" represents the incremental learning from scenario *X* to *Y*. The following figure represents the experimental relationships between those cases.
<img src="./Figures/relations.png" width="500" />

If you want to reproduce the code. Just using "./main" in each file. And before applying incremental learning, it is better to finish the base scenarios in case of errors and lack of dependency.
For base scenarios 3, 4 and 5, you can achieve:

<img src="./Figures/3.png" width="400" />
<img src="./Figures/4.png" width="400" />
<img src="./Figures/5.png" width="400" />
<img src="./Figures/6.png" width="400" />

And for 4-5, 4-5-6, 4-6, 5-6 and 4-3, you will see figures like


<img src="./Figures/4-5.png" width="400" />
<img src="./Figures/4-5-6.png" width="400" />
<img src="./Figures/4-6.png" width="400" />
<img src="./Figures/5-6.png" width="400" />
<img src="./Figures/4-3.png" width="400" />

Hope you think the code is useful. 

