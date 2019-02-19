My attempt to turn a "car approaching a red light" into an MDP to solve for the optimal approach trajectory. Still working out some kinks but I think the basic solver mostly works for the case of perfect information where you know exactly when the light will flip.

The state vector I'm using is [position relative to light, velocity, time until flip].

The output is a value function approximation which can be turned into a policy in the 3d state space.

The plan is, once this is working in the perfect knowledge case, to turn it into an actual POMDP, where you can't observe the time until the light flips, but have knowledge that the light flips with a given period, and you can observe that it hasn't flipped yet, which should be pretty easily convertible into a belief state transition model.