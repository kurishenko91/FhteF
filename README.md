# FhteF
the code for the FhteF model based on the paper "Fair Treatment Allocation via Tree Ensembles", Kseniia Kurishchenko

Abstract:

Nowadays, it is common to use data from observational studies in treatment allocation problems, where one has to decide which individuals will receive treatment and which not. However, observational data may be discriminating against a group defined by a sensitive attribute such as gender or age. If not carefully trained, the algorithm may provide unfair results, unequally allocating treatment to individuals in the sensitive and non-sensitive groups. In this paper, I propose to measure unfairness as the difference between the average treatment effects in the sensitive group and the non-sensitive group. I introduce a Mathematical Optimization model to have accurate heterogeneous treatment effect predictions and a good level of fairness, which will be the basis for the treatment allocation in forthcoming individuals. I present results on simulated datasets illustrating that my model provides fairer predictions of the treatment effect than the benchmark.

To test the FhteF I use a simulated dataset for both cases with and without fairness analysis.

