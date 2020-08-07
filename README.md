# JetsonBigBrother
This project is the subject of an exam for the Università Politecnica delle Marche.
We built a program using SCHED_DEADLINE for a real-time experiment.
The program execute a face detection and identification, using CUDA for hardware acceleration with the goal of running the program on Jetson board.

There are two branches:
* On master branch we choose a monolithical approach to the problem
* On the task_based_code branch we separated all the functions in single tasks and scheduled every task in different deadlines
