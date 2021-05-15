# Motion-Planning


## Table of contents
* [Introduction](#Introduction)
* [Prerequisites](#Prerequisites)
* [Usage](#Usage)
* [Sources](#Sources)

## Introduction
This is a Python implementation of Motion Planing which includes two parts
* Own implement of Search-based Motion Planning Algorithm:
    1. weighted A* with epsilon consistence
    2. One-to-many discrete collision/distance checking using 
       [python-fcl](https://github.com/BerkeleyAutomation/python-fcl)


* Sampling-based Motion Planning Algorithm:
    1. RRT* using the state-of-the-art sampling-based motion planning library [OMPL](https://ompl.kavrakilab.org/)
    2. Combination of Ray vs AABB Collision Checking and FCL's Continuous Collision Checking 

## Prerequisites
Before you continue, ensure you have met the following requirements:

* You have installed the version of`<python >= 3.6>` 
* You can install Python libraries using `pip install -r requirements.txt`. 
  
Note that you need to install [OMPL](https://ompl.kavrakilab.org/) and redirect it's path. 
Please follow the instruction in [download](https://ompl.kavrakilab.org/download.html) for help.

## Example
Enjoy A* Algorithm by running following script:

```
cd motion_planning
python3 enjoy_A_star.py  # The deafult heuristic is 2 (euclidean distance)
```

Enjoy RRT* Algorithm by running following script:
```
cd motion_planning
python3 enjoy_RRT_star.py  
```

## Sources
[ECE 276B Motion Planning](motion_planning/ECE276B_PR2.pdf)

[OMPL](https://ompl.kavrakilab.org/)

[python-fcl](https://github.com/BerkeleyAutomation/python-fcl)