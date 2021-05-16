# Motion-Planning


## Table of contents
* [Introduction](#Introduction)
* [Prerequisites](#Prerequisites)
* [Example](#Example)
* [Sources](#Sources)

## Introduction
This is a Python implementation of Motion Planing which includes two parts
* Own implement of Search-based Motion Planning Algorithm:
    1. Weighted A* with epsilon consistence
    2. One-to-many discrete collision/distance checking using 
       [python-fcl](https://github.com/BerkeleyAutomation/python-fcl)
       1. Continues collision checking required to perform in one-to-one manner 
       2. The time complexity will be O(n^2)
       3. Since A_star already discretize Configuration Space and obstacle's shape is simple,
    a one-to-many collision checking is used instead with time complexity O(n)


* Sampling-based Motion Planning Algorithm:
    1. RRT* using the state-of-the-art sampling-based motion planning library [OMPL](https://ompl.kavrakilab.org/)
    2. Combination of Ray vs AABB Collision Checking and FCL's Continuous Collision Checking
        1. Define robot trajectory as a line or piecewise-linear object.
        2. Define agent as a sphere with small radius.
## Prerequisites
Before you continue, ensure you have met the following requirements:

* You have installed the version of`<python >= 3.6>` 
* You can install Python libraries using `pip install -r requirements.txt`.

Note that you need to install [OMPL](https://ompl.kavrakilab.org/) and redirect it's path. 
Please follow the instruction in [download](https://ompl.kavrakilab.org/download.html) for help.

## 3rd Party Package explanation
```
pip install tqdm       # progress bar (which might not be very useful in this case)
pip install icecream   # use for print output
pip install numba      # use for speed up the calculation of cost and heuristic function
pip install pyrr       # use for line segment collision checking 
pip install python-pcl # use for one-to-many collision checking
                       # and continous collison checking
```

## Example
Enjoy A* Algorithm by running following script:

```
cd motion_planning

'''
Four type of heuristic: [1,2,3,4] wrt ['manhattan', 'euclidean', 'diagonal', 'octile']
'''
python3 enjoy_A_star.py  # The deafult heuristic is 2 (euclidean distance)


# with --showall plots 4 type of heuristic in each env(28 plots in total)                         
python3 enjoy_A_star.py --showall

# You own test can be run with
python3 run_Astar.py  
```


Enjoy RRT* Algorithm by running following script:
```
cd motion_planning
python3 enjoy_RRT_star.py  

# You own test can be run with
python3 run_OMPL_RRT.py
```

## Sources
[ECE 276B Motion Planning](motion_planning/ECE276B_PR2.pdf)

[OMPL](https://ompl.kavrakilab.org/)

[python-fcl](https://github.com/BerkeleyAutomation/python-fcl)