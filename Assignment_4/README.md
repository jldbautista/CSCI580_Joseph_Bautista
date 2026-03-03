# Author: Joseph Lorenzo Bautista
# Class: CSCI 580 - Artificial Intelligence
# Assignment: Assignment #4 - SA + GA


# ---------- Description ---------- #
This assignment solves the same TSP instance using two heuristic optimization methods and implemented:

1) Simulated Annealing (SA) with a 2-opt neighborhood.
2) Genetic Algorithm (GA) on permutation tours (a standard evolutionary approach for TSP).
3) Nearest Neighbor (NN), which is a greedy baseline heuristic.


# ---------- Requirements ---------- #
- Python 3.8 or higher
- matplotlib


# ---------- Installing ---------- #
- Install the required library using pip (Mac & Windows):

- For Mac / Linux:
    - pip3 install matplotlib

- For Windows:
    - pip install matplotlib
    

# ---------- Running ---------- #
- Run the script from the terminal:

- For Mac / Linux:
    - python3 assignment_4.py

- For Windows:
    - python assignment_4.py


# ---------- Output ---------- #
- When the code is ran it will:
1) Print the necessary progress logs for both SA and GA in the terminal.
2) Shows the plot windows and close each window to continue the execution.
3) Save all of the plots and figures as a PNG in the assigned folder.
4) Prints and shows the grade check verifying improvement thresholds.
5) Prints a final brief summary comparing all of the 3 methods.


# ---------- Saved Figures ---------- #

- The following are all the figures and plots created.

    - sa_delta_histogram.png
    - sa_current_vs_best_solution.png
    - sa_temperature_schedule.png
    - ga_pop_avg_fitness.png
    - ga_diversity_over_gen.png
    - sa_vs_ga_comparison.png
    - final_tour_plot.png
