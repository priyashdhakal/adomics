import numpy as np 
import matplotlib.pyplot as plt
import os

plots_path = "plots"
if not os.path.exists(plots_path):
    print("Creating plots directory...")
    os.mkdir(plots_path)
else:
    print("Plots directory already exists. Overwriting...")


class Plotter:
    def __init__(self, study_name):
        self.study_name = study_name
        self.study_path = os.path.join(plots_path, study_name)
        if not os.path.exists(self.study_path):
            print(f"Creating study directory {self.study_path}...")
            os.mkdir(self.study_path)
    
    def plot_history(self, history, metric):
        plt.plot(history.history[metric])
        plt.plot(history.history[f"val_{metric}"])
        plt.title(f"{metric} over epochs")
        plt.ylabel(metric)
        plt.xlabel("epoch")
        plt.legend(["train", "test"], loc="upper left")
        plt.savefig(os.path.join(self.study_path, f"{metric}.png"))
        plt.close() 