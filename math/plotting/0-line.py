#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

def line():

    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))
    plt.plot(np.arange(0, 11), y, 'r-')   # 'r-' = rouge, ligne continue
    plt.xlim(0, 10)                         # axe x : 0 à 10
    plt.ylim(0, y[-1])                      # axe y : 0 au dernier point (1000)
    plt.gca().set_facecolor('white')        # fond du graphique blanc
    plt.gcf().set_facecolor('white')        # fond de la figure blanc
    plt.show()
    