"""
Module de la première partie consacrée à la résoltuion exacte par force brute
"""
# Les imports

import networkx as nx
import matplotlib.pyplot as plt
from tools import permutations,intersects,uncross
import math
import random

random.seed(123)


# Réponse à la question Q1
def sample(N,xmin,xmax,ymin,ymax):
    """
    Ecrire une fonction sample prenant en entrée un entier, ainsi que 4 flottants xmin, xmax, ymin et ymax, et retournant
    un tableau de dimension n × 2, contenant les coordonnées 2D de n points échantillonnés aléatoirement et uniformément
    dans le carré [xmin, xmax] × [ymin, ymax]
    """
    Tableau = []
    for i in range(N):
        x = random.uniform(xmin,xmax)
        y = random.uniform(ymin,ymax)
        Tableau.append((x,y))
    return Tableau

Tab = sample(9,0,10,0,10)
#print(Tab)

# Réponse à la question Q2

# On pose n = 9. Résoudre le problème du voyageur de commerce en comparant les distances de toutes les permutations possibles de l’ordre de visite des n points. Indiquer la longueur de l’itinéraire optimal retenu


def distance_chemin(permut,Tab):
    """
    Renvoie la distance euclidienne du chemin de permut
    """
    dist = 0
    for i in range(len(permut)):
        if i == len(permut)-1:
            ind,ind1 = permut[i],permut[0]
        else :
            ind,ind1 = permut[i],permut[i+1]

        [X,Y] = Tab[ind]
        [X1,Y1] = Tab[ind1]
        dist += math.sqrt((X-X1)**2+(Y-Y1)**2)
    return dist



def itineraire_optimal(Tab):
    """
    Calcul la longueur de l'itineraire optimal 
    len(Tab) < 10 svp
    Balec de connaitre la permutation
    ¯\_(ツ)_/¯ 
    """
    permutation = permutations(len(Tab))
    Distances = []
    for permut in permutation:
        Distances.append(distance_chemin(permut,Tab))
    return min(Distances)

#print(itineraire_optimal(Tab))
          
# Réponse à la question Q3

# Estimation du temps mis par l'algorithme pour résoudre le problème avec n = 15 :
# On peut estimer que le nombre d'opération est égal au nombre de permutations possibles
# Pour n=9 on avait 362880 permutations, pour n=15 on en aurait 1.31.10^12
# L'algorithme mets 2 secondes pour n = 9
# Avec un produit en croix on peut alors estimer que l'algorithme mettrait 83 jours pour renvoyer la distance du chemin le plus cours 
