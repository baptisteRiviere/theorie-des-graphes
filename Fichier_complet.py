########################################################################
##  Partiel Théorie des Graphes - Problème du voyageur de commerce -  ##
##                Rivière Baptiste, Roineau Lubin                     ##
########################################################################

# -*- coding: utf-8 -*-


# Les réponses aux questions sont à la fin du document


###############
#   Imports   #
###############

import networkx as nx
import matplotlib.pyplot as plt
import math
import random
from tools import permutations, uncross


###########################################
#   Fonctions crées pour les questions    #
###########################################


def dist_eucl(x1,y1,x2,y2):
    """
    Renvoie la distance euclidienne entre les points (x1,y1) et (x2,y2)
    """
    dist = (x1-x2)**2 + (y1-y2)**2
    return dist**(1/2)

###############################################################################

def sample(N,xmin,xmax,ymin,ymax):
    """
    Prend en entrée un entier, ainsi que 4 flottants xmin, xmax, ymin et ymax, 
    et retourne un tableau de dimension N × 2, contenant les coordonnées 2D de n points échantillonnés aléatoirement et uniformément
    dans le carré [xmin, xmax] × [ymin, ymax]
    
    :param N: entier correspondant à la taille du tableau
    :type N: int
    :param xmin: minimum en x
    :type xmin: float
    :param xmax: maximum en x
    :type xmax: float
    :param ymin: minimum en y
    :type ymin: float
    :param ymax: maximum en y
    :type ymax: float
    :return: tableau de dimension N*2 
    :rtype: list
    """
    Tableau = []
    for i in range(N):
        x = random.uniform(xmin,xmax)
        y = random.uniform(ymin,ymax)
        Tableau.append((x,y))
    return Tableau

###############################################################################

def distance_chemin(permut,Tab):
    """
    Renvoie la distance euclidienne du chemin parcourru avec la permutation précisée
    
    :param permut: list, permutation des noeuds 
    :param Tab: tableau (liste de listes) contenant les positions des noeuds
    :type Tab: list
    :return: distance euclidienne
    :rtype: float
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
    Calcul la longueur de l'itineraire optimal solution du TSP pour le graphe dérivant de Tab 
    On recommande len(Tab) < 10 
    
    :param Tab: tableau contenant les valeurs du carré
    :type Tab: list
    :return: itineraire optimal
    :rtype: float
    """
    permutation = permutations(len(Tab))
    Distances = []
    for permut in permutation:
        Distances.append(distance_chemin(permut,Tab))
    return min(Distances)


###############################################################################
    
def création_graph_Kn(Tab):
    """
    Renvoie le graphe complet Kn avec les noeuds de Tab, dont chaque arête est munie d’un poids égal à sa longueur géométrique
     
    :param Tab: tableau (liste de listes) contenant les positions des noeuds
    :type Tab: list
    :return: Graph Kn
    :rtype: networkx.classes.graph.Graph
    
    """
    n = len(Tab)
    # Initialisation du graphe
    G = nx.Graph()
    
    for i in range(n):
        (x,y) = Tab[i]
        G.add_node(i) # Ajoute un sommet d'indice i 
        G.nodes[i]['pos'] = (x,y) # Position du noeud
        for j in range(0,i):
            [x_j,y_j] = Tab[j]
            dist_eucl = math.sqrt((x - x_j)**2+(y - y_j)**2)
            G.add_edge (i,j, weight = dist_eucl)
            # Ajoute un arc de poids la distance euclidienne entre les noeuds i et j.
    return G

def affiche(G):
    """
    Permet d'afficher le graphe G,
    l'emprise est spécifique au problème
    
    :param G: Graph
    :type G: networkx.classes.graph.Graph
    """
    plt.axis([0,1000,0,1000])
    pos = nx.get_node_attributes(G,'pos')
    nx.draw_networkx_nodes(G, pos, node_size=100)
    nx.draw_networkx_edges(G,pos,width=0.25)
    plt.show()

###############################################################################

def Kruskal(G):
    """
    Recherche de l'arbre recouvrant de poids minimum en utilisant l'algorithme de Krustal
    
    :param G: Graph
    :type G: networkx.classes.graph.Graph
    :return: Graph d'arbre de poids minimum
    :rtype: networkx.classes.graph.Graph
    """
    # On utilise la fonction proposée par networkx
    T_generator = nx.minimum_spanning_edges(G,algorithm='kruskal',weight='weight',keys=True,data=True,ignore_nan=False)
    T_list = list(T_generator)
    T = nx.Graph()

    for i in range(len(list(G.nodes))):
        node = list(G.nodes)[i]
        T.add_node(node)
        T.nodes[node]['pos'] = G.nodes[node]['pos']

    for i in range(len(T_list)):
        P1,P2 = T_list[i][0],T_list[i][1]
        poids = T_list[i][2]['weight']
        T.add_edge(P1,P2,weight = poids)
    return T


def ordre_visite(T):
    """
    Retourne le graphe obtenu grâce à l'ordre de visite obtenu à partir d’un parcours en profondeur de l’arbre T, tout en partant d'un sommet arbitraire
    
    :param T: Graphe d'un arbre recouvrant 
    :type T: networkx.classes.graph.Graph
    :return: Graph ordre de visite possible
    :rtype: networkx.classes.graph.Graph
    """
    # On initialise la liste INDICES (indices du graphe T, avec le premier noeud rajouté à la fin pour compléter la boucle)
    INDICES = list(nx.dfs_preorder_nodes(T))
    INDICES.append(INDICES[0])

    # On initialise le graphe proposant un chemin, il contient les mêmes points que T
    G_chemin = nx.Graph()  
    for i in range(len(list(T.nodes))):
        node = list(T.nodes)[i]
        G_chemin.add_node(node)
        G_chemin.nodes[node]['pos'] = T.nodes[node]['pos']

    # on parcourt tous les points et on ajoute les arcs correspondants, avec leur poids (distance euclidienne)
    for i in range(len(INDICES)-1):
        ind,ind_suiv = INDICES[i],INDICES[i+1]
        (x1,y1) = T.nodes[ind]['pos']
        (x2,y2) = T.nodes[ind_suiv]['pos']
        dist = ((x1-x2)**2+(y1-y2)**2)**(1/2)
        G_chemin.add_edge(ind,ind_suiv,weight = dist)

    return G_chemin

###############################################################################

def suppr_boucles(G):
    """
    Supprime toutes les intersections du graphe cycle PATH
    
    :param G: Graph cycle Path
    :type G: networkx.classes.graph.Graph
    :return: Graph sans intersections
    :rtype: networkx.classes.graph.Graph
    """
    Boucle = uncross(G)
    while Boucle:
        Boucle = uncross(G)
    return G

###############################################################################

def calc_longueur(G):
    """
    Renvoie la somme des longueurs des arcs d'un graphe
    
    :param G: Graph 
    :type G: networkx.classes.graph.Graph
    :return: longueur
    :rtype: float
    """
    S = 0
    for i in range(len(list(G.edges))):
        (ind,ind1) = list(G.edges)[i]
        S += G[ind][ind1]['weight']
    return S

###############################################################################

def solution_n_points(n,both = False):
    """
    Renvoie les solutions du TSP à un graphe de n points
    Cette fonction présente deux modes de fonctionnement :

    Dans le mode both (both == True) on renvoie le cycle obtenu avec la méthode "spanning tree", la longueur de ce cycle et la longueur optimale obtenue par la 'force brute' (pour n<9 seulement)

    Sans le mode both (both == False) et pour n quelconque, on ne renvoie que les deux premiers termes 

    :n: int, nombre de noeuds du graphe dont on veut la solution (on recommande n<9 si compararaison == True)
    :both: bool, permet de choisir le mode présenter ci-dessus
    """
    # On utilise la plupart des fonctions définies précédentes
    Tableau = sample(n,0,1000,0,1000)
    Graphe = création_graph_Kn(Tableau)
    G_final = ordre_visite(Kruskal(Graphe))
    suppr_boucles(G_final)
    Somme_avec_arbre = calc_longueur(G_final)
    if both == True:
        if n > 9 : 
            print("l'algorithme force-brute est trop lourd pour n supérieur à 9, la fonction va seulement renvoyer la solution avec l'arbre")
        else :
            Somme_force_brute = itineraire_optimal(Tableau)
            return G_final,Somme_avec_arbre,Somme_force_brute
    return G_final,Somme_avec_arbre

###############################################################################

def affiche_Paris(G):
    """
    Permet d'afficher les graphes propres à l'exemple de Paris, l'emprise est ainsi cadrée pour le graphe
    
    :param G: Graph 
    :type G: networkx.classes.graph.Graph
    """
    xmin = 642500.0
    xmax = 662000.0
    ymin = 6857000.0
    ymax = 6868000.0
    plt.axis([xmin,xmax,ymin,ymax])
    pos = nx.get_node_attributes(G,'pos')
    nx.draw_networkx_nodes(G, pos, node_size=1)
    nx.draw_networkx_edges(G,pos,width=0.25)
    plt.show()
 

###############################################################################

def Dijktra(R,V,i,j):
    """
    Prend en argument le graphe R, deux de ses noeuds pris dans la liste V, renvoie le plus court chemin entre ces deux noeuds grâce à l'algorithme de Dijktra

    :R: Graph, graphe étudié
    :V: list, liste de certains noeuds de 
    :i: et :j:, int, deux indices de la liste V
    :return: (float,list), le premier terme est la distance entre les deux noeuds, le deuxième est la liste des arcs qui permettent d'avoir le chemin avec la distance la plus courte 
    """
    node,target_node = V[i],V[j]
    distance = 0
    # On utilise Dijktra
    Path = nx.shortest_path(R,node,target_node,'weight')
    for i in range(len(Path)-1):
        etape,etape_suiv = Path[i],Path[i+1]
        distance += R[etape][etape_suiv]['weight']
    return distance,Path


###############################################################################

def creation_graphe_monuments(R,V):
    """
    Renvoie un graphe complet avec les points de V comme noeuds, le poids de chaque arc étant le plus court chemin entre ces points
    et un dictionnaire qui à la clé (i,j) associe la liste de points permettant d'avoir le plus court chemin entre les deux noeuds i et j

    Principe :
    Avec l'algorithme de Dijktra on trouve le chemin le plus court pour aller d'un point à un autre et ainsi créer le graphe complet de ces noeuds, avec comme poids la longueur réelle du chemin parcouru, et le chemin à parcourir entre deux points conservé hors de cette partie de l'algorithme
    
    :param R: graphe des noeuds de Paris
    :type R: networkx.classes.graph.Graph
    :param V: noeuds à visiter
    :type V: list
    :return: (Graph,Dict), Graphe complet avec les noeuds de V, dictionnaire recensant les "chemins" entre ces noeuds
    :rtype: networkx.classes.graph.Graph
    """
    Dico = {}
    G = nx.Graph()
    for node in V:
        G.add_node(node)
        G.nodes[node]['pos'] = R.nodes[node]['pos']
    for i in range(len(V)):
        for j in range(i+1,len(V)):
            node,next_node = V[i],V[j]
            dist,chemin = Dijktra(R,V,i,j)
            Dico[node,next_node] = chemin
            # dans l'autre sens on veut inverser le chemin, on n'a pas trouvé de solution pour le faire directement dans le dictionnaire, on le prendra en compte dans la fonction 'solution Paris'
            Dico[next_node,node] = chemin
            G.add_edge(node,next_node,weight = dist)
    return G,Dico


###############################################################################


def creation_monum_liste(G_simple,V):
    """
    Renvoie la liste "monum_list", celle-ci contient les noeuds de V (monuments) dans le bon "ordre" de parcourt
    On l'initialise avec le point de départ 
    C'est à dire V[0] ici

    :G_simple: graph, il s'agit du cycle dont les noeuds sont ceux de V, reliés avec l'algorithme proposant la solution du TSP
    :param V: noeuds à visiter
    :type V: list
    :return: list, liste des monuments à visiter dans le 'bon ordre', celui-ci étant trouvé avec le principe 'spanning tree'
    """
    monum_list =  [V[0]]
    # On supprime l'un des arcs contenant le premier noeud de la liste edge_list (utile pour la suite de l'algorithme)
    edge_list = list(G_simple.edges)
    for j in range(len(edge_list)):
            (node1,node2) = edge_list[j]
            if node1 == V[0] or node2 == V[0]:
                del(edge_list[j])
                break
    # On rentre les n-2 noeuds suivants dans monum_list
    for i in range(1,len(V)):
        # On va regarder dans les arcs de G_simple pour trouver quels noeuds sont reliés ou non
        for j in range(len(edge_list)):
            (node1,node2) = edge_list[j]
            # Si l'un des noeuds est le dernier élément de la liste, c'est que le deuxième noeud est le suivant dans la liste (on supprime l'arc pour ne pas revenir au point précédent)
            if node1 == monum_list[i-1]:
                del(edge_list[j])
                monum_list.append(node2)
                break
            elif node2 == monum_list[i-1]:
                del(edge_list[j])
                monum_list.append(node1)
                break
    # On veut que le point de départ soit également mis à la fin de la liste
    monum_list.append(monum_list[0])
    return monum_list


###############################################################################


def solution_Paris(R,V):
    """
    Renvoie le graphe final de la solution proposée au TSP dans le cas de Paris, et la longueur du trajet
    
    :param R: graphe, graphe des arcs et noeuds dans Paris
    :type R: networkx.classes.graph.Graph
    :param V: noeuds à visiter
    :type V: list
    :return: Graph monuments
    :rtype: networkx.classes.graph.Graph
    """
    # G_simple est le graphe ne contenant que les monuments en noeud
    R_monuments,Dico = creation_graphe_monuments(R,V)
    G_simple = ordre_visite(Kruskal(R_monuments))
    suppr_boucles(G_simple)
    long = calc_longueur(G_simple)

    # On construit la liste monum_list (liste ordonnée des noeuds) avec la fonction définie précédemment
    monum_list = creation_monum_liste(G_simple,V)
    # On constitue la liste du chemin final
    chemin = []
    for i in range(len(monum_list)-1):
        monum,monum_suiv = monum_list[i],monum_list[i+1]
        sous_chemin = Dico[monum,monum_suiv]
        if sous_chemin[0] == monum_suiv:
            # il faut inverser la liste si elle est à l'envers (on n'a pas pu le faire dans le dictionnaire directement)
            sous_chemin.reverse()
        # On supprime le dernier élément pour pouvoir coller autres listes
        del(sous_chemin[-1])
        chemin.extend(sous_chemin)
        
    # On peut créer le graphe final à partir de la liste constituée juste avant
    G_final = nx.Graph()
    for k in range(len(chemin)):
        etape = chemin[k]
        G_final.add_node(etape)
        G_final.nodes[etape]['pos'] = R.nodes[etape]['pos']
        if k != 0:
            etape_inf = chemin[k-1]
            dist = R[etape_inf][etape]['weight']
            G_final.add_edge(etape_inf,etape,weight = dist)
    # Il ne manque plus que l'arc entre le premier et le dernier point
    etape_av_der,etape_der = chemin[-1],chemin[0]
    dist = R[etape_av_der][etape_der]['weight']
    G_final.add_edge(etape_inf,etape,weight = dist)
    
    return G_final,long


def affiche_sol_Paris(G,V):
    """
    Permet d'afficher simultanément le trajet du touriste en rouge et le réseau de noeuds et arcs de Paris en bleu
    
    :param G: objet à afficher en fond
    :type G: networkx.classes.graph.Graph
    :param V: indice des sommets à visiter
    :type V: list
    """
    # On fixe l'emprise
    xmin = 642500.0
    xmax = 662000.0
    ymin = 6857000.0
    ymax = 6868000.0
    plt.axis([xmin,xmax,ymin,ymax])

    # Création du graphe solution
    R_sol,long = solution_Paris(R,V)
    list_edges_in_sol = list(R_sol.edges)

    # Création du graphe de tous les points/arcs non solution
    R_non_sol = nx.Graph()
    G_edges = list(G.edges)
    
    for k in range(len(list(G.nodes))):
        node = list(G.nodes)[k]
        R_non_sol.add_node(node)
        R_non_sol.nodes[node]['pos'] = G.nodes[node]['pos']
    for i in range(len(G_edges)):
        if G_edges[i] not in list_edges_in_sol:
            (n1,n2) = G_edges[i]
            dist = G[n1][n2]['weight']
            R_non_sol.add_edge(n1,n2,weight = dist)
    # Affichage du graphe R_non_sol
    pos1 = nx.get_node_attributes(R_non_sol,'pos')
    nx.draw_networkx_nodes(R_non_sol, pos1, node_size=0.5)
    nx.draw_networkx_edges(R_non_sol,pos1,width=0.25)

    
    # Affichage du graphe solution
    pos2 = nx.get_node_attributes(R_sol,'pos')
    nx.draw_networkx_nodes(R_sol, pos2, node_size=10,node_color = 'yellow')
    nx.draw_networkx_edges(R_sol,pos2,width=5,edge_color  = 'red')

    # affichage du graphe
    plt.show()
  






#########################################
#   Main pour réponses aux questions    #
#########################################


    
# Question 1 
Tab = sample(9,0,10,0,10)
print(Tab)

#_____________________________________________________________________________#   
  
# Question 2
print(itineraire_optimal(Tab))
    
#_____________________________________________________________________________#  
    
# Question 3
# Estimation du temps mis par l'algorithme pour résoudre le problème avec n = 15 :
# On peut estimer que le nombre d'opération est égal au nombre de permutations possibles
# Pour n=9 on avait 362880 permutations, pour n=15 on en aurait 1.31.10^12
# L'algorithme mets 2 secondes pour n = 9
# Avec un calcul de proportionalité on peut alors estimer que l'algorithme mettrait 83 jours pour renvoyer la distance du chemin le plus cours 

#_____________________________________________________________________________#

# Question 4
Tab = sample(16,0,1000,0,1000)
G = création_graph_Kn(Tab)
affiche(G)

#_____________________________________________________________________________#
    
# Question 5
T1 = Kruskal(G)
affiche(T1)

#_____________________________________________________________________________#    

# Question 6 
print(list(nx.dfs_preorder_nodes(T1)))
affiche(T1)
G_chemin = ordre_visite(T1)
affiche(G_chemin)
 
#_____________________________________________________________________________#
    
# Question 7 
# On a forcément OPT <= L*
# De plus, grâce au principe de l'algorithme
# et avec l'inégalité triangulaire 
# L* <= 2 * D   (avec D la longueur du chemin)
# Or on a D <= OPT  
# Donc OPT <= L* <= 2*OPT 
   
#_____________________________________________________________________________#
        
# Question 8 
#Supposons par l'absurde qu'une solution du problème comporte une auto-intersection
# Prenons les 4 points P1,P2,P3,P4 impliqués dans cette auto-intersection et O le point à l'intersection
# les arcs croisés sont (P1,P3) et (P2,P4) 
# Avec le principe d'inégalité triangulaire
# Si x et y sont deux côté opposés (par exemple x = (P1,P2) et y = (P3,P4))
# x <= OP1 + OP2
# y <= OP3 + OP4
# D'où le résultat
# x + y <= OP1 + OP2 + OP3 + OP4
# x + y <= somme de diagonales
# Donc il existe un chemin plus court, ce n'est pas une solution du problème

#_____________________________________________________________________________#
    
# Question 9 
"""
if intersection:
    CYCLE.remove_edge(n11,n12)
    CYCLE.remove_edge(n21,n22)

    # On calcule tout d'abord les distances des segments qui pourraient remplacer les diagonales
    L1a = dist_eucl(x11,y11,x21,y21) 
    L1b = dist_eucl(x22,y22,x12,y12)
    L2a = dist_eucl(x11,y11,x22,y22) 
    L2b = dist_eucl(x12,y12,x21,y21)

    # Cette condition est assez longue, on a deux choix différents, si le choix 1 raccourcit le trajet on va le préférer à l'autre
    # Mais ce choix n'est préférable que si un des arcs proposés n'est pas déjà présent
    # Le "or" à la fin permet cela dans le cas ou le choix 2 raccourcit le trajet
    # On ne doit pas rajouter un noeud déjà présent, cela pourrait engendrer des bugs dans la suite de l'algorithme

    if ((L1a+L1b < L2a+L2b) and ((n11,n21) not in CYCLE.edges) and ((n22,n12) not in CYCLE.edges)) or (((n11,n22) in CYCLE.edges) or ((n12,n21) in CYCLE.edges)):

        CYCLE.add_edge(n11,n21,weight = L1a)
        CYCLE.add_edge(n22,n12,weight = L1b)
        # Cette condition s'applique si l'algorithme sépare le cycle en plusieurs cycles, on change alors de choix de segment
        try :
            if len(nx.find_cycle(CYCLE)) < len(CYCLE.nodes):
            CYCLE.remove_edge(n11,n21)
            CYCLE.remove_edge(n22,n12)
            CYCLE.add_edge(n11, n22,weight = L2a)
            CYCLE.add_edge(n12, n21,weight = L2b)
                except : 
                    # Une erreur peut survenir
                    print('il y a eu une erreur')
            # De même pour l'autre cas
            else:
                CYCLE.add_edge(n11, n22,weight = L2a)
                CYCLE.add_edge(n12, n21,weight = L2b)
                try :
                    if len(nx.find_cycle(CYCLE)) < len(CYCLE.nodes):  
                        CYCLE.remove_edge(n11,n22)
                        CYCLE.remove_edge(n12,n21)
                        CYCLE.add_edge(n11, n21,weight = L1a)
                        CYCLE.add_edge(n22, n12,weight = L1b)
                except :
                    print('il y a eu une erreur')
            repaired = True
            break
"""
#_____________________________________________________________________________#
        
# Question 10 
# Cf fonction ligne 200
suppr_boucles(G_chemin)

#_____________________________________________________________________________#

# Question 11
affiche(G_chemin)
print(calc_longueur(G_chemin))

#_____________________________________________________________________________#

# Question 12
    
#affiche(solution_n_points(100, both=False)[0])
#affiche(solution_n_points(200, both=False)[0])
#affiche(solution_n_points(300, both=False)[0])
    

#_____________________________________________________________________________#

# Question 13


V = ['53726' ,'46780' ,'60159' ,'53796' ,'54342' ,'49811' ,'41385' ,'52805' ,'50444' ,'39710' ,'41399' ,'61108' ,'53843' ,'50117' ,'58033' ,'41806']
R = nx.read_gml("paris_network.gml")
R_monuments,Dico = creation_graphe_monuments(R,V)
R_sol,long = solution_Paris(R,V)
affiche_sol_Paris(R,V)

# La longueur du trajet obtenue est de 32619 m