"""
Module contenant la suite du sujet et la Résolution approchée par Minimum Spannig Tree
"""
# Les imports

import networkx as nx
import matplotlib.pyplot as plt
from tools import permutations,intersects,uncross
import math
import random
import numpy as np
from force_brute import sample,itineraire_optimal

random.seed(126)

# -- 0.1.2 Résolution approchée par Minimum Spannig Tree -- #


# Réponse à la question Q4


def création_graph_Kn(Tab):
    """
    pos = {0: (40, 20), 1: (20, 30), 2: (40, 30), 3: (30, 10)} 
    X.add_nodes_from(pos.keys())
    for n, p in pos.iteritems():
    X.nodes[n]['pos'] = p
    print(nx.spring_layout(G))
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
            # Ajoute un arc de cout wij entre les sommets i et j.
    return G

def affiche(G):
    plt.axis([0,1000,0,1000])
    #print(pos)
    pos = nx.get_node_attributes(G,'pos')
    nx.draw_networkx_nodes(G, pos, node_size=100)
    nx.draw_networkx_edges(G,pos,width=0.25)
    #nx.draw(G)
    plt.show()


#Tab = sample(16,0,1000,0,1000)
#G = création_graph_Kn(Tab)
#affiche(G)



#print(Tab)
#print('nombre de sommet', len(G))      
#print('nombre arrêts', G.size())

# Réponse à la question Q5


def Kruskal(G):
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


#T1 = Kruskal(G)
#affiche(T1)


#print(list(nx.dfs_preorder_nodes(T1)))

def ordre_visite(T):
    INDICES = list(nx.dfs_preorder_nodes(T))
    INDICES.append(INDICES[0])
    G_chemin = nx.Graph()
    #G_chemin.add_nodes_from(T)
    

    for i in range(len(list(T.nodes))):
        node = list(T.nodes)[i]
        G_chemin.add_node(node)
        G_chemin.nodes[node]['pos'] = T.nodes[node]['pos']


    for i in range(len(INDICES)-1):
        ind,ind_suiv = INDICES[i],INDICES[i+1]
        (x1,y1) = T.nodes[ind]['pos']
        (x2,y2) = T.nodes[ind_suiv]['pos']
        dist = ((x1-x2)**2+(y1-y2)**2)**(1/2)
        #dist = G[ind][ind_suiv]['weight']
        G_chemin.add_edge(ind,ind_suiv,weight = dist)

    return G_chemin

#affiche(T1)
#G_chemin = ordre_visite(T1)
#affiche(G_chemin)



# QUESTION 7

# On a forcément OPT <= L*
# De plus, grâce au principe de l'algorithme
#  et avec l'inégalité triangulaire 
# L* <= 2 * D   (avec D la longueur du chemin)
# Or on a D <= OPT  
# Donc OPT <= L* <= 2*OPT 
   
    
  
    
# Question 8 
# Supposons par l'absurde qu'une solution du problème comporte une auto-intersection
# Prenons les 4 points impliqués dans cette auto-intersection 
# On peut relier les 2 points de chaque partie du graphe de deux autres façons différentes (les parties sont les 2 graphes qu'il reste si l'on supprime les 2 arcs de l'auto-intesection)
# Dans le repère cartésien la somme des longueurs des deux diagonales est inférieure à celle des longueurs des deux autres tracés
# Donc il existe un chemin plus court, ce n'est pas une solution du problème




# Question 9 et 10

def suppr_boucles(G):
    Boucle = uncross(G)
    while Boucle:
        Boucle = uncross(G)
    return G



#suppr_boucles(G_chemin)


# QUESTION 11
def calc_longueur(G):
    S = 0
    for i in range(len(list(G.edges))):
        (ind,ind1) = list(G.edges)[i]
        S += G[ind][ind1]['weight']
    return S

#affiche(G_chemin)

#print(calc_longueur(G_chemin))
"ICI IL FAUT REPONDRE A UNE QUESTION"

# QUESTION 12
"""
# pour n = 9 :
def comparaison_methodes(n):
    Tableau = sample(n,0,1000,0,1000)
    Graphe = création_graph_Kn(Tableau)
    G_final = ordre_visite(Kruskal(Graphe))
    suppr_boucles(G_final)
    Somme_force_brute = itineraire_optimal(Tableau)
    Somme_avec_arbre = calc_longueur(G_final)
    return Somme_force_brute,Somme_avec_arbre
"""

#print(comparaison_methodes(9))

def solution_n_points(n,comparaison = False):
    Tableau = sample(n,0,1000,0,1000)
    Graphe = création_graph_Kn(Tableau)
    G_final = ordre_visite(Kruskal(Graphe))
    suppr_boucles(G_final)
    Somme_avec_arbre = calc_longueur(G_final)
    if comparaison == True:
        if n > 9 : 
            print("l'algorithme force-brute est trop lourd pour n supérieur à 9, la fonction va seulement renvoyer la solution avec l'arbre")
        else :
            Somme_force_brute = itineraire_optimal(Tableau)
            return G_final,Somme_avec_arbre,Somme_force_brute
    return G_final,Somme_avec_arbre

# On a parfois quelques bugs, si on affiche le graphe pour n=60 par exemple, on a deux points reliés à l'extérieur du cycle, celui-ci est coupé

#print(solution_n_points(8,comparaison = False)[1])





















# QUESTION 13

R = nx.read_gml("paris_network.gml")



def affiche_Paris(G):
    xmin = 642500.0
    xmax = 662000.0
    ymin = 6857000.0
    ymax = 6868000.0
    plt.axis([xmin,xmax,ymin,ymax])
    pos = nx.get_node_attributes(G,'pos')
    nx.draw_networkx_nodes(G, pos, node_size=1)
    nx.draw_networkx_edges(G,pos,width=0.25)
    plt.show()
 

#affiche_Paris(R)
V = ['53726' ,'46780' ,'60159' ,'53796' ,'54342' ,'49811' ,'41385' ,'52805' ,'50444' ,'39710' ,'41399' ,'61108' ,'53843' ,'50117' ,'58033' ,'41806']

def Dijktra(R,V,i,j):
    """
    Prend en arg le cycle R, deux noeuds et renvoie le plus court chemin entre ces deux noeuds (liste,distance)

    Ca c'est dans la fonction d'après :
    On va prendre tous les noeuds de R, on calcule le plus court chemin entre ce noeud et chaque autre noeud, on retient la distance qu'on va utiliser pour que ce soit le poids dans la fonction suivante
    Et on garde dans le stockage le chemin parcouru pour pouvoir le réutiliser après
    Ca va être extrement lourd mais bon, on est des pros
    """
    node,target_node = V[i],V[j]
    distance = 0
    # On utilise Dijktra
    Path = nx.shortest_path(R,node,target_node,'weight','dijkstra')
    for i in range(len(Path)-1):
        etape,etape_suiv = Path[i],Path[i+1]
        distance += R[etape][etape_suiv]['weight']
    return distance,Path

    #print(nx.shortest_path(R,'46780','60159','weight','dijkstra'))
    # >>> renvoie ['46780', '39510', '43722', ...]






#print(Dijktra(R,V,0,1))






def creation_graphe_monuments(R,V):
    """
    Renvoie le graphe complet avec seulement les monuments comme noeud, et un dictionnaire qui, à la clé (i,j), associe une liste de points du plus court chemin entre les deux stations
    """
    Dico = {}
    G = nx.Graph()
    for node in V:
        G.add_node(node)
        G.nodes[node]['pos'] = R.nodes[node]['pos']
    # Avec l'algorithme de Dijktra on trouve le chemin le plus court pour aller d'un point à un autre et ainsi créer le graphe complet de ces noeuds, avec comme poids la longueur réelle du chemin parcouru, et le chemin à parcourir entre deux points conservé hors de cette partie de l'algorithme
    for i in range(len(V)):
        for j in range(i+1,len(V)):
            node,next_node = V[i],V[j]
            # C'est ici qu'on va utiliser autre fct
            dist,chemin = Dijktra(R,V,i,j)
            Dico[node,next_node] = chemin
            # dans l'autre sens on veut inverser le chemin, on n'a pas troucé de solution pour le faire directement, on en prendra compte dans la fonction 'solution Paris'
            Dico[next_node,node] = chemin
            G.add_edge(node,next_node,weight = dist)
    return G,Dico

def creation_monum_liste(G_simple,V):
    # on veut créer la liste "monum_list", elle contient les noeuds (monuments) dans le bon "ordre"
    # On l'initialise avec le point de départ (V[0]) ici
    monum_list =  [V[0]]
    # On récupère ce qui sera le dernier arc
    edge_list = list(G_simple.edges)
    for j in range(len(edge_list)):
            (node1,node2) = edge_list[j]
            if node1 == V[0] or node2 == V[0]:
                del(edge_list[j])
                break
    # On rentre les n-2 noeuds suivants
    for i in range(1,len(V)):
        # On va regarder dans les arcs de G_simple pour trouver quels points sont reliés ou non
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

'''
test de la fonction :
R_monuments,Dico = creation_graphe_monuments(R,V)
G_simple = ordre_visite(Kruskal(R_monuments))
suppr_boucles(G_simple)
print(creation_monum_liste(G_simple,V))
'''

#R_monuments,Dico = creation_graphe_monuments(R,V)
#affiche_Paris(R_monuments)

def solution_Paris(R,V):
    """
    Renvoie le graphe final de la solution proposée, et sa longueur
    """
    # Gsimple est le graphe ne contenant que les monuments en noeud
    R_monuments,Dico = creation_graphe_monuments(R,V)
    G_simple = ordre_visite(Kruskal(R_monuments))
    suppr_boucles(G_simple)
    long = calc_longueur(G_simple)

    # On construit la liste monum_list (liste ordonnée)
    monum_list = creation_monum_liste(G_simple,V)
    # On constitue la liste du chemin final
    chemin = []
    for i in range(len(monum_list)-1):
        monum,monum_suiv = monum_list[i],monum_list[i+1]
        sous_chemin = Dico[monum,monum_suiv]
        #print(monum,monum_suiv,sous_chemin)
        if sous_chemin[0] == monum_suiv:
            # il faut inverser la liste si elle est à l'envers (on n'a pas pu le faire dans le dictionnaire directement)
            sous_chemin.reverse()
        # On supprime le dernier élément pour pouvoir coller autres listes
        del(sous_chemin[-1])
        #print(sous_chemin)
        chemin.extend(sous_chemin)
        
    
    # On peut créer le graphe
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
    G_final.add_edge(etape_av_der,etape_der,weight = dist)
    
    return G_final,long


#R_sol,long = solution_Paris(R,V)
#print(long)
#affiche_Paris(R_sol)




def affiche_sol_Paris(G,V):
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

affiche_sol_Paris(R,V)


"""
CE QU'IL RESTE A FAIRE SI ON A LE TEMPS
- REPONDRE AUX QUESTIONS THEORIQUES QU'ON A LAISSE
"""