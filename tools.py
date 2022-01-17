# ============================================================
# Fonctions pour le problem du TSP metrique
# ------------------------------------------------------------
# - permutation : liste les permutations de {0,1, ... N-1}
# - intersects : teste l'intersection de 2 segments
# - uncross : recherche et supprime (s'il existe) un 
# croisement dans un graphe circulaire (fonction a completer)
# ============================================================

import itertools
import networkx as nx
import matplotlib.pyplot as plt



def affiche_a_ne_pas_rendre(G):
    plt.axis([0,1000,0,1000])
    #print(pos)
    pos = nx.get_node_attributes(G,'pos')
    nx.draw_networkx_nodes(G, pos, node_size=100)
    nx.draw_networkx_edges(G,pos,width=0.25)
    #nx.draw(G)
    plt.show()

# ----------------------------------------
# Fonction de calcul des permutations
# ----------------------------------------
# Entree : entier N
# ----------------------------------------
# Sortie : permutations de {0,1, ... N-1}
# ----------------------------------------
def permutations(N):
	
	P = itertools.permutations(range(N))
	
	return list(P)

# ----------------------------------------
# Fonction equation cartesienne
# ----------------------------------------
# Entree : segment
# ----------------------------------------
# Sortie : liste de parametres (a,b,c)
# ----------------------------------------
def cartesienne(segment):
    
    parametres = list();
    
    x1 = segment[0]
    y1 = segment[1]
    x2 = segment[2]
    y2 = segment[3]
    
    u1 = x2-x1
    u2 = y2-y1
    
    b = -u1
    a = u2
    
    c = -(a*x1+b*y1)
    
    parametres.append(a)
    parametres.append(b)
    parametres.append(c)
    
    return parametres


# ----------------------------------------
# Fonction de test d'equation de droite
# ----------------------------------------
# Entrees : paramatres et coords (x,y)
# ----------------------------------------
# Sortie : en particulier 0 si le point 
# appartient a la droite
# ----------------------------------------
def eval(param, x, y):
    
    a = param[0]
    b = param[1]
    c = param[2]
    
    return a*x+b*y+c

# ----------------------------------------
# Fonction booleenne d'intersection
# ----------------------------------------
# Entrees : segment1 et segment2
# Sortie : true s'il y a intersection
# ----------------------------------------
# Note : les contacts par les extremites 
# sont comptes comme des intersections
# ----------------------------------------
def intersects(segment1, segment2):
    
    param_1 = cartesienne(segment1)
    param_2 = cartesienne(segment2)

    a1 = param_1[0]
    b1 = param_1[1]
    c1 = param_1[2]
    
    a2 = param_2[0]
    b2 = param_2[1]
    c2 = param_2[2]

    x11 = segment1[0]
    y11 = segment1[1]
    x12 = segment1[2]
    y12 = segment1[3]
    
    x21 = segment2[0]
    y21 = segment2[1]
    x22 = segment2[2]
    y22 = segment2[3]
    
    
    val11 = eval(param_1,x21,y21)
    val12 = eval(param_1,x22,y22)
    
    val21 = eval(param_2,x11,y11)
    val22 = eval(param_2,x12,y12)
    
    val1 = val11*val12
    val2 = val21*val22
    
    return (val1 <= 0) & (val2 <= 0)
    


def dist_eucl(x1,y1,x2,y2):
    dist = (x1-x2)**2 + (y1-y2)**2
    return dist**(1/2)



# ----------------------------------------
# Fonction pour supprimer un croisement
# dans un graphe circulaire
# ----------------------------------------
# Entrees : graphe circulaire
# ----------------------------------------
# Sortie : true s'il y a eu suppression
# ----------------------------------------
def uncross(CYCLE):

	# ------------------------------------------
	# Recherche d'une intersection
	# ------------------------------------------

    repaired = False
	
	# Parcours des arcs ei 
    for i in range(len(CYCLE.edges)):
        # Recuperation sommets de ei
        n11 = list(CYCLE.edges)[i][0]
        n12 = list(CYCLE.edges)[i][1]
        x11 = CYCLE.nodes[n11]["pos"][0]
        y11 = CYCLE.nodes[n11]["pos"][1]
        x12 = CYCLE.nodes[n12]["pos"][0]
        y12 = CYCLE.nodes[n12]["pos"][1]
        seg1 = [x11, y11, x12, y12]
		
		# Parcours des arcs ej
        for j in range(i+1,len(CYCLE.edges)):
            n21 = list(CYCLE.edges)[j][0]
            n22 = list(CYCLE.edges)[j][1]
            x21 = CYCLE.nodes[n21]["pos"][0]
            y21 = CYCLE.nodes[n21]["pos"][1]
            x22 = CYCLE.nodes[n22]["pos"][0]
            y22 = CYCLE.nodes[n22]["pos"][1]
            seg2 = [x21, y21, x22, y22]
			
			# Test intersection entre ei et ej
            intersection = intersects(seg1,seg2)
			
			# Gestion du cas "intersection par les extremites"
            intersection = intersection & (n11 != n21) & (n11 != n22)
            intersection = intersection & (n12 != n21) & (n12 != n22)
			
            if intersection:
                CYCLE.remove_edge(n11,n12)
                CYCLE.remove_edge(n21,n22)

                # Je calcule tout d'abord les distances des segments qui pourraient remplacer les diagonales
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
        if (repaired):
            break
    return repaired
	