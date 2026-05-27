# Projet PINN + Unreal Engine : aérodynamique temps réel d’un deltaplane 2D

## 1. Idée générale

Le projet consiste à créer une démonstration interactive dans Unreal Engine où un personnage/deltaplane 2D subit des forces aérodynamiques réalistes ou semi-réalistes. L’objectif n’est pas de faire une CFD industrielle, mais de construire une chaîne cohérente :

```text
modèle physique simple
    ↓
réseau de neurones supervisé
    ↓
PINN fluide offline
    ↓
extraction de forces aérodynamiques
    ↓
modèle neural runtime dans Unreal
```

L’idée principale est d’utiliser un **PINN** pour approximer un écoulement autour d’une forme de deltaplane simplifiée, puis de transférer cette information dans un modèle utilisable en temps réel dans Unreal.

Le deltaplane est supposé évoluer en 2D dans un plan vertical. Le joueur contrôle son inclinaison, et le mouvement résulte ensuite des forces physiques : gravité, portance et traînée.

---

## 2. Pourquoi un tel projet ?

Le point de départ est le suivant : dans un moteur de jeu, on ne veut pas résoudre Navier–Stokes en temps réel. C’est trop coûteux, trop instable, et pas adapté à une boucle de gameplay à 60 FPS.

Dans les jeux et simulateurs temps réel, on utilise normalement des modèles réduits :

$$
L = \frac12 \rho U^2 S C_L
$$

$$
D = \frac12 \rho U^2 S C_D
$$

où :
- $L$ est la portance ;
- $D$ est la traînée ;
- $\rho$ est la densité de l’air ;
- $U$ est la vitesse relative air/deltaplane ;
- $S$ est une surface de référence ;
- $C_L$ et $C_D$ sont les coefficients de portance et de traînée.

Dans un premier modèle, ces coefficients dépendent surtout de l’angle d’attaque :

$$
C_L = C_L(\alpha)
$$

$$
C_D = C_D(\alpha)
$$

Le problème est que si on s’arrête là, le projet ne contient pas vraiment de PINN. On aurait simplement un modèle aérodynamique empirique, éventuellement remplacé par un MLP.

Le but est donc d’aller plus loin : utiliser un PINN pour apprendre un **champ fluide local** autour du deltaplane, puis extraire ou distiller les forces qui en découlent.

---

## 3. Invariance utilisée : référentiel du deltaplane

Une difficulté importante est que le deltaplane bouge dans le monde. Si on essayait d’apprendre un champ fluide dans les coordonnées globales d’Unreal, le modèle serait vite inutilisable : il faudrait apprendre toutes les positions et orientations possibles.

On exploite donc une invariance physique : le fluide ne dépend pas de la position absolue de l’objet, mais de l’écoulement relatif autour de lui.

On travaille dans le **référentiel local du deltaplane**.

À chaque tick Unreal :
1. on récupère la vitesse du deltaplane ;
2. on calcule la vitesse relative par rapport à l’air ;
3. on exprime cette vitesse dans le repère local du deltaplane ;
4. on en déduit :
   - la norme $U$ ;
   - l’angle d’attaque $\alpha$.

Le problème local devient donc :

$$
(x,y,U,\alpha) \mapsto (u,v,p)
$$

où :
- $(x,y)$ sont les coordonnées locales autour du deltaplane ;
- $U$ est la vitesse relative ;
- $\alpha$ est l’angle d’attaque ;
- $u,v$ sont les composantes du champ de vitesse du fluide ;
- $p$ est la pression.

Cette formulation permet de faire bouger librement le deltaplane dans Unreal tout en utilisant un modèle local fixe.

---

## 4. Forme géométrique retenue

Pour simplifier, on ne modélise pas un vrai deltaplane 3D.

On part sur une forme 2D fixe, par exemple un triangle légèrement épais à l’arrière et pointu à l’avant :

```text
        pointe avant
             /\
            /  \
 arrière   /____\   arrière épais
```

Cette géométrie est exprimée dans le repère local du deltaplane.

On peut discrétiser sa surface en points :

$$
s_i
$$

avec pour chaque point :
- sa position locale ;
- sa normale extérieure ;
- un poids de longueur $\Delta s_i$.

Cette discrétisation servira plus tard à intégrer les forces locales.

---

## 5. Étape 0 — Modèle analytique dans Unreal, sans réseau

### Objectif

Vérifier que la physique de base marche dans Unreal.

Avant de faire du neural network ou du PINN, il faut que le deltaplane vole avec une loi simple.

### Modèle

On calcule :

$$
L = \frac12 \rho U^2 S C_L(\alpha)
$$

$$
D = \frac12 \rho U^2 S C_D(\alpha)
$$

avec par exemple :

$$
C_L(\alpha)
=
C_{L,\max}
\tanh \left(
\frac{C_{L,\alpha}\alpha}{C_{L,\max}}
\right)
$$

et :

$$
C_D(\alpha)
=
C_{D0}
+
k C_L(\alpha)^2
+
C_{D,\text{flat}}\sin^2(\alpha)
$$

Cette loi est simple mais crédible :
- $C_L$ est à peu près linéaire près de $\alpha=0$ ;
- $C_L$ sature pour éviter les valeurs absurdes ;
- $C_D$ augmente lorsque l’angle d’attaque devient grand ;
- les forces dépendent bien de $U^2$.

### Paramètres de départ

Une première série de valeurs raisonnables :

```text
rho       = 1.225   kg/m^3
mass      = 90      kg
wing_area = 14      m^2

CL_slope  = 4.5
CL_max    = 1.2
CD0       = 0.08
k         = 0.08
CD_flat   = 0.8
```

Avec $U \approx 10\ \text{m/s}$, la pression dynamique multipliée par une surface de 14 m² donne un ordre de grandeur proche du poids d’un humain + deltaplane. C’est donc cohérent pour une première simulation.

### Remarque Unreal

Unreal travaille en centimètres. On peut garder la gravité par défaut :

$$
g = 980 \ \text{cm/s}^2
$$

mais convertir la vitesse en m/s pour les formules aérodynamiques :

$$
U_m = \frac{U_{cm}}{100}
$$

Puis reconvertir l’accélération en cm/s² :

$$
a_{cm/s^2} = 100 \frac{F_N}{m}
$$

### Résultat attendu

Le joueur contrôle l’inclinaison :
- piquer vers le bas augmente la vitesse ;
- cabrer augmente l’angle d’attaque ;
- un angle raisonnable produit de la portance ;
- un angle trop fort produit beaucoup de traînée ;
- à basse vitesse, le deltaplane tombe.

Cette étape valide le gameplay physique.

---

## 6. Étape 1 — MLP global supervisé simple

### Objectif

Tester la pipeline réseau de neurones vers Unreal.

On entraîne un petit MLP :

$$
g_0(U,\alpha) \mapsto (C_L,C_D)
$$

sur les coefficients générés par les formules analytiques de l’étape 0.

Cette étape ne constitue pas encore un vrai PINN. Elle sert surtout à valider :
- l’entraînement en Python ;
- l’export du modèle ;
- l’import dans Unreal ;
- l’inférence runtime ;
- l’utilisation de la sortie du réseau pour appliquer des forces.

### Pourquoi inclure $U$ ?

Dans le modèle analytique simple, $C_L$ et $C_D$ peuvent ne dépendre que de $\alpha$. La dépendance en vitesse est déjà prise en compte par le facteur $U^2$ dans les forces.

Cependant, pour préparer la suite, on peut déjà donner au réseau les deux entrées :

$$
(U,\alpha)
$$

Même si la target initiale ne dépend presque pas de $U$, cela permet de garder la même interface que les modèles suivants.

### Limite

Cette étape est surtout technique. Elle ne justifie pas encore scientifiquement l’usage d’un PINN.

---

## 7. Rappel : Navier–Stokes et rôle de $u,v,p$

En 2D, le champ de vitesse du fluide est :

$$
\mathbf u(x,y) = (u(x,y), v(x,y))
$$

où :
- $u$ est la composante horizontale de la vitesse fluide ;
- $v$ est la composante verticale.

La pression est :

$$
p(x,y)
$$

Pour un fluide incompressible stationnaire, on a :

$$
u u_x + v u_y
=
-p_x + \nu (u_{xx}+u_{yy})
$$

$$
u v_x + v v_y
=
-p_y + \nu (v_{xx}+v_{yy})
$$

et l’incompressibilité :

$$
u_x + v_y = 0
$$

Ces équations sont locales : elles sont vérifiées en tout point du domaine fluide.

### Où intervient $\alpha$ ?

$\alpha$ n’est pas une coordonnée spatiale. C’est un paramètre des conditions aux bords.

Loin du deltaplane, le fluide arrive avec une vitesse de direction imposée par l’angle d’attaque :

$$
\mathbf U_\infty
=
U(\cos \alpha, \sin \alpha)
$$

Changer $\alpha$ change donc la manière dont le fluide contourne l’objet.

### Où intervient $U$ ?

La vitesse $U$ influence la pression dynamique et le nombre de Reynolds :

$$
Re = \frac{\rho U L}{\mu}
$$

Si le PINN ne dépendait pas de $U$, il ne pourrait représenter qu’un écoulement à une vitesse de référence. En particulier, si on intégrait la pression issue d’un tel PINN, on ne retrouverait pas automatiquement la loi en $U^2$.

C’est pourquoi le PINN final doit prendre $U$ en entrée :

$$
f(x,y,U,\alpha) \mapsto (u,v,p)
$$

---

## 8. Étape 2 — PINN fluide offline

### Objectif

Construire la vraie partie PINN du projet.

Le réseau :

$$
f_\theta(x,y,U,\alpha) \mapsto (u,v,p)
$$

apprend le champ fluide local autour de la forme triangulaire du deltaplane.

### Loss physique

On impose les résidus de Navier–Stokes :

$$
R_x =
u u_x + v u_y
+
p_x
-
\nu (u_{xx}+u_{yy})
$$

$$
R_y =
u v_x + v v_y
+
p_y
-
\nu (v_{xx}+v_{yy})
$$

et :

$$
R_c = u_x + v_y
$$

La loss PDE peut être :

$$
\mathcal L_{PDE}
=
\mathbb E[
R_x^2 + R_y^2 + R_c^2
]
$$

### Conditions aux bords

On ajoute des pertes de bord.

#### Bord lointain / inflow

Loin du deltaplane :

$$
(u,v)
\approx
U(\cos\alpha,\sin\alpha)
$$

#### Surface du deltaplane

Sur la surface, pour un modèle visqueux no-slip :

$$
(u,v) = (0,0)
$$

dans le référentiel de l’objet.

#### Pression de référence

La pression est définie à une constante près. On fixe par exemple :

$$
p(x_{ref},y_{ref}) = 0
$$

ou une moyenne de pression nulle sur une zone de référence.

### Remarque importante

Le PINN est entraîné offline en Python. Il n’a pas besoin de tourner directement dans Unreal dans la première version.

---

## 9. Étape 3 — Extraction des forces depuis le PINN

Une fois $f$ entraîné, on peut extraire des forces aérodynamiques.

La force du fluide sur la surface de l’objet vient du tenseur des contraintes :

$$
\sigma =
-pI
+
\mu(\nabla \mathbf u + \nabla \mathbf u^T)
$$

La traction locale sur la surface est :

$$
\mathbf t(s,U,\alpha)
=
\sigma \mathbf n
$$

où :
- $s$ désigne une position sur la surface ;
- $\mathbf n$ est la normale extérieure.

La force totale est :

$$
\mathbf F(U,\alpha)
=
\int_{\partial \Omega}
\mathbf t(s,U,\alpha)\,ds
$$

En pratique, avec des points de surface :

$$
\mathbf F(U,\alpha)
\approx
\sum_i
\mathbf t(s_i,U,\alpha)\Delta s_i
$$

### Subtilité importante : gradients et traînée

La pression seule donne :

$$
\mathbf F_p
=
-\int_{\partial\Omega}
p \mathbf n\,ds
$$

Cette force de pression donne déjà une partie importante de la portance et une partie de la traînée de pression.

Mais la traînée visqueuse dépend des gradients de vitesse :

$$
\mu(\nabla \mathbf u + \nabla \mathbf u^T)
$$

Pendant l’extraction offline, ce n’est pas un problème : on peut utiliser l’autodiff du PINN en Python pour calculer les gradients.

Dans Unreal, en revanche, calculer ces gradients en runtime serait beaucoup plus compliqué. D’où l’intérêt de ne pas exécuter directement cette étape complète dans Unreal.

---

## 10. Étape 4 — MLP global distillé depuis le PINN

### Objectif

Avoir un modèle runtime rapide.

À partir du PINN, on génère un dataset :

$$
(U_i,\alpha_i)
\mapsto
(C_{L,i},C_{D,i})
$$

où les coefficients proviennent de l’intégration des contraintes extraites du champ PINN.

On entraîne alors un modèle :

$$
g_1(U,\alpha)\mapsto(C_L,C_D)
$$

Ce modèle est utilisé dans Unreal.

### Interprétation

Le PINN sert de professeur physique offline.

Le MLP $g_1$ est un student runtime ultra rapide.

```text
PINN f
    ↓
champs u,v,p
    ↓
intégration surface
    ↓
dataset CL/CD
    ↓
MLP g1
    ↓
Unreal runtime
```

### Avantage

Cette étape est très robuste :
- runtime léger ;
- pas de gradient dans Unreal ;
- export facile ;
- forces simples à appliquer.

### Limite

On réduit toute l’aérodynamique à deux coefficients globaux. C’est efficace, mais on perd la distribution spatiale des forces.

---

## 11. Étape 5 — Student local de surface : $h(s,U,\alpha)$

### Motivation

Le modèle global :

$$
g_1(U,\alpha)\mapsto(C_L,C_D)
$$

est pratique, mais il écrase toute l’information spatiale.

Une version plus intéressante consiste à apprendre la traction locale sur la surface :

$$
h(s,U,\alpha)\mapsto \mathbf t(s)
$$

où $\mathbf t(s)$ est la force locale par unité de longueur exercée par le fluide sur la surface.

Dans Unreal, on échantillonne la surface du deltaplane en $N$ points :

```text
s_1, s_2, ..., s_N
```

et on évalue :

$$
h(s_i,U,\alpha)
$$

Puis on somme :

$$
\mathbf F
\approx
\sum_i
h(s_i,U,\alpha)\Delta s_i
$$

et on peut aussi calculer naturellement le moment :

$$
M
\approx
\sum_i
\mathbf r_i \times
h(s_i,U,\alpha)\Delta s_i
$$

### Pourquoi c’est plus intéressant ?

Parce qu’on ne prédit plus seulement deux nombres globaux.

On prédit une distribution de forces sur l’objet :
- plus interprétable ;
- plus proche d’une intégration CFD ;
- permet de visualiser les zones de force ;
- permet de calculer les torques naturellement ;
- conserve une structure spatiale.

### Comment entraîner $h$ ?

On utilise le PINN offline.

Pour chaque configuration $(U,\alpha)$, on calcule :

$$
\mathbf t_f(s,U,\alpha)
$$

à partir de :

$$
\sigma =
-pI + \mu(\nabla \mathbf u+\nabla \mathbf u^T)
$$

puis on entraîne :

$$
\mathcal L_h
=
\frac1N
\sum_i
\left\|
h(s_i,U_i,\alpha_i)
-
\mathbf t_f(s_i,U_i,\alpha_i)
\right\|^2
$$

Donc $h$ est une version compressée du PINN pour le runtime.

### Interprétation

```text
PINN f :
(x,y,U,alpha) -> u,v,p

offline :
autodiff + contraintes -> t(s,U,alpha)

student runtime :
h(s,U,alpha) -> t(s)

Unreal :
somme des tractions -> force + moment
```

Le PINN ne tourne pas directement dans Unreal, mais sa physique est absorbée dans $h$.

C’est probablement la version la plus élégante du projet si le temps le permet.

---

## 12. Étape 6 — Visualisation du champ fluide

En bonus, on peut utiliser le PINN $f$ pour produire une visualisation locale autour du deltaplane :

$$
f(x,y,U,\alpha)\mapsto(u,v,p)
$$

Dans Unreal, on peut échantillonner une petite grille locale autour du deltaplane :
- toutes les quelques frames ;
- ou à faible résolution ;
- ou uniquement pour des particules.

On peut afficher :
- des flèches de vitesse ;
- des lignes de courant ;
- des particules advectées ;
- une heatmap de pression ;
- les tractions locales sur la surface.

Cette partie n’est pas indispensable pour la dynamique du joueur, mais elle rend la démonstration beaucoup plus visuelle.

---

## 13. Discussion : PINN vs CFD classique

Un point important est de ne pas survendre les PINN.

Pour résoudre une simulation fluide unique avec précision, un solveur CFD classique est généralement :
- plus robuste ;
- plus précis ;
- plus mature ;
- plus efficace à entraînement nul.

Le PINN n’est pas choisi parce qu’il serait toujours meilleur que la CFD.

Il est intéressant ici parce qu’il fournit une représentation :
- continue ;
- compacte ;
- différentiable ;
- paramétrique en $U$ et $\alpha$ ;
- facilement distillable dans un modèle runtime.

Un solveur CFD classique donnerait plutôt :

```text
(U1, alpha1) -> simulation 1
(U2, alpha2) -> simulation 2
(U3, alpha3) -> simulation 3
...
```

Le PINN apprend directement une famille de solutions :

$$
(x,y,U,\alpha)\mapsto(u,v,p)
$$

Cela permet ensuite :
- d’interroger le champ en n’importe quel point ;
- d’extraire des forces pour des valeurs continues de $U,\alpha$ ;
- de distiller le comportement dans un réseau plus petit ;
- de visualiser un champ fluide local.

La présentation honnête du projet est donc :

> On ne prétend pas remplacer la CFD industrielle. On explore comment un PINN peut servir de représentation neuronale compacte d’un écoulement local, puis être compressé en modèles utilisables en temps réel dans Unreal Engine.

---

## 14. Progression finale retenue

La progression choisie est volontairement incrémentale, de sorte que le projet reste présentable même si les dernières étapes ne sont pas terminées.

### Version minimale

```text
0. Formules analytiques dans Unreal
1. MLP global appris sur ces formules
```

Cette version valide :
- la physique de gameplay ;
- la pipeline réseau vers Unreal.

### Version avec vraie composante PINN

```text
2. PINN offline : (x,y,U,alpha) -> u,v,p
3. Extraction offline : PINN -> CL/CD
4. MLP global distillé : (U,alpha) -> CL/CD
```

Cette version donne une vraie partie physics-informed.

### Version forte

```text
5. Student local : (s,U,alpha) -> traction locale
6. Visualisation du champ fluide
```

Cette version conserve la structure spatiale des forces et rend le projet beaucoup plus original.

---

## 15. Pipeline complet résumé

```text
Unreal prototype
---------------
Formule analytique CL/CD
    ↓
Forces lift/drag
    ↓
Deltaplane jouable


MLP simple
----------
Dataset analytique
    ↓
g0(U,alpha) -> CL/CD
    ↓
Export Unreal


PINN offline
------------
f(x,y,U,alpha) -> u,v,p
    ↓
Loss Navier-Stokes + conditions aux bords
    ↓
Champ fluide local


Extraction
----------
f + autodiff
    ↓
pression + gradients de vitesse
    ↓
traction locale t(s,U,alpha)
    ↓
force, moment, CL/CD


Runtime final possible A
------------------------
g1(U,alpha) -> CL/CD
    ↓
forces Unreal


Runtime final possible B
------------------------
h(s,U,alpha) -> traction locale
    ↓
somme sur la surface
    ↓
force + moment Unreal


Bonus visuel
------------
f(x,y,U,alpha) -> u,v,p
    ↓
flèches / particules / pression / streamlines
```

---

## 16. Ce qu’on pourra raconter dans le rapport

Le projet illustre une transition progressive :

1. **Modèle physique réduit classique**  
   Utilisation des coefficients aérodynamiques $C_L,C_D$.

2. **Surrogate neural simple**  
   Remplacement du modèle analytique par un MLP.

3. **PINN fluide**  
   Apprentissage d’un champ $(u,v,p)$ contraint par Navier–Stokes.

4. **Extraction de forces**  
   Calcul des contraintes de surface à partir du champ fluide.

5. **Distillation runtime**  
   Compression du PINN dans un modèle léger adapté au temps réel.

6. **Intégration moteur de jeu**  
   Application des forces dans Unreal pour obtenir un comportement interactif.

La thèse du projet pourrait être formulée ainsi :

> Peut-on utiliser un PINN comme représentation physique intermédiaire d’un écoulement aérodynamique, puis le distiller en modèles suffisamment rapides pour piloter un objet interactif dans Unreal Engine ?

---

## 17. Points de vigilance

### Entraînement PINN difficile

Navier–Stokes est difficile pour les PINN, surtout avec :
- Reynolds élevé ;
- séparation ;
- couches limites fines ;
- géométries complexes ;
- turbulence.

On restera donc sur :
- 2D ;
- stationnaire ;
- géométrie fixe ;
- Reynolds faible ou modéré ;
- plages raisonnables de $U,\alpha$.

### Traînée visqueuse

La traînée complète nécessite les gradients de vitesse à la surface. Ces gradients seront calculés offline avec autodiff.

En runtime, on évitera de calculer les gradients dans Unreal.

### Réalisme limité

Le deltaplane est 2D, rigide, et simplifié. Le projet vise une démonstration physique crédible, pas un simulateur aéronautique certifiable.

### Fallbacks

Si le PINN complet est trop dur :
- on garde le MLP analytique ;
- on montre un PINN sur un sous-problème plus simple ;
- on utilise l’intégration pression seule ;
- on ajoute une traînée empirique ;
- on limite la plage d’angles.

---

## 18. Prochaine étape pratique

La prochaine étape immédiate est de terminer l’étape 0 :

1. calculer $U$ et $\alpha$ dans Unreal ;
2. appliquer la loi analytique $C_L,C_D$ ;
3. vérifier que le deltaplane vole correctement ;
4. ajuster les paramètres physiques ;
5. logger les valeurs utiles :
   - $U$ ;
   - $\alpha$ ;
   - $C_L$ ;
   - $C_D$ ;
   - lift ;
   - drag ;
   - vitesse ;
   - altitude.

Une fois le gameplay stable, on pourra entraîner le premier MLP supervisé.
