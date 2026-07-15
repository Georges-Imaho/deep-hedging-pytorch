# Deep Hedging d'options avec PyTorch

## Présentation du projet

Ce dépôt contient un prototype exploratoire de couverture dynamique d'options avec PyTorch. Il rassemble des formules analytiques de Black--Scholes, des simulations de marché par mouvement brownien géométrique (GBM) et par modèle de Heston, ainsi qu'une stratégie de couverture séquentielle fondée sur un réseau LSTM.

L'objectif est méthodologique : étudier comment une stratégie apprise peut intégrer la dynamique du marché et des coûts de transaction. Le dépôt n'est pas présenté comme un produit fini et ne fournit, à ce stade, aucun résultat validé permettant d'affirmer que le deep hedging surpasse une couverture delta analytique.

## Question étudiée

La question de recherche qui motive le projet est la suivante :

> Une stratégie récurrente apprise sur des trajectoires simulées peut-elle réduire le risque terminal d'une position optionnelle lorsque la couverture est discrète et soumise à des coûts de transaction ?

L'objectif futur est de comparer cette stratégie de deep hedging à une couverture delta issue de Black--Scholes, sur les mêmes trajectoires et avec une convention de PnL commune. Cette comparaison reste à valider expérimentalement.

## Contexte financier

Le vendeur d'un call européen reçoit une prime initiale mais doit payer à maturité le payoff

\[
H(S_T) = (S_T-K)^+ = \max(S_T-K,0),
\]

où :

- \(S_t\) désigne le prix du sous-jacent à la date \(t\) ;
- \(K\) est le prix d'exercice ;
- \(T\) est la maturité ;
- \(\delta_t\) est la quantité de sous-jacent détenue entre deux dates de couverture ;
- \(V_0\) est la prime reçue à l'origine ;
- \(c\) est le taux de coût de transaction proportionnel.

Dans un marché idéal et sous les hypothèses de Black--Scholes, une couverture continue par le delta permet de répliquer l'option. En pratique, les rééquilibrages sont discrets et coûteux. Le deep hedging cherche alors directement une politique \(\delta_t\) adaptée au critère de risque choisi.

## Modèles utilisés

### Black--Scholes

`src/analytics_models.py` implémente le prix et le delta de calls et puts européens sous les hypothèses usuelles : volatilité et taux constants, absence de dividendes et dynamique lognormale du sous-jacent.

Cette implémentation sert d'oracle analytique et doit, à terme, fournir une stratégie de référence. Les cas limites tels que \(T=0\) ou \(\sigma=0\) ne disposent pas encore d'une validation automatisée complète.

### Mouvement brownien géométrique

`MarketSimulator.simulate_gbm` génère des trajectoires selon

\[
S_{t+\Delta t}
= S_t\exp\left[\left(r-\frac{1}{2}\sigma^2\right)\Delta t
+\sigma\sqrt{\Delta t}\,Z_t\right],
\qquad Z_t\sim\mathcal N(0,1).
\]

Le tableau retourné contient le prix initial puis les prix simulés. Sa forme est `(steps + 1, n_paths)` et l'horizon effectif est `steps * dt`.

### Modèle de Heston

`MarketSimulator.simulate_heston` représente une variance stochastique avec retour à la moyenne et corrélation entre les chocs du spot et de la variance :

\[
\begin{aligned}
dS_t &= rS_t\,dt + \sqrt{v_t}S_t\,dW_t^S,\\
dv_t &= \kappa(\theta-v_t)\,dt + \xi\sqrt{v_t}\,dW_t^v,\\
d\langle W^S,W^v\rangle_t &= \rho\,dt.
\end{aligned}
\]

Dans le code, `theta` est utilisé comme une variance de long terme et le second tableau retourné est sa racine carrée, donc une volatilité. La discrétisation actuelle est une approximation d'Euler simplifiée avec plancher numérique sur la variance. Elle convient à une exploration initiale, mais sa positivité, son biais de discrétisation et sa convergence doivent encore être étudiés.

### LSTM de couverture

`src/deep_hedger.py` contient un `LSTMCell` qui produit une décision à chaque date. Le modèle reçoit les variables de marché courantes ainsi que la position précédente. Dans le notebook expérimental, les variables utilisées sont notamment :

- la log-moneyness \(\log(S_t/K)\) ;
- le temps restant jusqu'à maturité ;
- une volatilité simulée et standardisée ;
- le dernier log-rendement observé.

Une couche sigmoïde contraint actuellement la position produite entre 0 et 1. Cette restriction est cohérente avec l'interprétation d'une couverture en sous-jacent d'un call vendu, mais elle constitue une hypothèse de modélisation et non une propriété générale du deep hedging.

## Fonction objectif et calcul du PnL

L'implémentation actuelle calcule le PnL terminal du vendeur du call sous la forme

\[
\mathrm{PnL}
= V_0
+ \sum_{t=0}^{N-1}\delta_t(S_{t+1}-S_t)
- \mathrm{TC}
- (S_T-K)^+.
\]

Deux objectifs d'entraînement sont présents :

1. une phase optionnelle de warm-up qui minimise la moyenne de \(\mathrm{PnL}^2\) ;
2. une forme logarithmique de risque exponentiel,

\[
\log\left(\frac{1}{B}\sum_{i=1}^{B}
\exp\left[-\lambda\,\frac{\mathrm{PnL}_i}{100}\right]\right),
\]

où \(B\) est la taille du batch et \(\lambda\) le paramètre d'aversion au risque. Le facteur 100 est une normalisation numérique fixée dans le code ; son choix et l'échelle économique de \(\lambda\) restent à justifier.

Le calcul actuel ne modélise pas explicitement le compte cash, le financement à taux non nul ou le coût de liquidation finale de la position en sous-jacent. Les expériences utilisent donc principalement un taux nul, mais la convention de PnL devra être formalisée avant une comparaison scientifique.

## Coûts de transaction

Les coûts sont proportionnels au montant de sous-jacent échangé :

\[
\mathrm{TC}
= c\sum_{t=0}^{N-1} S_t\,|\delta_t-\delta_{t-1}|,
\qquad \delta_{-1}=0.
\]

Cette convention inclut l'ouverture de la couverture et les rééquilibrages utilisés par le moteur actuel. Elle n'inclut pas explicitement la liquidation terminale. Les niveaux de coûts employés dans les notebooks sont des paramètres expérimentaux et ne proviennent pas encore d'une calibration sur données de marché.

## Architecture du dépôt

```text
deep-hedging-pytorch/
├── notebook/
│   ├── BlackScholes.ipynb   # Formule analytique et illustration GBM
│   ├── Data_Load.ipynb      # Génération d'un dataset synthétique Black--Scholes
│   └── test_LSTM.ipynb      # Entraînement et benchmark exploratoires
├── src/
│   ├── analytics_models.py  # Prix et deltas Black--Scholes
│   ├── deep_hedger.py       # Politique de couverture LSTM
│   ├── hedging_engine.py    # PnL, coûts et objectifs d'entraînement
│   └── market_simulator.py  # Simulations GBM et Heston
├── .gitignore
├── requirements.txt
└── README.md
```

Les notebooks sont conservés comme traces exploratoires. Ils ne constituent pas une suite de tests ni une démonstration de performance.



Le dépôt doit donc être lu comme une base de travail et une démonstration de démarche, non comme une conclusion empirique définitive.
