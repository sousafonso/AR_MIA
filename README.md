# AR_MIA

Repositório para a Unidade Curricular de **Aprendizagem por Reforço** (MIA).

O conteúdo resolvido foi consolidado numa única pasta [mia_rl/](./mia_rl/).

## Estrutura

Os exercícios resolvidos e o código reutilizável ficaram consolidados em [mia_rl/](./mia_rl/).

Estrutura principal:

- `core/` — abstrações genéricas como `Environment`, `Agent` e `Policy`
- `envs/` — ambientes como Blackjack, Windy Gridworld e Tic-Tac-Toe
- `mdps/` — abstrações de MDP para programação dinâmica
- `agents/` — algoritmos de RL, incluindo prediction, control, planning e policy gradient
- `features/` — representações de estado e feature engineering
- `policies/` — políticas reutilizáveis
- `experiments/` — loops de treino, avaliação e comparação
- `notebooks/` — notebooks consolidados das práticas
- `scripts/` — scripts executáveis dos experimentos
- `plots/` — helpers de visualização
- `outputs/` — resultados e gráficos gerados

O exercício de k-armed bandits também ficou incluído em [mia_rl/scripts/kbandits_incomplete.py](./mia_rl/scripts/kbandits_incomplete.py).
