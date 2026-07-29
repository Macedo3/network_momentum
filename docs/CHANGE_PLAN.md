# Plano de Alterações — Network Momentum (Itaú Quant)

**Data:** 2026-07-28. Derivado de `docs/AUDIT.md`. Nenhuma alteração remove a versão
interpretável (OLS + sign + vol target), que permanece como estratégia principal.

## Arquitetura-alvo

```
src/network_momentum/
  config.py        estende TOML: seed, moeda-base, custos, benchmarks, validação, solver
  universe.py      colunas novas opcionais (setor, datas, short, borrow, fonte, nota)
  data.py          OHLCV completo + FX + carimbo de download + verificações de qualidade
  fx.py            NOVO: conversão de retornos p/ moeda-base, proxy hedgeada, decomposição
  features.py      inalterado na matemática; expõe daily_return
  graph.py         elegibilidade por lookback, ensemble bruto no snapshot, solver CVXPY
                   opcional, threshold de aresta; propagação com normalização única
  topology.py      NOVO: esparsidade, grau, clustering, community ratio, Jaccard, máscaras
                   intra/inter região e setor
  model.py         OLS + erro-padrão clusterizado por data; Ridge/Lasso/ElasticNet próprios
  portfolio.py     NOVO: pesos, turnover (Eq. 13) incluindo alavancagem, custos por ticker,
                   vol targeting ex-ante, varredura de pseudo-custos (Eq. 14)
  costs.py         NOVO: modelo de custos por bolsa/ticker/lado a partir de CSV auditável,
                   cenários, borrow, participação/impacto, break-even
  splits.py        embargo opcional
  backtest.py      modos network/individual/combo; integra portfolio.py e costs.py
  benchmarks.py    NOVO: LongOnly, EW, MACD (Eq. 10), LinReg (Eq. 11), RegCombo, SignCombo,
                   índices externos com alpha/beta/TE/IR/captura
  metrics.py       métricas ampliadas (CAGR, Omega, skew, kurt, VaR, ES, exposições, etc.)
  validation.py    NOVO: bootstrap em bloco por data, permutação, DSR, PBO/CSCV quando
                   aplicável, learning curves, sensibilidade, correção de múltiplos testes
  robustness.py    NOVO: ablações (features, ativos, regiões, lookbacks, intra/inter),
                   long/short, regimes, contribuição, concentração
  plotting.py      NOVO: todos os gráficos, 300 dpi, PNG+SVG+CSV, com fonte e nota
  reporting.py     manifest completo (ambiente, seed, universo, período, hash)
  cli.py           mantido
config/
  default.toml     estendido (mantém valores atuais como perfil prático)
  paper_structure.toml  estendido (perfil fiel: 5 lookbacks, grafo diário, lag 0)
  universe.csv     enriquecido (setor aproximado, flags; dual listing HSBC documentado)
  costs.csv        NOVO: custos por bolsa com fonte, data de acesso e flag oficial/estimativa
  fx.csv           NOVO: mapa moeda → par Yahoo
  benchmarks.csv   NOVO: benchmarks externos com aviso PR vs TR
tests/             ampliados (custos, FX, alavancagem, topologia, modelos, validação, e2e)
docs/              AUDIT, CHANGE_PLAN, REFERENCES, CHANGELOG, LIMITATIONS, FUTURE_WORK,
                   FORM_ANSWERS
Network_Momentum_Professional_Colab.ipynb   notebook executável (Modos A e B)
```

## Ordem de execução e critérios de aceite

| Etapa | Conteúdo | Aceite |
|---|---|---|
| 1 | Configs + universo + custos + FX + benchmarks CSV | `load_config` compatível com TOML antigo |
| 2 | data.py OHLCV/FX/qualidade | checks retornam relatório; cache carimbado |
| 3 | fx.py | identidade (1+r_loc)(1+r_fx)−1 testada |
| 4 | graph.py + topology.py | testes de restrições continuam passando; Jaccard/others em grafo de brinquedo |
| 5 | model.py | ridge fecha com solução analítica; lasso zera coeficiente irrelevante |
| 6 | portfolio.py + costs.py | Eq. 13/14 numéricas; impostos por lado aplicados |
| 7 | backtest.py modos + benchmarks.py | e2e sintético produz GMOM/LinReg/Combos |
| 8 | metrics/validation/robustness | DSR contra valor conhecido; bootstrap com seed reprodutível |
| 9 | plotting/reporting | arquivos png+svg+csv gerados no e2e |
| 10 | notebook + docs | executa de ponta a ponta **no Colab** (validação local impossível: sem PyPI) |

## Decisões registradas

1. **Moeda-base USD**; três visões de retorno: local (proxy hedgeada sem custo de carry —
   aproximação documentada), USD sem hedge, decomposição ativo×câmbio.
2. **Custos**: apenas valores com fonte; tributos estatutários verificados nesta data
   (UK 50 bps compra — gov.uk; HK 10 bps/lado — gov.hk, vigente desde 17/11/2023);
   Índia STT 10 bps/lado e França FTT 30 bps compra (→40 bps a partir de 2025 pendente de
   confirmação oficial) marcados `verify`; microestrutura (spread/impacto) por **cenário**
   declarado como estimativa metodológica, nunca como fato.
3. **Solver do grafo**: L-BFGS-B continua padrão (sem dependência pesada); CVXPY+CLARABEL/SCS
   opcional para verificação de fidelidade; diferença numérica reportada quando ambos rodam.
4. **Grid α/β**: default científico 3×3 = {0.05, 0.1, 0.5}² com a grade completa do artigo
   documentada; seleção nested por fold (validação = últimos 10% do treino, como no artigo).
5. **Turnover**: custo calculado sobre a variação do **peso final** (incluindo alavancagem de
   portfólio), decomposto em compra/venda para tributos assimétricos.
6. **HSBC dual listing**: mantido no universo (não altero a escolha da equipe), documentado, e
   quantificado por ablação de dedução.
7. **Sem sklearn**: Ridge fechado em numpy; Lasso/ElasticNet por coordinate descent
   determinístico próprio (testado); evita dependência nova e mantém Colab leve.
8. **Seeds**: `seed` na config; todo processo estocástico (bootstrap, permutação, CSCV) o usa.
9. **Point-in-time**: impossível com Yahoo; viés de sobrevivência permanece e é declarado em
   todos os relatórios e nas respostas do formulário.
