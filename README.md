# Network Momentum com equities globais

Implementação profissionalizada do modelo **Network Momentum** de Pu, Roberts, Dong e
Zohren ([arXiv:2308.11294](https://arxiv.org/abs/2308.11294)), adaptada para ações de
diferentes bolsas com dados diários do `yfinance`.

> **Aviso de escopo:** isto é uma **adaptação** (ações globais, Yahoo Finance, 2005+),
> não uma replicação do artigo (64 futuros Pinnacle, 1990–2022). Os números não são
> comparáveis. O universo atual tem **viés de sobrevivência** — declarado e explorado
> por ablação, não corrigível com dados gratuitos. Ver `docs/AUDIT.md` e
> `docs/LIMITATIONS.md`.

## Como rodar tudo de uma só vez

**Opção 1 — Google Colab (recomendado):** abra
`Network_Momentum_Professional_Colab.ipynb`, escolha o perfil na primeira célula e
execute em ordem (Modo A: ZIP/clone; Modo B: Google Drive para cache).

**Opção 2 — linha de comando (um único comando):**

```bash
python -m pip install -e ".[dev]"
python -m network_momentum --config config/default.toml --profile fast
```

**Opção 3 — Windows:**

```powershell
.\run_all.bat                       # testes + pipeline (perfil fast)
.\run_all.bat -Profile full          # com todas as ablações
.\run_all.bat -Config config\paper_structure.toml
```

Perfis: `smoke` (dados sintéticos, sem internet, ~1–2 min — valida o encanamento),
`fast` (dados reais, sem as ablações caras) e `full` (tudo).

O pipeline único (`network_momentum.pipeline.run_full_pipeline`) executa: download
OHLCV+FX+benchmarks com cache carimbado → verificações de qualidade → 8 features
(Eqs. 1–3) → grafos (Eq. 4, ensemble Eq. 5, normalização Eq. 6) → propagação (Eq. 7)
→ OLS walk-forward com seleção nested de (α, β) (Eq. 8) → portfólio long-short com
vol targeting (Eq. 9) → custos reais por bolsa sobre variação efetiva de pesos
(Eqs. 13–14) → benchmarks (LinReg, RegCombo, SignCombo, MACD, LongOnly, EW +
ACWI/SPY/EFA/EEM/Ibovespa) → validação (bootstrap, permutação, DSR, PBO, learning
curve) → robustez (ablações, regimes, long/short, concentração, capacidade) →
topologia do grafo → ~30 gráficos (300 dpi, PNG+SVG+CSV) → tabelas → manifest →
respostas do formulário.

## Saídas (`outputs/<perfil>/`)

| Arquivo/pasta | Conteúdo |
|---|---|
| `run_manifest.json` | ambiente, versões, seed, hash do universo, período, commit |
| `metrics.csv`, `annual_metrics.csv` | métricas da estratégia principal |
| `tables/*.csv` | ~35 tabelas (benchmarks, custos, validação, robustez, topologia) |
| `figures/*.{png,svg,csv}` | todos os gráficos com dados de origem |
| `form_answers.md` | respostas do formulário (curta + técnica) |
| `daily_returns.csv`, `predictions.csv.gz`, `coefficients.csv`, `folds.csv` | séries e diagnósticos |

## Estrutura

```
src/network_momentum/
  pipeline.py    ← ponto de entrada único (run_full_pipeline)
  cli.py         ← python -m network_momentum
  data.py        OHLCV + FX + qualidade      features.py   Eqs. 1–3
  graph.py       Eqs. 4–6 (+ CVXPY opcional) topology.py   métricas §5.1
  backtest.py    walk-forward, modos GMOM/LinReg/RegCombo
  portfolio.py   Eq. 9 + turnover Eq. 13 + custos Eq. 14
  costs.py       custos por bolsa (config/costs.csv)        fx.py  conversão USD
  benchmarks.py  Eq. 10–12 + externos        model.py      OLS/Ridge/Lasso/ENet
  validation.py  bootstrap/permutação/DSR/PBO robustness.py ablações/regimes
  metrics.py     métricas completas          plotting.py   gráficos 300dpi
config/
  default.toml   perfil prático  |  paper_structure.toml  perfil fiel ao artigo
  universe.csv   36 ações, 9 moedas, metadados  |  costs.csv  custos com fontes
  fx.csv         mapa cambial    |  benchmarks.csv  externos (PR vs TR declarado)
docs/
  AUDIT.md (diagnóstico + Article Fidelity Matrix), CHANGE_PLAN.md, CHANGELOG.md,
  REFERENCES.md, LIMITATIONS.md, FUTURE_WORK.md
tests/           15 arquivos — look-ahead, Eq. 13/14, FX, grafo, modelos, e2e sintético
```

## Protocolo temporal

Walk-forward expansivo: treino inicial de 10 anos, blocos de teste de 5 anos, embargo
de 1 pregão. Os últimos 10% de cada treino escolhem (α, β) por Sharpe de validação
(grade 3×3 por padrão; a grade 11×11 do artigo está documentada em
`config/paper_structure.toml`). O teste nunca participa de escolha alguma.
`signal_lag_days = 1` por padrão (bolsas em fusos diferentes); o perfil
`paper_structure.toml` usa lag 0, cinco lookbacks e grafo diário como no artigo.

## Correspondência com o artigo

Ver a **Article Fidelity Matrix** completa em `docs/AUDIT.md`. Resumo: Eqs. 1–3, 6–9 e
13–14 reproduzidas fielmente; Eq. 4 com objetivo matematicamente equivalente mas solver
suavizado (L-BFGS-B; solver convexo exato opcional via `graph.solver="cvxpy"`); Eq. 5
com elegibilidade por lookback; universo/fonte/período são adaptações declaradas.

## Testes

```bash
python -m pytest -q            # suíte completa (inclui e2e sintético ~1-2 min)
python -m pytest -m "not slow" # apenas os rápidos
```

O notebook executa a suíte antes do pipeline e novamente na célula final.
