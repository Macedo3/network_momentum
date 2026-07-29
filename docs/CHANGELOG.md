# Changelog — profissionalização do Network Momentum

**Data:** 2026-07-28. Baseado no diagnóstico de `docs/AUDIT.md`. Versão 0.1.0 → 1.0.0.

## Correções (P0 — afetam a validade dos números)

1. **Moeda-base e câmbio** (`fx.py`, novo): antes, a Eq. (9) agregava retornos em 9
   moedas sem conversão. Agora há duas séries: local (proxy hedgeada aproximada, sem
   custo de carry — declarado) e USD sem hedge (preços convertidos por pares do Yahoo,
   com o trecho cambial cobrindo a mesma janela do retorno mesmo em feriados locais).
   Nenhuma agregação multi-moeda silenciosa: moeda sem taxa cadastrada → erro.
2. **Custos por bolsa/lado** (`costs.py` + `config/costs.csv`, novos): substituído o
   custo único de 1 bp por corretagem, emolumentos, clearing, taxas regulatórias,
   tributos (SDRT UK 0,5% compra — oficial; stamp HK 0,1%/lado — oficial; STT Índia e
   FTT França — flag verify), meio-spread em 3 cenários, conversão cambial, borrow para
   shorts e impacto square-root com limite de participação. Fontes e datas de acesso no
   CSV; oficial vs estimativa distinguido por coluna.
3. **Turnover sobre o peso final** (`portfolio.py`, novo): o custo agora incide sobre a
   variação efetiva dos pesos **incluindo a alavancagem do vol targeting** (antes a
   variação de leverage não pagava custo) e separa compras de vendas (tributos
   assimétricos). Ativo que sai do universo paga o custo de encerramento.
4. **Grade real de hiperparâmetros**: `alpha_grid`/`beta_grid` padrão 3×3 (antes 1×1 —
   a validação não selecionava nada). Grade 11×11 do artigo documentada.
5. **Benchmarks** (`benchmarks.py`, novo): LongOnly vol-scaled, EqualWeight, MACD
   (Eq. 10), LinReg (Eq. 11), RegCombo (Eq. 12), SignCombo — mesmos universo, período,
   suporte (data, ativo), Eq. (9) e custos — e externos (ACWI/SPY/EFA/EEM/^BVSP)
   convertidos para USD com tipo PR/TR declarado.

## Fidelidade ao artigo (P1)

6. **Elegibilidade por lookback no grafo** (`graph.py`): cada grafo δ usa os ativos com
   histórico completo naquele δ (antes: max δ para todos); o ensemble faz a média das
   arestas pelos grafos em que o par era elegível.
7. **Propagação com normalização única** (`graph.py`): o snapshot guarda o ensemble
   bruto (Eq. 5) e a normalização (Eq. 6) é aplicada uma única vez sobre o subconjunto
   de ativos válidos na data (antes: dupla normalização em dias de feriado parcial).
8. **Solver convexo opcional** (`graph.solver = "cvxpy"`): problema exato da Eq. (4)
   com CLARABEL/SCS/ECOS abertos (o artigo usa MOSEK proprietário); default continua
   L-BFGS-B (sem dependências novas), com threshold relativo de aresta para topologia.
9. **Embargo temporal** (`splits.py`): 1 pregão entre treino e teste (alvo t+1).
10. **Sharpe de treino por fold** (`backtest.py`): tabela treino/validação/teste.

## Novas capacidades

11. **Pipeline único** (`pipeline.py` + `python -m network_momentum`): dados →
    qualidade → features → grafos → GMOM → benchmarks → custos/cenários → validação →
    robustez → topologia → gráficos → tabelas → manifest → formulário, em um comando,
    com perfis full/fast/smoke (smoke = sintético, sem internet).
12. **Validação estatística** (`validation.py`): bootstrap circular em blocos por data,
    permutação por deslocamento circular, Deflated Sharpe Ratio, PBO/CSCV, learning
    curves temporais, estabilidade de coeficientes (com SE clusterizado por data),
    diagnóstico de resíduos (Ljung-Box), R² univariado, Bonferroni/BH.
13. **Robustez** (`robustness.py`): long/short, métricas por região/bolsa/moeda/setor,
    contribuição e concentração (HHI), remoção de melhores ativos/anos, sensibilidade à
    data inicial, regimes de volatilidade, janelas de crise, ablações de features,
    arestas intra/inter (região e setor) e lookbacks.
14. **Topologia** (`topology.py`): esparsidade, grau, clustering, community ratio,
    Jaccard — métricas da Seção 5.1 do artigo, com threshold declarado.
15. **Modelos auxiliares** (`model.py`): Ridge (fechado), Lasso/ElasticNet (coordinate
    descent determinístico próprio, sem sklearn) — robustez, não substitutos da OLS.
16. **OHLCV + qualidade de dados** (`data.py`): volume para ADV/impacto/capacidade;
    checks de preços não positivos, duplicatas, gaps, stale prices, retornos extremos,
    volume zero; cache carimbado por data efetiva (dados não envelhecem em silêncio).
17. **Gráficos de relatório** (`plotting.py`): ~30 figuras, 300 dpi, PNG+SVG+CSV,
    título/eixos/legenda/período/fonte/nota, paleta acessível (Okabe-Ito).
18. **Manifest de reprodutibilidade** (`reporting.py`): ambiente, versões, seed, hash
    do universo, período, commit git, carimbo do download.
19. **Universo enriquecido** (`config/universe.csv`): setor (aproximado), elegibilidade
    de short, borrow estimado, fonte, nota de dual listing (HSBC mantida por decisão de
    universo, quantificada por ablação).
20. **Notebook Colab profissional**: Modo A (ZIP/clone) e Modo B (Drive), detecção de
    ambiente, registro da execução, células didáticas em PT-BR, testes executados antes
    e depois do pipeline, respostas do formulário geradas dos artefatos da execução.
21. **Testes**: 8 → 15 arquivos (~35 casos): Eq. 13/14 numéricas, custos por lado,
    FX (identidade, feriados, inversão), alavancagem causal e com teto, modelos
    (ridge=forma fechada; lasso zera irrelevante; SE clusterizado), topologia em grafo
    de brinquedo, bootstrap reprodutível, permutação detecta look-ahead plantado, DSR
    monotônico em n_trials, PBO, BH, embargo, benchmarks (φ exata) e e2e sintético
    completo com verificação de todos os artefatos.

## Compatibilidade

- APIs antigas preservadas: `fit_ols`, `download_adjusted_close`, `save_results`,
  assinatura posicional de `run_backtest`, campos do `BacktestResult`.
- `run_all.ps1/bat` continuam funcionando; novo parâmetro `-Profile`.
- TOMLs antigos carregam (novas seções têm defaults).

## Não alterado (decisões deliberadas)

- Estratégia principal continua **OLS + sign + vol target** (interpretável).
- Universo definido pela equipe mantido (incl. dual listing HSBC, documentado).
- `yfinance` mantido como fonte (pesquisa), com limitações declaradas.
