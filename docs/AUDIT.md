# Auditoria Técnica — Network Momentum (Itaú Quant)

**Data da auditoria:** 2026-07-28
**Referência:** Pu, Roberts, Dong, Zohren — *Network Momentum across Asset Classes* (arXiv:2308.11294v1)
**Escopo:** todos os arquivos de `src/network_momentum/`, `config/`, `tests/`, `outputs/latest/` e o PDF do artigo (páginas 8–12 renderizadas + texto integral via ar5iv).

> Esta auditoria foi produzida **antes** de qualquer modificação no código, conforme o protocolo.
> Nota de ambiente: a máquina onde a auditoria foi executada não tem acesso ao PyPI; a
> validação dinâmica (execução do pipeline) foi feita com base nos artefatos de
> `outputs/latest/` gerados em execução anterior e em análise estática do código.

---

## 1. Sumário executivo

O projeto é uma adaptação **estruturalmente correta e honesta** do artigo para ações globais:
as oito features seguem as Eqs. (1)–(3), o objetivo do grafo é matematicamente equivalente à
Eq. (4) (verificado analiticamente, incluindo a reparametrização de escala), o ensemble/normalização
seguem as Eqs. (5)–(6), a propagação segue a Eq. (7), a OLS segue a Eq. (8), o portfólio segue a
Eq. (9) e o turnover/custo seguem as Eqs. (13)–(14). Os testes existentes cobrem vazamento
temporal, alvo futuro, restrições do grafo e o walk-forward sintético.

Os problemas encontrados **não são de fidelidade das equações**, e sim de:

1. **Agregação multi-moeda sem conversão cambial** (P0 — potencialmente incorreto): a Eq. (9) é
   aplicada à média de retornos denominados em 9 moedas diferentes, o que só é interpretável
   como uma carteira "100% hedgeada a custo zero e sem diferencial de juros". Não há moeda-base.
2. **Seleção de hiperparâmetros degenerada** (P0): os grids `alpha_grid`/`beta_grid` têm um único
   ponto (0.1, 0.1); a validação registra o Sharpe mas não seleciona nada. O grid do artigo tem
   11×11 pontos.
3. **Ausência total de benchmarks** (P0): não existem Long Only, MACD (Eq. 10), LinReg (Eq. 11),
   RegCombo, SignCombo, nem índices externos. Sem eles não é possível atribuir valor ao "network".
4. **Custo único em bps** (P0): 1 bp linear sobre turnover, sem corretagem/bolsa/spread/impacto/
   stamp duty/borrow; universo tem bolsas com tributos estatutários grandes (UK 50 bps de SDRT
   na compra; HK 10 bps por lado; França 30 bps FTT na compra; Índia 10 bps STT por lado) que
   tornam o resultado líquido atual **superestimado** para o universo escolhido.
5. **Solver do grafo aproximado** (P1): L-BFGS-B suavizado em vez do problema convexo exato
   (MOSEK/CVXPY no artigo). Zeros exatos viram valores minúsculos; a esparsidade — métrica central
   da Seção 5 do artigo — fica distorcida sem um threshold explícito.
6. **Viés de sobrevivência e universo minúsculo** (estrutural, documentado): 36 mega caps atuais,
   sem point-in-time; inclui HSBC duas vezes (HSBA.L e 0005.HK — mesma empresa em duas bolsas).
7. **Diagnósticos de overfitting/robustez inexistentes** (P1): sem bootstrap, DSR, permutação,
   ablações, métricas de topologia (esparsidade, grau, clustering, community ratio, Jaccard),
   sem decomposição long/short, sem análise por regime.

O Sharpe OOS reportado (1.24 com vol targeting, 2015–2026) **não é comparável** aos números do
artigo (universo, classe de ativos, período e frequência de atualização do grafo diferentes) e
está **bruto de custos realistas** — o custo de 1 bp usado é mais otimista que os tributos
estatutários de 4 das 11 bolsas do universo.

---

## 2. Tabela de auditoria por componente

Colunas: **Componente | Arquivo/função | Como está implementado | Correspondência com o artigo | Diferença vs. artigo | Risco | Prioridade | Alteração recomendada**

| # | Componente | Arquivo / função | Como está implementado | Artigo | Diferença | Risco quantitativo/operacional | Prior. | Alteração recomendada |
|---|---|---|---|---|---|---|---|---|
| 1 | Aquisição de dados | `data.py::download_adjusted_close` | `yf.download` só de `Close` ajustado (`auto_adjust=True`, `repair=True`), cache csv.gz por hash de (tickers, start, end, flags) | §2.1 usa 64 futuros contínuos ratio-adjusted da Pinnacle CLC, 1990–2022 | Fonte, classe de ativo e período diferentes; sem OHLCV/volume | Sem volume não há liquidez/spread/impacto/capacidade; cache com `end=""` nunca expira (dados envelhecem silenciosamente) | P0 | Baixar OHLCV completo; carimbar data de download no cache e no manifest; manter Close ajustado para retornos e Volume para liquidez |
| 2 | Eventos corporativos | `data.py` (delegado ao yfinance) | `auto_adjust=True` (splits+dividendos), `repair=True` | Futuros ratio-adjusted (equivalente funcional) | Adaptação necessária p/ ações | Yahoo revisa histórico (total-return backfill imperfeito, especialmente .SA e .T); sem verificação própria de saltos | P2 | Check de retornos extremos (>|60|% em 1d) com log; documentar dependência do ajuste do Yahoo |
| 3 | Viés de sobrevivência | `config/universe.csv` | 36 mega caps atuais de 11 bolsas, lista estática | Universo fixo de 64 futuros líquidos (sem delistagem relevante) | Ações exigiriam point-in-time; universo atual só tem sobreviventes vencedores | Sharpe OOS inflado por seleção ex-post; impossível corrigir com Yahoo | P0 (documentar) | Declarar o viés em todo output; ampliar universo; simular remoção dos melhores ativos (ablação) para dimensionar sensibilidade |
| 4 | Calendários/feriados | `features.py::build_feature_set` | Features por ativo no calendário local, depois `reindex(calendar)` com `ffill(limit=5)`; alvo só em observações reais | Futuros ~mesmo calendário | Adaptação correta | ffill de 5 dias pode propagar sinal velho p/ grafo; alvo protegido | OK | Manter; expor `max_stale_days` no relatório |
| 5 | Fusos horários | `data.py::_extract_close` + `signal_lag_days=1` | Datas normalizadas para date naive; features defasadas 1 dia na config padrão | Não aplicável (mesmo fechamento) | Adaptação conservadora | Com lag 0 (paper_structure) haveria uso de fechamento asiático "futuro" na visão de quem opera nos EUA — documentado, mas perigoso se usado sem entender | P1 | Manter lag 1 como padrão; teste automatizado do lag; alertar no notebook quando lag=0 |
| 6 | Signal lag | `features.py` linhas 111–112 | `features.shift(signal_lag_days)` após winsorização | Paper: sinal no fechamento t, retorno t→t+1 (lag 0) | Lag 1 padrão = 1 dia mais conservador que o artigo | Subestima levemente o Sharpe vs. artigo; é o preço da executabilidade global | OK | Manter; documentar como adaptação |
| 7 | Moeda dos retornos | ausente | Retornos locais; média transversal Eq. (9) mistura 9 moedas | Futuros em USD (margem), sem problema cambial equivalente | **Não implementado**: sem moeda-base, sem hedge, sem decomposição FX | Retorno "USD" reportado não existe; para BRL/INR o carrego cambial é material; comparação com benchmarks externos fica inválida | **P0** | Implementar conversão para USD via par Yahoo (`EURUSD=X` etc.); reportar 3 versões: local (≈hedged proxy), USD unhedged, decomposição ativo×câmbio |
| 8 | Oito features | `features.py::_features_for_ticker` | `pct_change(Δ)/(σ_d·√Δ)` p/ Δ∈{1,21,63,126,252}; MACD (8,24),(16,48),(32,96): EWMA α=1/S, /std63(preço), /std252(MACD_norm) | Eqs. (1)–(3) e §2.2 exatos | Nenhuma relevante | — | OK | Nenhuma (adicionar testes de valores contra implementação de referência) |
| 9 | Winsorização | `features.py::_winsorize_past_only` | clip em μ_EWM ± 5·σ_EWM, halflife 252, causal | §2.2: 5× EWMstd em torno da EWMA, halflife 252 | Fiel | — | OK | Nenhuma |
| 10 | Volatilidade | `features.py::_ewm_std` | `ewm(span=60, adjust=True).std(bias=True)` | Apêndice A.2: pesos (1−α)^τ normalizados, α=2/61 | Fiel (adjust=True reproduz a soma de pesos; bias=True = populacional) | — | OK | Nenhuma |
| 11 | Aprendizado do grafo | `graph.py::learn_adjacency` | Minimiza z·w − Σlog(deg) + (2βα/m²)‖w‖² sobre arestas w≥0 do triângulo superior via L-BFGS-B, com reparametrização w=(α/m)x | Eq. (4): min tr(V'LV) − α1'log(A1) + β‖A‖²_F, resolvido com MOSEK via CVXPY | Objetivo **matematicamente equivalente** (verificado); solver suavizado ≠ solver convexo exato: sem zeros exatos | Esparsidade (métrica-chave da §5.1) fica artificialmente ~densa; threshold atual = eps de máquina | P1 | Oferecer caminho CVXPY+CLARABEL/SCS opcional; threshold de aresta configurável e reportado; testes numéricos comparando soluções |
| 12 | Restrições do grafo | `graph.py` | Simetria e diagonal-zero por construção (parametrização em arestas); w≥0 por bounds; degree>0 garantido pelo log | Eq. (4) s.t. A=A', A≥0, diag=0 | Fiel | — | OK | Manter testes |
| 13 | Janela de observação V_t | `graph.py::_observation_matrix`, `_active_assets` | V ∈ R^{N×8δ} concatenado; ativo entra se **todas** as features estão completas na janela do **maior** lookback | §3.1: "assets consistently available throughout the lookback window" (por δ) | Elegibilidade usa max(δ) p/ todos os grafos → grafos de δ curto perdem ativos elegíveis | Menos ativos no grafo δ=252 do que o artigo teria; reduz amostra em universos jovens | P1 | Elegibilidade por lookback δ individual, com interseção somente no ensemble |
| 14 | Ensemble de lookbacks | `graph.py::build_graph_snapshots` | Média simples das adjacências dos lookbacks configurados; default 3 lookbacks (252/504/756); paper_structure com os 5 | Eq. (5): média de K=5, δ∈{252,…,1260} | Default reduzido (custo computacional) | Resultados default não comparáveis ao artigo; §5.3 do artigo mostra Sharpe decrescente com δ | P1 | Tornar 5 lookbacks o default científico; manter 3 como perfil rápido explicitamente rotulado |
| 15 | Normalização | `graph.py::normalize_adjacency` | D^{-1/2} A D^{-1/2}, diagonal zerada, degree 0 protegido | Eq. (6) | Fiel na construção do snapshot | — | OK | Nenhuma |
| 16 | Propagação | `graph.py::propagate_network_features` | ũ = Ã·u na data t com snapshot ≤ t; em dias com NaN, subamostra ativos válidos e **renormaliza a matriz já normalizada** | Eq. (7): ũ_i = Σ_j Ã_ij u_j | Dupla normalização no subconjunto (composição de duas D^{-1/2}·) | Viés pequeno mas evitável nos pesos em dias de feriado parcial | P2 | Guardar ensemble **bruto** no snapshot; subamostrar e normalizar uma única vez |
| 17 | Frequência do grafo | `graph.py` (`rebalance_every`) | Snapshot a cada 21 dias úteis (default) ou 1 (paper_structure) | §3.1: grafo re-aprendido diariamente | Adaptação por custo computacional | Grafo até 20 dias defasado; artigo indica Jaccard>0.99 entre dias consecutivos → impacto provavelmente pequeno, mas não medido | P1 | Suportar 1/5/21 e **medir** a diferença (já configurável); reportar comparação |
| 18 | Regressão | `model.py::fit_ols` | OLS pooled com intercepto via `lstsq`, mínimo de amostras, rank reportado | Eq. (8), solução analítica | Fiel | Sem erro-padrão; multicolinearidade das 8 network features não diagnosticada | P1 | Adicionar SE robusto clusterizado por data; Ridge/Lasso/ElasticNet como robustez (não substitutos) |
| 19 | Posições | `backtest.py` linha 286 | `position = sign(prediction)` | Eq. (9): x = sign(y) | Fiel | — | OK | Opcional: threshold de convicção como estudo (não default) |
| 20 | Volatility targeting (ativo) | `backtest.py::_portfolio_for_fold` | `weight = sign × σ_tgt/σ_ann,i`; retorno diário = média entre ativos ativos | Eq. (9): (1/N)Σ x σ_tgt/σ_i r | Fiel | — | OK | Nenhuma |
| 21 | Volatility targeting (portfólio) | `backtest.py::_apply_portfolio_volatility_scaling` | Alavancagem ex-ante = σ_tgt/EWMstd(span 60, shift 1) do retorno líquido, clip [0,5] | §4.2 Panel B: "camada adicional de vol scaling p/ 15%" sem fórmula (provavelmente ex-post) | Versão ex-ante implementável (mais honesta que reescalar ex-post); cap 5 não existe no artigo | Turnover da alavancagem não entra no custo (ver #23) | P1 | Manter ex-ante; reportar também a versão ex-post do paper p/ comparabilidade; incluir Δleverage no turnover |
| 22 | Turnover | `backtest.py` linhas 151–154 | ζ_i = \|w_t − w_{t−1}\| por ativo (primeira obs = \|w\|), média transversal | Eq. (13): ζ = σ_tgt\|x_t/σ_t − x_{t−1}/σ_{t−1}\| | Fiel (idêntico) | Fronteira de fold conta custo cheio (conservador, ok); turnover da alavancagem de portfólio fora | P1 | Computar turnover no peso **final** (com leverage) para o custo |
| 23 | Custos | `backtest.py` linha 163 | custo = turnover × bps/10⁴, bps único global (default 1.0; paper_structure 0.0) | Eq. (14): c·ζ, c∈{0,0.5,…,5} bps (análise de sensibilidade) | Sem estrutura por bolsa/lado/tributo; universo tem SDRT/FTT/STT/stamp de 10–50 bps | Resultado líquido **superestimado**; break-even não calculado | **P0** | Modelo de custos por bolsa/ticker/lado em CSV com fonte e data; cenários conservador/base/otimista; curva Sharpe×custo; break-even |
| 24 | Walk-forward | `splits.py::expanding_windows` | Expansivo, 10y inicial + blocos de teste de 5y, sem sobreposição (testado) | §4.1: 1990–99 treino, blocos de 5y, expansão | Fiel na estrutura (2005+ vs 1990+) | Sem embargo (alvo de 1 dia → risco de leakage marginal ~1 dia na fronteira) | P2 | Embargo de 1 dia opcional; nested walk-forward p/ hiperparâmetros |
| 25 | Seleção de hiperparâmetros | `backtest.py::_fit_candidate_on_validation` | Últimos 10% do treino como validação; escolhe (α,β) por Sharpe de validação; **grid atual = 1 ponto** | §4.1: mesmos 10%, grid {1e-4…10} 11×11 | Grid degenerado → nenhuma seleção real acontece | Hiperparâmetros fixados sem evidência; risco simultâneo de under/overfitting não medido | **P0** | Grid real (subconjunto viável, ex. 3×3 a 5×5), nested, com registro por fold; validação nunca vê teste |
| 26 | Métricas | `metrics.py` | ann_return aritmético ×252, vol ddof=0, Sharpe rf=0, Sortino, Calmar, MDD+duração, hit rate, P/L médio | §4.2 usa o mesmo conjunto | Fiel ao artigo; faltam as métricas extras exigidas (CAGR, skew, kurtosis, VaR, ES, Omega, turnover anualizado, alpha/beta, IR, TE, exposições, capacidade) | Relato incompleto p/ competição | P1 | Ampliar módulo de métricas com rf documentada e frequência de anualização explícita |
| 27 | Benchmarks | ausentes | — | §4.1: Long Only, MACD (Eq. 10), LinReg (Eq. 11); §4.3: RegCombo (Eq. 12), SignCombo | **Não implementado** | Impossível saber se o grafo adiciona valor sobre momentum individual; sem comparação externa (ACWI/SPX/IBOV) | **P0** | Implementar todos os metodológicos + externos via tickers de índice/ETF com aviso PR vs TR |
| 28 | Testes | `tests/` (7 testes) | Leakage de features, alvo t+1, restrições/normalização do grafo, propagação, janelas, e2e sintético, drawdown, cert TLS | — | Boa base; faltam turnover/custos/FX/leverage/timezone/reprodutibilidade/benchmarks | Regressões silenciosas em áreas novas | P1 | Ampliar suíte (ver plano) |
| 29 | Saídas/gráficos | `reporting.py` | 8 CSVs + 1 PNG (160 dpi, sem fonte/nota), manifest mínimo | — | Muito aquém do exigido (32 gráficos, 300 dpi, PNG+SVG+CSV, manifest com seed/universo/ambiente) | Relatório não auditável | P1 | Novo módulo de plotting/manifest |
| 30 | Config | `config.py`, `default.toml` | TOML por seções; validação básica | — | Sem seed, sem moeda-base, sem custos estruturados, sem benchmarks | Reprodutibilidade parcial | P1 | Estender config preservando compatibilidade |
| 31 | Universo (metadados) | `universe.csv` | 6 colunas (ticker, name, region, country, exchange, currency) | Tabela A.5 do artigo tem classe e datas | Sem setor, datas, ADV, short eligibility, borrow, fonte | Custos/ablações setoriais impossíveis | P1 | Enriquecer CSV (setor GICS aproximado, datas, flags) |
| 32 | HSBC duplicado | `universe.csv` linhas 11 e 23 | HSBA.L e 0005.HK simultâneos | — | Mesma empresa em duas bolsas/moedas | Dupla contagem de risco idiossincrático; infla arestas do grafo entre "dois nós" que são o mesmo ativo | P2 | Manter apenas uma listagem ou documentar como estudo de dual listing |
| 33 | Aleatoriedade/seed | n/a | Pipeline determinístico (L-BFGS, lstsq); sem seção de seed | — | Bootstrap/permutação (a criar) precisarão de seed | Irreprodutibilidade futura | P1 | `seed` na config, usado por todo processo estocástico |

---

## 3. Article Fidelity Matrix

Classificações: **fiel** (reprodução fiel), **adaptação** (justificável, documentada), **aproximação** (difere numericamente), **não implementado**, **potencialmente incorreto**.

| Item do artigo | Referência | Implementação | Classificação | Observações |
|---|---|---|---|---|
| Features de momentum: retornos vol-scaled 1/21/63/126/252d | Eq. §2.2 | `features.py` `volret_*` | **fiel** | σ diária EWMstd span 60, bias populacional, idêntico ao Apêndice A.2 |
| MACD normalizado (8,24),(16,48),(32,96) | Eqs. (1)–(3) | `features.py` `macd_*` | **fiel** | EWMA α=1/J recursiva (adjust=False), std 63 do preço, std 252 do MACD_norm |
| Winsorização 5σ EWM, halflife 252 | §2.2 | `_winsorize_past_only` | **fiel** | Causal |
| Volatilidade EWMstd span 60 | Ap. A.2 | `_ewm_std` | **fiel** | — |
| Aprendizado do grafo | Eq. (4) | `learn_adjacency` | **aproximação** | Objetivo equivalente (verificação analítica da reparametrização); solver L-BFGS-B suave em vez do problema convexo exato via MOSEK — sem zeros exatos; esparsidade dependente de threshold |
| Restrições simetria / não-negatividade / diag 0 | Eq. (4) s.t. | parametrização por arestas | **fiel** | Testado |
| Ensemble de 5 lookbacks | Eq. (5) | `build_graph_snapshots` | **adaptação** | Média fiel; default usa 3 lookbacks (custo); perfil com 5 existe (`paper_structure.toml`); elegibilidade por max(δ) é **aproximação** adicional |
| Normalização D^{-1/2}ĀD^{-1/2} | Eq. (6) | `normalize_adjacency` | **fiel** | No snapshot; a re-normalização do subconjunto na propagação é **aproximação** |
| Network features ũ=Ãu | Eq. (7) | `propagate_network_features` | **fiel** (com a ressalva acima) | Grafo defasado até 21d na config default (artigo: diário) — **adaptação** |
| Regressão transversal pooled, alvo r_{t:t+1}/σ_t | Eq. (8) | `fit_ols` + target | **fiel** | Alvo idêntico (σ diária) |
| Portfólio x=sign(y), peso σ_tgt/σ_i, média 1/N | Eq. (9) | `_portfolio_for_fold` | **fiel** | σ_tgt=15% |
| Vol scaling adicional no portfólio (Panel B) | §4.2 | `_apply_portfolio_volatility_scaling` | **adaptação** | Artigo não dá fórmula (presumivelmente ex-post); implementação é ex-ante (EWM 60 defasado, cap 5×) — mais realista, números não comparáveis diretamente |
| MACD benchmark φ(y)=y·exp(−y²/4)/0.89 | Eq. (10) | — | **não implementado** | Necessário como benchmark |
| LinReg com features individuais | Eq. (11) | — | **não implementado** | Necessário: é o teste direto de "o grafo adiciona valor?" |
| RegCombo (u e ũ juntos) | Eq. (12) | — | **não implementado** | — |
| SignCombo (média dos sinais) | §4.3 | — | **não implementado** | — |
| Turnover ζ=σ_tgt\|x_t/σ_t−x_{t−1}/σ_{t−1}\| | Eq. (13) | `_portfolio_for_fold` | **fiel** | Turnover da alavancagem de portfólio não incluído (artigo também não tem essa camada com custo) |
| Retorno líquido r−c·ζ, c∈{0,…,5} bps | Eq. (14) | custo bps único | **aproximação** | Falta a curva de sensibilidade e custos realistas por bolsa (P0) |
| Walk-forward 5y expandindo, validação 10% | §4.1 | `splits.py` + `backtest.py` | **fiel** (estrutura) | Grid α/β degenerado → seleção **não implementada** de fato |
| Grid α,β ∈ {1e-4,…,10} | §4.1 | grids de 1 ponto | **não implementado** | Documentado no TOML, mas nunca executado |
| Universo 64 futuros Pinnacle 1990–2022 | §2.1 | 36 ações Yahoo 2005– | **adaptação** (declarada) | Muda classe de ativo, fonte, período, N e introduz sobrevivência — projeto **não replica** os números do artigo e não deve alegar isso |
| Métricas §4.2 | §4.2 | `metrics.py` | **fiel** | — |
| Topologia: esparsidade, grau, clustering, community ratio, Jaccard | §5.1 | — | **não implementado** | Exigido pelo formulário/relatório |
| Agregação multi-moeda | n/a no artigo | média direta de retornos em 9 moedas | **potencialmente incorreto** | Sem moeda-base; único item classificado como potencialmente incorreto |

---

## 4. Riscos priorizados

**P0 — corrigir antes de qualquer conclusão quantitativa**
1. Moeda-base e conversão FX (retorno local vs USD; decomposição câmbio).
2. Modelo de custos por bolsa/lado com tributos estatutários (UK/HK/FR/IN) + cenários e break-even.
3. Benchmarks metodológicos (LongOnly, EW, MACD, LinReg, RegCombo, SignCombo) e externos (ACWI, SPX TR, IBOV) — sem eles o formulário não pode ser respondido com honestidade.
4. Grid real de (α, β) com validação nested; registrar seleção por fold.
5. Declaração explícita e quantificada do viés de sobrevivência em todos os relatórios.

**P1 — fidelidade e diagnóstico**
6. Solver convexo opcional (CVXPY+CLARABEL/SCS) + threshold de aresta configurável; métricas de topologia.
7. Elegibilidade por lookback δ; ensemble com 5 lookbacks como perfil científico.
8. Turnover incluindo variação da alavancagem; custo aplicado ao peso final.
9. Suíte de overfitting/underfitting (bootstrap por data, DSR, permutação, learning curves, ablações, regimes).
10. Métricas ampliadas + erros-padrão clusterizados por data; Ridge/Lasso/ElasticNet como robustez.

**P2 — qualidade de engenharia**
11. Propagação sem dupla normalização; embargo opcional; HSBC único; checks de dados (stale, extremos, gaps); manifest completo com seed/versões/hashes.

---

## 5. Nota sobre o que **não** está errado (para evitar retrabalho)

- Não há look-ahead detectável no caminho features → grafo → propagação → OLS → posição:
  em cada data t, tudo depende de preços ≤ t−1 (lag 1) ou ≤ t (lag 0, perfil paper), e o alvo é
  t→t+1. O teste `test_eight_features_and_no_future_leakage` cobre o caso principal.
- O turnover Eq. (13) e o custo Eq. (14) estão matematicamente corretos para a camada de ativo.
- A reparametrização do solver do grafo **não** altera o ótimo da Eq. (4) (prova nas notas do
  código; conferida nesta auditoria).
- O vol targeting de portfólio é causal (EWM defasado em 1 dia).
- A validação usa apenas o fim do treino; o teste nunca é tocado na seleção.
