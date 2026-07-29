# Referências

Bibliografia auditável. Para cada item: autores, título, ano, veículo, identificador,
tipo, seção do modelo em que é usada, justificativa e data de acesso (quando aplicável).

## Artigo-base

1. **Pu, X.; Roberts, S.; Dong, X.; Zohren, S.** — *Network Momentum across Asset
   Classes* (2023). arXiv:2308.11294. [preprint revisado por grupo acadêmico de Oxford]
   — **Uso:** todo o núcleo do modelo (Eqs. 1–14: features, grafo, ensemble,
   normalização, propagação, regressão, portfólio, turnover e custos).
   Acesso: 2026-07-28 (PDF local `2308.11294v1.pdf` + ar5iv).

## Metodologia — momentum e features

2. **Jegadeesh, N.; Titman, S.** — *Returns to Buying Winners and Selling Losers:
   Implications for Stock Market Efficiency* (1993), Journal of Finance, 48(1).
   DOI:10.1111/j.1540-6261.1993.tb04702.x. [peer-reviewed] — **Uso:** fundamentação
   econômica do momentum (notebook §1). 
3. **Moskowitz, T.; Ooi, Y. H.; Pedersen, L. H.** — *Time Series Momentum* (2012),
   Journal of Financial Economics, 104(2). DOI:10.1016/j.jfineco.2011.11.003.
   [peer-reviewed] — **Uso:** convenção de retornos ajustados por volatilidade e
   vol targeting de 15% (Eq. 9 do artigo-base deriva desta literatura).
4. **Baz, J.; Granger, N.; Harvey, C. R.; Le Roux, N.; Rattray, S.** — *Dissecting
   Investment Strategies in the Cross Section and Time Series* (2015), SSRN 2695101.
   [working paper de instituição reconhecida (Man Group/Duke)] — **Uso:** definição do
   MACD normalizado (Eqs. 1–3) e da função φ do benchmark MACD (Eq. 10).
5. **Lim, B.; Zohren, S.; Roberts, S.** — *Enhancing Time Series Momentum Strategies
   Using Deep Neural Networks* (2019), Journal of Financial Data Science.
   DOI:10.3905/jfds.2019.1.015. [peer-reviewed] — **Uso:** citada pelo artigo-base como
   origem das 8 features e da regularização de turnover (aqui: estudo de thresholds).

## Metodologia — grafos

6. **Kalofolias, V.** — *How to Learn a Graph from Smooth Signals* (2016), AISTATS.
   arXiv:1601.02513. [peer-reviewed] — **Uso:** formulação da Eq. (4) (suavidade
   Laplaciana + barreira log + Frobenius); base do solver próprio e do caminho CVXPY.
7. **Diamond, S.; Boyd, S.** — *CVXPY: A Python-Embedded Modeling Language for Convex
   Optimization* (2016), JMLR 17(83). [peer-reviewed] — **Uso:** solver convexo
   opcional (`graph.solver="cvxpy"`), substituindo o MOSEK proprietário do artigo por
   CLARABEL/SCS/ECOS abertos. Diferenças numéricas documentadas em docs/AUDIT.md §11.

## Metodologia — validação estatística

8. **Bailey, D. H.; López de Prado, M.** — *The Deflated Sharpe Ratio: Correcting for
   Selection Bias, Backtest Overfitting and Non-Normality* (2014), Journal of Portfolio
   Management 40(5). DOI:10.3905/jpm.2014.40.5.094. [peer-reviewed] — **Uso:**
   `validation.deflated_sharpe_ratio`.
9. **Bailey, D. H.; Borwein, J.; López de Prado, M.; Zhu, Q. J.** — *The Probability of
   Backtest Overfitting* (2017), Journal of Computational Finance 20(4).
   DOI:10.21314/JCF.2016.322. [peer-reviewed] — **Uso:** CSCV/PBO
   (`validation.probability_of_backtest_overfitting`).
10. **Politis, D.; Romano, J.** — *The Stationary Bootstrap* (1994), JASA 89(428).
    DOI:10.1080/01621459.1994.10476870. [peer-reviewed] — **Uso:** base conceitual do
    bootstrap em blocos circulares por data (`validation.block_bootstrap_sharpe`).
11. **Benjamini, Y.; Hochberg, Y.** — *Controlling the False Discovery Rate* (1995),
    JRSS-B 57(1). [peer-reviewed] — **Uso:** correção de múltiplos testes.
12. **Hastie, T.; Tibshirani, R.; Friedman, J.** — *The Elements of Statistical
    Learning* (2009, 2ª ed.), Springer. [livro-texto] — **Uso:** Ridge/Lasso/Elastic
    Net como diagnósticos de multicolinearidade (model.py).
13. **Almgren, R.; Thum, C.; Hauptmann, E.; Li, H.** — *Direct Estimation of Equity
    Market Impact* (2005), Risk 18(7). [artigo de prática amplamente citado] — **Uso:**
    forma funcional de impacto (lei de raiz quadrada) em costs.py — declarada como
    cenário metodológico, não calibração.

## Custos — documentos oficiais e fontes

14. **HM Revenue & Customs (Reino Unido)** — *Tax when you buy shares*,
    https://www.gov.uk/tax-buy-shares. [documento oficial de regulador] — **Uso:**
    SDRT 0,5% na compra eletrônica (config/costs.csv, LSE). Acesso: 2026-07-28.
15. **Governo de Hong Kong** — *Stamp Duty Rates*,
    https://www.gov.hk/en/residents/taxes/stamp/stamp_duty_rates.htm. [oficial] —
    **Uso:** stamp duty de 0,1% por lado desde 17/11/2023 (HKEX). Acesso: 2026-07-28.
16. **França — art. 235 ter ZD do CGI (taxe sur les transactions financières)** —
    0,3% desde 2017; a Loi de finances 2025 elevou para 0,4% a partir de 2025-04-01.
    [lei; verificada nesta data apenas em fonte secundária → flag `verify` no CSV] —
    **Uso:** tax_buy Euronext Paris. Acesso: 2026-07-28.
17. **Índia — Securities Transaction Tax (Finance Act)** — 0,1% por lado em operações
    de entrega. [estatutário; verificado em fonte secundária → flag `verify`] —
    **Uso:** NSE. Acesso: 2026-07-28.
18. **SEC (EUA)** — *Section 31 Fee Rate Advisories*, sec.gov. [oficial; taxa revisada
    ao longo do ano → registrada como estimativa ~0,3 bp na venda] — **Uso:** NYSE/Nasdaq.
19. **B3 — Tabela de tarifas de ações à vista**, b3.com.br. [oficial; valor agregado
    aproximado, varia por volume/tipo de investidor → estimativa] — **Uso:** B3.
20. **HKEX — Transaction cost tables** (trading fee 0,00565%, SFC levy 0,0027%),
    hkex.com.hk. [oficial via segunda fonte → estimativa] — **Uso:** HKEX.

Comissões, spreads, borrow e impacto são **estimativas metodológicas institucionais**
por cenário (conservador/base/otimista), declaradas como tal na coluna `notes` de
`config/costs.csv` — nunca apresentadas como fatos medidos.

## Software

21. **Harris, C. R. et al.** — *Array programming with NumPy* (2020), Nature 585.
    DOI:10.1038/s41586-020-2649-2. — numérico.
22. **The pandas development team** — *pandas* (Zenodo DOI:10.5281/zenodo.3509134). —
    séries temporais.
23. **Virtanen, P. et al.** — *SciPy 1.0* (2020), Nature Methods 17.
    DOI:10.1038/s41592-019-0686-2. — L-BFGS-B e distribuições.
24. **Hunter, J. D.** — *Matplotlib* (2007), CiSE 9(3). — gráficos.
25. **yfinance** — https://github.com/ranaroussi/yfinance. [biblioteca comunitária,
    não oficial da Yahoo] — **Uso:** dados; adequada para pesquisa, não produção
    (limitação declarada).

## Critério de uso de referências

Nenhuma alteração metodológica foi feita com base em blog comercial. Cada melhoria
própria referencia a fonte, explica o problema que resolve, se a evidência é in-sample
ou out-of-sample, e o custo em hiperparâmetros adicionais (tabela na Seção 11 do
notebook).
