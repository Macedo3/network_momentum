# Limitações

Lista honesta do que este projeto **não** resolve. Nenhum item abaixo é escondido nos
resultados; os que são quantificáveis têm ablação ou tabela correspondente.

## Dados

1. **Viés de sobrevivência (o mais grave).** O universo é uma lista estática de 36
   ações grandes *hoje*. Empresas deslistadas, adquiridas ou que encolheram não estão
   lá. Todo o desempenho reportado herda esse viés para cima. Correção exigiria
   universo point-in-time (CRSP, Datastream) — indisponível gratuitamente. A ablação
   "remoção dos melhores ativos" dimensiona a sensibilidade; não corrige o viés.
2. **Yahoo Finance / yfinance.** Preços ajustados podem ser revisados retroativamente;
   qualidade heterogênea fora dos EUA; limites de acesso; biblioteca comunitária sem
   SLA. Adequado para pesquisa; inadequado para produção.
3. **Universo pequeno (36 ativos).** O artigo usa 64 futuros; redes com poucos nós têm
   topologia menos informativa e a OLS transversal tem menos amostras por data.
4. **HSBC em duas bolsas** (HSBA.L, 0005.HK). Mesma empresa; o grafo cria aresta forte
   entre elas (informação real de dual listing, mas duplica risco idiossincrático).
   Mantida por decisão de universo; quantificada por ablação de ativos.
5. **Sem dados intradiários.** Spread e impacto são cenários metodológicos; não há
   medição de fill real. O carimbo "estimativa" acompanha esses números em todo lugar.
6. **Proxies de benchmark.** ETFs (ACWI/SPY/EFA/EEM) via adjusted close aproximam
   índices de retorno total; o índice oficial MSCI ACWI TR não é distribuído
   gratuitamente. O Ibovespa é TR por construção, em BRL (convertido para USD).

## Metodologia

7. **Solver do grafo.** O default (L-BFGS-B) resolve um problema matematicamente
   equivalente à Eq. (4), mas não produz zeros exatos; a esparsidade reportada depende
   de um threshold declarado. O caminho `cvxpy` (CLARABEL/SCS) resolve o problema
   exato; MOSEK (usado no artigo) não é exigido.
8. **Grade de hiperparâmetros reduzida.** 3×3 por padrão vs 11×11 do artigo (custo
   computacional). A grade completa está documentada e é executável conscientemente.
9. **Grafo mensal no perfil prático** (diário no artigo e no perfil paper_structure).
   O Jaccard >0,99 entre grafos diários sugere impacto pequeno; medido, não assumido.
10. **Proxy hedgeada ignora o custo do hedge.** A série "local" não desconta o
    diferencial de juros dos forwards — para BRL/INR isso é material. A série é um
    limite superior declarado da carteira hedgeada real.
11. **Ablações sem reajuste.** Remoções de ativos/regiões da agregação não reestimam
    grafo e regressão (aproximação declarada nas tabelas).
12. **PBO sobre a família de estratégias**, não sobre a grade (α, β) — não guardamos
    séries OOS por candidato de grade (custo); o DSR usa a contagem total de
    configurações como n_trials.

## Execução

13. **Custos parcialmente estimados.** Tributos de UK/HK verificados em fonte oficial
    nesta data; Índia/França bem estabelecidos mas com flag `verify`; corretagem,
    spread, borrow e impacto são estimativas institucionais por cenário.
14. **Short em mercados restritos.** Índia (SLB restrito a estrangeiros) e custos de
    aluguel no Brasil variam por papel/dia — borrow usa estimativas conservadoras.
15. **Execução multi-fuso.** O lag de 1 dia elimina o look-ahead informacional, mas a
    execução real no fechamento de 11 bolsas exige infraestrutura não modelada.
16. **Volatility targeting com estimador defasado** — implementável, porém difere do
    "reescalonamento para 15%" (provavelmente ex-post) do Painel B do artigo; números
    do Painel B do paper não são diretamente comparáveis.

## Estatística

17. **OOS sem bear market secular.** 2015–presente não contém um urso de múltiplos
    anos (2000–02, 2008). Momentum historicamente sofre em reversões (momentum crash);
    janelas de crise curtas (2020, 2022) estão nas tabelas de regime, mas o risco de
    cauda longa não está representado no período.
18. **Testes múltiplos residuais.** Mesmo com DSR/PBO/BH, o processo de pesquisa
    (deste projeto e do próprio artigo) envolve escolhas não contabilizáveis
    (features herdadas da literatura, lookbacks "redondos" etc.).
