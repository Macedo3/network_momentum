# Melhorias futuras

Ordenadas por relação impacto/custo. Cada uma indica o problema que ataca e o risco
metodológico que adiciona.

## Dados
1. **Universo point-in-time** (constituintes históricos de índices; delistings com
   retorno de saída). Ataca a limitação nº 1 (sobrevivência). Risco: nenhum; custo:
   fonte paga.
2. **Ampliar o universo** (100–300 ações líquidas por região, com filtros de ADV).
   Melhora a rede e a OLS transversal. Risco: custos de short/borrow menos uniformes.
3. **Fonte de dados de produção** (EODHD, Refinitiv, Bloomberg) com auditoria de
   eventos corporativos.

## Metodologia
4. **Turnover regularization** na regressão (Lim et al., 2019) — o próprio artigo
   deixa como trabalho futuro. Reduz custos na fonte. Risco: +1 hiperparâmetro.
5. **Grade (α, β) completa 11×11 com paralelização** (joblib por candidato) e PBO
   sobre a grade (guardando séries de validação por candidato).
6. **Hedge cambial explícito** com forwards sintéticos (juros locais via proxies) —
   transforma a "proxy hedgeada" em estimativa com custo de carry.
7. **Grafos direcionados / lead-lag** (assimetria de spillover) — literatura de
   momentum spillover sugere direção; muda a Eq. (4) (perde simetria). Risco: mais
   parâmetros e perda da convexidade original.
8. **Desempate estatístico GMOM vs LinReg** com teste de Diebold-Mariano/White reality
   check sobre as séries diárias.

## Execução
9. **Calibração de impacto com dados intradiários** (substituir cenários por medição).
10. **Agendamento de execução multi-fuso** (rebalanceamento regional escalonado; medir
    o custo do atraso adicional por região).
11. **Borrow real por papel** (feeds de securities lending) para o custo de short.

## Engenharia
12. **Cache Parquet + versionamento de dados** (DVC ou hash por partição).
13. **CI (GitHub Actions)** rodando a suíte + smoke pipeline a cada commit.
14. **Paralelizar o aprendizado dos grafos** por snapshot (processos) — reduz o custo
    do perfil paper_structure (grafo diário).

## Desafiadores (sem alterar a identidade da estratégia)
15. **Modelos não lineares como challengers separados** (GBM/rede rasa sobre as mesmas
    8+8 features, mesmo protocolo temporal) — comparação, nunca substituição
    silenciosa da OLS. Risco: data mining; exigiria DSR/PBO próprios.
