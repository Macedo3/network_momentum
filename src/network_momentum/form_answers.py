from __future__ import annotations

"""Geração automática das respostas do formulário da competição (perguntas 9–15).

As respostas são preenchidas com os valores da EXECUÇÃO (universo, período,
frequência, benchmark), nunca com números do artigo. Duas versões: curta
(para colar no Microsoft Forms) e técnica (para o relatório).
"""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class RunFacts:
    n_assets: int
    n_regions: int
    regions: tuple[str, ...]
    exchanges: tuple[str, ...]
    currencies: tuple[str, ...]
    oos_start: str
    oos_end: str
    base_currency: str
    target_volatility: float
    rebalance_frequency: str
    graph_lookbacks: tuple[int, ...]
    net_sharpe: float | None
    benchmark_names: tuple[str, ...]


def build_run_facts(
    universe: pd.DataFrame,
    daily_returns: pd.DataFrame,
    *,
    base_currency: str,
    target_volatility: float,
    graph_lookbacks: tuple[int, ...],
    rebalance_every: int,
    benchmark_names: tuple[str, ...],
    net_sharpe: float | None = None,
) -> RunFacts:
    frequency = {1: "diária"}.get(rebalance_every, f"a cada {rebalance_every} pregões")
    return RunFacts(
        n_assets=int(universe.shape[0]),
        n_regions=int(universe["region"].nunique()),
        regions=tuple(sorted(universe["region"].unique())),
        exchanges=tuple(sorted(universe["exchange"].unique())),
        currencies=tuple(sorted(universe["currency"].unique())),
        oos_start=str(daily_returns.index.min().date()),
        oos_end=str(daily_returns.index.max().date()),
        base_currency=base_currency,
        target_volatility=target_volatility,
        rebalance_frequency=frequency,
        graph_lookbacks=graph_lookbacks,
        net_sharpe=net_sharpe,
        benchmark_names=benchmark_names,
    )


def form_answers(facts: RunFacts) -> dict[str, dict[str, str]]:
    robot_name = "NEXO — Network Momentum Global Equities"
    lookbacks = ", ".join(str(x) for x in facts.graph_lookbacks)
    regions = ", ".join(facts.regions)

    short = {
        "9. Nome do Robô": robot_name,
        "10. Explicação do Nome": (
            "NEXO remete a 'nexo/conexão': a estratégia opera o momentum que se propaga "
            "pelas conexões (arestas) de um grafo de ações aprendido dos dados."
        ),
        "11. Lógica da Estratégia": (
            "Long-short sistemático de momentum em rede (Pu et al., arXiv:2308.11294) adaptado a ações globais: "
            "8 sinais de momentum (retornos ajustados por volatilidade e MACD) alimentam um grafo aprendido "
            f"por otimização convexa (lookbacks de {lookbacks} pregões); os sinais dos vizinhos são propagados "
            "e uma regressão linear transversal prevê o retorno do dia seguinte ajustado por volatilidade. "
            "Posição = sinal da previsão, com peso inverso à volatilidade (alvo "
            f"{facts.target_volatility:.0%} a.a.), walk-forward expansivo sem look-ahead e custos por bolsa."
        ),
        "12. Classe de Ativos": (
            "Ações (equities) listadas — apenas ações à vista; sem futuros, FX ou renda fixa "
            "(o artigo original usa futuros multiativos; esta implementação executa somente ações)."
        ),
        "13. Universo de Investimento": (
            f"{facts.n_assets} ações de grande capitalização em {facts.n_regions} regiões ({regions}), "
            f"moedas {', '.join(facts.currencies)}, moeda-base {facts.base_currency}. "
            "Universo estático definido em config/universe.csv — sujeito a viés de sobrevivência (declarado)."
        ),
        "14. Frequência da Estratégia": (
            f"Sinais e rebalanceamento diários (posições recalculadas a cada pregão); "
            f"o grafo é reestimado {facts.rebalance_frequency}."
        ),
        "15. Benchmark": (
            f"Principal: MSCI ACWI via proxy investível (ETF ACWI, retorno total, {facts.base_currency}). "
            "Metodológicos: Long Only vol-scaled, Equal Weight, MACD e LinReg (momentum individual) no mesmo universo."
        ),
    }

    technical = {
        "9. Nome do Robô": robot_name,
        "10. Explicação do Nome": (
            "O nome resume o mecanismo econômico central: momentum spillover. Em vez de operar apenas o momentum "
            "próprio de cada ação, o robô aprende, por otimização convexa (Eq. 4 do artigo), uma rede de similaridade "
            "entre ativos e opera o momentum que 'flui' pelos nexos dessa rede — daí NEXO."
        ),
        "11. Lógica da Estratégia": (
            "Pipeline (Pu, Roberts, Dong e Zohren, 2023): (i) oito features de momentum por ativo — retornos de "
            "1/21/63/126/252 pregões normalizados por vol EWM (span 60) e MACD normalizado (8,24)/(16,48)/(32,96) — "
            "winsorizadas em ±5σ EWM (meia-vida 252); (ii) grafo simétrico, não negativo e sem autoarestas aprendido "
            f"da Eq. (4) com ensemble de lookbacks ({lookbacks} pregões, Eq. 5) e normalização espectral (Eq. 6); "
            "(iii) propagação das features pelos vizinhos (Eq. 7); (iv) OLS transversal pooled prevendo o retorno de "
            "t+1 ajustado por volatilidade (Eq. 8), reestimada em walk-forward expansivo com validação interna para "
            "(α, β); (v) posição = sign(previsão), peso σ_alvo/σ_i (Eq. 9), camada de vol targeting de portfólio "
            f"({facts.target_volatility:.0%} a.a.) com teto de alavancagem; (vi) custos por bolsa (corretagem, "
            "emolumentos, tributos como SDRT/FTT/STT/stamp, spread por cenário, borrow) aplicados sobre a variação "
            "efetiva dos pesos (Eqs. 13–14). Sem dados futuros: features defasadas 1 pregão para lidar com fusos."
        ),
        "12. Classe de Ativos": (
            "Somente ações à vista (cash equities) de bolsas desenvolvidas e emergentes. A versão executada NÃO é "
            "multiativos: a adaptação de futuros (artigo) para ações é uma decisão declarada do projeto, com efeitos "
            "sobre custos (tributos por bolsa, borrow para short) e sobre a comparabilidade com o paper."
        ),
        "13. Universo de Investimento": (
            f"{facts.n_assets} ações de alta capitalização/liquidez em {facts.n_regions} regiões ({regions}); "
            f"bolsas: {', '.join(facts.exchanges)}; moedas: {', '.join(facts.currencies)}. Resultados na moeda-base "
            f"{facts.base_currency} (sem hedge) e em moeda local (proxy hedgeada aproximada). Limitação declarada: "
            "lista estática atual (sem point-in-time) → viés de sobrevivência quantificado por ablação no relatório. "
            f"Período fora da amostra: {facts.oos_start} a {facts.oos_end}."
        ),
        "14. Frequência da Estratégia": (
            "Estratégia diária: previsão e reponderação a cada pregão (posições mudam quando o sinal cruza zero ou a "
            f"volatilidade se move). Grafo reestimado {facts.rebalance_frequency} "
            f"(no perfil fiel ao artigo, diariamente). Horizonte da previsão: 1 pregão."
        ),
        "15. Benchmark": (
            "Externo principal: MSCI ACWI Total Return via proxy investível (ETF ACWI em USD; série de índice TR "
            "oficial não é distribuída gratuitamente — diferença declarada). Complementares: S&P 500 TR (SPY), "
            "EAFE (EFA), EM (EEM) e Ibovespa (índice de retorno total em BRL, convertido para USD). Benchmarks "
            "metodológicos no mesmo universo e com a mesma Eq. (9): Long Only vol-scaled, Equal Weight, MACD "
            "(Eq. 10), LinReg individual (Eq. 11), RegCombo (Eq. 12) e SignCombo — comparações sempre no mesmo "
            "período, moeda e metodologia de métricas, bruto e líquido."
        ),
    }
    if facts.net_sharpe is not None:
        technical["15. Benchmark"] += (
            f" Sharpe líquido da estratégia no período avaliado: {facts.net_sharpe:.2f} "
            "(cenário base de custos; ver tabelas de sensibilidade)."
        )
    return {"curta": short, "tecnica": technical}


def answers_markdown(answers: dict[str, dict[str, str]]) -> str:
    lines = ["# Respostas do Formulário", ""]
    for version, block in (("Versão curta (Microsoft Forms)", answers["curta"]),
                           ("Versão técnica (relatório)", answers["tecnica"])):
        lines.append(f"## {version}")
        lines.append("")
        for question, answer in block.items():
            lines.append(f"**{question}**")
            lines.append("")
            lines.append(answer)
            lines.append("")
    return "\n".join(lines)
