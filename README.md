# Crypto Swing Bot — Laboratório de Backtesting e Estratégias

Projeto estruturado para validar estatisticamente estratégias de **Swing Trade em Cripto** (Spot CEX, timeframes `4h` e `1d`) antes de qualquer execução em conta demo ou capital real.

## Estrutura do Projeto

- [`data_loader.py`](data_loader.py): Coleta histórica paginada da API pública da Binance (`data-api.binance.vision`) com cache local em `data/`.
- [`indicators.py`](indicators.py): Indicadores técnicos (SMA, EMA, RSI, ATR, ADX, Donchian, Supertrend) calculados sem *look-ahead bias*.
- [`backtest_engine.py`](backtest_engine.py): Simulador causal para ativo único com taxas (0.10%), *slippage* (0.05%) e prioridade conservadora de Stop Loss sobre Take Profit intrabar.
- [`portfolio_backtest.py`](portfolio_backtest.py): Simulador multi-ativo com banca compartilhada em USDT e limite de posições simultâneas.
- [`strategies.py`](strategies.py): Estratégias clássicas de Swing Trade com filtro de regime macro do Bitcoin (`BTC > SMA200`).
- [`run_backtest.py`](run_backtest.py): Benchmark histórico completo (2021–2026) por portfólio, ano a ano e por ativo.
- [`validate_refinements.py`](validate_refinements.py): Estudo anti-overfitting em 16 ativos (Tendência Própria, Risk Sizing, Controle de Volatilidade, *Out-of-Sample* 2024–2026, platô e *stress test*).
- [`config.py`](config.py), [`market_api.py`](market_api.py), [`telegram_client.py`](telegram_client.py): Configuração, leitor público em tempo real da Binance e cliente Telegram.
- [`paper_portfolio.py`](paper_portfolio.py): Contabilidade única (*Single Source of Truth*) em `paper_state.json`.
- [`paper_trader.py`](paper_trader.py): Motor de Paper Trading (`US$ 300` virtuais, monitoramento de stop a cada 60s, avaliação diária às 21h BRT e comandos no Telegram).
- [`telegram_webapp.py`](telegram_webapp.py) e [`twa/index.html`](twa/index.html): Servidor do Telegram Mini App + Dashboard híbrido (Local e GitHub Pages com overlay de preços ao vivo da Binance).
- [`cloud_runner.py`](cloud_runner.py) e [`.github/workflows/swingbot.yml`](.github/workflows/swingbot.yml): Execução 100% na nuvem via GitHub Actions + deploy automático no GitHub Pages.
- [`run_swingbot.py`](run_swingbot.py) e [`Iniciar_SwingBot_Paper.bat`](Iniciar_SwingBot_Paper.bat): Inicializador unificado local.
- [`test_engine_integrity.py`](test_engine_integrity.py) e [`test_paper_trader.py`](test_paper_trader.py): Testes automatizados de causalidade, contabilidade, nuvem e API.

## Como Executar

```powershell
.\.venv\Scripts\python.exe -X utf8 test_engine_integrity.py
.\.venv\Scripts\python.exe -X utf8 test_paper_trader.py
.\.venv\Scripts\python.exe -X utf8 run_swingbot.py
```
