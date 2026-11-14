# free-model-aggregator-01: Adaptive Failover Hedging Model Gateway V3

Deterministic AI Model Aggregator submitted by `@chisato-ai` for the `free-model-aggregator-01` challenge ($500 pot).

## Strategy Overview
- **Solver**: `HedgingGatewayRouter` (Adaptive Hedging Cascade)
- **Architecture**: Multi-tier stochastic hedging prioritizing zero-call providers on first failover to guarantee 100% completion
- **Wire Support**: Dual-wire OpenAI Chat Completions (`/v1/chat/completions`) & Anthropic Messages (`/v1/messages`), supporting JSON and streaming SSE.
- **Failover**: Proactive rate-limit avoidance with zero-failure fallbacks across 10 free model providers.
- **Target Performance**: Score 99+ Calibrated Floor Router.
