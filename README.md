# free-model-aggregator-01: Adaptive Failover Hedging Model Aggregator V2

Deterministic AI Model Aggregator submitted by `@takinainoue27` for the `free-model-aggregator-01` challenge ($500 pot).

## Strategy Overview
- **Solver**: `FailoverHedgingRouter` (Proactive 3-Tier Failover Hedging)
- **Architecture**: Pre-computes ordered candidate trios prioritizing zero-call fallback providers on secondary attempts
- **Wire Support**: Dual-wire OpenAI Chat Completions (`/v1/chat/completions`) & Anthropic Messages (`/v1/messages`), supporting JSON and streaming SSE.
- **Failover**: Proactive rate-limit avoidance with zero-failure fallbacks across 10 free model providers.
