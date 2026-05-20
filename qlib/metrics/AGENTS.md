Keep growth and performance separate.

Growth is about how a thing changed against itself over time: CAGR, trailing
CAGR, NAV growth, revenue growth, BTC-per-share growth, that kind of thing.

Performance is about judging a return stream: Sharpe, Sortino, drawdown,
benchmark-relative stuff, or anything that asks whether the path was good,
risky, smooth, or ugly.

If a metric is just start value to end value, it probably belongs in
`growth.py`. If it needs periodic returns or a risk/quality standard, it
probably belongs in `performance.py`.
