# Phase 3A Design Constraints

"PnL can only be computed from confirmed acquisition and disposal events with deterministic direction, amount, timestamp, token identity, and valuation basis."

## Allowed future outputs
- Eligibility flags and evidence-gap diagnostics.

## Disallowed future outputs
- Realized/unrealized PnL values, ROI, win rate, profitability labels.

## Minimum evidence standard
- Confirmed acquisition + disposal events with deterministic side and valuation basis.

## Cost basis standard
- Explicit reconstructable method is mandatory for realized modes.

## Trade-side standard
- Direction must be deterministic and directly evidenced.

## Transfer-handling standard
- Transfers/flows are movement evidence, not trade proof.

## CEX-flow-handling standard
- CEX counterparties do not prove trade intent without execution evidence.

## Balance-snapshot-handling standard
- Balance snapshots support exposure tracking only.

## HistoricalUSD-handling standard
- historicalUSD cannot be assumed to be cost basis.

## Rule for Phase 3B computation
- Phase 3B may compute only after trade-side + cost-basis evidence expansion closes blockers.
