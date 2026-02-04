#!/usr/bin/env python3
"""
panda_phase3_2_recompute.py

PATCHED whale event recomputation matching DB semantics EXACTLY:
- One event per (wallet, window, direction, anchor_time) at FIRST threshold crossing
- After crossing at anchor_time, DO NOT include later same-time flows
- flow_ref = signature of the crossing flow (first flow where sum >= threshold)
- supporting_flow_count = count up to and including crossing flow
- sol_amount_lamports = sum up to and including crossing flow

This eliminates phantom events and amount/count mismatches from forensicsv4.
"""

import sqlite3
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import sys

# Constants - EXACT same as forensicsv4
SINGLE_TX_THRESHOLD = 10_000_000_000  # 10 SOL
CUM_24H_THRESHOLD = 50_000_000_000    # 50 SOL
CUM_7D_THRESHOLD = 200_000_000_000    # 200 SOL

WINDOW_SECONDS = {
    '24h': 86400,
    '7d': 604800,
    'lifetime': None
}


class FlowEvent:
    """Single wallet token flow."""
    
    def __init__(self, wallet: str, block_time: int, direction: str, 
                 amount_lamports: int, signature: str):
        self.wallet = wallet if wallet else ''
        self.block_time = block_time
        self.direction = direction.upper() if direction else ''
        self.amount_lamports = abs(amount_lamports) if amount_lamports else 0
        self.signature = signature if signature else ''
    
    def __repr__(self):
        return f"Flow({self.direction} {self.amount_lamports} @ {self.block_time} sig={self.signature[:8]})"


class WhaleEvent:
    """Whale event (baseline or recomputed)."""
    
    def __init__(self, wallet: str, window: str, event_type: str, 
                 event_time: int, flow_ref: str, amount: int, count: int):
        self.wallet = wallet if wallet else ''
        self.window = window if window else ''
        self.event_type = event_type if event_type else ''
        self.event_time = event_time if event_time else 0
        self.flow_ref = flow_ref if flow_ref else ''
        self.amount = amount if amount else 0
        self.count = count if count else 0
    
    def key(self) -> Tuple:
        """Return tuple key for comparison."""
        return (self.wallet, self.window, self.event_type, 
                self.event_time, self.flow_ref)
    
    def __repr__(self):
        return f"Whale({self.event_type} {self.window} @ {self.event_time})"


def load_baseline_events(conn: sqlite3.Connection) -> Dict[Tuple, WhaleEvent]:
    """Load baseline whale_events from DB."""
    print("\n" + "="*80)
    print("LOADING BASELINE WHALE_EVENTS")
    print("="*80)
    
    query = """
        SELECT wallet, window, event_type, event_time, flow_ref,
               sol_amount_lamports, supporting_flow_count
        FROM whale_events
        ORDER BY wallet, event_time, window, event_type
    """
    
    cursor = conn.execute(query)
    events = {}
    
    for row in cursor:
        event = WhaleEvent(
            wallet=row[0],
            window=row[1],
            event_type=row[2],
            event_time=row[3],
            flow_ref=row[4],
            amount=row[5],
            count=row[6]
        )
        events[event.key()] = event
    
    print(f"Total baseline events: {len(events)}")
    
    # Count by (window, event_type)
    counts = defaultdict(int)
    for event in events.values():
        counts[(event.window, event.event_type)] += 1
    
    print("\nCounts by (window, event_type):")
    for (window, event_type), count in sorted(counts.items()):
        print(f"  {window:10s} {event_type:25s} : {count:6d}")
    
    return events


def load_wallet_flows(conn: sqlite3.Connection) -> Dict[str, List[FlowEvent]]:
    """Load wallet token flows with DETERMINISTIC ordering."""
    print("\n" + "="*80)
    print("LOADING WALLET_TOKEN_FLOW")
    print("="*80)
    
    # CRITICAL: Sort by (scan_wallet, block_time ASC, signature ASC)
    query = """
        SELECT scan_wallet, block_time, sol_direction, sol_amount_lamports, signature
        FROM wallet_token_flow
        ORDER BY scan_wallet, block_time ASC, signature ASC
    """
    
    cursor = conn.execute(query)
    flows_by_wallet = defaultdict(list)
    total_flows = 0
    
    for row in cursor:
        wallet, block_time, direction, amount, signature = row
        
        direction = direction.upper() if direction else ''
        if direction not in ('BUY', 'SELL'):
            continue
        
        flow = FlowEvent(wallet, block_time, direction, amount, signature)
        flows_by_wallet[wallet].append(flow)
        total_flows += 1
    
    print(f"Total flows loaded: {total_flows}")
    print(f"Distinct wallets: {len(flows_by_wallet)}")
    
    return flows_by_wallet


def compute_strict_whale_events(flows_by_wallet: Dict[str, List[FlowEvent]]) -> Dict[Tuple, WhaleEvent]:
    """
    STRICT recomputation matching DB semantics EXACTLY.
    
    Key rules:
    1. Sort flows by (block_time ASC, signature ASC) within wallet+direction
    2. For each potential anchor time, scan flows in window
    3. Accumulate sum/count in order
    4. At FIRST crossing (sum >= threshold):
       - Emit event with crossing flow's signature
       - Stop including further flows at same timestamp
    5. Dedupe: one event per (wallet, window, event_type, event_time)
    """
    events = {}
    emitted_keys = set()  # Track (wallet, window, event_type, event_time) to dedupe
    
    total_wallets = len(flows_by_wallet)
    
    for wallet_idx, (wallet, all_flows) in enumerate(flows_by_wallet.items(), 1):
        if wallet_idx % 25 == 0 or wallet_idx == total_wallets:
            pct = (wallet_idx / total_wallets) * 100
            print(f"  Processing wallet {wallet_idx}/{total_wallets} ({pct:.1f}%)...", flush=True)
        
        # Separate by direction and sort deterministically
        for direction in ['BUY', 'SELL']:
            dir_flows = [f for f in all_flows if f.direction == direction]
            if not dir_flows:
                continue
            
            # Sort by (block_time ASC, signature ASC) - CRITICAL for determinism
            dir_flows.sort(key=lambda f: (f.block_time, f.signature))
            
            # Track which anchor times we've already emitted events for
            emitted_anchor_times_24h = set()
            emitted_anchor_times_7d = set()
            
            # Process each flow as potential anchor
            for anchor_idx, anchor in enumerate(dir_flows):
                anchor_time = anchor.block_time
                anchor_sig = anchor.signature
                
                # SINGLE-TX events (lifetime window)
                if anchor.amount_lamports >= SINGLE_TX_THRESHOLD:
                    event_type = f"WHALE_TX_{direction}"
                    dedupe_key = (wallet, 'lifetime', event_type, anchor_time)
                    
                    if dedupe_key not in emitted_keys:
                        # For single-tx, if multiple same-time flows cross threshold,
                        # emit for the FIRST one only
                        event = WhaleEvent(
                            wallet=wallet,
                            window='lifetime',
                            event_type=event_type,
                            event_time=anchor_time,
                            flow_ref=anchor_sig,
                            amount=anchor.amount_lamports,
                            count=1
                        )
                        events[event.key()] = event
                        emitted_keys.add(dedupe_key)
                
                # CUMULATIVE 24h
                if anchor_time not in emitted_anchor_times_24h:
                    result = find_first_crossing_in_window(
                        dir_flows, anchor_time, 
                        WINDOW_SECONDS['24h'], CUM_24H_THRESHOLD
                    )
                    
                    if result:
                        crossing_sig, crossing_amount, crossing_count = result
                        event_type = f"WHALE_CUM_24H_{direction}"
                        dedupe_key = (wallet, '24h', event_type, anchor_time)
                        
                        if dedupe_key not in emitted_keys:
                            event = WhaleEvent(
                                wallet=wallet,
                                window='24h',
                                event_type=event_type,
                                event_time=anchor_time,
                                flow_ref=crossing_sig,
                                amount=crossing_amount,
                                count=crossing_count
                            )
                            events[event.key()] = event
                            emitted_keys.add(dedupe_key)
                            emitted_anchor_times_24h.add(anchor_time)
                
                # CUMULATIVE 7d
                if anchor_time not in emitted_anchor_times_7d:
                    result = find_first_crossing_in_window(
                        dir_flows, anchor_time,
                        WINDOW_SECONDS['7d'], CUM_7D_THRESHOLD
                    )
                    
                    if result:
                        crossing_sig, crossing_amount, crossing_count = result
                        event_type = f"WHALE_CUM_7D_{direction}"
                        dedupe_key = (wallet, '7d', event_type, anchor_time)
                        
                        if dedupe_key not in emitted_keys:
                            event = WhaleEvent(
                                wallet=wallet,
                                window='7d',
                                event_type=event_type,
                                event_time=anchor_time,
                                flow_ref=crossing_sig,
                                amount=crossing_amount,
                                count=crossing_count
                            )
                            events[event.key()] = event
                            emitted_keys.add(dedupe_key)
                            emitted_anchor_times_7d.add(anchor_time)
    
    return events


def find_first_crossing_in_window(flows: List[FlowEvent], anchor_time: int, 
                                   window_seconds: int, 
                                   threshold: int) -> Optional[Tuple[str, int, int]]:
    """
    Find FIRST threshold crossing in window [anchor_time - window_seconds, anchor_time].
    
    CRITICAL RULES (matching DB semantics):
    1. Flows must be sorted by (block_time ASC, signature ASC)
    2. Accumulate sum/count in order
    3. Stop at FIRST flow where sum >= threshold
    4. Record crossing flow signature, amount, and count at that point
    5. STOP processing - do not include any more flows after crossing
    
    Returns: (crossing_signature, amount_at_crossing, count_at_crossing) or None
    """
    start_time = anchor_time - window_seconds
    
    running_sum = 0
    running_count = 0
    
    # Scan all flows in chronological order
    for flow in flows:
        # Only consider flows in window [start_time, anchor_time] inclusive
        if flow.block_time < start_time:
            continue
        if flow.block_time > anchor_time:
            break
        
        # Add this flow to running totals
        running_sum += flow.amount_lamports
        running_count += 1
        
        # Check if we just crossed threshold
        if running_sum >= threshold:
            # FIRST crossing - return immediately
            return (flow.signature, running_sum, running_count)
    
    # Never crossed threshold
    return None


def compare_events(baseline: Dict[Tuple, WhaleEvent], 
                  recomputed: Dict[Tuple, WhaleEvent]) -> Dict:
    """Compare baseline vs recomputed events."""
    baseline_keys = set(baseline.keys())
    recomputed_keys = set(recomputed.keys())
    
    common_keys = baseline_keys & recomputed_keys
    missing_keys = baseline_keys - recomputed_keys
    phantom_keys = recomputed_keys - baseline_keys
    
    amount_mismatches = set()
    count_mismatches = set()
    
    for key in common_keys:
        b_event = baseline[key]
        r_event = recomputed[key]
        
        if b_event.amount != r_event.amount:
            amount_mismatches.add(key)
        if b_event.count != r_event.count:
            count_mismatches.add(key)
    
    mismatched_keys = amount_mismatches | count_mismatches
    perfect_matches = common_keys - mismatched_keys
    
    return {
        'recomputed_total': len(recomputed),
        'common_keys': len(common_keys),
        'missing_keys': missing_keys,
        'phantom_keys': phantom_keys,
        'amount_mismatches': amount_mismatches,
        'count_mismatches': count_mismatches,
        'perfect_matches': len(perfect_matches),
        'total_errors': len(missing_keys) + len(phantom_keys) + len(mismatched_keys)
    }


def print_comparison(baseline: Dict[Tuple, WhaleEvent],
                    recomputed: Dict[Tuple, WhaleEvent],
                    comparison: Dict,
                    sample_size: int = 20):
    """Print detailed comparison results."""
    print("\n" + "="*80)
    print("STRICT RECOMPUTE COMPARISON")
    print("="*80)
    
    print(f"\nBaseline events:   {len(baseline):,}")
    print(f"Recomputed events: {comparison['recomputed_total']:,}")
    print(f"Common keys:       {comparison['common_keys']:,}")
    print(f"Perfect matches:   {comparison['perfect_matches']:,}")
    
    # Calculate success rate
    if len(baseline) > 0:
        success_rate = (comparison['perfect_matches'] / len(baseline)) * 100
        print(f"SUCCESS RATE:      {success_rate:.2f}%")
    
    print(f"\nERRORS:")
    print(f"  Missing:         {len(comparison['missing_keys']):,}")
    print(f"  Phantom:         {len(comparison['phantom_keys']):,}")
    print(f"  Amount mismatch: {len(comparison['amount_mismatches']):,}")
    print(f"  Count mismatch:  {len(comparison['count_mismatches']):,}")
    print(f"  TOTAL ERRORS:    {comparison['total_errors']:,}")
    
    # Recomputed counts by (window, event_type)
    print("\n" + "="*80)
    print("RECOMPUTED COUNTS BY (WINDOW, EVENT_TYPE)")
    print("="*80)
    
    counts = defaultdict(int)
    for event in recomputed.values():
        counts[(event.window, event.event_type)] += 1
    
    for (window, event_type), count in sorted(counts.items()):
        print(f"  {window:10s} {event_type:25s} : {count:6d}")
    
    # Sample errors
    if comparison['missing_keys']:
        print(f"\n--- MISSING EVENTS (first {min(sample_size, len(comparison['missing_keys']))}) ---")
        for i, key in enumerate(sorted(comparison['missing_keys'])[:sample_size]):
            event = baseline[key]
            wallet_trunc = event.wallet[:12] + '...' if len(event.wallet) > 12 else event.wallet
            ref_trunc = event.flow_ref[:12] + '...' if len(event.flow_ref) > 12 else event.flow_ref
            print(f"{i+1}. {wallet_trunc:16s} | {event.window:8s} | {event.event_type:25s} | "
                  f"time={event.event_time} | ref={ref_trunc:16s} | "
                  f"amt={event.amount:,} | cnt={event.count}")
    
    if comparison['phantom_keys']:
        print(f"\n--- PHANTOM EVENTS (first {min(sample_size, len(comparison['phantom_keys']))}) ---")
        for i, key in enumerate(sorted(comparison['phantom_keys'])[:sample_size]):
            event = recomputed[key]
            wallet_trunc = event.wallet[:12] + '...' if len(event.wallet) > 12 else event.wallet
            ref_trunc = event.flow_ref[:12] + '...' if len(event.flow_ref) > 12 else event.flow_ref
            print(f"{i+1}. {wallet_trunc:16s} | {event.window:8s} | {event.event_type:25s} | "
                  f"time={event.event_time} | ref={ref_trunc:16s} | "
                  f"amt={event.amount:,} | cnt={event.count}")
    
    if comparison['amount_mismatches']:
        print(f"\n--- AMOUNT MISMATCHES (first {min(sample_size, len(comparison['amount_mismatches']))}) ---")
        for i, key in enumerate(sorted(comparison['amount_mismatches'])[:sample_size]):
            b_event = baseline[key]
            r_event = recomputed[key]
            wallet_trunc = b_event.wallet[:12] + '...' if len(b_event.wallet) > 12 else b_event.wallet
            ref_trunc = b_event.flow_ref[:12] + '...' if len(b_event.flow_ref) > 12 else b_event.flow_ref
            diff = r_event.amount - b_event.amount
            print(f"{i+1}. {wallet_trunc:16s} | {b_event.window:8s} | {b_event.event_type:25s} | "
                  f"time={b_event.event_time} | ref={ref_trunc:16s}")
            print(f"    Baseline:    amt={b_event.amount:15,} cnt={b_event.count:4d}")
            print(f"    Recomputed:  amt={r_event.amount:15,} cnt={r_event.count:4d}")
            print(f"    Diff:        amt={diff:+15,} cnt={r_event.count - b_event.count:+4d}")
    
    # Count-only mismatches (amount matches but count differs)
    count_only = comparison['count_mismatches'] - comparison['amount_mismatches']
    if count_only:
        print(f"\n--- COUNT-ONLY MISMATCHES (first {min(sample_size, len(count_only))}) ---")
        for i, key in enumerate(sorted(count_only)[:sample_size]):
            b_event = baseline[key]
            r_event = recomputed[key]
            wallet_trunc = b_event.wallet[:12] + '...' if len(b_event.wallet) > 12 else b_event.wallet
            ref_trunc = b_event.flow_ref[:12] + '...' if len(b_event.flow_ref) > 12 else b_event.flow_ref
            diff = r_event.count - b_event.count
            print(f"{i+1}. {wallet_trunc:16s} | {b_event.window:8s} | {b_event.event_type:25s} | "
                  f"time={b_event.event_time} | ref={ref_trunc:16s}")
            print(f"    Baseline:    amt={b_event.amount:15,} cnt={b_event.count:4d}")
            print(f"    Recomputed:  amt={r_event.amount:15,} cnt={r_event.count:4d}")
            print(f"    Diff:        amt=              0 cnt={diff:+4d}")


def main():
    parser = argparse.ArgumentParser(
        description='STRICT whale event recomputation matching DB semantics'
    )
    parser.add_argument('--db', required=True, help='Path to SQLite database')
    parser.add_argument('--sample', type=int, default=20, 
                       help='Number of sample errors to display')
    
    args = parser.parse_args()
    
    # Connect to database
    try:
        conn = sqlite3.connect(args.db)
    except Exception as e:
        print(f"ERROR: Failed to connect to database: {e}")
        sys.exit(1)
    
    # Load baseline
    baseline = load_baseline_events(conn)
    if not baseline:
        print("ERROR: No baseline events found")
        sys.exit(1)
    
    # Load flows
    flows_by_wallet = load_wallet_flows(conn)
    if not flows_by_wallet:
        print("ERROR: No wallet flows found")
        sys.exit(1)
    
    # Recompute with STRICT semantics
    print("\n" + "="*80)
    print("RECOMPUTING WITH STRICT FIRST-CROSS SEMANTICS")
    print("="*80)
    print("Rules:")
    print("  - Sort flows by (block_time ASC, signature ASC)")
    print("  - Emit at FIRST threshold crossing only")
    print("  - Do NOT include post-cross same-time flows")
    print("  - flow_ref = crossing flow signature")
    print("  - amount/count = values at crossing")
    print()
    
    recomputed = compute_strict_whale_events(flows_by_wallet)
    
    # Compare
    comparison = compare_events(baseline, recomputed)
    
    # Print results
    print_comparison(baseline, recomputed, comparison, args.sample)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    
    # Print actionable summary
    if len(baseline) > 0:
        success_rate = (comparison['perfect_matches'] / len(baseline)) * 100
        
        print(f"\nSUMMARY:")
        print(f"  {comparison['perfect_matches']:,} / {len(baseline):,} events matched perfectly ({success_rate:.2f}%)")
        
        if success_rate >= 99.9:
            print("\n  ✓ EXCELLENT! The strict first-cross semantics match your DB nearly perfectly.")
            print("    Remaining errors are likely due to:")
            print("    - Edge cases with exact timestamp boundaries")
            print("    - Floating point precision in amounts")
            print("    - Race conditions in original data capture")
        elif success_rate >= 95.0:
            print("\n  ✓ GOOD! The strict first-cross semantics are very close.")
            print("    Review remaining mismatches to identify additional semantic differences.")
        elif success_rate >= 80.0:
            print("\n  ⚠ PARTIAL MATCH. First-cross semantics help but aren't complete.")
            print("    There may be additional logic differences to identify.")
        else:
            print("\n  ✗ LOW MATCH. The DB uses significantly different semantics.")
            print("    Review missing/phantom events to understand the actual logic.")
    
    conn.close()


if __name__ == '__main__':
    main()
