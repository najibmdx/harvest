#!/usr/bin/env python3
"""
panda_phase3_2_semantics.py

Deterministically infer the exact semantics used to generate DB table `whale_events`
from `wallet_token_flow`, by testing candidate emission + flow_ref + window boundary
rules and ranking them by match quality.

Usage:
    python panda_phase3_2_semantics.py --db masterwalletsdb.db
    python panda_phase3_2_semantics.py --db masterwalletsdb.db --fast
    python panda_phase3_2_semantics.py --db masterwalletsdb.db --fast-all
    python panda_phase3_2_semantics.py --db masterwalletsdb.db --limit-wallets 10
"""

import sqlite3
import argparse
import os
import sys
from collections import defaultdict, deque
from datetime import datetime, timedelta
from itertools import product

# --- Constants ---
THRESHOLDS = {
    'TX': 10_000_000_000,
    'CUM_24H': 50_000_000_000,
    'CUM_7D': 200_000_000_000,
}

WINDOW_SECONDS = {
    '24h': 24 * 3600,
    '7d': 7 * 24 * 3600,
    'lifetime': None,  # Special handling
}

EVENT_TYPE_MAP = {
    ('24h', 'BUY'): 'WHALE_CUM_24H_BUY',
    ('24h', 'SELL'): 'WHALE_CUM_24H_SELL',
    ('7d', 'BUY'): 'WHALE_CUM_7D_BUY',
    ('7d', 'SELL'): 'WHALE_CUM_7D_SELL',
    ('lifetime', 'BUY'): 'WHALE_TX_BUY',
    ('lifetime', 'SELL'): 'WHALE_TX_SELL',
}

OUTPUT_DIR = 'exports_phase3_2_semantics_probe'

# --- Semantic Variant Enums ---
class EmissionRule:
    PERSIST = 'E_PERSIST'
    FIRST_CROSS = 'E_FIRST_CROSS'
    TRANSITION = 'E_TRANSITION'

class AnchorTime:
    EACH_FLOW = 'T_EACH_FLOW'
    EACH_UNIQUE_TIME = 'T_EACH_UNIQUE_TIME'
    DB_EVENT_TIMES = 'T_DB_EVENT_TIMES'

class WindowBoundary:
    INCL_START_INCL_END = 'W_INCL_START_INCL_END'
    EXCL_START_INCL_END = 'W_EXCL_START_INCL_END'
    INCL_START_EXCL_END = 'W_INCL_START_EXCL_END'

class SameTimeInclusion:
    INCLUDE_ALL_SAME_TIME = 'S_INCLUDE_ALL_SAME_TIME'
    EVAL_AFTER_EACH_FLOW = 'S_EVAL_AFTER_EACH_FLOW'

class FlowRefChoice:
    ANCHOR_LAST_SIG = 'R_ANCHOR_LAST_SIG'
    ANCHOR_FIRST_SIG = 'R_ANCHOR_FIRST_SIG'
    CROSSING_SIG = 'R_CROSSING_SIG'
    MAX_SIG_IN_WINDOW = 'R_MAX_SIG_IN_WINDOW'
    MIN_SIG_IN_WINDOW = 'R_MIN_SIG_IN_WINDOW'

class PayloadSemantics:
    SUM_EXACT = 'P_SUM_EXACT'
    THRESHOLD_ONLY = 'P_THRESHOLD_ONLY'


# --- Schema Discovery ---
def discover_schema(cursor, table_name):
    """Discover columns and return mapping."""
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = {row[1]: row[0] for row in cursor.fetchall()}
    return columns


def validate_whale_events_schema(columns):
    """Validate and map whale_events columns."""
    required = ['wallet', 'window', 'event_time', 'event_type', 'sol_amount_lamports',
                'supporting_flow_count', 'flow_ref']
    mapping = {}
    for req in required:
        if req not in columns:
            print(f"ERROR: whale_events missing required column: {req}")
            sys.exit(1)
        mapping[req] = req
    return mapping


def validate_wallet_token_flow_schema(columns):
    """Validate and map wallet_token_flow columns."""
    required = ['scan_wallet', 'block_time', 'sol_direction', 'sol_amount_lamports', 'signature']
    mapping = {}
    for req in required:
        if req not in columns:
            print(f"ERROR: wallet_token_flow missing required column: {req}")
            sys.exit(1)
        mapping[req] = req
    return mapping


# --- Data Loading ---
def load_baseline_events(cursor, whale_map, limit_wallets=None):
    """Load baseline events from whale_events."""
    wallet_col = whale_map['wallet']
    window_col = whale_map['window']
    event_type_col = whale_map['event_type']
    event_time_col = whale_map['event_time']
    flow_ref_col = whale_map['flow_ref']
    amount_col = whale_map['sol_amount_lamports']
    count_col = whale_map['supporting_flow_count']
    
    query = f"""
        SELECT {wallet_col}, {window_col}, {event_type_col}, {event_time_col},
               {flow_ref_col}, {amount_col}, {count_col}
        FROM whale_events
        ORDER BY {wallet_col}, {event_time_col}
    """
    
    cursor.execute(query)
    rows = cursor.fetchall()
    
    baseline_events = []
    baseline_keys_strict = set()
    baseline_keys_relax = set()
    wallet_set = set()
    limited_wallets = set()
    
    # First pass: determine which wallets to include
    if limit_wallets:
        for row in rows:
            wallet = row[0]
            limited_wallets.add(wallet)
            if len(limited_wallets) >= limit_wallets:
                break
    
    for row in rows:
        wallet, window, event_type, event_time, flow_ref, amount, count = row
        
        # Skip if limiting and wallet not in limited set
        if limit_wallets and wallet not in limited_wallets:
            continue
        
        wallet_set.add(wallet)
        
        event = {
            'wallet': wallet,
            'window': window,
            'event_type': event_type,
            'event_time': event_time,
            'flow_ref': flow_ref,
            'amount': amount,
            'count': count,
        }
        baseline_events.append(event)
        
        key_strict = (wallet, window, event_type, event_time, flow_ref)
        key_relax = (wallet, window, event_type, event_time)
        baseline_keys_strict.add((key_strict, amount, count))
        baseline_keys_relax.add((key_relax, amount, count))
    
    print(f"Loaded {len(baseline_events)} baseline events from whale_events")
    print(f"Unique wallets in baseline: {len(wallet_set)}")
    
    return baseline_events, baseline_keys_strict, baseline_keys_relax, wallet_set


def load_flows(cursor, flow_map, wallet_set, fast_all=False, fast_max_flows=None, wallet_time_ranges=None):
    """Load flows from wallet_token_flow for wallets in baseline."""
    wallet_col = flow_map['scan_wallet']
    time_col = flow_map['block_time']
    direction_col = flow_map['sol_direction']
    amount_col = flow_map['sol_amount_lamports']
    sig_col = flow_map['signature']
    
    # Build WHERE clause for wallets
    wallet_list = list(wallet_set)
    placeholders = ','.join('?' * len(wallet_list))
    
    query = f"""
        SELECT {wallet_col}, {time_col}, {direction_col}, {amount_col}, {sig_col}
        FROM wallet_token_flow
        WHERE {wallet_col} IN ({placeholders})
          AND {time_col} IS NOT NULL
          AND {sig_col} IS NOT NULL
          AND {amount_col} IS NOT NULL
        ORDER BY {wallet_col}, {time_col}, {sig_col}
    """
    
    cursor.execute(query, wallet_list)
    rows = cursor.fetchall()
    
    # Discover distinct directions
    directions = set(row[2] for row in rows)
    print(f"Distinct sol_direction values: {directions}")
    
    # Normalize direction to BUY/SELL
    def normalize_direction(d):
        d_upper = str(d).upper()
        if 'BUY' in d_upper or d_upper == 'IN':
            return 'BUY'
        elif 'SELL' in d_upper or d_upper == 'OUT':
            return 'SELL'
        else:
            return d_upper
    
    # Group by wallet and direction
    flows_by_wallet_dir = defaultdict(list)
    
    for row in rows:
        wallet, block_time, direction, amount, signature = row
        direction = normalize_direction(direction)
        
        if direction not in ['BUY', 'SELL']:
            continue
        
        flow = {
            'block_time': block_time,
            'direction': direction,
            'amount': amount,
            'signature': signature,
        }
        flows_by_wallet_dir[(wallet, direction)].append(flow)

    if fast_all and fast_max_flows is not None:
        truncated_flows_by_wallet_dir = defaultdict(list)
        for (wallet, direction), flows in flows_by_wallet_dir.items():
            if wallet_time_ranges and wallet in wallet_time_ranges:
                min_time, max_time = wallet_time_ranges[wallet]
                flows = [f for f in flows if min_time <= f['block_time'] <= max_time]
            if fast_max_flows and len(flows) > fast_max_flows:
                flows = flows[-fast_max_flows:]
            truncated_flows_by_wallet_dir[(wallet, direction)] = flows
        flows_by_wallet_dir = truncated_flows_by_wallet_dir
    
    print(f"Loaded {len(rows)} flows across {len(flows_by_wallet_dir)} wallet-direction pairs")
    
    return flows_by_wallet_dir


# --- Recomputation Engine ---
def get_threshold(window, event_type):
    """Get threshold for window/event_type."""
    if window == 'lifetime':
        return THRESHOLDS['TX']
    elif window == '24h':
        return THRESHOLDS['CUM_24H']
    elif window == '7d':
        return THRESHOLDS['CUM_7D']
    else:
        return None


def compute_rolling_window_events(
    flows,
    window,
    direction,
    threshold,
    variant,
    db_anchor_times=None,
    fast_all=False,
    fast_anchors=None,
):
    """
    Compute events for a rolling window using the given semantic variant.
    
    Returns list of events: [(event_time, flow_ref, amount, count), ...]
    """
    if not flows:
        return []
    
    window_seconds = WINDOW_SECONDS.get(window)
    if window_seconds is None:
        # Lifetime (TX events)
        return compute_tx_events(flows, threshold, variant)
    
    # Choose anchors
    if variant['anchor_time'] == AnchorTime.DB_EVENT_TIMES:
        if not db_anchor_times:
            return []
        anchor_times = sorted(db_anchor_times)
    elif variant['anchor_time'] == AnchorTime.EACH_UNIQUE_TIME:
        anchor_times = sorted(set(f['block_time'] for f in flows))
    else:  # EACH_FLOW
        # For EACH_FLOW with INCLUDE_ALL_SAME_TIME, we still need unique times
        # Otherwise we'd emit duplicate events for same-time flows
        if variant['same_time'] == SameTimeInclusion.INCLUDE_ALL_SAME_TIME:
            anchor_times = sorted(set(f['block_time'] for f in flows))
        else:
            # For EVAL_AFTER_EACH_FLOW, we want each flow as a separate anchor
            # So we keep duplicates but will handle them in the loop
            anchor_times = sorted([f['block_time'] for f in flows])
    
    if fast_all and fast_anchors is not None:
        anchor_times = anchor_times[:fast_anchors]

    events = []
    state = {
        'last_emitted_below': True,  # For TRANSITION rule
        'first_cross_emitted': False  # For FIRST_CROSS rule
    }
    
    for anchor_time in anchor_times:
        # Compute window boundaries
        start_time = anchor_time - window_seconds
        
        # Determine which flows are included
        included_flows = []
        for f in flows:
            ft = f['block_time']
            include = False
            
            if variant['window_boundary'] == WindowBoundary.INCL_START_INCL_END:
                include = start_time <= ft <= anchor_time
            elif variant['window_boundary'] == WindowBoundary.EXCL_START_INCL_END:
                include = start_time < ft <= anchor_time
            elif variant['window_boundary'] == WindowBoundary.INCL_START_EXCL_END:
                include = start_time <= ft < anchor_time
            
            if include:
                included_flows.append(f)
        
        if not included_flows:
            continue
        
        # Handle same-time inclusion
        if variant['same_time'] == SameTimeInclusion.INCLUDE_ALL_SAME_TIME:
            # Evaluate once with all same-time flows
            rolling_sum = sum(f['amount'] for f in included_flows)
            
            should_emit = False
            if variant['emission'] == EmissionRule.PERSIST:
                should_emit = rolling_sum >= threshold
            elif variant['emission'] == EmissionRule.FIRST_CROSS:
                # Emit only at first crossing
                if rolling_sum >= threshold and not state['first_cross_emitted']:
                    should_emit = True
                    state['first_cross_emitted'] = True
            elif variant['emission'] == EmissionRule.TRANSITION:
                # Emit when transitioning from below to above threshold
                should_emit = rolling_sum >= threshold and state['last_emitted_below']
            
            if should_emit:
                flow_ref = choose_flow_ref(included_flows, variant['flow_ref'], threshold)
                amount = rolling_sum if variant['payload'] == PayloadSemantics.SUM_EXACT else threshold
                count = len(included_flows)
                events.append((anchor_time, flow_ref, amount, count))
            
            # Update TRANSITION state after emission decision
            if variant['emission'] == EmissionRule.TRANSITION:
                state['last_emitted_below'] = rolling_sum < threshold
        
        else:  # EVAL_AFTER_EACH_FLOW
            # Sort same-time flows by signature
            same_time_flows = [f for f in included_flows if f['block_time'] == anchor_time]
            same_time_flows.sort(key=lambda f: f['signature'])
            
            for stf in same_time_flows:
                # Recompute included flows up to and including this flow
                included_up_to = [f for f in included_flows if (
                    f['block_time'] < anchor_time or
                    (f['block_time'] == anchor_time and f['signature'] <= stf['signature'])
                )]
                
                rolling_sum = sum(f['amount'] for f in included_up_to)
                
                should_emit = False
                if variant['emission'] == EmissionRule.PERSIST:
                    should_emit = rolling_sum >= threshold
                elif variant['emission'] == EmissionRule.FIRST_CROSS:
                    if rolling_sum >= threshold and not state['first_cross_emitted']:
                        should_emit = True
                        state['first_cross_emitted'] = True
                elif variant['emission'] == EmissionRule.TRANSITION:
                    should_emit = rolling_sum >= threshold and state['last_emitted_below']
                
                if should_emit:
                    flow_ref = choose_flow_ref(included_up_to, variant['flow_ref'], threshold)
                    amount = rolling_sum if variant['payload'] == PayloadSemantics.SUM_EXACT else threshold
                    count = len(included_up_to)
                    events.append((anchor_time, flow_ref, amount, count))
                
                # Update TRANSITION state
                if variant['emission'] == EmissionRule.TRANSITION:
                    state['last_emitted_below'] = rolling_sum < threshold
    
    return events


def compute_tx_events(flows, threshold, variant):
    """Compute TX (lifetime) events."""
    events = []
    for f in flows:
        if f['amount'] >= threshold:
            events.append((f['block_time'], f['signature'], f['amount'], 1))
    return events


def choose_flow_ref(included_flows, ref_rule, threshold):
    """Choose flow_ref based on the rule."""
    if not included_flows:
        return None
    
    if ref_rule == FlowRefChoice.ANCHOR_LAST_SIG:
        # Last flow at anchor time (max time, then max sig)
        anchor_time = max(f['block_time'] for f in included_flows)
        anchor_flows = [f for f in included_flows if f['block_time'] == anchor_time]
        return max(anchor_flows, key=lambda f: f['signature'])['signature']
    
    elif ref_rule == FlowRefChoice.ANCHOR_FIRST_SIG:
        anchor_time = max(f['block_time'] for f in included_flows)
        anchor_flows = [f for f in included_flows if f['block_time'] == anchor_time]
        return min(anchor_flows, key=lambda f: f['signature'])['signature']
    
    elif ref_rule == FlowRefChoice.CROSSING_SIG:
        # Find the flow where cumulative sum first crosses threshold
        cumsum = 0
        sorted_flows = sorted(included_flows, key=lambda x: (x['block_time'], x['signature']))
        for f in sorted_flows:
            cumsum += f['amount']
            if cumsum >= threshold:
                return f['signature']
        # Fallback: last flow in time order
        return sorted_flows[-1]['signature']
    
    elif ref_rule == FlowRefChoice.MAX_SIG_IN_WINDOW:
        return max(included_flows, key=lambda f: f['signature'])['signature']
    
    elif ref_rule == FlowRefChoice.MIN_SIG_IN_WINDOW:
        return min(included_flows, key=lambda f: f['signature'])['signature']
    
    return None


def recompute_all_events(flows_by_wallet_dir, variant, baseline_events, fast_all=False, fast_anchors=None):
    """Recompute all events using the variant semantics."""
    recomputed = []
    recomputed_keys_strict = set()
    recomputed_keys_relax = set()
    
    # Build db_anchor_times lookup for DB_EVENT_TIMES mode
    db_anchors_by_wallet_window_type = defaultdict(set)
    if variant['anchor_time'] == AnchorTime.DB_EVENT_TIMES:
        for event in baseline_events:
            key = (event['wallet'], event['window'], event['event_type'])
            db_anchors_by_wallet_window_type[key].add(event['event_time'])
    
    # Process each wallet-direction
    for (wallet, direction), flows in flows_by_wallet_dir.items():
        for window in ['24h', '7d', 'lifetime']:
            event_type = EVENT_TYPE_MAP.get((window, direction))
            if not event_type:
                continue
            
            threshold = get_threshold(window, event_type)
            if threshold is None:
                continue
            
            # Get db anchor times for this combination
            db_anchor_times = None
            if variant['anchor_time'] == AnchorTime.DB_EVENT_TIMES:
                db_anchor_times = db_anchors_by_wallet_window_type.get((wallet, window, event_type))
            
            # Compute events
            events = compute_rolling_window_events(
                flows,
                window,
                direction,
                threshold,
                variant,
                db_anchor_times,
                fast_all=fast_all,
                fast_anchors=fast_anchors,
            )
            
            for event_time, flow_ref, amount, count in events:
                event = {
                    'wallet': wallet,
                    'window': window,
                    'event_type': event_type,
                    'event_time': event_time,
                    'flow_ref': flow_ref,
                    'amount': amount,
                    'count': count,
                }
                recomputed.append(event)
                
                key_strict = (wallet, window, event_type, event_time, flow_ref)
                key_relax = (wallet, window, event_type, event_time)
                recomputed_keys_strict.add((key_strict, amount, count))
                recomputed_keys_relax.add((key_relax, amount, count))
    
    return recomputed, recomputed_keys_strict, recomputed_keys_relax


# --- Scoring ---
def score_variant(baseline_keys_strict, baseline_keys_relax,
                  recomputed_keys_strict, recomputed_keys_relax,
                  baseline_events, recomputed_events):
    """Score the match quality."""
    # Strict scoring
    perfect_strict = baseline_keys_strict & recomputed_keys_strict
    missing_strict = baseline_keys_strict - recomputed_keys_strict
    phantom_strict = recomputed_keys_strict - baseline_keys_strict
    
    # Relaxed scoring
    perfect_relax = baseline_keys_relax & recomputed_keys_relax
    missing_relax = baseline_keys_relax - recomputed_keys_relax
    phantom_relax = recomputed_keys_relax - baseline_keys_relax
    
    # Amount/count mismatches (strict)
    # Events matching on key but differing on payload
    baseline_key_to_payload_strict = {k: (amt, cnt) for k, amt, cnt in baseline_keys_strict}
    recomputed_key_to_payload_strict = {k: (amt, cnt) for k, amt, cnt in recomputed_keys_strict}
    
    common_keys_strict = set(baseline_key_to_payload_strict.keys()) & set(recomputed_key_to_payload_strict.keys())
    amount_mismatch_strict = sum(
        1 for k in common_keys_strict
        if baseline_key_to_payload_strict[k][0] != recomputed_key_to_payload_strict[k][0]
    )
    count_mismatch_strict = sum(
        1 for k in common_keys_strict
        if baseline_key_to_payload_strict[k][1] != recomputed_key_to_payload_strict[k][1]
    )
    
    # Relaxed
    baseline_key_to_payload_relax = {k: (amt, cnt) for k, amt, cnt in baseline_keys_relax}
    recomputed_key_to_payload_relax = {k: (amt, cnt) for k, amt, cnt in recomputed_keys_relax}
    
    common_keys_relax = set(baseline_key_to_payload_relax.keys()) & set(recomputed_key_to_payload_relax.keys())
    amount_mismatch_relax = sum(
        1 for k in common_keys_relax
        if baseline_key_to_payload_relax[k][0] != recomputed_key_to_payload_relax[k][0]
    )
    count_mismatch_relax = sum(
        1 for k in common_keys_relax
        if baseline_key_to_payload_relax[k][1] != recomputed_key_to_payload_relax[k][1]
    )
    
    total_errors_strict = len(missing_strict) + len(phantom_strict) + amount_mismatch_strict + count_mismatch_strict
    total_errors_relax = len(missing_relax) + len(phantom_relax) + amount_mismatch_relax + count_mismatch_relax
    
    return {
        'baseline_total': len(baseline_keys_strict),
        'recomputed_total': len(recomputed_keys_strict),
        'perfect_strict': len(perfect_strict),
        'missing_strict': len(missing_strict),
        'phantom_strict': len(phantom_strict),
        'amount_mismatch_strict': amount_mismatch_strict,
        'count_mismatch_strict': count_mismatch_strict,
        'perfect_relax': len(perfect_relax),
        'missing_relax': len(missing_relax),
        'phantom_relax': len(phantom_relax),
        'amount_mismatch_relax': amount_mismatch_relax,
        'count_mismatch_relax': count_mismatch_relax,
        'total_errors_strict': total_errors_strict,
        'total_errors_relax': total_errors_relax,
    }


# --- Variant Generation ---
def generate_variants(fast_mode=False):
    """Generate all semantic variant combinations."""
    if fast_mode:
        # Reduced set for fast mode
        variants = [
            {
                'emission': EmissionRule.PERSIST,
                'anchor_time': AnchorTime.DB_EVENT_TIMES,
                'window_boundary': WindowBoundary.EXCL_START_INCL_END,
                'same_time': SameTimeInclusion.INCLUDE_ALL_SAME_TIME,
                'flow_ref': FlowRefChoice.ANCHOR_LAST_SIG,
                'payload': PayloadSemantics.SUM_EXACT,
            },
            {
                'emission': EmissionRule.FIRST_CROSS,
                'anchor_time': AnchorTime.DB_EVENT_TIMES,
                'window_boundary': WindowBoundary.EXCL_START_INCL_END,
                'same_time': SameTimeInclusion.INCLUDE_ALL_SAME_TIME,
                'flow_ref': FlowRefChoice.CROSSING_SIG,
                'payload': PayloadSemantics.SUM_EXACT,
            },
            {
                'emission': EmissionRule.TRANSITION,
                'anchor_time': AnchorTime.DB_EVENT_TIMES,
                'window_boundary': WindowBoundary.EXCL_START_INCL_END,
                'same_time': SameTimeInclusion.INCLUDE_ALL_SAME_TIME,
                'flow_ref': FlowRefChoice.ANCHOR_LAST_SIG,
                'payload': PayloadSemantics.SUM_EXACT,
            },
        ]
    else:
        # Full combinatorial search
        emissions = [EmissionRule.PERSIST, EmissionRule.FIRST_CROSS, EmissionRule.TRANSITION]
        anchors = [AnchorTime.EACH_FLOW, AnchorTime.EACH_UNIQUE_TIME, AnchorTime.DB_EVENT_TIMES]
        boundaries = [WindowBoundary.INCL_START_INCL_END, WindowBoundary.EXCL_START_INCL_END, WindowBoundary.INCL_START_EXCL_END]
        same_times = [SameTimeInclusion.INCLUDE_ALL_SAME_TIME, SameTimeInclusion.EVAL_AFTER_EACH_FLOW]
        flow_refs = [FlowRefChoice.ANCHOR_LAST_SIG, FlowRefChoice.ANCHOR_FIRST_SIG, 
                     FlowRefChoice.CROSSING_SIG, FlowRefChoice.MAX_SIG_IN_WINDOW, 
                     FlowRefChoice.MIN_SIG_IN_WINDOW]
        payloads = [PayloadSemantics.SUM_EXACT, PayloadSemantics.THRESHOLD_ONLY]
        
        variants = []
        for e, a, b, s, f, p in product(emissions, anchors, boundaries, same_times, flow_refs, payloads):
            variants.append({
                'emission': e,
                'anchor_time': a,
                'window_boundary': b,
                'same_time': s,
                'flow_ref': f,
                'payload': p,
            })
    
    return variants


# --- Output ---
def print_ranking_table(results):
    """Print top 10 variants ranked by total errors."""
    print("\n" + "=" * 160)
    print("VARIANT RANKING (Top 10 by TOTAL_ERRORS_STRICT)")
    print("=" * 160)
    
    header = f"{'Rank':<6} {'Emission':<18} {'Anchor':<25} {'Boundary':<28} {'SameTime':<30} {'FlowRef':<25} {'Payload':<20} {'Err_S':<10} {'Err_R':<10}"
    print(header)
    print("-" * 160)
    
    for i, result in enumerate(results[:10], 1):
        v = result['variant']
        s = result['score']
        row = f"{i:<6} {v['emission']:<18} {v['anchor_time']:<25} {v['window_boundary']:<28} {v['same_time']:<30} {v['flow_ref']:<25} {v['payload']:<20} {s['total_errors_strict']:<10} {s['total_errors_relax']:<10}"
        print(row)
    
    print("=" * 160)


def write_best_variant_summary(result, output_dir):
    """Write summary for best variant."""
    os.makedirs(output_dir, exist_ok=True)
    
    filepath = os.path.join(output_dir, 'best_variant_summary.txt')
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("BEST VARIANT SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        v = result['variant']
        f.write("Variant Configuration:\n")
        for key, val in v.items():
            f.write(f"  {key}: {val}\n")
        f.write("\n")
        
        s = result['score']
        f.write("Match Scores:\n")
        f.write(f"  Baseline Total:        {s['baseline_total']}\n")
        f.write(f"  Recomputed Total:      {s['recomputed_total']}\n")
        f.write(f"  Perfect Strict:        {s['perfect_strict']}\n")
        f.write(f"  Missing Strict:        {s['missing_strict']}\n")
        f.write(f"  Phantom Strict:        {s['phantom_strict']}\n")
        f.write(f"  Amount Mismatch:       {s['amount_mismatch_strict']}\n")
        f.write(f"  Count Mismatch:        {s['count_mismatch_strict']}\n")
        f.write(f"  TOTAL ERRORS STRICT:   {s['total_errors_strict']}\n")
        f.write(f"\n")
        f.write(f"  Perfect Relax:         {s['perfect_relax']}\n")
        f.write(f"  Missing Relax:         {s['missing_relax']}\n")
        f.write(f"  Phantom Relax:         {s['phantom_relax']}\n")
        f.write(f"  Amount Mismatch Relax: {s['amount_mismatch_relax']}\n")
        f.write(f"  Count Mismatch Relax:  {s['count_mismatch_relax']}\n")
        f.write(f"  TOTAL ERRORS RELAX:    {s['total_errors_relax']}\n")
    
    print(f"\nWrote best variant summary to: {filepath}")


def write_mismatches(baseline_events, recomputed_events, baseline_keys_strict, 
                     recomputed_keys_strict, output_dir):
    """Write mismatch details."""
    filepath = os.path.join(output_dir, 'best_variant_mismatches.tsv')
    
    # Build lookups
    baseline_lookup = {}
    for e in baseline_events:
        key = (e['wallet'], e['window'], e['event_type'], e['event_time'], e['flow_ref'])
        baseline_lookup[key] = e
    
    recomputed_lookup = {}
    for e in recomputed_events:
        key = (e['wallet'], e['window'], e['event_type'], e['event_time'], e['flow_ref'])
        recomputed_lookup[key] = e
    
    # Extract keys
    baseline_keys_only = {k for k, amt, cnt in baseline_keys_strict}
    recomputed_keys_only = {k for k, amt, cnt in recomputed_keys_strict}
    
    # Build payload lookups for mismatches
    baseline_key_to_payload = {k: (amt, cnt) for k, amt, cnt in baseline_keys_strict}
    recomputed_key_to_payload = {k: (amt, cnt) for k, amt, cnt in recomputed_keys_strict}
    
    missing = baseline_keys_only - recomputed_keys_only
    phantom = recomputed_keys_only - baseline_keys_only
    
    # Find amount and count mismatches among common keys
    common_keys = baseline_keys_only & recomputed_keys_only
    amount_mismatches = []
    count_mismatches = []
    
    for key in common_keys:
        base_amt, base_cnt = baseline_key_to_payload[key]
        recomp_amt, recomp_cnt = recomputed_key_to_payload[key]
        
        if base_amt != recomp_amt:
            amount_mismatches.append((key, base_amt, recomp_amt, base_cnt, recomp_cnt))
        
        if base_cnt != recomp_cnt:
            count_mismatches.append((key, base_amt, recomp_amt, base_cnt, recomp_cnt))
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("TYPE\tWALLET\tWINDOW\tEVENT_TYPE\tEVENT_TIME\tFLOW_REF\tBASE_AMOUNT\tRECOMP_AMOUNT\tBASE_COUNT\tRECOMP_COUNT\n")
        
        # Missing
        for key in sorted(missing)[:50]:
            e = baseline_lookup.get(key)
            if e:
                f.write(f"MISSING\t{e['wallet']}\t{e['window']}\t{e['event_type']}\t{e['event_time']}\t{e['flow_ref']}\t{e['amount']}\t\t{e['count']}\t\n")
        
        # Phantom
        for key in sorted(phantom)[:50]:
            e = recomputed_lookup.get(key)
            if e:
                f.write(f"PHANTOM\t{e['wallet']}\t{e['window']}\t{e['event_type']}\t{e['event_time']}\t{e['flow_ref']}\t\t{e['amount']}\t\t{e['count']}\n")
        
        # Amount mismatches
        for key, base_amt, recomp_amt, base_cnt, recomp_cnt in sorted(amount_mismatches)[:50]:
            wallet, window, event_type, event_time, flow_ref = key
            f.write(f"AMOUNT_MISMATCH\t{wallet}\t{window}\t{event_type}\t{event_time}\t{flow_ref}\t{base_amt}\t{recomp_amt}\t{base_cnt}\t{recomp_cnt}\n")
        
        # Count mismatches
        for key, base_amt, recomp_amt, base_cnt, recomp_cnt in sorted(count_mismatches)[:50]:
            wallet, window, event_type, event_time, flow_ref = key
            f.write(f"COUNT_MISMATCH\t{wallet}\t{window}\t{event_type}\t{event_time}\t{flow_ref}\t{base_amt}\t{recomp_amt}\t{base_cnt}\t{recomp_cnt}\n")
    
    print(f"Wrote mismatch details to: {filepath}")


def write_wallet_deepdive(wallet_prefix, baseline_events, flows_by_wallet_dir, variant, output_dir):
    """Write deep dive for a specific wallet (e.g., 215nhcAH prefix)."""
    # Find full wallet address
    target_wallet = None
    for event in baseline_events:
        if event['wallet'].startswith(wallet_prefix):
            target_wallet = event['wallet']
            break
    
    if not target_wallet:
        print(f"Warning: No wallet found with prefix '{wallet_prefix}'")
        return
    
    # Get first 5 baseline events for this wallet
    wallet_events = [e for e in baseline_events if e['wallet'] == target_wallet]
    wallet_events.sort(key=lambda e: e['event_time'])
    first_5 = wallet_events[:5]
    
    if not first_5:
        print(f"Warning: No events found for wallet {target_wallet}")
        return
    
    filepath = os.path.join(output_dir, f'wallet_{wallet_prefix}_deepdive.txt')
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"WALLET DEEP DIVE: {target_wallet}\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"Variant Configuration:\n")
        for key, val in variant.items():
            f.write(f"  {key}: {val}\n")
        f.write("\n")
        
        f.write(f"Analyzing first {len(first_5)} baseline events:\n")
        f.write("=" * 100 + "\n\n")
        
        for idx, event in enumerate(first_5, 1):
            f.write(f"EVENT #{idx}:\n")
            f.write(f"  Window: {event['window']}\n")
            f.write(f"  Event Type: {event['event_type']}\n")
            f.write(f"  Event Time: {event['event_time']}\n")
            f.write(f"  Flow Ref: {event['flow_ref']}\n")
            f.write(f"  Amount: {event['amount']:,}\n")
            f.write(f"  Count: {event['count']}\n")
            f.write("\n")
            
            # Get direction from event_type
            if 'BUY' in event['event_type']:
                direction = 'BUY'
            elif 'SELL' in event['event_type']:
                direction = 'SELL'
            else:
                direction = None
            
            if direction:
                # Get flows for this wallet-direction
                flows = flows_by_wallet_dir.get((target_wallet, direction), [])
                
                if flows:
                    window = event['window']
                    window_seconds = WINDOW_SECONDS.get(window)
                    anchor_time = event['event_time']
                    threshold = get_threshold(window, event['event_type'])
                    
                    if window == 'lifetime':
                        # TX event - just find the specific flow
                        matching_flow = None
                        for flow in flows:
                            if flow['signature'] == event['flow_ref']:
                                matching_flow = flow
                                break
                        
                        f.write("  LIFETIME (TX) EVENT:\n")
                        if matching_flow:
                            f.write(f"    Flow: time={matching_flow['block_time']}, "
                                   f"amount={matching_flow['amount']:,}, sig={matching_flow['signature']}\n")
                        else:
                            f.write("    WARNING: Matching flow not found\n")
                    
                    else:
                        # Rolling window event
                        start_time = anchor_time - window_seconds
                        
                        # Find included flows
                        included = []
                        for flow in flows:
                            ft = flow['block_time']
                            include = False
                            
                            if variant['window_boundary'] == WindowBoundary.INCL_START_INCL_END:
                                include = start_time <= ft <= anchor_time
                            elif variant['window_boundary'] == WindowBoundary.EXCL_START_INCL_END:
                                include = start_time < ft <= anchor_time
                            elif variant['window_boundary'] == WindowBoundary.INCL_START_EXCL_END:
                                include = start_time <= ft < anchor_time
                            
                            if include:
                                included.append(flow)
                        
                        included.sort(key=lambda f: (f['block_time'], f['signature']))
                        
                        f.write(f"  ROLLING WINDOW: {window} (threshold={threshold:,})\n")
                        f.write(f"    Start: {start_time}, Anchor: {anchor_time}\n")
                        f.write(f"    Included flows ({len(included)}):\n")
                        
                        cumsum = 0
                        crossed_at = None
                        for flow in included:
                            cumsum += flow['amount']
                            marker = ""
                            if cumsum >= threshold and crossed_at is None:
                                crossed_at = flow['signature']
                                marker = " <-- CROSSES THRESHOLD"
                            
                            f.write(f"      {flow['block_time']}: {flow['amount']:>15,} "
                                   f"(cumsum={cumsum:>15,}) sig={flow['signature']}{marker}\n")
                        
                        f.write(f"\n    Flow_ref selection ({variant['flow_ref']}):\n")
                        chosen_ref = choose_flow_ref(included, variant['flow_ref'], threshold)
                        f.write(f"      Chosen: {chosen_ref}\n")
                        f.write(f"      Baseline: {event['flow_ref']}\n")
                        f.write(f"      Match: {'YES' if chosen_ref == event['flow_ref'] else 'NO'}\n")
            
            f.write("\n" + "-" * 100 + "\n\n")
    
    print(f"Wrote wallet deep dive to: {filepath}")


# --- Main ---
def main():
    parser = argparse.ArgumentParser(description='Infer whale_events semantic generation rules')
    parser.add_argument('--db', required=True, help='Path to SQLite database')
    parser.add_argument('--limit-wallets', type=int, help='Limit number of wallets for testing')
    parser.add_argument('--limit-variants', type=int, help='Limit number of variants to test')
    parser.add_argument('--fast', action='store_true', help='Fast mode: fewer variants, DB_EVENT_TIMES only')
    parser.add_argument('--fast-all', action='store_true', help='Fast-all mode: all variants with sampling caps')
    parser.add_argument('--fast-wallets', type=int, default=15, help='Fast-all cap for wallets')
    parser.add_argument('--fast-anchors', type=int, default=200, help='Fast-all cap for anchor times')
    parser.add_argument('--fast-max-flows', type=int, help='Fast-all cap for flows per wallet-direction')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.db):
        print(f"ERROR: Database not found: {args.db}")
        sys.exit(1)

    if args.fast_all:
        print(
            "FAST-ALL MODE: testing 540 variants with caps: "
            f"wallets={args.fast_wallets} anchors={args.fast_anchors} max_flows={args.fast_max_flows}"
        )
    
    print("Connecting to database...")
    conn = sqlite3.connect(args.db)
    cursor = conn.cursor()
    
    # Discover schemas
    print("\n--- Schema Discovery ---")
    whale_cols = discover_schema(cursor, 'whale_events')
    flow_cols = discover_schema(cursor, 'wallet_token_flow')
    
    whale_map = validate_whale_events_schema(whale_cols)
    flow_map = validate_wallet_token_flow_schema(flow_cols)
    
    print(f"whale_events columns: {list(whale_cols.keys())}")
    print(f"wallet_token_flow columns: {list(flow_cols.keys())}")
    
    # Load data
    print("\n--- Loading Baseline Events ---")
    limit_wallets = args.limit_wallets
    if args.fast_all and limit_wallets is None:
        limit_wallets = args.fast_wallets
    baseline_events, baseline_keys_strict, baseline_keys_relax, wallet_set = load_baseline_events(
        cursor, whale_map, limit_wallets
    )
    
    print("\n--- Loading Flows ---")
    wallet_time_ranges = None
    if args.fast_all and args.fast_max_flows is not None:
        wallet_time_ranges = {}
        for event in baseline_events:
            wallet = event['wallet']
            event_time = event['event_time']
            if wallet not in wallet_time_ranges:
                wallet_time_ranges[wallet] = (event_time, event_time)
            else:
                min_time, max_time = wallet_time_ranges[wallet]
                wallet_time_ranges[wallet] = (min(min_time, event_time), max(max_time, event_time))
    flows_by_wallet_dir = load_flows(
        cursor,
        flow_map,
        wallet_set,
        fast_all=args.fast_all,
        fast_max_flows=args.fast_max_flows,
        wallet_time_ranges=wallet_time_ranges,
    )
    
    conn.close()
    
    # Generate variants
    print("\n--- Generating Variants ---")
    if args.fast:
        print("FAST MODE ENABLED: Using reduced variant set and DB_EVENT_TIMES anchors only")

    variants = generate_variants(args.fast)
    if args.fast_all:
        variants = generate_variants(False)
    
    if args.limit_variants:
        variants = variants[:args.limit_variants]
    
    print(f"Testing {len(variants)} semantic variants...")
    
    # Test each variant
    results = []
    for i, variant in enumerate(variants, 1):
        if i == 1 or i % 10 == 0 or i == len(variants):
            print(f"  Tested {i}/{len(variants)} variants...")
        
        recomputed, recomputed_keys_strict, recomputed_keys_relax = recompute_all_events(
            flows_by_wallet_dir,
            variant,
            baseline_events,
            fast_all=args.fast_all,
            fast_anchors=args.fast_anchors,
        )
        
        score = score_variant(
            baseline_keys_strict, baseline_keys_relax,
            recomputed_keys_strict, recomputed_keys_relax,
            baseline_events, recomputed
        )
        
        results.append({
            'variant': variant,
            'score': score,
            'recomputed': recomputed,
        })
    
    # Sort by total errors
    results.sort(key=lambda r: (r['score']['total_errors_strict'], r['score']['total_errors_relax']))
    
    # Print ranking
    if results:
        print_ranking_table(results)
    else:
        print("\nWARNING: No variants produced any results!")
    
    # Write best variant
    if results:
        best = results[0]
        print("\n--- Writing Best Variant Outputs ---")
        
        # Ensure output directory exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        write_best_variant_summary(best, OUTPUT_DIR)
        
        # Get recomputed_keys_strict from recomputation
        _, recomputed_keys_strict_best, _ = recompute_all_events(
            flows_by_wallet_dir,
            best['variant'],
            baseline_events,
            fast_all=args.fast_all,
            fast_anchors=args.fast_anchors,
        )
        
        write_mismatches(baseline_events, best['recomputed'], 
                        baseline_keys_strict, recomputed_keys_strict_best, OUTPUT_DIR)
        
        write_wallet_deepdive('215nhcAH', baseline_events, flows_by_wallet_dir, 
                            best['variant'], OUTPUT_DIR)
    
    print("\n--- COMPLETE ---")


if __name__ == '__main__':
    main()
