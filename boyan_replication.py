"""Run: python boyan_replication.py

Algorithm:
1. Validate the circuit and cut boundaries; identify original shared nodes.
2. Label each node by the earliest arrival among its allowed candidate cuts.
3. Trace the chosen cuts back from every output to extract a complete cover.
4. Check available signals; count LUTs and repeated interior logic.
5. Compare shared-boundary and duplication-allowed modes; verify small examples.
"""
from itertools import product


# 1. Signal helpers
# Remove the inversion marker to find the source node, for example !v -> v.
def node_name(signal):
    return signal[1:] if signal.startswith('!') else signal


# Read the node value; XOR with 1 flips the bit when ! is present.
def signal_value(signal, values):
    return values[node_name(signal)] ^ int(signal.startswith('!'))


# 2. Circuit data and basic checks
# inputs lists input names; gates maps each AND node to two signals; outputs lists outputs.
class Circuit:
    def __init__(self, inputs, gates, outputs):

        # Store the circuit without generating cuts or rewriting its logic.
        self.inputs = tuple(inputs)
        self.gates = dict(gates)
        self.outputs = tuple(outputs)

        # Reject empty input lists, duplicate names, and invalid node names.
        if not self.inputs or len(set(self.inputs)) != len(self.inputs):
            raise ValueError('Supply unique primary inputs')
        known = set(self.inputs)
        if any(not isinstance(n, str) or not n or n.startswith('!') for n in known):
            raise ValueError('Node names must be nonempty and cannot start with !')
        
        # Gates must follow dependency order: define u before y if y uses u.
        # known tracks defined nodes; forward references and cycles are rejected here.
        # Dictionary order is the circuit's topological order.
        for root, signals in self.gates.items():
            if not isinstance(root, str) or not root or root.startswith('!') or root in known:
                raise ValueError(f'Invalid or duplicate gate name: {root}')
            if len(signals) != 2 or any(not isinstance(s, str) or node_name(s) not in known
                                        for s in signals):
                raise ValueError(f'{root}: expected two signals from earlier nodes')
            known.add(root)
        if not self.outputs or not set(self.outputs) <= known:
            raise ValueError('Outputs must be existing, uncomplemented node names')
        
        # rank stores topological position; users stores distinct consuming gates.
        self.rank = {n: i for i, n in enumerate([*self.inputs, *self.gates])}
        users = {n: set() for n in known}
        for root, signals in self.gates.items():
            for signal in signals:
                users[node_name(signal)].add(root)

        # Mark a gate as shared when gate users plus output-port references exceed one.
        # This counts connections; it does not model capacitive fanout delay.
        self.shared = {n for n in self.gates
                       if len(users[n]) + self.outputs.count(n) > 1}

    # 3. Validate a cut and find its interior
    # Example: y with leaves (a,b,c) contains u,y; with leaves (u,c), it contains only y.
    def cut_inside(self, root, leaves, k):
        """Find the logic inside a supplied cut; do not generate new cuts."""

        # A cut has at most k distinct inputs, all earlier than root; self-cuts are rejected.
        if (root not in self.gates or not 1 <= len(leaves) <= k
                or len(set(leaves)) != len(leaves)
                or any(n not in self.rank or self.rank[n] >= self.rank[root] for n in leaves)):
            raise ValueError(f'Invalid cut for {root}: {leaves}')
        
        # inside stores internal nodes; reached stores boundary inputs actually encountered.
        inside, reached = set(), set()


        # Trace backward from root and stop at leaves; reaching an uncut primary input is invalid.
        # Using a set prevents counting a reconvergent node twice inside one LUT.
        def visit(node):
            if node in leaves:
                reached.add(node)
            elif node in self.inputs:
                raise ValueError(f'Cut for {root} misses input {node}')
            elif node not in inside:
                inside.add(node)
                for signal in self.gates[node]:
                    visit(node_name(signal))

        visit(root)

        # Reject listed boundary inputs that were never reached.
        if reached != set(leaves):
            raise ValueError(f'Cut for {root} has an unused boundary input')
        return inside


# 4. Automatic cover selection: the core algorithm
def select_cover(circuit, candidates, k=3, preserve_shared=False):
    """Automatically select a cover from supplied candidate boundaries."""
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError('k must be a positive integer')
    
    # Candidate lists must refer to existing gates.
    if not set(candidates) <= set(circuit.gates):
        raise ValueError('Candidates contain an unknown gate')
    
    # Forward pass: inputs arrive at 0; best stores the chosen cut for each node.
    arrival = dict.fromkeys(circuit.inputs, 0)
    best = {}
    for root in circuit.gates:
        choices = []

        # Evaluate the supplied choices; this function does not enumerate new cuts.
        for candidate in candidates.get(root, []):
            leaves = tuple(candidate)
            inside = circuit.cut_inside(root, leaves, k)

            # Shared mode: reject a cut that absorbs another original shared node.
            # The root itself may be shared, so remove it before checking.
            if preserve_shared and (inside - {root}) & circuit.shared:
                continue
            
            # Skip a candidate if any boundary signal has no available implementation.
            if any(n not in arrival for n in leaves):
                continue
            
            # Wait for the latest input, then add one LUT delay.
            time = max(arrival[n] for n in leaves) + 1

            # Prefer smaller delay; ties use input count and order, not global area cost.
            # Fewer leaves, then input order: deterministic ties, not area recovery.
            score = (time, len(leaves), tuple(circuit.rank[n] for n in leaves))
            choices.append((score, leaves))

        # Store this node's earliest arrival for later nodes: the dynamic-programming recurrence.
        if choices:
            score, leaves = min(choices, key=lambda item: item[0])
            arrival[root], best[root] = score[0], leaves


    # Backward pass: start at all primary outputs and keep only needed LUTs.
    # Computing best[u] does not force a u LUT; output cuts may absorb its logic.
    # This is cover extraction, not full top-down timing optimization.
    selected = {}
    pending = list(circuit.outputs)
    while pending:
        root = pending.pop()
        if root in circuit.inputs or root in selected:
            continue
        if root not in best:
            raise ValueError(f'No usable supplied cut for {root} in this mode')
        selected[root] = best[root]

        # Boundary signals must also be implemented; primary inputs need no new LUT.
        pending.extend(best[root])
    return selected


# 5. Cover legality, area, and delay
# This function can also check a complete cover supplied separately.
def analyze_cover(circuit, selected, k=3):
    """Validate a complete cover and measure its area, delay, and replication."""
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError('k must be a positive integer')
    
    # owners[node] lists the LUTs containing that original logic node.
    owners = {}
    for root, leaves in selected.items():
        inside = circuit.cut_inside(root, leaves, k)

        # A LUT input must be a primary input or the output of another selected LUT.
        # Having u inside y does not provide an external u for z; each LUT outputs only its root.
        for leaf in leaves:
            if leaf not in circuit.inputs and leaf not in selected:
                raise ValueError(f'Unavailable signal {leaf} used by LUT {root}')
            
        # Logic appearing in multiple LUTs is recorded as replication across LUTs.
        for node in sorted(inside, key=circuit.rank.get):
            owners.setdefault(node, []).append(root)


    # Check output dependencies and reject unused LUTs before computing area.
    used, pending = set(), list(circuit.outputs)
    while pending:
        root = pending.pop()
        if root in circuit.inputs or root in used:
            continue
        if root not in selected:
            raise ValueError(f'Missing implementation of {root}')
        used.add(root)
        pending.extend(selected[root])
    if used != set(selected):
        raise ValueError('Remove unused LUTs before comparing area')


    # Recompute arrivals on the selected cover; original AND count is not LUT count.
    arrival = dict.fromkeys(circuit.inputs, 0)
    for root in circuit.gates:
        if root in selected:
            arrival[root] = max(arrival[n] for n in selected[root]) + 1

    # area counts selected LUTs; worst_delay is the latest arrival across all outputs.
    # replicated_nodes keeps repeated logic; do not add that count again to physical LUT area.
    return {
        'selected': selected,
        'area': len(selected),
        'output_arrivals': {n: arrival[n] for n in circuit.outputs},
        'worst_delay': max(arrival[n] for n in circuit.outputs),
        'replicated_nodes': {n: sorted(roots, key=circuit.rank.get)
                             for n, roots in sorted(owners.items(), key=lambda x: circuit.rank[x[0]])
                             if len(roots) > 1},
    }


# 6. Functional verification for small circuits
def verify_cover(circuit, selected, k=3):
    """Compile LUT truth tables and exhaustively check a small circuit."""
    analyze_cover(circuit, selected, k)

    # n inputs have 2^n patterns; cap this demo at 12 inputs to limit exhaustive checking.
    if len(circuit.inputs) > 12:
        raise ValueError('Exhaustive checking is limited to 12 primary inputs')
    
    # First compile each LUT truth table from its internal logic, including inversions.
    tables = {}
    for root, leaves in selected.items():
        table = {}
        inside = circuit.cut_inside(root, leaves, k)

        # word assigns bits to the boundary inputs; keep the leaf order consistent.
        for word in product((0, 1), repeat=len(leaves)):
            values = dict(zip(leaves, word))
            for node in sorted(inside, key=circuit.rank.get):
                a, b = circuit.gates[node]
                values[node] = signal_value(a, values) & signal_value(b, values)
            table[word] = values[root]
        tables[root] = table


    # Then simulate the original AND network and mapped LUT network for all input patterns.
    checked = 0
    for word in product((0, 1), repeat=len(circuit.inputs)):
        original = dict(zip(circuit.inputs, word))
        mapped = dict(original)
        for root, (a, b) in circuit.gates.items():
            original[root] = signal_value(a, original) & signal_value(b, original)

        # The mapped circuit reads actual LUT outputs, not hidden values from the original circuit.
        for root in circuit.gates:
            if root in selected:
                row = tuple(mapped[n] for n in selected[root])
                mapped[root] = tables[root][row]

        # Any output mismatch fails; return the pattern count only after all checks pass.
        if any(original[n] != mapped[n] for n in circuit.outputs):
            raise AssertionError(f'Output mismatch for inputs {word}')
        checked += 1
    return checked


# 7. Compare the two policies
# Both modes use the same circuit and candidates; False permits replication but does not force it.
def compare_modes(circuit, candidates, k=3):
    results = {}
    for name, preserve in [('Shared boundary', True), ('Duplication allowed', False)]:
        selected = select_cover(circuit, candidates, k, preserve)
        results[name] = analyze_cover(circuit, selected, k)
    return results


# 8.  Example 1: one shared node
# y and z both use u; their candidate cuts either read u or compute it inside each LUT.
def small_example():
    circuit = Circuit(['a', 'b', 'c', 'd'], {
        'u': ('a', 'b'), 'y': ('u', 'c'), 'z': ('u', 'd')
    }, ['y', 'z'])

    # These are hand-written choices, not final answers; select_cover makes the selection.
    # Test candidates only. The algorithm chooses the final cover.
    candidates = {
        'u': [('a', 'b')],
        'y': [('u', 'c'), ('a', 'b', 'c')],
        'z': [('u', 'd'), ('a', 'b', 'd')],
    }
    return circuit, candidates


# 9. Example 2: multiple shared nodes and inversion
# u, v, q are shared; u's branches reconverge at y; !v is an internal inverted signal.
def larger_example():
    circuit = Circuit(['a', 'b', 'c', 'd', 'e', 'f'], {
        'u': ('a', 'b'), 'v': ('c', 'd'),
        'p': ('u', 'e'), 'q': ('u', 'f'),
        'r': ('p', 'v'), 's': ('q', '!v'),
        'y': ('r', 'q'), 'z': ('s', 'e'),
    }, ['y', 'z'])

    # This is a subset of legal candidates, not all possible cuts; Jay's output can replace it later.
    # A supplied subset of legal cuts, not an exhaustive cut enumeration.
    candidates = {
        'u': [('a', 'b')], 'v': [('c', 'd')],
        'p': [('u', 'e'), ('a', 'b', 'e')],
        'q': [('u', 'f'), ('a', 'b', 'f')],
        'r': [('p', 'v'), ('u', 'v', 'e')],
        's': [('q', 'v')],
        'y': [('r', 'q'), ('v', 'p', 'q')],
        'z': [('s', 'e'), ('v', 'q', 'e')],
    }
    return circuit, candidates


# 10. Demo output
# Show the circuit and candidates, then selected LUTs, repeated logic, costs, and verification.
def run_demo(name, circuit, candidates):
    print(f'\n{name}')
    print('Model: 3-input LUTs, unit area/delay, zero input arrival')
    print('Candidate cuts: supplied test data; cover selection: automatic')
    print('Original shared nodes:', sorted(circuit.shared, key=circuit.rank.get))
    for root, (a, b) in circuit.gates.items():
        print(f'  {root} = {a} AND {b}; candidates: {candidates[root]}')
    results = compare_modes(circuit, candidates)
    for mode, result in results.items():
        print(f'\n{mode}')
        for root in circuit.gates:
            if root in result['selected']:
                print(f"  {root} <- {result['selected'][root]}")

        # Repeating logic need not increase LUT count; absorption can also reduce LUT stages.
        print('  Replicated logic:', result['replicated_nodes'])
        print('  Area:', result['area'], 'LUTs')
        print('  Output arrivals:', result['output_arrivals'])
        print('  Worst delay:', result['worst_delay'])
        patterns = verify_cover(circuit, result['selected'])
        print(f'  Equivalence: PASS for all {patterns} input patterns')
    return results


# 11. Demo entry point
# Expected: small 3 LUTs/2 levels -> 2 LUTs/1 level; larger 6/3 -> 5/2.
def main():
    circuit, candidates = small_example()
    results = run_demo('Example 1: one shared node', circuit, candidates)

    # Delete the u LUT while y,z still need it, to demonstrate missing-signal detection.
    invalid = dict(results['Shared boundary']['selected'])
    del invalid['u']
    try:
        analyze_cover(circuit, invalid)

    # Catch the expected error; correctly rejected means the check succeeded, not a crash.
    except ValueError as error:
        print('\nInvalid cover correctly rejected:', error)
    else:
        raise AssertionError('Missing boundary signal was not detected')
    circuit, candidates = larger_example()
    run_demo('Example 2: multiple shared nodes', circuit, candidates)


# Run the demo only when this file is executed directly, not when imported.
if __name__ == '__main__':
    main()
