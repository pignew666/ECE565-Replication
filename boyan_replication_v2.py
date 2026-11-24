"""ECE 565 - Boyan Li: replication and fanout handling.
Run: python boyan_replication_submit.py

expand_replication is the integration module. The remaining search driver is a
small standalone demo harness, not the team's full mapper. Cut/gate matches are
supplied, not generated. Costs are illustrative, not measured library data.
"""
from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from math import isfinite


# 1. Circuit input: dependency-ordered AND nodes and optional inverted edges.
def source(signal):
    return signal[1:] if signal.startswith('!') else signal


def signal_value(signal, values):
    return values[source(signal)] ^ int(signal.startswith('!'))


class Circuit:
    def __init__(self, inputs, gates, outputs):
        self.inputs, self.gates, self.outputs = tuple(inputs), dict(gates), tuple(outputs)
        names = [*self.inputs, *self.gates]
        if not self.inputs or len(set(names)) != len(names):
            raise ValueError('Input and gate names must be unique')
        if any(not isinstance(n, str) or not n.isidentifier() for n in names):
            raise ValueError('Use identifier names such as a or gate1')
        known = set(self.inputs)
        for root, pins in self.gates.items():
            if len(pins) != 2 or any(source(s) not in known for s in pins):
                raise ValueError('Each AND must use two previously defined signals')
            known.add(root)
        if not self.outputs or any(source(s) not in known for s in self.outputs):
            raise ValueError('Invalid outputs')
        live, pending = set(), [source(s) for s in self.outputs]
        while pending:
            n = pending.pop()
            if n in self.gates and n not in live:
                live.add(n)
                pending.extend(source(s) for s in self.gates[n])
        self.order = tuple(n for n in self.gates if n in live)
        self.rank = {n: i for i, n in enumerate([*self.inputs, *self.order])}
        self.fanout = dict.fromkeys(self.order, 0)
        for n in self.order:
            for s in self.gates[n]:
                if source(s) in self.fanout:
                    self.fanout[source(s)] += 1
        for s in self.outputs:
            if source(s) in self.fanout:
                self.fanout[source(s)] += 1

    def cut_logic(self, root, leaves):
        # Stop at the supplied cut boundary; do not generate new cuts.
        if (root not in self.order or not leaves or len(set(leaves)) != len(leaves)
                or any(n not in self.rank or self.rank[n] >= self.rank[root] for n in leaves)):
            raise ValueError('Invalid cut boundary')
        inside, reached = set(), set()

        def visit(n):
            if n in leaves:
                reached.add(n)
            elif n in self.inputs:
                raise ValueError('Cut misses a primary-input path')
            elif n not in inside:
                inside.add(n)
                for s in self.gates[n]:
                    visit(source(s))

        visit(root)
        if reached != set(leaves):
            raise ValueError('Unused cut boundary signal')
        table = []
        for bits in product((0, 1), repeat=len(leaves)):
            values = dict(zip(leaves, bits))
            for n in sorted(inside, key=self.rank.get):
                a, b = self.gates[n]
                values[n] = signal_value(a, values) & signal_value(b, values)
            table.append(values[root])
        return tuple(sorted(inside, key=self.rank.get)), tuple(table)


# 2. Supplied match: leaves are already ordered by physical input-pin assignment.
@dataclass(frozen=True)
class Option:
    root: str
    name: str
    leaves: tuple
    truth: tuple
    caps: tuple
    intrinsic: float = 1.0
    resistance: float = 0.3
    area: float = 1.0


def check_option(circuit, option):
    inside, expected = circuit.cut_logic(option.root, option.leaves)
    numbers = (*option.caps, option.intrinsic, option.resistance, option.area)
    if len(option.caps) != len(option.leaves) or any(not isfinite(x) or x < 0 for x in numbers):
        raise ValueError('Invalid pin capacitances or cell costs')
    if option.truth != expected:
        raise ValueError('Gate truth table does not match the cut function')
    return inside


def demo_option(circuit, root, leaves, name='demo_cell'):
    # A synthetic truth-table cell for testing; replace with actual library matches.
    _, truth = circuit.cut_logic(root, leaves)
    return Option(root, name, tuple(leaves), truth, (1.0,) * len(leaves))


# 3. Partition sinks exactly once, without permutations of identical group labels.
def partitions(items, limit):
    groups = []

    def assign(i):
        if i == len(items):
            yield tuple(tuple(g) for g in groups)
            return
        for group in groups:
            group.append(items[i])
            yield from assign(i + 1)
            group.pop()
        if len(groups) < limit:
            groups.append([items[i]])
            yield from assign(i + 1)
            groups.pop()

    yield from assign(0)


# 4. State S: requests, physical instances, explicit wires, area, and path bound H.
# A request is (consumer ID, pin index, pin capacitance, downstream delay).
def initial_state(circuit, output_load=1.0):
    if not isfinite(output_load) or output_load < 0:
        raise ValueError('Output load must be finite and nonnegative')
    state = {'requests': {}, 'instances': {}, 'wires': {}, 'area': 0.0, 'H': 0.0,
             'output_loads': (output_load,) * len(circuit.outputs)}
    for i, signal in enumerate(circuit.outputs):
        n = source(signal)
        if n in circuit.inputs:
            state['wires'][('@out', i)] = n
        else:
            state['requests'].setdefault(n, []).append(('@out', i, output_load, 0.0))
    return state


def validate_connections(circuit, state, complete=False):
    # Every required sink has exactly one wire or one pending request.
    expected = {('@out', i): source(s) for i, s in enumerate(circuit.outputs)}
    for inst, data in state['instances'].items():
        for pin, leaf in enumerate(data['option'].leaves):
            expected[(inst, pin)] = leaf
    assigned = {}
    for sink, driver in state['wires'].items():
        if driver in circuit.inputs:
            logical_signal = driver
        elif driver in state['instances']:
            logical_signal = state['instances'][driver]['option'].root
        else:
            raise ValueError('Missing producer instance')
        if sink not in expected or expected[sink] != logical_signal:
            raise ValueError('Driver does not provide the required exposed signal')
        assigned[sink] = logical_signal
    for logical_signal, requests in state['requests'].items():
        for consumer, pin, cap, tail in requests:
            sink = (consumer, pin)
            if sink in assigned or expected.get(sink) != logical_signal:
                raise ValueError('Duplicate or incorrect pending request')
            if consumer == '@out':
                if cap != state['output_loads'][pin] or tail != 0.0:
                    raise ValueError('Incorrect output request load or delay')
            else:
                cell = state['instances'][consumer]
                if cap != cell['option'].caps[pin] or abs(tail-cell['F']) > 1e-9:
                    raise ValueError('Incorrect request load or downstream delay')
            assigned[sink] = logical_signal
    if set(assigned) != set(expected) or (complete and state['requests']):
        raise ValueError('Unresolved or missing connections')
    for inst in state['instances']:
        if inst not in state['wires'].values():
            raise ValueError('Unused physical instance')
    if abs(state['area']-sum(d['option'].area for d in state['instances'].values())) > 1e-9:
        raise ValueError('Incorrect area accounting')


# 5. Boyan's integration module: return every legal replication branch at p.
def expand_replication(circuit, p, state, options, allow_replication=True,
                       consistency_check=None):
    if p not in state['requests']:
        raise ValueError('p must be pending')
    if p != max(state['requests'], key=circuit.rank.get):
        raise ValueError('Process the latest pending node first')
    if any(d['option'].root == p for d in state['instances'].values()):
        raise ValueError('All requests for p must be expanded together')
    validate_connections(circuit, state)
    choices = tuple(options.get(p, ()))
    for option in choices:
        if option.root != p:
            raise ValueError('Option root differs from p')
        check_option(circuit, option)
    requests = state['requests'][p]
    limit = min(circuit.fanout[p], len(requests)) if allow_replication else 1
    for groups in partitions(requests, limit):
        # Assign a cut/cell option independently to each physical copy.
        for assignment in product(choices, repeat=len(groups)):
            branch = deepcopy(state)
            del branch['requests'][p]
            for number, (group, option) in enumerate(zip(groups, assignment)):
                inst = f'{p}#{number}'
                load = sum(q[2] for q in group)
                tail = max(q[3] for q in group)
                forward_delay = option.intrinsic + option.resistance * load + tail
                inside = check_option(circuit, option)
                branch['instances'][inst] = {'option': option, 'load': load,
                                            'F': forward_delay, 'inside': inside}
                branch['area'] += option.area
                for consumer, pin, _, _ in group:
                    branch['wires'][(consumer, pin)] = inst
                for pin, leaf in enumerate(option.leaves):
                    if leaf in circuit.inputs:
                        branch['wires'][(inst, pin)] = leaf
                        branch['H'] = max(branch['H'], forward_delay)
                    else:
                        request = (inst, pin, option.caps[pin], forward_delay)
                        branch['requests'].setdefault(leaf, []).append(request)
            validate_connections(circuit, branch)
            # Optional extra legality rules from the team's mapper.
            if consistency_check is None or consistency_check(circuit, branch):
                yield branch


# 6. Demo harness ONLY: recurse over the integration module on tiny examples.
def run_search(circuit, options, allow_replication=True, max_states=100000):
    results, visited = [], 0

    def visit(state):
        nonlocal visited
        visited += 1
        if visited > max_states:
            raise RuntimeError('Search limit exceeded; no partial result returned')
        if not state['requests']:
            validate_connections(circuit, state, complete=True)
            results.append(state)
            return
        p = max(state['requests'], key=circuit.rank.get)
        for branch in expand_replication(circuit, p, state, options, allow_replication):
            visit(branch)

    visit(initial_state(circuit))
    if not results:
        raise ValueError('No complete mapping for the supplied options')
    return results


# 7. Independent forward timing check and exhaustive Boolean verification.
def verify(circuit, state):
    validate_connections(circuit, state, complete=True)
    if len(circuit.inputs) > 12:
        raise ValueError('Exhaustive verification is limited to 12 primary inputs')
    order = sorted(state['instances'], key=lambda i: circuit.rank[state['instances'][i]['option'].root])
    arrivals = dict.fromkeys(circuit.inputs, 0.0)
    for inst in order:
        cell = state['instances'][inst]
        option = cell['option']
        load = 0.0
        for (consumer, pin), driver in state['wires'].items():
            if driver == inst:
                load += state['output_loads'][pin] if consumer == '@out' else state['instances'][consumer]['option'].caps[pin]
        if abs(load-cell['load']) > 1e-9:
            raise AssertionError('Stored load differs from actual load')
        arrivals[inst] = max(arrivals[state['wires'][(inst, pin)]] for pin in range(len(option.leaves))) + option.intrinsic + option.resistance * load
    outputs = tuple(arrivals[state['wires'][('@out', i)]] for i in range(len(circuit.outputs)))
    if abs(max(outputs)-state['H']) > 1e-9:
        raise AssertionError('Top-down H differs from independent forward timing')
    for bits in product((0, 1), repeat=len(circuit.inputs)):
        original = dict(zip(circuit.inputs, bits))
        mapped = dict(original)
        for n in circuit.order:
            a, b = circuit.gates[n]
            original[n] = signal_value(a, original) & signal_value(b, original)
        for inst in order:
            option = state['instances'][inst]['option']
            row = 0
            for pin in range(len(option.leaves)):
                row = 2 * row + mapped[state['wires'][(inst, pin)]]
            mapped[inst] = option.truth[row]
        for i, signal in enumerate(circuit.outputs):
            got = mapped[state['wires'][('@out', i)]] ^ int(signal.startswith('!'))
            if got != signal_value(signal, original):
                raise AssertionError(f'Output mismatch: {bits}')
    return outputs, 2 ** len(circuit.inputs)


# 8. Small demonstration of the returned candidates; team retention is separate.
def keep_candidates(results, delta=0.6, r=16):
    if delta < 0 or r < 1:
        raise ValueError('Use delta >= 0 and R >= 1')
    best = min(s['H'] for s in results)
    return sorted((s for s in results if s['H'] <= best + delta + 1e-9),
                  key=lambda s: (round(s['H'], 9), s['area']))[:r]


def recover_area(candidates, deadline):
    allowed = [s for s in candidates if s['H'] <= deadline + 1e-9]
    if not allowed:
        raise ValueError('No retained candidate meets the deadline')
    return min(allowed, key=lambda s: (s['area'], s['H']))


# 9. Supplied fixtures stand in for the group's loader and cut/cell matches.
def examples():
    first = Circuit(['a', 'b', 'c', 'd', 'e'], {
        'u': ('a', 'b'), 'y': ('u', 'c'), 'z': ('u', 'd'), 'w': ('u', '!e')
    }, ['y', 'z', 'w'])
    second = Circuit(['a', 'b', 'c', 'd', 'e', 'f'], {
        't': ('a', 'b'), 'u': ('t', 'c'), 'y': ('u', 'd'), 'z': ('u', 'e'), 'w': ('u', '!f')
    }, ['y', 'z', 'w'])
    third = Circuit(['a', 'b', 'c', 'd'], {
        'u': ('a', 'b'), 'y': ('u', 'c'), 'z': ('u', 'd')
    }, ['y', 'z'])
    cases = []
    for circuit in (first, second, third):
        options = {n: [demo_option(circuit, n, tuple(dict.fromkeys(source(s) for s in circuit.gates[n])))] for n in circuit.order}
        cases.append((circuit, options))
    cases[2][1]['y'].append(demo_option(third, 'y', ('a', 'b', 'c'), 'absorbed_y'))
    cases[2][1]['z'].append(demo_option(third, 'z', ('a', 'b', 'd'), 'absorbed_z'))
    return cases


def describe(state):
    for inst, cell in sorted(state['instances'].items()):
        option = cell['option']
        sinks = [f'{n}[{pin}]' for (n, pin), driver in state['wires'].items() if driver == inst]
        print(f"  {inst}: {option.name} inputs={option.leaves}, load={cell['load']:.1f}, F={cell['F']:.2f}; drives {', '.join(sinks)}")


# 10. Run the complete standalone demo without external files or packages.
def main():
    print('Boyan: replication / fanout module with a small top-down demo harness')
    print('Synthetic cells: d=1, R=0.3, input caps=1, area=1; output load=1')
    for index, (circuit, options) in enumerate(examples(), 1):
        results = run_search(circuit, options)
        for state in results:
            _, patterns = verify(circuit, state)
        print(f'\nExample {index}: {len(results)} complete candidates')
        if index <= 2:
            print('Extra u copies | Area | Worst delay')
            for copies in (1, 2, 3):
                group = [s for s in results if sum(d['option'].root == 'u' for d in s['instances'].values()) == copies]
                chosen = min(group, key=lambda s: s['H'])
                print(f"{copies - 1:14} | {chosen['area']:4.0f} | {chosen['H']:.2f}")
            retained = keep_candidates(results)
            deadlines = (2.6, 2.9, 3.2) if index == 1 else (4.5,)
            for deadline in deadlines:
                chosen = recover_area(retained, deadline)
                print(f"Deadline {deadline:.2f}: area={chosen['area']:.0f}, delay={chosen['H']:.2f}")
        else:
            # Include a mixed cover: absorbed u in y, independent u feeding z.
            mixed = next(s for s in results if any(d['option'].name == 'absorbed_y' for d in s['instances'].values()) and any(d['option'].root == 'u' for d in s['instances'].values()))
            print('Legal mixed cover: y absorbs u; z still has an exposed u producer')
            describe(mixed)
            fastest = min(results, key=lambda s: (s['H'], s['area']))
            print(f"Fastest supplied-option result: area={fastest['area']:.0f}, delay={fastest['H']:.2f}")
        print(f'PASS: all {len(results)} candidates, {patterns} input patterns each; H matches forward timing')
    print('\nAll demos completed successfully.')


if __name__ == '__main__':
    main()
