"""Meter-tree reconciliation engine.

The carbon engineer receives, for one site and month, readings for every
metering point (总表与分表同时入账).  Because a parent meter already measures
its whole subtree, summing every row double counts.  A non-duplicating total
corresponds to a *frontier* of the meter tree: a set of meters such that

* the selected meters are pairwise incomparable (no ancestor/descendant pair
  is ever counted together), and
* every leaf (末端点位) is covered by exactly one selected subtree.

For every node the engine keeps two structural options — take the node's own
reading, or recurse into its children — and uses tree DP to find, across all
feasible frontiers, the minimum and the maximum electricity total together
with the meters each extreme uses.  When the two extremes differ the period
cannot be settled automatically and is flagged for manual review.

Inputs are strict: duplicate or negative readings and an invalid meter tree
reject the whole run and no total is ever produced.  When no frontier exists
because a whole branch lacks readings, the uncovered branches are reported.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation

from .engine import DECIMAL_RE, MONTH_RE, _is_blank_line, _raise_csv_field_limit
from .spec import normalize_header

MAX_METERS = 200

# Fixed standard columns plus a few common source aliases.  Matching uses the
# same NFKC/case/whitespace folding as the ledger standardizer.
READING_ALIASES: dict[str, tuple[str, ...]] = {
    "厂区": ("厂区", "工厂", "厂站", "site"),
    "月份": ("月份", "统计月份", "年月", "period"),
    "计量点": ("计量点", "表计", "计量点名称", "meter", "meter_point"),
    "用电量": ("用电量", "用电量(kWh)", "电量", "kwh", "electricity_kwh"),
}
TREE_ALIASES: dict[str, tuple[str, ...]] = {
    "计量点": ("计量点", "表计", "计量点名称", "meter", "meter_point"),
    "上级计量点": ("上级计量点", "上级", "父表", "父计量点", "parent", "parent_meter"),
}


class ReconcileError(ValueError):
    """Fatal problem that prevents the reconciliation from even starting."""


@dataclass(frozen=True)
class ReadingRow:
    line: int
    site: str
    month: str
    meter: str
    value_text: str
    value: Decimal


@dataclass
class ReadingUpload:
    rows: list[ReadingRow]
    sites: list[str]
    months: list[str]
    ignored_columns: list[str] = field(default_factory=list)


@dataclass
class TreeUpload:
    tree: MeterTree | None
    problems: list[str] = field(default_factory=list)
    ignored_columns: list[str] = field(default_factory=list)


@dataclass
class MeterTree:
    order: list[str]
    parent: dict[str, str | None]
    children: dict[str, list[str]]
    root: str
    preorder: list[str]
    depth: dict[str, int]
    leaves: set[str]

    @property
    def node_count(self) -> int:
        return len(self.order)


@dataclass(frozen=True)
class Combo:
    total: Decimal
    meters: tuple[str, ...]


@dataclass(frozen=True)
class GapBranch:
    """A maximal subtree that contains no readable meter at all."""

    node: str
    path: tuple[str, ...]
    leaves: tuple[str, ...]


@dataclass(frozen=True)
class SliceRow:
    meter: str
    value_text: str
    value: Decimal
    is_leaf: bool
    in_min: bool
    in_max: bool


@dataclass
class ReconcileResult:
    site: str
    month: str
    feasible: bool
    reading_count: int
    tree_node_count: int
    leaf_count: int
    combo_count: int = 0
    min_combo: Combo | None = None
    max_combo: Combo | None = None
    needs_review: bool = False
    gaps: list[GapBranch] = field(default_factory=list)
    slice_rows: list[SliceRow] = field(default_factory=list)
    unread_meters: list[str] = field(default_factory=list)
    ignored_columns: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Shared CSV helpers
# --------------------------------------------------------------------------- #


def _decode(raw: bytes, kind: str) -> str:
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ReconcileError(
            f"{kind}不是 UTF-8 编码（支持 UTF-8 与带 BOM 的 UTF-8）"
        ) from exc


def _records(text: str, kind: str) -> list[tuple[list[str], int]]:
    _raise_csv_field_limit()
    out: list[tuple[list[str], int]] = []
    reader = csv.reader(io.StringIO(text, newline=""))
    previous = 0
    try:
        for row in reader:
            out.append((list(row), previous + 1))
            previous = reader.line_num
    except csv.Error as exc:
        raise ReconcileError(f"{kind}CSV 解析失败：{exc}") from exc
    return out


def _resolve_headers(
    headers: list[str],
    aliases: dict[str, tuple[str, ...]],
    kind: str,
) -> tuple[dict[str, int] | None, list[str], list[str]]:
    """Map the fixed standard columns onto source columns 1:1.

    Returns (standard -> source index, problems, ignored columns).  Duplicate
    headers, missing standards and several source columns claiming the same
    standard column are all fatal — nothing is guessed from column order.
    """
    problems: list[str] = []
    normalized = [normalize_header(h) for h in headers]

    seen: dict[str, list[int]] = {}
    for index, norm in enumerate(normalized):
        if norm:
            seen.setdefault(norm, []).append(index)
    for positions in seen.values():
        if len(positions) > 1:
            cols = "、".join(f"第 {p + 1} 列" for p in positions)
            label = headers[positions[0]].strip() or "（空表头）"
            problems.append(f"{kind}存在重复表头 {label!r}（{cols}）")

    accepted: dict[str, set[str]] = {
        std: {normalize_header(a) for a in names if normalize_header(a)}
        for std, names in aliases.items()
    }
    mapping: dict[str, int] = {}
    matched_sources: set[int] = set()
    for std, accepted_headers in accepted.items():
        hits = [i for i, norm in enumerate(normalized) if norm and norm in accepted_headers]
        if not hits:
            problems.append(f"{kind}缺少必填标准列 {std!r}")
        elif len(hits) > 1:
            labels = "、".join(repr(headers[i].strip() or "（空表头）") for i in hits)
            problems.append(f"{kind}的标准列 {std!r} 同时命中多个来源列：{labels}")
        else:
            mapping[std] = hits[0]
            matched_sources.add(hits[0])

    ignored = [
        f"第 {i + 1} 列 {headers[i].strip() or '（空表头）'}"
        for i in range(len(headers))
        if i not in matched_sources and normalized[i]
    ]
    return (None if problems else mapping), problems, ignored


# --------------------------------------------------------------------------- #
# Meter tree
# --------------------------------------------------------------------------- #


def parse_meter_tree(raw: bytes) -> TreeUpload:
    """Parse and validate the single-root meter tree CSV.

    A fatal parse/encoding problem raises :class:`ReconcileError`; structural
    problems (unknown parent, cycles, multiple roots, size cap) are collected
    on the returned upload so the web layer can show them all at once.
    """
    text = _decode(raw, "表计树文件")
    records = _records(text, "表计树文件")
    if not records:
        return TreeUpload(tree=None, problems=["表计树文件为空：缺少表头行"])

    headers, _ = records[0]
    mapping, problems, ignored = _resolve_headers(headers, TREE_ALIASES, "表计树文件")
    if mapping is None:
        return TreeUpload(tree=None, problems=problems, ignored_columns=ignored)

    name_col = mapping["计量点"]
    parent_col = mapping["上级计量点"]
    order: list[str] = []
    parent: dict[str, str | None] = {}
    seen_names: set[str] = set()

    for row, line_number in records[1:]:
        if _is_blank_line(row):
            continue
        if len(row) != len(headers):
            problems.append(
                f"表计树第 {line_number} 行列数不符：表头 {len(headers)} 列，该行 {len(row)} 列"
            )
            continue
        name = row[name_col].strip()
        parent_name = row[parent_col].strip() or None
        if not name:
            problems.append(f"表计树第 {line_number} 行计量点名称为空")
            continue
        if name in seen_names:
            problems.append(f"表计树第 {line_number} 行计量点 {name!r} 重复定义")
            continue
        seen_names.add(name)
        order.append(name)
        parent[name] = parent_name

    if len(order) > MAX_METERS:
        problems.append(
            f"表计树共有 {len(order)} 个计量点，超过单厂区 {MAX_METERS} 个点位上限"
        )
    # Referential integrity.
    roots: list[str] = []
    for name in order:
        parent_name = parent[name]
        if parent_name is None:
            roots.append(name)
        elif parent_name == name:
            problems.append(f"计量点 {name!r} 的上级计量点指向自身")
        elif parent_name not in seen_names:
            problems.append(
                f"计量点 {name!r} 的上级计量点 {parent_name!r} 未在计量点列中定义"
            )
    if len(roots) > 1:
        problems.append("表计树必须只有一个根表（上级计量点留空），当前有多个根表："
                        + "、".join(repr(r) for r in roots))
    if not roots and not problems:
        problems.append("表计树没有根表（上级计量点留空的计量点）")

    if problems:
        return TreeUpload(tree=None, problems=problems, ignored_columns=ignored)

    children: dict[str, list[str]] = {name: [] for name in order}
    for name in order:
        parent_name = parent[name]
        if parent_name is not None:
            children[parent_name].append(name)

    root = roots[0]
    preorder: list[str] = []
    depth: dict[str, int] = {}
    stack: list[tuple[str, int]] = [(root, 0)]
    while stack:
        name, level = stack.pop()
        preorder.append(name)
        depth[name] = level
        for child in reversed(children[name]):
            stack.append((child, level + 1))

    if len(preorder) != len(order):
        # Every non-root has a known, distinct parent, so a short DFS means a
        # cycle (or a chain leading into one).  Name the nodes on the cycle.
        reachable = set(preorder)
        cyclic = [name for name in order if name not in reachable]
        leaders = cyclic[:5]
        more = f" 等 {len(cyclic)} 个" if len(cyclic) > 5 else ""
        problems.append("表计树存在环路，涉及计量点：" + "、".join(repr(n) for n in leaders) + more)
        return TreeUpload(tree=None, problems=problems, ignored_columns=ignored)

    leaves = {name for name in order if not children[name]}
    tree = MeterTree(
        order=order,
        parent=parent,
        children=children,
        root=root,
        preorder=preorder,
        depth=depth,
        leaves=leaves,
    )
    return TreeUpload(tree=tree, ignored_columns=ignored)


# --------------------------------------------------------------------------- #
# Readings
# --------------------------------------------------------------------------- #


def parse_readings(raw: bytes) -> tuple[ReadingUpload | None, list[str]]:
    """Parse the standardized readings CSV and run whole-file validation.

    Returns the parsed upload plus structural/value problems.  Fatal
    encoding/CSV-syntax problems raise :class:`ReconcileError`.
    """
    text = _decode(raw, "读数文件")
    records = _records(text, "读数文件")
    if not records:
        return None, ["读数文件为空：缺少表头行"]

    headers, _ = records[0]
    mapping, problems, ignored = _resolve_headers(headers, READING_ALIASES, "读数文件")
    if mapping is None:
        return None, problems

    site_col = mapping["厂区"]
    month_col = mapping["月份"]
    meter_col = mapping["计量点"]
    value_col = mapping["用电量"]

    rows: list[ReadingRow] = []
    first_seen: dict[tuple[str, str, str], int] = {}
    sites: set[str] = set()
    months: set[str] = set()

    for row, line_number in records[1:]:
        if _is_blank_line(row):
            continue
        if len(row) != len(headers):
            problems.append(
                f"读数文件第 {line_number} 行列数不符：表头 {len(headers)} 列，该行 {len(row)} 列"
            )
            continue
        site = row[site_col].strip()
        month = row[month_col].strip()
        meter = row[meter_col].strip()
        value_text = row[value_col].strip()

        missing = [
            label
            for value, label in (
                (site, "厂区"),
                (month, "月份"),
                (meter, "计量点"),
                (value_text, "用电量"),
            )
            if not value
        ]
        if missing:
            problems.append(
                f"读数文件第 {line_number} 行必填单元格为空：{'、'.join(missing)}"
            )
            continue
        if not MONTH_RE.fullmatch(month):
            problems.append(
                f"读数文件第 {line_number} 行月份 {month!r} 不符合 YYYY-MM 格式"
            )
            continue
        if not DECIMAL_RE.fullmatch(value_text):
            problems.append(
                f"读数文件第 {line_number} 行计量点 {meter!r} 的用电量 {value_text!r} 不是有限小数"
            )
            continue
        try:
            value = Decimal(value_text)
        except InvalidOperation:  # grammar already rejects NaN/inf; defensive
            problems.append(
                f"读数文件第 {line_number} 行计量点 {meter!r} 的用电量 {value_text!r} 无法解析"
            )
            continue
        if value < 0:
            problems.append(
                f"读数文件第 {line_number} 行计量点 {meter!r} 的用电量 {value_text} 为负数"
            )
            continue

        key = (site, month, meter)
        if key in first_seen:
            problems.append(
                f"读数文件第 {line_number} 行计量点 {meter!r} 在 {site} / {month} "
                f"重复读数（首次出现于第 {first_seen[key]} 行）"
            )
            continue
        first_seen[key] = line_number
        rows.append(
            ReadingRow(
                line=line_number,
                site=site,
                month=month,
                meter=meter,
                value_text=value_text,
                value=value,
            )
        )
        sites.add(site)
        months.add(month)

    if problems:
        return None, problems
    return (
        ReadingUpload(
            rows=rows,
            sites=sorted(sites),
            months=sorted(months),
            ignored_columns=ignored,
        ),
        [],
    )


# --------------------------------------------------------------------------- #
# Tree DP over feasible frontiers
# --------------------------------------------------------------------------- #


@dataclass
class _NodeState:
    feasible: bool
    count: int = 0
    min_sum: Decimal | None = None
    min_combo: tuple[str, ...] = ()
    max_sum: Decimal | None = None
    max_combo: tuple[str, ...] = ()


def _better_min(
    current: _NodeState | None,
    total: Decimal,
    combo: tuple[str, ...],
) -> tuple[Decimal, tuple[str, ...]]:
    if current is None or current.min_sum is None:
        return total, combo
    candidate_key = (total, len(combo), combo)
    current_key = (current.min_sum, len(current.min_combo), current.min_combo)
    return (total, combo) if candidate_key < current_key else (
        current.min_sum,
        current.min_combo,
    )


def _better_max(
    current: _NodeState | None,
    total: Decimal,
    combo: tuple[str, ...],
) -> tuple[Decimal, tuple[str, ...]]:
    if current is None or current.max_sum is None:
        return total, combo
    candidate_key = (total, -len(combo), combo)
    current_key = (current.max_sum, -len(current.max_combo), current.max_combo)
    return (total, combo) if candidate_key > current_key else (
        current.max_sum,
        current.max_combo,
    )


def _gap_branches(
    tree: MeterTree,
    states: dict[str, _NodeState],
    readings: dict[str, Decimal],
) -> list[GapBranch]:
    """Maximal unreadable subtrees whose leaves can never be covered.

    A gap branch is the highest node whose whole subtree is unreadable (the
    node itself carries no reading) and infeasible.  When only some child
    subtrees fail, recursion pinpoints those branches instead of blaming the
    common ancestor, whose other children still cover their leaves.
    """
    gaps: list[GapBranch] = []

    def walk(name: str, path: tuple[str, ...]) -> None:
        if states[name].feasible:
            return  # the subtree is covered, nothing to report inside
        if readings.get(name) is not None:
            return  # defensive: a readable node itself is always feasible
        kids = tree.children[name]
        infeasible_kids = [c for c in kids if not states[c].feasible]
        if not infeasible_kids:
            # An unread leaf (no children) is its own one-leaf gap.
            gaps.append(GapBranch(node=name, path=path + (name,), leaves=(name,)))
            return
        if len(infeasible_kids) == len(kids):
            # Every child fails and this node cannot help: maximal gap here.
            gaps.append(
                GapBranch(
                    node=name,
                    path=path + (name,),
                    leaves=_collect_leaves(tree, name),
                )
            )
            return
        # Mixed children: descend only into the failing subtrees.
        for child in infeasible_kids:
            walk(child, path + (name,))

    walk(tree.root, ())
    return gaps


def _is_within(descendant: str, ancestor: str, tree: MeterTree) -> bool:
    name: str | None = descendant
    while name is not None:
        if name == ancestor:
            return True
        name = tree.parent[name]
    return False


def _collect_leaves(tree: MeterTree, node: str) -> tuple[str, ...]:
    return tuple(
        name for name in tree.preorder
        if name in tree.leaves and _is_within(name, node, tree)
    )


def reconcile(
    tree: MeterTree,
    upload: ReadingUpload,
    site: str,
    month: str,
) -> ReconcileResult:
    site = (site or "").strip()
    month = (month or "").strip()
    if not site or not month:
        raise ReconcileError("请选择厂区与月份")
    if site not in upload.sites:
        raise ReconcileError(f"读数文件中不存在厂区 {site!r}，请重新选择")
    if month not in upload.months:
        raise ReconcileError(f"读数文件中不存在月份 {month!r}，请重新选择")

    slice_rows = [r for r in upload.rows if r.site == site and r.month == month]
    if not slice_rows:
        raise ReconcileError(f"{site} 在 {month} 没有任何读数记录，无法核对")

    readings: dict[str, Decimal] = {}
    raw_text: dict[str, str] = {}
    unknown: list[str] = []
    for row in slice_rows:
        if row.meter not in tree.parent:
            unknown.append(row.meter)
            continue
        readings[row.meter] = row.value
        raw_text[row.meter] = row.value_text
    if unknown:
        raise ReconcileError(
            "读数中的计量点不在表计树内：" + "、".join(repr(n) for n in sorted(set(unknown)))
        )

    states: dict[str, _NodeState] = {}
    for name in reversed(tree.preorder):  # children before parents
        state = _NodeState(feasible=False)
        kids = tree.children[name]
        child_option_feasible = bool(kids) and all(states[c].feasible for c in kids)

        if readings.get(name) is not None:
            # Option A: this meter represents its whole subtree.
            state.feasible = True
            state.count += 1
            state.min_sum, state.min_combo = readings[name], (name,)
            state.max_sum, state.max_combo = readings[name], (name,)

        if child_option_feasible:
            # Option B: drop this node and combine the children's frontiers.
            total = sum((states[c].min_sum for c in kids), Decimal("0"))
            combo = tuple(m for c in kids for m in states[c].min_combo)
            state.min_sum, state.min_combo = _better_min(state, total, combo)
            total_max = sum((states[c].max_sum for c in kids), Decimal("0"))
            combo_max = tuple(m for c in kids for m in states[c].max_combo)
            state.max_sum, state.max_combo = _better_max(state, total_max, combo_max)
            child_count = 1
            for child in kids:
                child_count *= states[child].count
            state.count += child_count
            state.feasible = True

        states[name] = state

    root_state = states[tree.root]
    result = ReconcileResult(
        site=site,
        month=month,
        feasible=root_state.feasible,
        reading_count=len(slice_rows),
        tree_node_count=tree.node_count,
        leaf_count=len(tree.leaves),
        ignored_columns=list(upload.ignored_columns),
    )

    if not root_state.feasible:
        result.gaps = _gap_branches(tree, states, readings)
        # Defensive: a failed root must yield at least one gap branch.
        if not result.gaps:
            result.gaps = [
                GapBranch(
                    node=tree.root,
                    path=(tree.root,),
                    leaves=_collect_leaves(tree, tree.root),
                )
            ]
        return result

    result.combo_count = root_state.count
    result.min_combo = Combo(total=root_state.min_sum, meters=root_state.min_combo)
    result.max_combo = Combo(total=root_state.max_sum, meters=root_state.max_combo)
    result.needs_review = root_state.min_sum != root_state.max_sum

    min_set = set(root_state.min_combo)
    max_set = set(root_state.max_combo)
    result.slice_rows = [
        SliceRow(
            meter=row.meter,
            value_text=raw_text[row.meter],
            value=row.value,
            is_leaf=row.meter in tree.leaves,
            in_min=row.meter in min_set,
            in_max=row.meter in max_set,
        )
        for row in slice_rows
    ]
    result.unread_meters = [
        name for name in tree.preorder if name not in readings
    ]
    return result
