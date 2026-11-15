"""Tests for the meter-tree reconciliation engine."""

from decimal import Decimal

import pytest

from app.reconcile import (
    MAX_METERS,
    ReconcileError,
    reconcile,
    parse_meter_tree,
    parse_readings,
)

# 厂区总表
# ├── 车间A总
# │   ├── A-进线1
# │   └── A-进线2
# └── 车间B总
#     ├── B-进线1
#     └── B-进线2
TREE_CSV = (
    "计量点,上级计量点\n"
    "厂区总表,\n"
    "车间A总,厂区总表\n"
    "A-进线1,车间A总\n"
    "A-进线2,车间A总\n"
    "车间B总,厂区总表\n"
    "B-进线1,车间B总\n"
    "B-进线2,车间B总\n"
)

ALL_NAMES = ["厂区总表", "车间A总", "A-进线1", "A-进线2", "车间B总", "B-进线1", "B-进线2"]


def make_tree(text=TREE_CSV):
    upload = parse_meter_tree(text.encode("utf-8"))
    assert upload.tree is not None, upload.problems
    return upload.tree


def make_readings(rows, header="厂区,月份,计量点,用电量\n"):
    lines = [header]
    lines.extend(rows)
    upload, problems = parse_readings("\n".join(lines).encode("utf-8"))
    assert upload is not None, problems
    return upload


def row(meter, value, site="华东一厂", month="2025-01"):
    return f"{site},{month},{meter},{value}"


def run(rows, site="华东一厂", month="2025-01", tree_text=TREE_CSV):
    tree = make_tree(tree_text)
    upload = make_readings(rows)
    return reconcile(tree, upload, site, month)


# --------------------------------------------------------------------------- #
# Frontier / DP semantics
# --------------------------------------------------------------------------- #


def test_all_feasible_frontiers_min_max_and_combo_count():
    # Five feasible frontiers; extremes come from distinct structures.
    rows = [
        row("厂区总表", 12500),
        row("车间A总", 6200),
        row("A-进线1", 3000),
        row("A-进线2", 3200),
        row("车间B总", 6000),
        row("B-进线1", 2700),
        row("B-进线2", 3300),
    ]
    result = run(rows)
    assert result.feasible
    assert result.combo_count == 5
    assert result.needs_review is True
    assert result.min_combo.total == Decimal("12200")
    assert set(result.min_combo.meters) == {"车间A总", "车间B总"}
    assert result.max_combo.total == Decimal("12500")
    assert result.max_combo.meters == ("厂区总表",)
    # Ancestor and descendant never co-occur in either extreme.
    for combo in (result.min_combo.meters, result.max_combo.meters):
        assert "厂区总表" not in combo or len(combo) == 1


def test_equal_extremes_means_no_review():
    rows = [
        row("厂区总表", 12000),
        row("车间A总", 6000),
        row("A-进线1", 3000),
        row("A-进线2", 3000),
        row("车间B总", 6000),
        row("B-进线1", 3000),
        row("B-进线2", 3000),
    ]
    result = run(rows)
    assert result.feasible
    assert result.needs_review is False
    assert result.min_combo.total == result.max_combo.total == Decimal("12000")


def test_leaf_only_readings_have_single_frontier():
    rows = [
        row("A-进线1", 3000),
        row("A-进线2", 3200),
        row("B-进线1", 2700),
        row("B-进线2", 3300),
    ]
    result = run(rows)
    assert result.feasible
    assert result.combo_count == 1
    assert result.min_combo.meters == result.max_combo.meters
    assert set(result.min_combo.meters) == {"A-进线1", "A-进线2", "B-进线1", "B-进线2"}
    assert result.min_combo.total == Decimal("12200")


def test_root_reading_alone_covers_every_leaf():
    result = run([row("厂区总表", 9999)])
    assert result.feasible
    assert result.combo_count == 1
    assert result.max_combo.meters == ("厂区总表",)
    assert result.max_combo.total == Decimal("9999")
    assert result.unread_meters == [n for n in ALL_NAMES if n != "厂区总表"]


def test_mixed_internal_and_leaf_frontiers():
    # A covered by its parent meter; B covered by its leaves.
    rows = [
        row("车间A总", 6200),
        row("B-进线1", 2700),
        row("B-进线2", 3300),
    ]
    result = run(rows)
    assert result.feasible
    assert result.combo_count == 1
    assert set(result.min_combo.meters) == {"车间A总", "B-进线1", "B-进线2"}
    assert result.min_combo.total == Decimal("12200")


def test_decimal_precision_is_exact():
    rows = [
        row("厂区总表", "10000.10"),
        row("A-进线1", "0.1"),
        row("A-进线2", "0.2"),
        row("B-进线1", "0.3"),
        row("B-进线2", "9999.5"),
    ]
    result = run(rows)
    assert result.feasible
    # Leaf sum 0.1+0.2+0.3+9999.5 must not drift via binary floats.
    assert result.min_combo.total == Decimal("10000.10")
    assert result.max_combo.total == Decimal("10000.10")
    assert result.needs_review is False


def test_slice_rows_flag_extreme_membership():
    rows = [
        row("厂区总表", 12500),
        row("车间A总", 6200),
        row("车间B总", 6000),
        row("B-进线1", 2700),
        row("B-进线2", 3300),
    ]
    result = run(rows)
    flags = {r.meter: (r.in_min, r.in_max, r.is_leaf) for r in result.slice_rows}
    assert flags["厂区总表"] == (False, True, False)
    assert flags["车间A总"] == (True, False, False)
    assert flags["车间B总"] == (True, False, False)
    assert flags["B-进线1"] == (False, False, True)


def test_missing_leaf_reading_pinpoints_gap_branch():
    # B-进线1 missing; 车间B总 unread so its subtree cannot be completed.
    rows = [
        row("车间A总", 6100),
        row("A-进线1", 2950),
        row("A-进线2", 3150),
        row("B-进线2", 3300),
    ]
    result = run(rows)
    assert result.feasible is False
    assert result.min_combo is None and result.max_combo is None
    assert len(result.gaps) == 1
    gap = result.gaps[0]
    assert gap.node == "B-进线1"
    assert gap.path == ("厂区总表", "车间B总", "B-进线1")
    assert gap.leaves == ("B-进线1",)


def test_unread_internal_with_unread_children_reports_highest_subtree():
    # Whole B subtree absent: the maximal gap is 车间B总, not separate leaves.
    rows = [
        row("车间A总", 6100),
        row("A-进线1", 2950),
        row("A-进线2", 3150),
    ]
    result = run(rows)
    assert result.feasible is False
    assert len(result.gaps) == 1
    gap = result.gaps[0]
    assert gap.node == "车间B总"
    assert set(gap.leaves) == {"B-进线1", "B-进线2"}


def test_unread_root_everything_missing_reports_root():
    result = run([row("A-进线1", 1)])
    assert result.feasible is False
    assert len(result.gaps) == 1
    assert result.gaps[0].node == "厂区总表"


def test_other_sites_and_months_are_ignored_in_slice():
    rows = [
        row("厂区总表", 100, month="2025-02"),
        row("厂区总表", 200, site="华北二厂"),
        row("厂区总表", 12500),
        row("车间A总", 6200),
        row("A-进线1", 3000),
        row("A-进线2", 3200),
        row("车间B总", 6000),
        row("B-进线1", 2700),
        row("B-进线2", 3300),
    ]
    result = run(rows)
    assert result.reading_count == 7
    assert result.max_combo.total == Decimal("12500")


def test_unknown_meter_in_slice_raises():
    with pytest.raises(ReconcileError, match="不在表计树内"):
        run([row("幽灵表", 10)])


def test_empty_slice_raises():
    # The site and month both exist in the file, just never together.
    upload = make_readings(
        [
            row("厂区总表", 1, site="华北二厂", month="2025-01"),
            row("厂区总表", 2, site="华东一厂", month="2025-02"),
        ]
    )
    tree = make_tree()
    with pytest.raises(ReconcileError, match="没有任何读数"):
        reconcile(tree, upload, "华东一厂", "2025-01")


@pytest.mark.parametrize("site,month", [("", "2025-01"), ("华东一厂", "")])
def test_missing_selection_raises(site, month):
    tree = make_tree()
    upload = make_readings([row("厂区总表", 1)])
    with pytest.raises(ReconcileError, match="请选择"):
        reconcile(tree, upload, site, month)


def test_unknown_site_raises():
    tree = make_tree()
    upload = make_readings([row("厂区总表", 1)])
    with pytest.raises(ReconcileError, match="不存在厂区"):
        reconcile(tree, upload, "不存在的厂", "2025-01")


# --------------------------------------------------------------------------- #
# Readings validation
# --------------------------------------------------------------------------- #


def test_duplicate_reading_rejected_with_line_numbers():
    text = (
        "厂区,月份,计量点,用电量\n"
        "华东一厂,2025-01,厂区总表,100\n"
        "华东一厂,2025-01,厂区总表,120\n"
    )
    upload, problems = parse_readings(text.encode("utf-8"))
    assert upload is None
    assert any("重复读数" in p and "第 3 行" in p and "第 2 行" in p for p in problems)


def test_same_meter_different_month_is_not_duplicate():
    text = (
        "厂区,月份,计量点,用电量\n"
        "华东一厂,2025-01,厂区总表,100\n"
        "华东一厂,2025-02,厂区总表,110\n"
    )
    upload, problems = parse_readings(text.encode("utf-8"))
    assert upload is not None, problems
    assert upload.months == ["2025-01", "2025-02"]


@pytest.mark.parametrize("bad", ["-1", "-0.01"])
def test_negative_reading_rejected(bad):
    upload, problems = parse_readings(
        ("厂区,月份,计量点,用电量\n" f"华东一厂,2025-01,厂区总表,{bad}\n").encode("utf-8")
    )
    assert upload is None
    assert any("负数" in p for p in problems)


def test_zero_reading_is_allowed():
    upload, problems = parse_readings(
        "厂区,月份,计量点,用电量\n华东一厂,2025-01,厂区总表,0\n".encode("utf-8")
    )
    assert upload is not None, problems
    assert upload.rows[0].value == Decimal("0")


@pytest.mark.parametrize("bad", ["NaN", "inf", "1e3", "abc"])
def test_non_decimal_reading_rejected(bad):
    upload, problems = parse_readings(
        ("厂区,月份,计量点,用电量\n" f"华东一厂,2025-01,厂区总表,{bad}\n").encode("utf-8")
    )
    assert upload is None
    assert any("有限小数" in p for p in problems)


def test_bad_month_rejected():
    upload, problems = parse_readings(
        "厂区,月份,计量点,用电量\n华东一厂,2025-13,厂区总表,1\n".encode("utf-8")
    )
    assert upload is None
    assert any("YYYY-MM" in p for p in problems)


def test_empty_required_cell_rejected():
    upload, problems = parse_readings(
        "厂区,月份,计量点,用电量\n华东一厂,2025-01,厂区总表,\n".encode("utf-8")
    )
    assert upload is None
    assert any("用电量" in p and "为空" in p for p in problems)


def test_missing_standard_column_rejected():
    upload, problems = parse_readings(
        "厂区,月份,计量点\n华东一厂,2025-01,厂区总表\n".encode("utf-8")
    )
    assert upload is None
    assert any("用电量" in p for p in problems)


def test_duplicate_reading_headers_rejected():
    upload, problems = parse_readings(
        "厂区,月份,计量点,用电量,电量\n"
        "华东一厂,2025-01,厂区总表,1,2\n".encode("utf-8")
    )
    assert upload is None
    assert any("用电量" in p and "多个来源列" in p for p in problems)


def test_extra_columns_are_ignored_and_listed():
    upload, problems = parse_readings(
        "厂区,月份,计量点,用电量,备注\n"
        "华东一厂,2025-01,厂区总表,1,ok\n".encode("utf-8")
    )
    assert upload is not None, problems
    assert any("备注" in c for c in upload.ignored_columns)


def test_alias_headers_accepted():
    text = (
        "site,period,meter,kwh\n"
        "华东一厂,2025-01,厂区总表,42.5\n"
    )
    upload, problems = parse_readings(text.encode("utf-8"))
    assert upload is not None, problems
    assert upload.rows[0].value == Decimal("42.5")


def test_ragged_reading_row_rejected():
    upload, problems = parse_readings(
        "厂区,月份,计量点,用电量\n华东一厂,2025-01,厂区总表,1,extra\n".encode("utf-8")
    )
    assert upload is None
    assert any("列数不符" in p for p in problems)


def test_thousands_separator_becomes_ragged_row():
    # An unquoted 1,000 is two CSV cells, so the row fails the column-count check.
    upload, problems = parse_readings(
        "厂区,月份,计量点,用电量\n华东一厂,2025-01,厂区总表,1,000\n".encode("utf-8")
    )
    assert upload is None
    assert any("列数不符" in p for p in problems)


def test_non_utf8_readings_fatal():
    with pytest.raises(ReconcileError):
        parse_readings("厂区,月份,计量点,用电量\n".encode("gbk"))


# --------------------------------------------------------------------------- #
# Tree validation
# --------------------------------------------------------------------------- #


def test_tree_parses_root_depths_and_leaves():
    tree = make_tree()
    assert tree.root == "厂区总表"
    assert tree.node_count == 7
    assert tree.depth["厂区总表"] == 0
    assert tree.depth["B-进线1"] == 2
    assert tree.leaves == {"A-进线1", "A-进线2", "B-进线1", "B-进线2"}


def test_tree_unknown_parent_rejected():
    text = "计量点,上级计量点\n甲,幽灵\n"
    upload = parse_meter_tree(text.encode("utf-8"))
    assert upload.tree is None
    assert any("幽灵" in p and "未在计量点列中定义" in p for p in upload.problems)


def test_tree_multiple_roots_rejected():
    text = "计量点,上级计量点\n甲,\n乙,\n"
    upload = parse_meter_tree(text.encode("utf-8"))
    assert upload.tree is None
    assert any("多个根表" in p for p in upload.problems)


def test_tree_no_root_rejected():
    text = "计量点,上级计量点\n甲,乙\n乙,甲\n"
    upload = parse_meter_tree(text.encode("utf-8"))
    assert upload.tree is None
    assert any("根表" in p or "环路" in p for p in upload.problems)


def test_tree_self_parent_rejected():
    text = "计量点,上级计量点\n甲,甲\n"
    upload = parse_meter_tree(text.encode("utf-8"))
    assert upload.tree is None
    assert any("指向自身" in p for p in upload.problems)


def test_tree_cycle_rejected():
    text = (
        "计量点,上级计量点\n"
        "root,\n甲,root\n乙,甲\n丙,乙\n甲,丙\n"
    )
    upload = parse_meter_tree(text.encode("utf-8"))
    # Duplicate 甲 also surfaces; either way no tree is produced and a cycle is named.
    assert upload.tree is None
    assert any("环路" in p or "重复定义" in p for p in upload.problems)


def test_tree_duplicate_meter_rejected():
    text = "计量点,上级计量点\n甲,\n甲,\n"
    upload = parse_meter_tree(text.encode("utf-8"))
    assert upload.tree is None
    assert any("重复定义" in p for p in upload.problems)


def test_tree_empty_meter_name_rejected():
    text = "计量点,上级计量点\n甲,\n,甲\n"
    upload = parse_meter_tree(text.encode("utf-8"))
    assert upload.tree is None
    assert any("名称为空" in p for p in upload.problems)


def test_tree_missing_standard_column():
    upload = parse_meter_tree("计量点\n甲\n".encode("utf-8"))
    assert upload.tree is None
    assert any("上级计量点" in p for p in upload.problems)


def test_tree_size_cap_enforced():
    lines = ["计量点,上级计量点", "root,"]
    for i in range(MAX_METERS):  # root + 200 children => 201 nodes
        lines.append(f"m{i},root")
    upload = parse_meter_tree("\n".join(lines).encode("utf-8"))
    assert upload.tree is None
    assert any(str(MAX_METERS) in p and "上限" in p for p in upload.problems)


def test_tree_with_exactly_max_nodes_accepted():
    lines = ["计量点,上级计量点", "root,"]
    for i in range(MAX_METERS - 1):
        lines.append(f"m{i},root")
    upload = parse_meter_tree("\n".join(lines).encode("utf-8"))
    assert upload.tree is not None, upload.problems
    assert upload.tree.node_count == MAX_METERS


def test_tree_alias_headers_and_blank_root_parent():
    text = "meter,parent\nroot,\nchild,root\n"
    upload = parse_meter_tree(text.encode("utf-8"))
    assert upload.tree is not None, upload.problems
    assert upload.tree.root == "root"
    assert upload.tree.parent["child"] == "root"


def test_chain_tree_dp_performance_and_counts():
    # A pure chain of 200 nodes: at every level choose node or descend.
    lines = ["计量点,上级计量点", "m0,"]
    for i in range(1, MAX_METERS):
        lines.append(f"m{i},m{i-1}")
    tree = make_tree("\n".join(lines))
    upload = make_readings([row(f"m{i}", 1) for i in range(MAX_METERS)])
    result = reconcile(tree, upload, "华东一厂", "2025-01")
    assert result.feasible
    # Each level offers one binary choice -> 200 frontiers for a 200-node chain.
    assert result.combo_count == MAX_METERS
    assert result.min_combo.total == Decimal("1")
    assert result.max_combo.total == Decimal("1")
