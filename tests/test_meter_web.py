"""End-to-end browser-flow tests for the meter reconciliation page."""

import pytest
from fastapi.testclient import TestClient

from app.main import app

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

READINGS_CSV = (
    "厂区,月份,计量点,用电量\n"
    "华东一厂,2025-01,厂区总表,12500\n"
    "华东一厂,2025-01,车间A总,6200\n"
    "华东一厂,2025-01,A-进线1,3000\n"
    "华东一厂,2025-01,A-进线2,3200\n"
    "华东一厂,2025-01,车间B总,6000\n"
    "华东一厂,2025-01,B-进线1,2700\n"
    "华东一厂,2025-01,B-进线2,3300\n"
    "华东一厂,2025-02,车间A总,6100\n"
    "华东一厂,2025-02,A-进线1,2950\n"
    "华东一厂,2025-02,A-进线2,3150\n"
    "华东一厂,2025-02,B-进线2,3300\n"
    "华北二厂,2025-01,厂区总表,8800\n"
)


@pytest.fixture()
def client():
    with TestClient(app) as test_client:
        yield test_client


def prepare(client, readings=READINGS_CSV, tree=TREE_CSV):
    return client.post(
        "/meters/prepare",
        files={
            "readings": ("readings.csv", readings.encode("utf-8"), "text/csv"),
            "tree": ("tree.csv", tree.encode("utf-8"), "text/csv"),
        },
    )


def extract_token(response) -> str:
    return response.text.split('name="token" value="', 1)[1].split('"', 1)[0]


def test_nav_link_and_meters_page(client):
    response = client.get("/meters")
    assert response.status_code == 200
    assert "表计核对" in response.text
    assert 'enctype="multipart/form-data"' in response.text
    home = client.get("/")
    assert '/meters' in home.text


def test_prepare_shows_site_and_month_selectors(client):
    response = prepare(client)
    assert response.status_code == 200
    assert "华东一厂" in response.text
    assert "华北二厂" in response.text
    assert "2025-01" in response.text and "2025-02" in response.text
    assert "厂区总表" in response.text  # tree root displayed


def test_happy_path_shows_min_max_and_review_flag(client):
    token = extract_token(prepare(client))
    response = client.post(
        "/meters/reconcile",
        data={"token": token, "site": "华东一厂", "month": "2025-01"},
    )
    assert response.status_code == 200
    text = response.text
    assert "待核查" in text
    assert "12200" in text and "12500" in text
    assert "厂区总表" in text and "车间A总" in text
    assert "5 种完整覆盖" in text
    # Both extreme combinations are rendered with their meters.
    assert "最小组合" in text and "最大组合" in text


def test_consistent_period_is_not_flagged(client):
    token = extract_token(prepare(client))
    response = client.post(
        "/meters/reconcile",
        data={"token": token, "site": "华北二厂", "month": "2025-01"},
    )
    assert response.status_code == 200
    assert "核对一致" in response.text
    assert "待核查" not in response.text


def test_infeasible_period_names_gap_branch(client):
    token = extract_token(prepare(client))
    response = client.post(
        "/meters/reconcile",
        data={"token": token, "site": "华东一厂", "month": "2025-02"},
    )
    assert response.status_code == 200
    text = response.text
    assert "无法补足的分支" in text
    assert "B-进线1" in text
    assert "无法给出总量" in text
    # No totals rendered on a failed run.
    assert "最小组合" not in text


def test_duplicate_reading_blocks_preparation(client):
    bad = READINGS_CSV + "华东一厂,2025-01,厂区总表,9999\n"
    response = prepare(client, readings=bad)
    assert response.status_code == 422
    assert "重复读数" in response.text


def test_negative_reading_blocks_preparation(client):
    bad = READINGS_CSV + "华北二厂,2025-02,厂区总表,-5\n"
    response = prepare(client, readings=bad)
    assert response.status_code == 422
    assert "负数" in response.text


def test_invalid_tree_blocks_preparation(client):
    bad_tree = "计量点,上级计量点\n甲,幽灵\n"
    response = prepare(client, tree=bad_tree)
    assert response.status_code == 422
    assert "未在计量点列中定义" in response.text


def test_missing_file_reported(client):
    response = client.post(
        "/meters/prepare",
        files={"tree": ("tree.csv", TREE_CSV.encode("utf-8"), "text/csv")},
    )
    assert response.status_code == 422
    assert "请上传标准列读数 CSV" in response.text


def test_empty_file_reports_missing_columns(client):
    response = client.post(
        "/meters/prepare",
        files={
            "readings": ("r.csv", b" ", "text/csv"),
            "tree": ("t.csv", b" ", "text/csv"),
        },
    )
    assert response.status_code == 422
    assert "缺少必填标准列" in response.text


def test_alias_headers_and_extra_columns_flow(client):
    readings = (
        "site,period,meter,kwh,备注\n"
        "华东一厂,2025-01,厂区总表,100,x\n"
    ).encode("utf-8")
    tree = "meter,parent,备注列\n厂区总表,,y\n".encode("utf-8")
    response = client.post(
        "/meters/prepare",
        files={
            "readings": ("r.csv", readings, "text/csv"),
            "tree": ("t.csv", tree, "text/csv"),
        },
    )
    assert response.status_code == 200
    assert "被忽略的额外列" in response.text
    token = extract_token(response)
    result = client.post(
        "/meters/reconcile",
        data={"token": token, "site": "华东一厂", "month": "2025-01"},
    )
    assert result.status_code == 200
    assert "100" in result.text and "待核查" not in result.text


def test_reconcile_with_unknown_token_is_404(client):
    response = client.post(
        "/meters/reconcile",
        data={"token": "nope", "site": "华东一厂", "month": "2025-01"},
    )
    assert response.status_code == 404
    assert "已过期或不存在" in response.text


def test_reconcile_time_errors_show_banner_without_total(client):
    token = extract_token(prepare(client))
    # Tampered selection values are rejected at reconcile time, not silently.
    response = client.post(
        "/meters/reconcile",
        data={"token": token, "site": "幽灵厂", "month": "2025-01"},
    )
    assert response.status_code == 422
    assert "无法核对" in response.text
    assert "最小组合" not in response.text


def test_session_is_reusable_within_its_lifetime(client):
    token = extract_token(prepare(client))
    first = client.post(
        "/meters/reconcile",
        data={"token": token, "site": "华东一厂", "month": "2025-01"},
    )
    second = client.post(
        "/meters/reconcile",
        data={"token": token, "site": "华北二厂", "month": "2025-01"},
    )
    assert first.status_code == 200 and second.status_code == 200
    assert "12500" in first.text and "8800" in second.text


def test_upload_size_limit(client):
    rows = "".join(f"x,2025-01,m{i},1\n" for i in range(750_000))
    big = "厂区,月份,计量点,用电量\n".encode("utf-8") + rows.encode("utf-8")
    assert len(big) > 10 * 1024 * 1024
    response = client.post(
        "/meters/prepare",
        files={
            "readings": ("big.csv", big, "text/csv"),
            "tree": ("tree.csv", TREE_CSV.encode("utf-8"), "text/csv"),
        },
    )
    assert response.status_code == 422
    assert "上限" in response.text


def test_standardizer_flow_still_works(client):
    # The original feature must be untouched by the new page.
    response = client.get("/")
    assert response.status_code == 200
    assert "字段合同" in response.text
    assert client.get("/healthz").json() == {"status": "ok"}
