"""The two filing sources: archive bundles (XML) and the API (JSON rows)."""

import io
import json
import tarfile
from datetime import date, datetime

import pytest

from cac40_mood.data import amf_api, dila_archive
from cac40_mood.data.schema import FILING_FIELDS, doc_id_from_pdf_path, normalise_language

OUR_ISIN = "FR0000121014"

# A real metadata file from the archive, shortened (distribution list cut to one entry).
XML_TEMPLATE = """<?xml version="1.0"?>
<UniteInformation UIN_VER_FIN="FI_v1.1_20120515" UIN_IDT_UIN="123955_20200304"
    UIN_DAT_AMF="2020-03-04T14:33:03+01:00" UIN_DAT_MAR="2020-03-04T18:00:00+01:00">
  <identificationDiffuseur IDI_COD_DIF="BWR"/>
  <identificationSociete ISO_PAY_SS="FR" ISO_NOM_SOC="LVMH" ISO_CD_ISI="{isin}" ISO_CD_LEI="">
    <ISO_CODE_TKR ISO_CD_TKR="MC" ISO_CD_MCH="99"/>
  </identificationSociete>
  <InformationDeposee INF_DAT_EMT="2020-03-04T14:33:03+01:00" INF_STP_PRI="001007"
      INF_TIT_INF="Résultats annuels" INF_LNG_INF="FR" INF_COD_DIF="OBL">
    <INF_UPG_INF INF_UPG_STS="NEW" INF_UPG_REL=""/>
    <INF_STP_INF>001007</INF_STP_INF>
    <INF_STP_INF>005001</INF_STP_INF>
    <INF_LST_DIF>AMF</INF_LST_DIF>
  </InformationDeposee>
  <FichierDeContenu INF_FIC_NOM="{pdf_path}"/>
</UniteInformation>
"""


def metadata_xml(isin: str, pdf_path: str) -> bytes:
    return XML_TEMPLATE.format(isin=isin, pdf_path=pdf_path).encode()


def make_tarball(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


# ------------------------------------------------------------------- schema


def test_doc_id_and_language_helpers() -> None:
    assert doc_id_from_pdf_path("MKW/2024/01/FCMKW110450_20240125.pdf") == "FCMKW110450_20240125"
    assert [normalise_language(v) for v in ["FR", "Français", "EN", "Anglais", None]] == [
        "fr",
        "fr",
        "en",
        "en",
        "",
    ]


# ------------------------------------------------------------------ archive


def test_listing_is_parsed_and_sorted_by_bundle_time() -> None:
    html = (
        '<a href="amf-gz_02012019_0900.tar.gz">amf-gz_02012019_0900.tar.gz</a> 5.8M '
        '<a href="amf-gz_01012019_2100.tar.gz">amf-gz_01012019_2100.tar.gz</a> 312K '
        '<a href="MKW/">MKW/</a>'
    )
    tarballs = dila_archive.parse_listing(html)
    assert [t.name for t in tarballs] == [
        "amf-gz_01012019_2100.tar.gz",
        "amf-gz_02012019_0900.tar.gz",
    ]
    assert tarballs[0].bundled_at == datetime(2019, 1, 1, 21, 0)


def test_window_selection_includes_trailing_days_for_late_arrivals() -> None:
    names = ["31122018_2300", "01012019_0100", "30062020_2300", "07072020_2300", "08072020_0100"]
    html = " ".join(f"amf-gz_{name}.tar.gz" for name in names)
    chosen = dila_archive.select_tarballs(
        dila_archive.parse_listing(html), date(2019, 1, 1), date(2020, 6, 30)
    )
    assert [t.name[7:20] for t in chosen] == ["01012019_0100", "30062020_2300", "07072020_2300"]


def test_metadata_xml_becomes_a_complete_record() -> None:
    record = dila_archive.parse_metadata_xml(
        metadata_xml(OUR_ISIN, "BWR/2020/03/FCBWR123955_20200304.pdf")
    )
    assert set(record) == set(FILING_FIELDS)
    assert record["doc_id"] == "FCBWR123955_20200304"
    assert record["source"] == "tarball"
    assert record["isin"] == OUR_ISIN
    assert record["lang"] == "fr"
    assert record["type_code"] == "001007"
    assert record["type_codes"] == "001007;005001"
    assert record["ts_amf"] == "2020-03-04T14:33:03+01:00"
    assert record["ts_market"] == "2020-03-04T18:00:00+01:00"
    assert record["distributor"] == "BWR"
    assert record["feed_ticker"] == "MC"


def test_only_our_filings_are_kept_from_a_bundle(tmp_path) -> None:
    ours = "BWR/2020/03/FCBWR000001_20200304.pdf"
    theirs = "BWR/2020/03/FCBWR000002_20200304.pdf"
    bundle = make_tarball(
        {
            # The PDF comes before its XML on purpose: order inside a bundle is not guaranteed.
            "pdf/" + ours: b"%PDF-ours",
            "xml/BWR/2020/03/MDBWR000001_20200304.xml": metadata_xml(OUR_ISIN, ours),
            "xml/BWR/2020/03/MDBWR000002_20200304.xml": metadata_xml("FR0000031775", theirs),
            "pdf/" + theirs: b"%PDF-theirs",
            "xml/BWR/2020/03/MDBWR000003_20200304.xml": b"<broken",
            "xml/../../escape.xml": metadata_xml(OUR_ISIN, ours),
        }
    )
    counts = dila_archive.extract_our_filings(
        bundle, {OUR_ISIN}, tmp_path / "xml", tmp_path / "pdf"
    )
    assert counts == {
        "n_xml": 3,
        "n_ours": 1,
        "n_pdf_saved": 1,
        "n_bad_xml": 1,
        "n_unsafe_names": 1,
    }
    assert (tmp_path / "pdf" / ours).read_bytes() == b"%PDF-ours"
    assert not (tmp_path / "pdf" / theirs).exists()
    assert [p.name for p in tmp_path.rglob("*.xml")] == ["MDBWR000001_20200304.xml"]
    assert not (tmp_path.parent / "escape.xml").exists()

    records = dila_archive.read_raw_records(tmp_path)
    assert [record["doc_id"] for record in records] == ["FCBWR000001_20200304"]


def test_non_pdf_attachments_of_our_filings_are_kept(tmp_path) -> None:
    attachment = "182/2019/07/FC182293779_20190725.docx"
    bundle = make_tarball(
        {
            "xml/182/2019/07/MD182293779_20190725.xml": metadata_xml(OUR_ISIN, attachment),
            "pdf/" + attachment: b"docx bytes",
        }
    )
    counts = dila_archive.extract_our_filings(
        bundle, {OUR_ISIN}, tmp_path / "xml", tmp_path / "pdf"
    )
    assert counts["n_pdf_saved"] == 1 and counts["n_unsafe_names"] == 0


def test_manifest_is_read_back_by_bundle_name(tmp_path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text('{"name": "a.tar.gz", "n_ours": 2}\n{"name": "b.tar.gz", "error": "x"}\n')
    entries = dila_archive.read_manifest(manifest)
    assert entries["a.tar.gz"]["n_ours"] == 2 and "error" in entries["b.tar.gz"]
    assert dila_archive.read_manifest(tmp_path / "absent.jsonl") == {}


# ---------------------------------------------------------------------- API

API_ROW = {
    "uin_dat_amf": "2024-01-25T16:45:52+00:00",
    "uin_dat_mar": "8887-12-31T23:00:00+00:00",
    "informationdeposee_inf_dat_emt": "2024-01-25T16:45:52+00:00",
    "identificationdiffuseur_idi_cod_dif": "MKW",
    "identificationsociete_iso_nom_soc": "LVMH MOET HENNESSY-LOUIS VUITTON",
    "identificationsociete_iso_cd_isi": OUR_ISIN,
    "identificationsociete_iso_cd_lei": None,
    "identificationsociete_iso_code_tkr_iso_cd_tkr": "MC",
    "fichierdecontenu_inf_fic_nom": "MKW/2024/01/FCMKW130450_20240125.pdf",
    "informationdeposee_inf_tit_inf": "2023 : Nouvelle année record pour LVMH",
    "informationdeposee_inf_stp_pri": "001007",
    "informationdeposee_inf_stp_inf": "001007",
    "informationdeposee_inf_lng_inf": "Français",
    "informationdeposee_inf_cod_dif": "OBL",
    "informationdeposee_inf_upg_inf_inf_upg_sts": "NEW",
}


def test_api_row_becomes_the_same_record_layout_as_the_archive() -> None:
    record = amf_api.normalise_api_row(API_ROW)
    assert set(record) == set(FILING_FIELDS)
    assert record["doc_id"] == "FCMKW130450_20240125"
    assert record["source"] == "api"
    assert record["lang"] == "fr"
    assert record["feed_lei"] == ""  # a missing value becomes an empty string, never "None"
    assert record["ts_market"] == "8887-12-31T23:00:00+00:00"  # kept raw; cleaned later


@pytest.mark.parametrize(
    ("raw_codes", "expected"),
    [
        ("001007", "001007"),
        (["300005", "003000"], "300005;003000"),
        ('["006000", "001008"]', "006000;001008"),
        (None, ""),
    ],
)
def test_api_type_codes_come_in_three_shapes(raw_codes, expected) -> None:
    row = dict(API_ROW, informationdeposee_inf_stp_inf=raw_codes)
    assert amf_api.normalise_api_row(row)["type_codes"] == expected


def test_where_clause_asks_for_a_day_of_margin_on_each_side() -> None:
    where = amf_api.build_where(OUR_ISIN, date(2019, 1, 1), date(2025, 12, 31))
    assert f"identificationsociete_iso_cd_isi='{OUR_ISIN}'" in where
    assert "uin_dat_amf>=date'2018-12-31'" in where
    assert "uin_dat_amf<date'2026-01-02'" in where


class FakeResponse:
    def __init__(self, payload: object) -> None:
        self._payload = payload
        self.text = payload if isinstance(payload, str) else json.dumps(payload)

    def raise_for_status(self) -> None:
        return None

    def json(self) -> object:
        return self._payload


class FakeApi:
    """Stands in for PoliteSession: answers count and export requests from a script."""

    def __init__(self, live_counts: list[int], catalog_count: int = 100) -> None:
        self._live_counts = live_counts  # one value per call to the live row count
        self._catalog_count = catalog_count

    def get(self, url: str, params: dict | None = None, timeout: float = 0) -> FakeResponse:
        if url.endswith("/exports/jsonl"):
            return FakeResponse(json.dumps(API_ROW) + "\n")
        if url.endswith("/records"):
            return FakeResponse({"total_count": self._live_counts.pop(0)})
        return FakeResponse({"metas": {"default": {"records_count": self._catalog_count}}})


def test_snapshot_is_stored_when_the_index_is_complete(tmp_path) -> None:
    out_dir = tmp_path / "amf_api"
    counts = amf_api.fetch_snapshot(
        FakeApi([100, 100]), [OUR_ISIN], date(2024, 1, 1), date(2024, 12, 31), out_dir
    )
    assert counts == {OUR_ISIN: 1}
    assert json.loads((out_dir / "manifest.json").read_text())["index_rows"] == 100
    records = amf_api.read_raw_records(out_dir)
    assert [record["doc_id"] for record in records] == ["FCMKW130450_20240125"]


def test_snapshot_is_refused_while_the_index_is_rebuilding(tmp_path) -> None:
    out_dir = tmp_path / "amf_api"
    with pytest.raises(amf_api.IndexRebuildingError):
        amf_api.fetch_snapshot(
            FakeApi([60]), [OUR_ISIN], date(2024, 1, 1), date(2024, 12, 31), out_dir
        )
    assert not out_dir.exists()


def test_snapshot_is_discarded_if_the_index_changes_during_the_download(tmp_path) -> None:
    out_dir = tmp_path / "amf_api"
    out_dir.mkdir()
    (out_dir / "previous.jsonl").write_text("")  # an older, good snapshot must survive
    with pytest.raises(amf_api.IndexRebuildingError):
        amf_api.fetch_snapshot(
            FakeApi([100, 70]), [OUR_ISIN], date(2024, 1, 1), date(2024, 12, 31), out_dir
        )
    assert (out_dir / "previous.jsonl").exists()
    assert not (tmp_path / "amf_api.partial").exists()
