from pathlib import Path

from network_momentum import data


def test_ascii_ca_bundle_workaround(monkeypatch, safe_tmp_path: Path) -> None:
    unicode_directory = safe_tmp_path / "área"
    unicode_directory.mkdir()
    source = unicode_directory / "cacert.pem"
    source.write_text("certificate", encoding="utf-8")
    destination = safe_tmp_path / "ascii-temp"

    monkeypatch.setattr(data.tempfile, "gettempdir", lambda: str(destination))
    monkeypatch.setattr("certifi.where", lambda: str(source))
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.delenv("CURL_CA_BUNDLE", raising=False)
    data._ensure_ascii_ca_bundle()

    expected = destination / "network_momentum_certs" / "cacert.pem"
    assert expected.read_text(encoding="utf-8") == "certificate"
    assert data.os.environ["SSL_CERT_FILE"] == str(expected)
