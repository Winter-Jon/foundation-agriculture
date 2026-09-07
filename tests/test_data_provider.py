import ast
from pathlib import Path

from agrinet.data.contracts import TeacherDataProvider
from agrinet.data.providers.vloom import VloomTeacherDataProvider
from agrinet.data.vloom_tools.agent import OptionalImageBasicAgent


def test_vloom_provider_satisfies_public_protocol() -> None:
    assert isinstance(VloomTeacherDataProvider(Path.cwd()), TeacherDataProvider)


def test_only_vloom_adapter_mentions_legacy_runner() -> None:
    data_root = Path("src/agrinet/data")
    offenders = []
    for path in data_root.rglob("*.py"):
        if path.is_relative_to(data_root / "vloom_tools"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                rendered = ast.unparse(node)
                if "agrinet.data.vloom_tools" in rendered or "vloom." in rendered:
                    offenders.append(str(path))
    assert offenders == []


def test_current_data_cli_does_not_use_slurm() -> None:
    content = Path("src/agrinet/cli/data.py").read_text(encoding="utf-8")
    assert "sbatch" not in content
    assert "scripts/slurm" not in content


def test_vloom_is_installed_from_vendor_tree() -> None:
    import vloom

    package_path = Path(vloom.__path__[0]).resolve()
    assert package_path == Path("vendor/VLooM/vloom").resolve()


def test_vloom_metadata_envelope_is_normalized_and_parsed() -> None:
    payload = {
        "sample": {
            "metadata": {
                "raw_response": '{"final_label": "N04001"}',
                "result": None,
                "template_vars": {"query_image": "image.jpg"},
                "thinking": None,
            }
        }
    }
    OptionalImageBasicAgent._normalize_result_shape(payload)
    OptionalImageBasicAgent._repair_markdown_json(payload)
    assert payload["sample"]["result"] == {"final_label": "N04001"}
    assert payload["sample"]["template_vars"]["query_image"] == "image.jpg"
