"""Source-qualified evidence quarantine. Never relabel ambiguous legacy rows."""
import hashlib


def verified_repair(evidence):
    metadata = evidence.get("metadata") or {}
    return (evidence.get("artifact_id") == "Disease_pest_dataset_seg_wiki::N04056"
            and metadata.get("identity_version") == "source-scoped-wiki/v1"
            and metadata.get("source_code") == "N04056"
            and metadata.get("canonical_class_code") == "N04063"
            and metadata.get("code") == "N04063"
            and metadata.get("english_name") == "grape black rot"
            and hashlib.sha256(str(metadata.get("public_description", "")).encode()).hexdigest()
            == "470defa046dcc1bc42834190a86e1ffe1d805f0d082a5d7cf6a79b73a0026911")
QUARANTINED_ARTIFACTS = {
    'Disease_pest_dataset_seg_wiki::N04056':
        'Canonical fig-rust label conflicts with grape-black-rot source text and image lineage',
}


def partition_evidence(response):
    retained, excluded = [], []
    for evidence in response.get('evidence', []):
        artifact = evidence.get('artifact_id')
        if artifact in QUARANTINED_ARTIFACTS and not verified_repair(evidence):
            excluded.append({'artifact_id': artifact, 'reason': QUARANTINED_ARTIFACTS[artifact]})
        else:
            retained.append(evidence)
    return {**response, 'evidence': retained}, excluded
