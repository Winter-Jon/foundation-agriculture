"""Repository pytest module groups used by the optional local test suite."""

UNIT_MODULES = {
    "test_artifacts.py",
    "test_config.py",
    "test_contracts.py",
    "test_credentials.py",
    "test_data_pipeline.py",
    "test_experiments.py",
    "test_network.py",
    "test_repository_audit.py",
    "test_test_taxonomy.py",
}
INTEGRATION_MODULES = {
    "test_cli.py",
    "test_data_provider.py",
    "test_formal_eval_runtime.py",
    "test_local_runtime.py",
    "test_rag_http.py",
    "test_rag_index.py",
    "test_rag_service.py",
    "test_vlm_adapters.py",
    "test_vlm_evaluate.py",
    "test_vlm_export.py",
}
