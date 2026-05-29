import argparse
import asyncio
import copy
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

import draccus

from vloom.agents.registry import AgentRegistry
from vloom.config import PipelineConfig
from vloom.core.llm_client import LLMClient
from vloom.dataset import create_dataset
from vloom.utils.importer import import_modules_from_strings


logger = logging.getLogger(__name__)


def _resolve_api_key(cfg: PipelineConfig) -> str:
    explicit_key = (cfg.model.api_key or "").strip()
    if explicit_key and explicit_key != "EMPTY":
        return explicit_key

    for env_name in ("YUNWU_API_KEY", "OPENAI_API_KEY"):
        env_key = os.environ.get(env_name)
        if env_key:
            logger.info("Using API key from %s", env_name)
            return env_key

    if cfg.dry_run or cfg.model.dry_run:
        return explicit_key or "EMPTY"

    raise RuntimeError(
        "Real teacher collection requires YUNWU_API_KEY or OPENAI_API_KEY in the environment; "
        "the key is not read from repository config."
    )


def _dump_redacted_config(cfg: PipelineConfig, path: Path) -> None:
    redacted = copy.deepcopy(cfg)
    if redacted.model.api_key:
        redacted.model.api_key = ""
    with path.open("w", encoding="utf-8") as f:
        draccus.dump(redacted, f)


async def _run(cfg: PipelineConfig) -> None:
    if cfg.imports:
        import_modules_from_strings(cfg.imports)

    llm_client = LLMClient(
        api_key=_resolve_api_key(cfg),
        base_url=cfg.model.base_url,
        dry_run=cfg.dry_run or cfg.model.dry_run,
    )
    registry = AgentRegistry(
        template_root=cfg.template_dir,
        llm_client=llm_client,
        log_interval=cfg.log_interval,
    )
    registry.load_from_config(cfg)

    try:
        for dataset_config in cfg.datasets:
            dataset = create_dataset(dataset_config)
            logger.info("Dataset %s has %d items", dataset.name, len(dataset))
            for task_name in cfg.run_tasks:
                agent = registry.get_agent(task_name)
                if agent is None:
                    raise ValueError(f"Unknown task: {task_name}")
                await _run_task(cfg, dataset, task_name, agent)
    finally:
        await llm_client.close()


async def _run_task(cfg: PipelineConfig, dataset, task_name: str, agent) -> None:
    save_dir = cfg.save_dir / dataset.name
    save_dir.mkdir(parents=True, exist_ok=True)
    result_path = save_dir / f"{task_name}_results.json"
    results: Dict[str, Any] = {}
    semaphore = asyncio.Semaphore(cfg.max_concurrent)

    logger.info(
        "Running VLooM batch task %s on %s with concurrency=%d",
        task_name,
        dataset.name,
        cfg.max_concurrent,
    )

    parse_failures = 0
    error_samples: list[tuple[str, str]] = []

    async def worker(index: int):
        async with semaphore:
            item = await dataset.get_item(index)
            if not item.valid:
                return None
            try:
                return await agent.run(item, cfg.model.model_name, **(cfg.model.generation_kwargs or {}))
            except Exception as exc:
                logger.exception("Worker failed for index=%d: %s", index, exc)
                return None

    tasks = [asyncio.create_task(worker(index)) for index in range(len(dataset))]
    completed = 0
    for future in asyncio.as_completed(tasks):
        result = await future
        completed += 1
        if result:
            results.update(result)
            # Real-time validation: check for empty parsed results
            for key, row in result.items():
                if not row or not row.get("result"):
                    parse_failures += 1
                    raw = (row.get("raw_response") or "")[:500] if row else ""
                    error_samples.append((key, raw))
                    logger.warning("Parse failure for %s: raw_response=%r...", key, raw[:200])
        if completed <= 5 or completed % cfg.log_interval == 0 or completed == len(tasks):
            logger.info("Completed %d/%d items for %s", completed, len(tasks), task_name)

    if parse_failures:
        logger.warning("Total parse failures: %d/%d", parse_failures, len(tasks))
        # Persist failure details for offline analysis
        fail_path = save_dir / f"{task_name}_parse_failures.json"
        with fail_path.open("w", encoding="utf-8") as f:
            json.dump({k: v for k, v in error_samples}, f, ensure_ascii=False, indent=2)
        logger.info("Wrote %d parse failure details to %s", len(error_samples), fail_path)

    with result_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info("Wrote %d results to %s", len(results), result_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Project-local VLooM runner for AgriNet contrast CoT collection")
    parser.add_argument("--config_path", default="configs/vloom/agrinet_insect_contrast_cot.yaml")
    parser.add_argument("--max-concurrent", type=int, help="Override PipelineConfig.max_concurrent for VLooM batch collection")
    args = parser.parse_args()

    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
    import_modules_from_strings(["tools.vloom_agrinet"])
    cfg = draccus.parse(config_class=PipelineConfig, args=["--config_path", args.config_path])
    if args.max_concurrent is not None:
        cfg.max_concurrent = args.max_concurrent
    logging.getLogger().setLevel(cfg.log_level.value)
    cfg.save_dir.mkdir(parents=True, exist_ok=True)
    _dump_redacted_config(cfg, cfg.save_dir / "config.yaml")
    asyncio.run(_run(cfg))


if __name__ == "__main__":
    main()
