import logging
import sys
import asyncio
import draccus
from pathlib import Path

# Add current directory to sys.path to ensure vloom is importable if running from here
sys.path.insert(0, str(Path(__file__).parent))

from vloom.config import PipelineConfig, DefaultTaskConfig
from vloom.core.llm_client import LLMClient
from vloom.agents.registry import AgentRegistry
from vloom.core.pipeline import get_pipeline_class
from vloom.utils import preload_imports

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("main")

def main():
    preload_imports()

    cfg: PipelineConfig = draccus.parse(config_class=PipelineConfig)

    logging.getLogger().setLevel(cfg.log_level.value)
    
    # Suppress httpx/httpcore INFO logs (too verbose)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    save_dir = cfg.save_dir
    config_out_path = save_dir / "config.yaml"
    with open(config_out_path, 'w', encoding='utf-8') as f:
        draccus.dump(cfg, f)
        
    logger.info("Starting Pipeline Execution...")
    try:
        asyncio.run(async_main(cfg))
    except KeyboardInterrupt:
        logger.info("User interrupted.")
    except Exception as e:
        logger.exception(f"Pipeline execution failed: {e}")

async def async_main(cfg: PipelineConfig):
    # Load custom strategy class if specified
    strategy_class = None
    if cfg.model.strategy_class:
        import importlib
        module_path, class_name = cfg.model.strategy_class.rsplit(".", 1)
        module = importlib.import_module(module_path)
        strategy_class = getattr(module, class_name)
        logger.info(f"Using custom LLM strategy: {cfg.model.strategy_class}")
    
    # Initialize Core Components
    llm_client = LLMClient(
        api_key=cfg.model.api_key, 
        base_url=cfg.model.base_url,
        dry_run=cfg.dry_run or cfg.model.dry_run,
        strategy_class=strategy_class
    )
    
    registry = AgentRegistry(
        template_root=cfg.template_dir,
        llm_client=llm_client,
        log_interval=cfg.log_interval
    )
    
    # Load tasks from config (including auto-discovery)
    registry.load_from_config(cfg)

    try:
        # Get pipeline class (custom or default)
        pipeline_cls = get_pipeline_class(cfg.pipeline_class)
        logger.info(f"Using pipeline: {pipeline_cls.__name__}")
        
        # Instantiate and run pipeline
        pipeline = pipeline_cls(cfg, registry, llm_client)
        await pipeline.run()
    finally:
        # Cleanup within the same loop
        await llm_client.close()

if __name__ == "__main__":
    main()