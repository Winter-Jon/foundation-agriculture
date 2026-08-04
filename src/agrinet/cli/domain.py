from __future__ import annotations

from typing import Annotated

import typer
import yaml

from agrinet.common.config import ConfigError, list_experiments, load_experiment
from agrinet.common.contracts import Domain


def domain_app(domain: Domain) -> typer.Typer:
    app = typer.Typer(help=f"Discover and run {domain.value} experiments.")

    @app.command("list")
    def list_command() -> None:
        """List registered experiments in this domain."""
        for spec in list_experiments(domain):
            typer.echo(
                f"{spec.id}\t{spec.display_name.en}\t{spec.display_name.zh}\t{spec.lifecycle.value}"
            )

    @app.command("show")
    def show_command(experiment_id: Annotated[str, typer.Argument()]) -> None:
        """Show one registered experiment."""
        try:
            spec = load_experiment(experiment_id)
        except ConfigError as exc:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(2) from exc
        if spec.domain != domain:
            typer.echo(f"error: experiment belongs to {spec.domain.value}, not {domain.value}", err=True)
            raise typer.Exit(2)
        typer.echo(yaml.safe_dump(spec.model_dump(mode="json"), sort_keys=False, allow_unicode=True))

    return app
