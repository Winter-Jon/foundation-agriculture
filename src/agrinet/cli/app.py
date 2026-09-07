import typer

from agrinet.cli.data import app as data_app
from agrinet.cli.rag import app as rag_app
from agrinet.cli.vlm import app as vlm_app
from agrinet.cli.vision import app as vision_app

app = typer.Typer(help="Unified AgriNet data, RAG, and VLM workflows.", no_args_is_help=True)
app.add_typer(data_app, name="data")
app.add_typer(rag_app, name="rag")
app.add_typer(vlm_app, name="vlm")
app.add_typer(vision_app, name="vision")


if __name__ == "__main__":
    app()
