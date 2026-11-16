"""Policy simulation CLI: policy documents → knowledge graph → personas → ToC.

Local IAD/IG graphs are the default workflow. Zep ingestion, grounded Inputs,
and graph/ToC ablations are available as optional commands and flags.
"""
import json
import os
import time
from datetime import datetime
from pathlib import Path

import typer
import yaml
from dotenv import load_dotenv

from llm.client import USAGE_TRACKER

load_dotenv(Path(__file__).resolve().parent / ".env")

app = typer.Typer()


def start_run() -> float:
    """Reset the LLM usage tracker and return a wall-clock start timestamp."""
    USAGE_TRACKER.reset()
    return time.time()


def write_run_log(
    run_dir: Path,
    command: str,
    config: dict,
    started_at: float,
    config_path: str | None = None,
    extra: dict | None = None,
) -> None:
    """Persist a run_log.json next to the run's primary outputs."""
    ended_at = time.time()
    log = {
        "command": command,
        "model": config.get("llm", {}).get("model"),
        "config_url": config.get("llm", {}).get("url"),
        "config_path": config_path,
        "started_at": datetime.fromtimestamp(started_at).isoformat(timespec="seconds"),
        "ended_at": datetime.fromtimestamp(ended_at).isoformat(timespec="seconds"),
        "duration_seconds": round(ended_at - started_at, 2),
        "usage": USAGE_TRACKER.snapshot(),
        "config": config,
    }
    if extra:
        log.update(extra)
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "run_log.json", "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_run_dir(config: dict, case: str, tag: str | None = None) -> Path:
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    model = str(config["llm"]["model"]).replace("/", "_")
    suffix = f"_{tag}" if tag else ""
    run_dir = Path(config["paths"]["outputs"][case]) / f"{ts}_{model}_result{suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_json(data: dict | list, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# Country prefix in --policy → matching Nemotron dataset directory + HF dataset folder.
_COUNTRY_TO_NEMOTRON = {
    "kor":       ("nemotron_kor", "Nemotron-Personas-Korea"),
    "singapore": ("nemotron_sg",  "Nemotron-Personas-Singapore"),
    "usa":       ("nemotron_usa", "Nemotron-Personas-USA"),
    "fr":        ("nemotron_fr",  "Nemotron-Personas-France"),
    "jp":        ("nemotron_jp",  "Nemotron-Personas-Japan"),
}


def _apply_policy_override(config: dict, policy: str | None) -> dict:
    """If `--policy <subpath>` is provided (e.g. 'kor/bk21', 'singapore/nais'),
    point input/past policy dirs at `dataset/<subpath>/input_policy/` AND auto-
    select the matching Nemotron dataset (parquet path + values module) based on
    the country prefix. Mutates `config` in place and returns it.

    Lets a single config file serve multiple policies *and* multiple countries;
    switch via CLI without editing yaml or code.
    """
    if not policy:
        return config
    target = f"dataset/{policy.strip('/').strip()}/input_policy"
    if not Path(target).is_dir():
        raise typer.BadParameter(
            f"--policy '{policy}' resolves to '{target}' which does not exist. "
            f"Expected structure: dataset/<country>/<policy_name>/input_policy/*.pdf"
        )
    config.setdefault("paths", {})["input_policy_dir"] = target
    config["paths"]["past_policy_dir"] = target

    country = policy.strip("/").split("/", 1)[0]
    if country in _COUNTRY_TO_NEMOTRON:
        pkg, hf_dir = _COUNTRY_TO_NEMOTRON[country]
        parquet_glob = f"dataset/processed/{pkg}/{hf_dir}/data/*.parquet"
        values_module = f"dataset.processed.{pkg}"
        scope_cfg = config.setdefault("persona", {}).setdefault("scope_definer", {})
        scope_cfg["parquet_glob"] = parquet_glob
        scope_cfg["values_module"] = values_module
        # scope_definer.py reads NEMOTRON_VALUES_MODULE at import time, so set it
        # in the environment before any persona module is imported.
        os.environ["NEMOTRON_VALUES_MODULE"] = values_module
        persona_cfg = config.setdefault("persona", {})
        persona_cfg.setdefault("kg_gen", {})["country"] = country
        nemotron_cfg = persona_cfg.setdefault("nemotron", {})
        nemotron_cfg["dataset_name"] = f"nvidia/{hf_dir}"
        nemotron_cfg["local_data_dir"] = f"dataset/processed/{pkg}/{hf_dir}/data"
        nemotron_cfg["cache_dir"] = f"dataset/processed/{pkg}"


    # Re-root run output dirs under outputs/<country>/<policy>/<stage>/ so each
    # policy keeps its artifacts isolated. Without --policy the legacy
    # `outputs/<stage>/...` layout from the yaml config is preserved.
    sub = policy.strip("/")  # e.g. "singapore/nais"
    outputs_cfg = config.setdefault("paths", {}).setdefault("outputs", {})
    for case in ("stage1", "stage2_toc"):
        current = outputs_cfg.get(case) or f"outputs/{case}"
        leaf = current.rstrip("/").split("/")[-1]  # "stage1" / "stage2_toc"
        outputs_cfg[case] = f"outputs/{sub}/{leaf}"
    return config


@app.command("gen-ontology")
def gen_ontology(
    simulation_requirement: str = typer.Argument(..., help="What this simulation is about (e.g., 'AI talent development policy impact')"),
    config_path: str = typer.Option("config/base.yaml", help="Path to config file"),
    policy: str = typer.Option(None, "--policy", help="Policy subpath under dataset/ (e.g. 'kor/bk21', 'kor/ssf', 'singapore/nais'). Overrides config.paths.past_policy_dir."),
    out: str = typer.Option(None, help="Output path for graph_ontology.json (default: outputs/stage1/<timestamp>_<model>_result_ontology_only/graph_ontology.json)"),
):
    """STEP 1 only - generate the GRAPH ONTOLOGY via LLM (no Zep call).

    Produces graph_ontology.json containing {entity_types, edge_types}. Inspect or edit it before piping into build-zep.
    """
    from llm.pdf_loader import load_pdfs
    from graph.ontology_generator import generate_ontology

    config = load_config(config_path)
    _apply_policy_override(config, policy)
    started_at = start_run()
    typer.echo("Loading past policy PDFs...")
    past_text = load_pdfs(config["paths"]["past_policy_dir"], config)

    typer.echo("Generating graph ontology (1 LLM call)...")
    ontology = generate_ontology([past_text], simulation_requirement, config)
    typer.echo(f"  Graph ontology: {len(ontology['entity_types'])} entity types, {len(ontology['edge_types'])} edge types.")

    if out:
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = make_run_dir(config, "stage1", tag="ontology_only")
        out_path = run_dir / "graph_ontology.json"
    save_json(ontology, out_path)
    typer.echo(f"Graph ontology saved to: {out_path}")
    write_run_log(
        out_path.parent,
        "gen-ontology",
        config,
        started_at,
        config_path=config_path,
        extra={
            "simulation_requirement": simulation_requirement,
            "outputs": {"graph_ontology": str(out_path.name)},
            "ontology_summary": {
                "entity_types": len(ontology["entity_types"]),
                "edge_types": len(ontology["edge_types"]),
            },
        },
    )


@app.command("build-zep")
def build_zep(
    graph_ontology: str = typer.Option(..., "--graph-ontology", help="Path to graph_ontology.json produced by gen-ontology (or any compatible JSON with entity_types + edge_types)"),
    config_path: str = typer.Option("config/base.yaml", help="Path to config file"),
    policy: str = typer.Option(None, "--policy", help="Policy subpath under dataset/ (e.g. 'kor/bk21', 'kor/ssf', 'singapore/nais')."),
    out: str = typer.Option(None, help="Output path for zep_graph_built.json (default: outputs/stage1/<timestamp>_<model>_result_zep_only/zep_graph_built.json)"),
):
    """STEP 2 only - build the ZEP GRAPH using a pre-existing graph ontology (no LLM call for ontology).

    Reads past-policy PDFs and ingests them into Zep with the provided ontology. Writes zep_graph_built.json
    containing {graph_id, ontology} - the manifest of what was built.
    """
    from llm.pdf_loader import load_pdfs
    from graph.graph_builder import build_graph as _build_graph

    config = load_config(config_path)
    _apply_policy_override(config, policy)
    started_at = start_run()
    ontology = load_json(graph_ontology)
    if "entity_types" not in ontology or "edge_types" not in ontology:
        typer.echo("Error: graph ontology JSON must contain 'entity_types' and 'edge_types' top-level keys.", err=True)
        raise typer.Exit(1)

    typer.echo("Loading past policy PDFs...")
    past_text = load_pdfs(config["paths"]["past_policy_dir"], config)

    typer.echo("Building Zep graph (no LLM - uses provided ontology)...")
    graph_id = _build_graph(past_text, ontology, config)
    typer.echo(f"  Zep graph built: {graph_id}")

    if out:
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = make_run_dir(config, "stage1", tag="zep_only")
        out_path = run_dir / "zep_graph_built.json"
    save_json({"graph_id": graph_id, "ontology": ontology}, out_path)
    typer.echo(f"Zep graph manifest saved to: {out_path}")
    typer.echo(json.dumps({"graph_id": graph_id}, indent=2))
    write_run_log(
        out_path.parent,
        "build-zep",
        config,
        started_at,
        config_path=config_path,
        extra={
            "graph_id": graph_id,
            "outputs": {"zep_graph_built": str(out_path.name)},
        },
    )


@app.command("build-kg")
def build_kg(
    simulation_requirement: str = typer.Argument(..., help="What this simulation is about (e.g., 'AI talent development policy impact')"),
    graph_ontology: str = typer.Option(None, "--graph-ontology", help="Optional graph_ontology.json path. If omitted, ontology is generated first."),
    config_path: str = typer.Option("config/base.yaml", help="Path to config file"),
    policy: str = typer.Option(None, "--policy", help="Policy subpath under dataset/ (e.g. 'kor/bk21', 'kor/ssf', 'singapore/nais')."),
    out: str = typer.Option(None, help="Output path for policy_graph.json (default: outputs/stage1/<timestamp>_<model>_result_local_kg/policy_graph.json)"),
    no_iadig: bool = typer.Option(False, "--no-iadig", help="Ablation: build a free-form KG without IAD/IG schema constraints (no iad_category/node_family/deontic/edge_category enums; ontology generation skipped). metadata.builder='local_freeform'."),
):
    """Build a local IAD/IG policy KG without Zep.

    Produces policy_graph.json containing {nodes, edges, ontology, metadata}. If --graph-ontology is omitted,
    graph_ontology.json is generated and saved in the same run directory.
    """
    from llm.pdf_loader import load_pdfs
    from graph.ontology_generator import generate_ontology
    from graph.local_graph_builder import build_policy_graph

    config = load_config(config_path)
    _apply_policy_override(config, policy)
    if no_iadig:
        config.setdefault("kg", {})
        config["kg"]["mode"] = "freeform"
        config["kg"]["skip_iadig_validation"] = True
    started_at = start_run()
    tag = "local_freeform_kg" if no_iadig else "local_kg"
    run_dir = Path(out).parent if out else make_run_dir(config, "stage1", tag=tag)
    run_dir.mkdir(parents=True, exist_ok=True)

    typer.echo("Loading past policy PDFs...")
    past_text = load_pdfs(config["paths"]["past_policy_dir"], config)

    if no_iadig:
        typer.echo("Free-form mode: skipping IAD/IG ontology generation")
        ontology = {"entity_types": [], "edge_types": [], "_freeform": True}
    elif graph_ontology:
        ontology = load_json(graph_ontology)
        typer.echo(f"Using graph ontology: {graph_ontology}")
    else:
        typer.echo("Generating IAD/IG graph ontology...")
        ontology = generate_ontology([past_text], simulation_requirement, config)
        save_json(ontology, run_dir / "graph_ontology.json")
        typer.echo(f"  Graph ontology saved to: {run_dir / 'graph_ontology.json'}")

    typer.echo(
        "Building local free-form policy graph (no IAD/IG)..." if no_iadig
        else "Building local IAD/IG policy graph (no Zep)..."
    )
    policy_graph = build_policy_graph(past_text, ontology, config, simulation_requirement)

    out_path = Path(out) if out else run_dir / "policy_graph.json"
    save_json(policy_graph, out_path)
    validation_audit = policy_graph.get("metadata", {}).get("validation_audit")
    if validation_audit:
        save_json(validation_audit, out_path.parent / "validation_report.json")
    typer.echo(f"Policy graph saved to: {out_path}")
    typer.echo(
        json.dumps(
            {
                "graph_id": policy_graph["graph_id"],
                "nodes": len(policy_graph["nodes"]),
                "edges": len(policy_graph["edges"]),
                "validation_warnings": len(policy_graph.get("validation_warnings", [])),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    write_run_log(
        out_path.parent,
        "build-kg",
        config,
        started_at,
        config_path=config_path,
        extra={
            "simulation_requirement": simulation_requirement,
            "graph_id": policy_graph["graph_id"],
            "graph_summary": {
                "nodes": len(policy_graph["nodes"]),
                "edges": len(policy_graph["edges"]),
                "validation_warnings": len(policy_graph.get("validation_warnings", [])),
            },
            "ontology_source": graph_ontology if graph_ontology else "generated_in_run",
            "outputs": {"policy_graph": str(out_path.name)},
        },
    )


@app.command()
def build_graph(
    simulation_requirement: str = typer.Argument(..., help="What this simulation is about (e.g., 'AI talent development policy impact')"),
    config_path: str = typer.Option("config/base.yaml", help="Path to config file"),
    policy: str = typer.Option(None, "--policy", help="Policy subpath under dataset/ (e.g. 'kor/bk21', 'kor/ssf', 'singapore/nais')."),
):
    """Convenience: gen-ontology + build-zep in one command.

    Writes BOTH graph_ontology.json (the LLM-generated ontology, standalone)
    AND zep_graph_built.json (the Zep manifest with {graph_id, ontology}).
    """
    from llm.pdf_loader import load_pdfs
    from graph.ontology_generator import generate_ontology
    from graph.graph_builder import build_graph as _build_graph

    config = load_config(config_path)
    _apply_policy_override(config, policy)
    started_at = start_run()
    run_dir = make_run_dir(config, "stage1")

    typer.echo("Loading past policy PDFs...")
    past_text = load_pdfs(config["paths"]["past_policy_dir"], config)

    typer.echo("Generating graph ontology from policy documents...")
    ontology = generate_ontology([past_text], simulation_requirement, config)
    typer.echo(f"  Graph ontology: {len(ontology['entity_types'])} entity types, "
               f"{len(ontology['edge_types'])} edge types.")
    save_json(ontology, run_dir / "graph_ontology.json")
    typer.echo(f"  Graph ontology saved to: {run_dir / 'graph_ontology.json'}")

    typer.echo("Building Zep graph...")
    graph_id = _build_graph(past_text, ontology, config)
    typer.echo(f"  Zep graph built: {graph_id}")

    save_json({"graph_id": graph_id, "ontology": ontology}, run_dir / "zep_graph_built.json")
    typer.echo(f"  Zep graph manifest saved to: {run_dir / 'zep_graph_built.json'}")
    typer.echo(json.dumps({"graph_id": graph_id}, indent=2))
    write_run_log(
        run_dir,
        "build-graph",
        config,
        started_at,
        config_path=config_path,
        extra={
            "simulation_requirement": simulation_requirement,
            "graph_id": graph_id,
            "ontology_summary": {
                "entity_types": len(ontology["entity_types"]),
                "edge_types": len(ontology["edge_types"]),
            },
            "outputs": {
                "graph_ontology": "graph_ontology.json",
                "zep_graph_built": "zep_graph_built.json",
            },
        },
    )


@app.command()
def dump_graph(
    zep_graph_built: str = typer.Option(..., "--zep-graph-built", help="Path to zep_graph_built.json produced by build-zep or build-graph"),
    output: str = typer.Option(None, help="Output path for zep_graph_result.json (default: same dir as zep_graph_built.json)"),
    config_path: str = typer.Option("config/base.yaml", help="Path to config file"),
):
    """STEP 3 - fetch all nodes and edges from the built Zep graph and save the ZEP GRAPH RESULT to disk."""
    from graph.graph_dumper import dump_graph as _dump_graph

    config = load_config(config_path)
    info = load_json(zep_graph_built)
    graph_id = info["graph_id"]
    typer.echo(f"Dumping Zep graph: {graph_id}")

    data = _dump_graph(graph_id, config)

    out_path = Path(output) if output else Path(zep_graph_built).parent / "zep_graph_result.json"
    save_json(data, out_path)
    typer.echo(f"Zep graph result saved to: {out_path}")
    typer.echo(f"  {len(data['nodes'])} nodes, {len(data['edges'])} edges")


@app.command()
def stage1(
    simulation_requirement: str = typer.Argument(..., help="Short description of the downstream simulation target or requirement"),
    zep_graph_built: str = typer.Option(None, "--zep-graph-built", help="Path to zep_graph_built.json produced by build-zep or build-graph (not required with --no-graph)"),
    policy_graph: str = typer.Option(None, "--policy-graph", help="Path to local policy_graph.json produced by build-kg (not required with --no-graph)"),
    no_graph: bool = typer.Option(False, "--no-graph", help="Ablation: generate personas from past policy text only, skip Zep graph"),
    config_path: str = typer.Option("config/base.yaml", help="Path to config file"),
    policy: str = typer.Option(None, "--policy", help="Policy subpath under dataset/ (e.g. 'kor/bk21', 'kor/ssf', 'singapore/nais')."),
    run_tag: str = typer.Option(None, help="Optional tag appended to the output dir name"),
):
    """STEP 4 - run Stage 1 pipeline: economic context + policy plan + persona handoff for Stage 2."""
    config = load_config(config_path)
    _apply_policy_override(config, policy)
    started_at = start_run()
    run_dir = make_run_dir(config, "stage1", tag=run_tag)

    from stage1.pipeline import run_stage1
    if no_graph:
        result = run_stage1(config, run_dir, simulation_requirement, use_graph=False)
    else:
        if policy_graph:
            info = load_json(policy_graph)
            ontology = info.get("ontology")
            typer.echo(f"Using local policy graph: {policy_graph}")
            result = run_stage1(
                config,
                run_dir,
                simulation_requirement,
                ontology=ontology,
                policy_graph_path=policy_graph,
            )
        else:
            if not zep_graph_built:
                typer.echo("Error: --policy-graph or --zep-graph-built is required unless --no-graph is set.", err=True)
                raise typer.Exit(1)
            info = load_json(zep_graph_built)
            graph_id = info["graph_id"]
            ontology = info["ontology"]
            typer.echo(f"Using Zep graph: {graph_id}")
            result = run_stage1(config, run_dir, simulation_requirement, graph_id=graph_id, ontology=ontology)
    typer.echo(f"Stage 1 complete -> {run_dir}")
    pool_count = sum(len(group.personas) for group in result.persona_pool)
    selected_count = sum(len(group.personas) for group in result.simulation_participants)
    typer.echo(json.dumps({
        "run_dir": str(run_dir),
        "persona_pool": pool_count,
        "simulation_participants": selected_count,
        "stage1_result": str(run_dir / "stage1_result.json"),
    }, indent=2))
    write_run_log(
        run_dir,
        "stage1",
        config,
        started_at,
        config_path=config_path,
        extra={
            "simulation_requirement": simulation_requirement,
            "run_tag": run_tag,
            "mode": "no_graph" if no_graph else ("policy_graph" if policy_graph else "zep_graph"),
            "policy_graph": policy_graph,
            "zep_graph_built": zep_graph_built,
            "persona_pool": pool_count,
            "simulation_participants": selected_count,
            "outputs": {"stage1_result": "stage1_result.json"},
        },
    )


@app.command("stage2-toc")
def stage2_toc(
    simulation_target: str = typer.Argument(..., help="What is being simulated (e.g. 'number of SCI(E) publications in 2021-2022')"),
    stage1_result: str = typer.Option(..., "--stage1-result", help="Path to stage1_result.json from a previous stage1 run"),
    simulation_personas: str = typer.Option(None, "--simulation-personas", help="Path to scenario-grouped simulation_personas.json (defaults to alongside stage1_result.json)"),
    config_path: str = typer.Option("config/base.yaml", help="Path to config file"),
    policy: str = typer.Option(None, "--policy", help="Policy subpath under dataset/ (e.g. 'kor/ssf', 'singapore/nais'). Re-roots output dir as outputs/<country>/<policy>/stage2_toc/."),
    scenarios: str = typer.Option(None, "--scenarios", help="Scenario index range to run, format '<lo>-<hi>' (1-indexed, inclusive). Output dir tag gets '_sc<lo>-<hi>' appended. Default: all scenarios."),
    run_tag: str = typer.Option(None, help="Optional tag appended to the output dir name"),
    no_forward: bool = typer.Option(False, "--no-forward", help="Skip the forward (Inputs->Impact) pass"),
    no_backward: bool = typer.Option(False, "--no-backward", help="Skip the backward (Impact->Inputs) pass"),
    inner_schema: str = typer.Option(None, "--inner-schema", help="Override LLM-built inner schema with a JSON file"),
    resume_from: str = typer.Option(None, "--resume-from", help="Resume scenarios already completed under another output dir"),
    policy_md_override: str = typer.Option(None, "--policy-md-override", help="Path to raw policy markdown file. When set, replaces the KG-derived policy_context JSON with the markdown text in simulation prompts (KG-ablation experiment)."),
    simple_prediction: bool = typer.Option(False, "--simple-prediction", help="Ablation: skip 5-phase ToC and run one 'Impact'-only round predicting the target directly (forces --no-backward, no refinement, no seq chains)."),
    stance: str = typer.Option(None, "--stance", help="Persona reasoning framing condition: 'optimistic' | 'neutral' | 'conservative' (main 3-way study), or 'baseline' (empty directive, reproduces default). Omit ⇒ default == baseline. Same scenario/personas/inner-schema across stances → independent perspective band."),
    parallel_scenarios: bool = typer.Option(False, "--parallel-scenarios", help="Run scenario groups concurrently. Within-scenario interaction order is unchanged."),
    interaction_mode: str = typer.Option(None, "--interaction-mode", help="Override phase interaction: 'sequential' | 'independent' | 'independent_revision'."),
):
    """STEP 5 - Stage 2 ToC simulation: forward + backward causal chains per scenario."""
    config = load_config(config_path)
    _apply_policy_override(config, policy)
    if stance is not None:
        valid_stances = {"baseline", "optimistic", "neutral", "conservative"}
        if stance not in valid_stances:
            raise typer.BadParameter(f"--stance must be one of {sorted(valid_stances)}; got {stance!r}")
        config.setdefault("stage2_toc", {})["stance"] = stance
        typer.echo(f"  stance condition: {stance}")
    if parallel_scenarios:
        config.setdefault("stage2_toc", {})["parallel_scenarios"] = True
        typer.echo("  scenario execution: parallel")
    if interaction_mode is not None:
        valid_modes = {"sequential", "independent", "independent_revision"}
        if interaction_mode not in valid_modes:
            raise typer.BadParameter(
                f"--interaction-mode must be one of {sorted(valid_modes)}; got {interaction_mode!r}"
            )
        config.setdefault("stage2_toc", {})["phase_interaction_mode"] = interaction_mode
        typer.echo(f"  phase interaction: {interaction_mode}")
    if policy_md_override:
        md_path = Path(policy_md_override)
        if not md_path.exists():
            raise typer.BadParameter(f"--policy-md-override file not found: {md_path}")
        md_text = md_path.read_text(encoding="utf-8")
        config.setdefault("stage2_toc", {})["policy_md_text"] = md_text
        typer.echo(f"  policy_context override: {md_path} ({len(md_text):,} chars)")
    started_at = start_run()

    # Label the output dir by stance so the per-stance runs are self-labeled and never collide.
    if stance is not None:
        run_tag = f"{run_tag}_stance-{stance}" if run_tag else f"stance-{stance}"

    # Parse --scenarios range and append to run_tag so each cut gets its own dir.
    scenario_range: tuple[int, int] | None = None
    if scenarios:
        try:
            lo_str, hi_str = scenarios.split("-", 1)
            lo, hi = int(lo_str), int(hi_str)
        except ValueError as exc:
            raise typer.BadParameter(f"--scenarios must be '<lo>-<hi>' (e.g. '1-50'); got {scenarios!r}") from exc
        if lo < 1 or hi < lo:
            raise typer.BadParameter(f"--scenarios range must satisfy 1 ≤ lo ≤ hi; got {lo}-{hi}")
        scenario_range = (lo, hi)
        tag_suffix = f"sc{lo}-{hi}"
        run_tag = f"{run_tag}_{tag_suffix}" if run_tag else tag_suffix

    run_dir = make_run_dir(config, "stage2_toc", tag=run_tag)

    from stage2_toc.pipeline import run_stage2_toc

    stage1_result_path = Path(stage1_result)
    sim_personas_path = Path(simulation_personas) if simulation_personas else None
    inner_schema_path = Path(inner_schema) if inner_schema else None
    resume_from_path = Path(resume_from) if resume_from else None

    final = run_stage2_toc(
        config=config,
        run_dir=run_dir,
        simulation_target=simulation_target,
        stage1_result_path=stage1_result_path,
        simulation_personas_path=sim_personas_path,
        run_forward_flag=False if no_forward else None,
        run_backward_flag=False if no_backward else None,
        inner_schema_override=inner_schema_path,
        resume_from=resume_from_path,
        scenario_range=scenario_range,
        simple_prediction=simple_prediction,
    )

    typer.echo(f"Stage 2 ToC complete -> {run_dir}")
    write_run_log(
        run_dir,
        "stage2-toc",
        config,
        started_at,
        config_path=config_path,
        extra={
            "simulation_target": simulation_target,
            "run_tag": run_tag,
            "stage1_result": stage1_result,
            "simulation_personas": str(sim_personas_path or stage1_result_path.parent / "simulation_personas.json"),
            "scenario_count": len(final.scenarios) if final else 0,
            "outputs": {
                "stage2_toc_result": str((run_dir / config.get("stage2_toc", {}).get("output_file", "stage2_toc_result.json")).name),
                "final_report": "final_report.json",
            },
        },
    )


@app.command("grounded-inputs")
def grounded_inputs(
    stage1_result: str = typer.Option(..., "--stage1-result", help="Path to stage1_result.json"),
    config_path: str = typer.Option("config/base.yaml", help="Path to config file"),
    policy_graph: str = typer.Option(None, "--policy-graph", help="Optional policy_graph.json for causal linking"),
    output: str = typer.Option("outputs/grounded_input_state.json", "--output", help="Output JSON path"),
):
    """Extract and validate document-grounded ToC Inputs before schema generation."""
    from models.schemas import Stage1Result
    from stage2_toc.grounded_inputs import build_grounded_input_state

    config = load_config(config_path)
    stage1_path = Path(stage1_result)
    if not stage1_path.exists():
        raise typer.BadParameter(f"stage1_result not found: {stage1_path}")
    payload = Stage1Result.model_validate(load_json(stage1_path))
    if payload.policy_context is None:
        raise typer.BadParameter("stage1_result.policy_context is required")
    graph_path = Path(policy_graph) if policy_graph else None
    if graph_path is not None and not graph_path.exists():
        raise typer.BadParameter(f"policy_graph not found: {graph_path}")
    started_at = start_run()
    state = build_grounded_input_state(config, payload.policy_context, graph_path)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(state.model_dump(mode="json"), output_path)
    typer.echo(json.dumps({
        "output": str(output_path),
        "fields": state.values,
        "policy_parameters": {
            parameter.name: parameter.value for parameter in state.policy_parameters
        },
        "unresolved_candidates": len(state.unresolved_candidates),
    }, ensure_ascii=False, indent=2))
    write_run_log(
        output_path.parent,
        "grounded-inputs",
        config,
        started_at,
        config_path=config_path,
        extra={
            "stage1_result": str(stage1_path),
            "policy_graph": str(graph_path) if graph_path else None,
            "outputs": {"grounded_input_state": output_path.name},
        },
    )

if __name__ == "__main__":
    app()
