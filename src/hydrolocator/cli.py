"""Command line: hydrolocator who | where | catalog | ais | blindtest | serve."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="Match ship passages heard on a hydrophone against AIS tracks.")
ais_app = typer.Typer(no_args_is_help=True, help="Manage the AIS store.")
app.add_typer(ais_app, name="ais")
console = Console()


def parse_time(text: str) -> float:
    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    return (dt if dt.tzinfo else dt.replace(tzinfo=UTC)).timestamp()


def hm(text: str | None) -> str:
    return "" if not text else text[11:16]


def _clip(audio: Path | None, site: str | None, start: str | None, hours: float, source: str):
    from .pipeline import clip_from_audio
    from .public import public_clip

    if audio:
        return clip_from_audio(audio, parse_time(start) if start else None)
    if not (site and start):
        raise typer.BadParameter("give --audio FILE, or --site ID with --start")
    t0 = parse_time(start)
    return public_clip(site, t0, t0 + hours * 3600.0, source=source)


def _candidates(clip, hidden: str | None = None):
    from . import catalog
    from .pipeline import Candidate

    return [Candidate(s.id, d.lat, d.lon, s.name, s.network)
            for s, d in catalog.active_sites(clip.t_start, clip.t_end)]


def _print_passages(rep: dict) -> None:
    table = Table(show_edge=False, header_style="bold")
    for col in ("status", "AIS CPA", "heard", "dt min", "range km", "kn", "p", "dB", "vessel"):
        table.add_column(col, justify="right" if col not in ("status", "vessel") else "left")
    style = {"matched": "green", "also_present": "cyan", "not_heard": "yellow", "dark": "magenta"}
    for r in rep["passages"]:
        vessel = "" if r["mmsi"] is None else (
            f"{r['name'] or r['mmsi']}  {r['type']}" + (f" {r['length_m']:.0f} m" if r["length_m"] else ""))
        table.add_row(f"[{style[r['status']]}]{r['status'].replace('_', ' ')}[/]", hm(r["cpa_time"]), hm(r["heard_time"]),
                      "" if r["dt_s"] is None else f"{r['dt_s'] / 60:+.1f}",
                      "" if r["cpa_range_km"] is None else f"{r['cpa_range_km']:.1f}",
                      "" if r["speed_kn"] is None else f"{r['speed_kn']:.1f}",
                      "" if r["detect_prob"] is None else f"{r['detect_prob']:.2f}",
                      "" if r["prominence_db"] is None else f"{r['prominence_db']:.0f}", vessel)
    console.print(table)
    c = rep["counts"]
    console.print(f"{c['heard']} passages heard, {c['ais_passages']} AIS passages within 20 km: "
                  f"{c['matched']} matched, {c['dark']} with no AIS match, {c['not_heard']} expected but not heard. "
                  f"Level-curve match: rho {rep['match']['rho']}, z {rep['match']['z']}.")
    if rep["ais_days_missing"]:
        console.print(f"[yellow]No AIS held for {', '.join(rep['ais_days_missing'])}; run `hydrolocator ais ingest`.[/]")


@app.command()
def who(site: str | None = typer.Option(None, help="Catalog id, for example sanctsound:sb01"),
        start: str | None = typer.Option(None, help="UTC start, ISO 8601"),
        hours: float = typer.Option(24.0, help="Clip length when reading a public recording"),
        audio: Path | None = typer.Option(None, exists=True, help="Local recording instead of a public one"),
        lat: float | None = typer.Option(None), lon: float | None = typer.Option(None),
        source: str = typer.Option("auto", help="auto | spectra | audio"),
        as_json: bool = typer.Option(False, "--json")):
    """Mode A: which ships passed a known hydrophone."""
    from . import catalog, report
    from .pipeline import who_passed

    clip = _clip(audio, None if audio else site, start, hours, source)
    if lat is None or lon is None:
        if not site:
            raise typer.BadParameter("a local recording needs --site or --lat/--lon")
        s = catalog.get(site)
        dep = s.active(clip.t_start, clip.t_end)
        lat, lon = (dep.lat, dep.lon) if dep else (s.lat, s.lon)
    rep = report.site_report(who_passed(clip, lat, lon), with_curves=as_json, with_tracks=as_json)
    if as_json:
        typer.echo(json.dumps(rep))
        return
    console.print(f"[bold]{clip.label}[/]  {clip.hours:.1f} h at {lat:.4f}, {lon:.4f}")
    _print_passages(rep)


@app.command()
def where(audio: Path | None = typer.Option(None, exists=True, help="Recording to locate"),
          start: str | None = typer.Option(None, help="UTC start (approximate is fine with --clock-hours)"),
          clock_hours: float = typer.Option(0.0, help="How wrong the start time could be, +/- hours (14 covers any timezone)"),
          site: str | None = typer.Option(None, help="Blind test: read this public site, then hide it"),
          hours: float = typer.Option(24.0), source: str = typer.Option("auto"),
          refine: bool = typer.Option(False, help="Grid-search around the best site"),
          as_json: bool = typer.Option(False, "--json")):
    """Mode B: where was this recorded."""
    from . import report
    from .pipeline import grid_search, locate, who_passed

    clip = _clip(audio, site, start, hours, source)
    cands = _candidates(clip)
    result = locate(clip, cands, clock_uncertainty_s=clock_hours * 3600.0)
    out = report.locate_result(result)
    best = result.best
    evidence = grid = None
    if best and result.verdict in ("located", "probable", "ambiguous") and best.z > 0:
        evidence = report.site_report(who_passed(clip, best.candidate.lat, best.candidate.lon, offset_s=best.offset_s),
                                      with_curves=as_json, with_tracks=as_json)
        if refine:
            grid = report.grid_result(grid_search(clip, best.candidate.lat, best.candidate.lon, offset_s=best.offset_s))
    if as_json:
        typer.echo(json.dumps({"locate": out, "evidence": evidence, "grid": grid}))
        return
    console.print(f"[bold]{clip.label}[/]  {clip.hours:.1f} h, {len(cands)} candidate sites, "
                  f"{out['clock_offsets_searched']} clock offsets")
    table = Table(show_edge=False, header_style="bold")
    for col in ("site", "confidence", "z", "rho", "clock offset", "where"):
        table.add_column(col, justify="left" if col in ("site", "where") else "right")
    for r in out["ranking"][:5]:
        table.add_row(f"{r['id']}  {r['name']}", f"{r['confidence']:.0%}", f"{r['z']:.1f}", f"{r['rho']:.2f}",
                      f"{r['clock_offset_s'] / 60:+.0f} min", f"{r['lat']:.3f}, {r['lon']:.3f}")
    console.print(table)
    verdicts = {"located": "Located.", "probable": "Probably the top site, but not certain.", "ambiguous": "Best guess only: no site stands clear of the rest.",
                "not_enough_ships": "Not enough ship passages in this clip to say.",
                "no_ais_coverage": "No AIS held for this time, so nothing can be matched."}
    console.print(f"[bold]{verdicts[result.verdict]}[/] None-of-these: {out['none_of_these']:.0%}.")
    if site:
        truth = next((k + 1 for k, r in enumerate(out["ranking"]) if r["id"].endswith(site.split(':')[-1])), None)
        console.print(f"Hidden truth {site}: rank {truth}.")
    if evidence:
        console.print(f"\nShips that explain the match at {best.candidate.id}:")
        _print_passages(evidence)
    if grid:
        console.print(f"Grid refinement: best cell at {grid['best']['lat']:.4f}, {grid['best']['lon']:.4f}")


@app.command("catalog")
def catalog_cmd(on: str | None = typer.Option(None, "--date", help="Only sites recording on this UTC date")):
    """List the public hydrophones Mode B can name."""
    from . import catalog

    t = parse_time(on) if on else None
    table = Table(show_edge=False, header_style="bold")
    for col in ("id", "name", "network", "lat", "lon", "AIS region", "deployments"):
        table.add_column(col)
    for s in catalog.load():
        if t is not None and not s.active(t, t + 86400.0):
            continue
        first = datetime.fromtimestamp(s.deployments[0].start, UTC).date()
        last = s.deployments[-1].end
        span = f"{first} to {datetime.fromtimestamp(last, UTC).date() if last else 'now'}"
        table.add_row(s.id, s.name, s.network, f"{s.lat:.4f}", f"{s.lon:.4f}", s.region or "none", f"{len(s.deployments)}  {span}")
    console.print(table)


@ais_app.command("ingest")
def ais_ingest(days: list[str] = typer.Argument(..., help="UTC dates, YYYY-MM-DD"),
               keep_zip: bool = typer.Option(False, help="Keep the raw MarineCadastre zip")):
    """Download MarineCadastre AIS for the given days and index the v1 regions."""
    from . import ais

    for text in days:
        d = date.fromisoformat(text)
        with console.status(f"{d}: downloading and indexing"):
            path = ais.ingest_day(d, keep_zip=keep_zip)
        console.print(f"{d}: {path.stat().st_size / 1e6:.1f} MB")


@ais_app.command("days")
def ais_days():
    """List the UTC days held in the AIS store."""
    from . import ais

    days = ais.available_days()
    console.print(f"{len(days)} days: " + ", ".join(str(d) for d in days))


@app.command()
def blindtest(hours: float = typer.Option(24.0, help="Clip length"),
              refit: bool = typer.Option(False, help="Refit the level model on the training days first"),
              as_json: bool = typer.Option(False, "--json")):
    """Blind site identification on held-out days (the main accuracy test)."""
    from . import ais
    from . import blindtest as bt
    from .soundscape import LevelParams

    days = ais.available_days()
    train_days, test_days = days[0::2], days[1::2]
    params = LevelParams()
    if refit:
        with console.status("fitting on training days"):
            params = bt.fit(bt.build_cases(train_days))
        console.print({k: round(v, 2) for k, v in params.to_dict().items()})
    with console.status("building held-out cases"):
        cases = bt.build_cases(test_days)
    if hours < 24:
        cases = bt.subclips(cases, hours)
    res = bt.evaluate(cases, params)
    if as_json:
        typer.echo(json.dumps(res))
        return
    table = Table(show_edge=False, header_style="bold")
    for col in ("clip", "source", "rho", "z", "rank", "top pick", "top z"):
        table.add_column(col)
    for r in res["rows"]:
        mark = "green" if r["rank"] == 1 else "red"
        table.add_row(r["label"], r["source"], f"{r['rho']:.2f}", f"{r['z']:.1f}", f"[{mark}]{r['rank']}[/]", r["top"], f"{r['top_z']:.1f}")
    console.print(table)
    console.print(f"[bold]top-1 {res['top1']}/{res['n']} = {res['top1_rate']:.0%}[/], top-3 {res['top3']}/{res['n']}, "
                  f"{hours:g} h clips, {len(test_days)} held-out days")


@app.command()
def tracks(site_a: str = typer.Argument(..., help="First catalog id, for example sanctsound:sb01"),
           site_b: str = typer.Argument(..., help="Second catalog id in the same waters"),
           start: str = typer.Option(..., help="UTC start, ISO 8601"), hours: float = typer.Option(24.0),
           as_json: bool = typer.Option(False, "--json")):
    """Link sounds with no AIS at two hydrophones and infer each ship's track."""
    from . import multisite

    t0 = parse_time(start)
    with console.status("reading both recordings"):
        found = multisite.analyse(site_a, site_b, t0, t0 + hours * 3600.0)
    if as_json:
        typer.echo(json.dumps(found))
        return
    a, b = found["sites"]
    console.print(f"[bold]{a['name']} and {b['name']}[/], {found['baseline_km']} km apart; "
                  f"{found['library']['crossings']} AIS crossings in the store")
    table = Table(show_edge=False, header_style="bold")
    for col in ("kind", "at A", "at B", "kn", "range A km", "range B km", "vessel or route"):
        table.add_column(col)
    for c in found["on_ais"]:
        table.add_row("on AIS", hm(c["t_a"]), hm(c["t_b"]), f"{c['speed_kn']:.1f}", f"{c['range_a_km']:.1f}", f"{c['range_b_km']:.1f}",
                      f"{c['name'] or c['mmsi']} (heard at {'both' if c['heard_a'] and c['heard_b'] else 'one or neither'})")
    for t in found["inferred"]:
        table.add_row("[magenta]possible[/]", hm(t["t_a"]), hm(t["t_b"]), f"{t['speed_kn']:.1f}", f"{t['range_a_km']:.1f}",
                      f"{t['range_b_km']:.1f}", f"route of {t['route_like']} ({t['routes_used']} AIS crossings)")
    console.print(table)
    v = found["validation"]
    console.print(f"Tested on {v['ships']} AIS ships with their AIS hidden: speed within {v['speed_err_kn']} kn and range within "
                  f"{v['range_err_km']} km given the right pair; the right pair was chosen {v['links_right']} times of {v['ships']}.")


@app.command()
def voiceprints():
    """Rebuild the library of vessel signatures from every AIS-matched passage held."""
    from . import voiceprint

    with console.status("building"):
        lib = voiceprint.build_library()
    console.print(f"{len(lib)} signatures from {len(set(lib.mmsi.tolist()))} vessels.")
    for threshold, right in lib.calibration:
        console.print(f"  nearest match at similarity {threshold:.1f} or more named the right vessel {right:.0%} of the time")


@app.command()
def live(node: str | None = typer.Argument(None, help="Orcasound node, for example orcasound-lab. Omit to list them."),
         minutes: float = typer.Option(3.0, help="How long to listen")):
    """Listen to a live Orcasound hydrophone and report ship noise as it happens."""
    from . import live as live_mod
    from .sources import orcasound

    nodes = orcasound.nodes()
    pick = next((n for n in nodes if n.slug == node), None)
    if pick is None:
        for n in nodes:
            console.print(f"{n.slug:18s} {n.name}  ({n.lat:.4f}, {n.lon:.4f})")
        return
    console.print(f"[bold]{pick.name}[/] live. Naming the ship needs a live AIS feed, which is not connected.")
    levels: list[float] = []
    for t, level in live_mod.listen(pick.stream_url(), minutes * 60.0):
        levels.append(level)
        excess = level - min(levels)
        bar = "#" * int(min(40, max(0, excess * 2)))
        console.print(f"{datetime.fromtimestamp(t, UTC):%H:%M:%S} UTC  {excess:5.1f} dB over the quietest so far  {bar:40s} {live_mod.describe(levels)}")


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000):
    """Run the JSON API (docs at /api/docs)."""
    import uvicorn

    uvicorn.run("hydrolocator.api:app", host=host, port=port)


if __name__ == "__main__":
    app()
