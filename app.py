from pathlib import Path
from threading import RLock
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from dsa_engine import MetroGraph

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="Ahmedabad Metro DSA Route Planner")
metro = MetroGraph()

# The network is one shared in-memory object. FastAPI runs sync endpoints in a thread
# pool, so track edits and route searches are serialised to avoid reading a half-updated graph.
_lock = RLock()

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class RouteRequest(BaseModel):
    origin: int
    destination: int
    algorithm: Literal["dijkstra", "bfs"] = "dijkstra"
    is_peak: bool = True


class TrackUpdateRequest(BaseModel):
    u: int
    v: int
    is_closed: bool
    delay_mins: int = Field(default=0, ge=0, le=60)


@app.get("/")
def serve_ui():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/network")
def get_network():
    with _lock:
        return metro.serialize_network()


@app.post("/api/route")
def compute_route(req: RouteRequest):
    if req.origin not in metro.stations or req.destination not in metro.stations:
        raise HTTPException(status_code=400, detail="Invalid station ID.")
    if req.origin == req.destination:
        raise HTTPException(status_code=400, detail="Origin and Destination cannot be the same station.")

    with _lock:
        if req.algorithm == "bfs":
            return metro.bfs_minimum_stops(req.origin, req.destination, req.is_peak)
        return metro.dijkstra_fastest_route(req.origin, req.destination, req.is_peak)


@app.post("/api/track")
def update_track_status(req: TrackUpdateRequest):
    with _lock:
        ok = metro.update_track(req.u, req.v, req.is_closed, req.delay_mins)
        if not ok:
            raise HTTPException(status_code=404, detail="No direct track connects the specified stations.")
        return {"status": "updated", "network": metro.serialize_network()}


@app.post("/api/reset")
def reset_network():
    with _lock:
        metro.reset_all_tracks()
        return {"status": "reset", "network": metro.serialize_network()}
