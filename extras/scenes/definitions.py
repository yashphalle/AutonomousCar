"""
Scene definitions — NPC layouts for testing behaviour planning phases.

Usage:
    python3.10 main.py --scene <name>

Each NPCSpec is placed relative to ego's start position along the route.
route_offset_m > 0 = ahead, < 0 = behind.
lateral_offset_m > 0 = left of lane centre, < 0 = right.

NOTE on stop signs:
    Stop signs are CARLA map fixtures — they cannot be spawned via this file.
    Town01 spawn[0]→spawn[50] may not have stop signs on the route.
    Use the 'stop_sign_sim' scene which places a stationary blocker to
    exercise the stop-and-hold behaviour, or change the route in config.py
    to one that passes through a stop-sign intersection.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class NPCSpec:
    route_offset_m: float           # metres ahead (+) or behind (-) along route
    lateral_offset_m: float = 0.0   # metres left (+) or right (-) of lane centre
    target_speed_mps: float = 5.0   # cruise speed for vehicles; walk speed for peds
    blueprint: str = "vehicle.tesla.model3"
    label: str = ""
    is_pedestrian: bool = False
    walk_to_lateral_m: float | None = None  # directed destination (None = random nav)


@dataclass
class SceneConfig:
    name: str
    npcs: list[NPCSpec] = field(default_factory=list)
    description: str = ""


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------

SCENES: dict[str, SceneConfig] = {

    # ── 1. LANE FOLLOW ───────────────────────────────────────────────────────
    # Clean run: no NPCs, ego should cruise at speed limit the whole way.
    # Expected FSM: lane_follow throughout.
    # Safety planner: speed_limit / junction_speed_cap / red_light_stop as usual.
    "test_lane_follow": SceneConfig(
        name="test_lane_follow",
        description="[TEST 1] Empty road — pure lane follow, no obstacles",
        npcs=[],
    ),

    # ── 2. ACC / FOLLOW VEHICLE ──────────────────────────────────────────────
    # Lane change (Phase 3) is not yet implemented.
    # This scene tests the prerequisite: ego gets stuck behind a slow car and
    # maintains a safe following gap with ACC.
    # Expected FSM: lane_follow → follow_vehicle (once close enough).
    # Watch: lead dist stays ≥ 6 m floor; speed matches lead at ~10.8 km/h.
    "test_acc_follow": SceneConfig(
        name="test_acc_follow",
        description="[TEST 2] Slow lead car 20m ahead — ACC gap-keeping (prerequisite for lane change)",
        npcs=[
            NPCSpec(route_offset_m=20.0, target_speed_mps=3.0, label="lead_slow"),
        ],
    ),

    # ACC stress: two cars at different speeds — only closest in-lane triggers ACC.
    "test_acc_multi": SceneConfig(
        name="test_acc_multi",
        description="[TEST 2b] Two lead cars at 20m and 40m — ACC follows closest",
        npcs=[
            NPCSpec(route_offset_m=20.0, target_speed_mps=3.0, label="lead_close"),
            NPCSpec(route_offset_m=40.0, target_speed_mps=6.0, label="lead_far"),
        ],
    ),

    # ── 3. STOP SIGN ─────────────────────────────────────────────────────────
    # stop_and_go FSM requires real traffic.stop actors in the CARLA map.
    # Town01 spawn[0]→spawn[50] has none — run extras/find_stop_signs.py first
    # to identify which spawn pairs bracket a stop sign, then:
    #   python3.10 main.py --start X --end Y --scene test_lane_follow
    #
    # This scene is an empty placeholder — the stop sign FSM will fire
    # automatically whenever the route passes a map stop sign actor.
    "test_stop_sign": SceneConfig(
        name="test_stop_sign",
        description="[TEST 3] No NPCs — stop_and_go fires from map actors. Use Town03: python3.10 extras/find_stop_signs.py --town Town03",
        npcs=[],
    ),

    # ── 4. PEDESTRIAN ────────────────────────────────────────────────────────
    # 4a. Blocking ped: stationary in lane at 12m.
    # Expected FSM: lane_follow → emergency_stop → (3s clear dwell) → lane_follow.
    "test_ped_block": SceneConfig(
        name="test_ped_block",
        description="[TEST 4a] Stationary pedestrian 12m ahead — emergency stop + 3s dwell",
        npcs=[
            NPCSpec(route_offset_m=12.0, lateral_offset_m=0.0,
                    target_speed_mps=0.0, is_pedestrian=True, label="ped_static"),
        ],
    ),

    # 4b. Crossing ped: spawns left sidewalk, walks right across the road.
    # Expected FSM: lane_follow → emergency_stop (as ped crosses into ±20° cone)
    #               → clearing dwell → lane_follow (ped exits cone).
    "test_ped_crossing": SceneConfig(
        name="test_ped_crossing",
        description="[TEST 4b] Pedestrian crossing left→right at 15m ahead",
        npcs=[
            NPCSpec(route_offset_m=15.0, lateral_offset_m=3.5,
                    target_speed_mps=1.2, is_pedestrian=True,
                    walk_to_lateral_m=-3.5, label="ped_crossing"),
        ],
    ),

    # ── 5. COMBINED GAUNTLET ─────────────────────────────────────────────────
    # Exercises all implemented behaviours in one run:
    #   ~0–25 m   : open road         → lane_follow
    #   ~25 m     : slow lead car     → follow_vehicle (ACC)
    #   ~60 m     : another slow car  → ACC re-engages after first passes junction
    #   ~100 m    : ped blocking path → emergency_stop → 3s dwell → resume
    #   ~130 m    : ped crossing      → emergency_stop (brief) → resume
    #
    # Stop sign depends on map — will engage if route passes one.
    # Run with viz for full picture: python3.10 extras/viz.py in a second terminal.
    "test_gauntlet": SceneConfig(
        name="test_gauntlet",
        description="[TEST ALL] Full scenario: lane follow → ACC → ped block → ped crossing",
        npcs=[
            # Lead vehicles (ACC)
            NPCSpec(route_offset_m=25.0, target_speed_mps=3.5, label="lead_1"),
            NPCSpec(route_offset_m=60.0, target_speed_mps=4.0, label="lead_2"),
            # Pedestrians
            NPCSpec(route_offset_m=100.0, lateral_offset_m=0.0,
                    target_speed_mps=0.0, is_pedestrian=True, label="ped_block"),
            NPCSpec(route_offset_m=130.0, lateral_offset_m=3.5,
                    target_speed_mps=1.2, is_pedestrian=True,
                    walk_to_lateral_m=-3.5, label="ped_cross"),
        ],
    ),

    # ── REGRESSION / NEGATIVE TESTS ─────────────────────────────────────────
    # Car behind ego: must NOT trigger follow_vehicle.
    "test_neg_vehicle_behind": SceneConfig(
        name="test_neg_vehicle_behind",
        description="[NEG] Fast vehicle 15m behind — must stay lane_follow",
        npcs=[
            NPCSpec(route_offset_m=-15.0, target_speed_mps=10.0, label="chaser"),
        ],
    ),

    # Adjacent lane vehicle: same road, different lane_id — must NOT trigger ACC.
    "test_neg_adjacent_lane": SceneConfig(
        name="test_neg_adjacent_lane",
        description="[NEG] Vehicle in adjacent lane 15m ahead — lead filter must reject it",
        npcs=[
            NPCSpec(route_offset_m=15.0, lateral_offset_m=3.5,
                    target_speed_mps=3.0, label="adjacent"),
        ],
    ),

    # Ped to the side (outside ±20° cone): must NOT trigger emergency_stop.
    "test_neg_ped_sidewalk": SceneConfig(
        name="test_neg_ped_sidewalk",
        description="[NEG] Pedestrian 5m to the side at 15m ahead — must stay lane_follow",
        npcs=[
            NPCSpec(route_offset_m=15.0, lateral_offset_m=5.5,
                    target_speed_mps=0.0, is_pedestrian=True, label="ped_sidewalk"),
        ],
    ),

    # ── LEGACY NAMES (kept for backward compat) ──────────────────────────────
    "empty":                SceneConfig(name="empty", description="Alias: test_lane_follow", npcs=[]),
    "follow_vehicle_close": SceneConfig(name="follow_vehicle_close", description="Alias: test_acc_follow",
                                        npcs=[NPCSpec(route_offset_m=20.0, target_speed_mps=3.0, label="lead_slow")]),
    "pedestrian_in_path":   SceneConfig(name="pedestrian_in_path", description="Alias: test_ped_block",
                                        npcs=[NPCSpec(route_offset_m=10.0, lateral_offset_m=0.0,
                                                      target_speed_mps=0.0, is_pedestrian=True, label="ped_blocking")]),
    "pedestrian_crossing":  SceneConfig(name="pedestrian_crossing", description="Alias: test_ped_crossing",
                                        npcs=[NPCSpec(route_offset_m=15.0, lateral_offset_m=3.5,
                                                      target_speed_mps=1.2, is_pedestrian=True,
                                                      walk_to_lateral_m=-3.5, label="ped_crossing")]),
}
