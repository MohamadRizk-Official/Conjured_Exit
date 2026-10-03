"""Resolve log/param names against a TOC whose names were mangled by nRF firmware 2024.10.

Every BLE downlink packet longer than 20 bytes loses one byte at packet index 19 (see README.md,
"Verified BLE facts"). In a TOC item reply that is offset 14 of the "group\\0name\\0" string, so
long entries arrive with one character missing ("stabilizer.estimator" -> "estmator",
"kalman.resetEstimation" -> "resetEsimation") and sometimes a stray trailing NUL. Entries whose
whole reply fits in 20 bytes are intact, and the ids are always intact. cflib keys its TOC by
the mangled names, so we look up what cflib stored and hand cflib its own name back.

    from toc_names import resolve
    cf.param.set_value(resolve(cf.param.toc, "stabilizer.estimator"), "2")
    logconf.add_variable(resolve(cf.log.toc, "stabilizer.roll"), "FP16")

Names whose lost character cannot be reconstructed unambiguously (e.g. "stateEstimate.x",
"stateEstimate.vx" and "stateEstimate.qx" all collapse to the same entry) raise KeyError;
use another variable (kalman.stateX/Y/Z) instead.
"""
from __future__ import annotations

LOST_OFFSET = 14          # offset in "group\0name\0" of the byte the nRF drops
PACKET_PREFIX = 5         # CRTP header + command + id (2) + type byte, before the strings
WHOLE_PACKET_MAX = 20     # a reply this short is a single BLE fragment and arrives intact


def is_intact(group: str, name: str) -> bool:
    """True when the TOC reply for this entry fits in one BLE notification (never corrupted)."""
    return PACKET_PREFIX + len(group) + 1 + len(name) + 1 <= WHOLE_PACKET_MAX


def as_seen(group: str, name: str) -> tuple[str, str]:
    """Predict how a true (group, name) shows up after the nRF 2024.10 corruption."""
    if is_intact(group, name):
        return group, name
    s = f"{group}\x00{name}\x00"
    s = s[:LOST_OFFSET] + s[LOST_OFFSET + 1:]
    parts = s.split("\x00")
    return parts[0], parts[1] if len(parts) > 1 else ""


def _entries(toc) -> dict:
    """Accept a cflib Toc object (has .toc) or a plain {group: {name: ...}} dict."""
    return getattr(toc, "toc", toc)


def resolve(toc, complete_name: str) -> str:
    """Return the 'group.name' string cflib knows for the true variable name, or raise KeyError."""
    if "." not in complete_name:
        raise KeyError(f"{complete_name!r} is not a 'group.name'")
    group, name = complete_name.split(".", 1)
    groups = _entries(toc)
    entries = groups.get(group)
    if entries is None:
        raise KeyError(f"group {group!r} not in TOC")
    if name in entries and is_intact(group, name):
        return complete_name
    norm = {}
    for key in entries:
        norm.setdefault(key.replace("\x00", ""), key)
    candidates: list[str] = []
    if name in norm and is_intact(group, name):
        return f"{group}.{norm[name]}"
    # the dropped byte sits at offset LOST_OFFSET of "group\0name\0"
    i = LOST_OFFSET - (len(group) + 1)
    if 0 <= i < len(name):
        damaged = name[:i] + name[i + 1:]
        if damaged in norm:
            candidates.append(norm[damaged])
    elif i == len(name) or i < 0:
        # the lost byte is the terminating NUL (or inside the group): the name itself survives
        if name in norm:
            candidates.append(norm[name])
    if name in norm and norm[name] not in candidates:
        candidates.append(norm[name])   # exact match that we cannot prove intact
    candidates = list(dict.fromkeys(candidates))
    if len(candidates) == 1:
        # an exact-looking match may also be another, longer name with its first char dropped
        if 0 <= i < len(name) and candidates[0].replace("\x00", "") == name and i == 0:
            raise KeyError(f"{complete_name}: ambiguous after BLE TOC corruption (first character of the "
                           f"name is the lost byte); use a different variable")
        return f"{group}.{candidates[0]}"
    if len(candidates) > 1:
        raise KeyError(f"{complete_name}: ambiguous after BLE TOC corruption: {candidates}")
    raise KeyError(f"{complete_name}: not in TOC (BLE TOC corruption or wrong name)")


def resolve_many(toc, names) -> dict[str, str]:
    """{true_name: cflib_name} for every name; raises on the first failure."""
    return {n: resolve(toc, n) for n in names}
