"""HARD enforcement guard — the one place we deliberately hard-code, because a SAFETY boundary
must be GUARANTEED, not understood (understanding is never a proof). The brain stays free to think
and talk; the guard only sits at the ACTION layer and blocks forbidden actions before they run.

Set a constraint with a plain "don't" command:
  "do not alter files config.py, secrets.env"   "don't touch the data folder"
  "never delete x"   "do not run deploy.sh"      "do not modify y or z"

Then any attempt (a request in chat now, or a real file op once the brain has hands) is checked
against the constraints and BLOCKED if forbidden.
"""
import os
import re

_PROHIBIT = r"(?:do not|do n't|don'?t|never|do not ever|please do ?n'?t|please do not)"
# verb -> canonical action category (so "edit"/"modify"/"change" all map to "alter")
_VERB = {}
for _cat, _vs in {
    "alter": ("alter", "change", "modify", "edit", "overwrite", "write", "touch", "rewrite"),
    "delete": ("delete", "remove", "erase", "rm", "wipe", "destroy", "drop"),
    "read": ("read", "open", "access", "view", "see"),
    "run": ("run", "execute", "launch", "exec", "start"),
    "move": ("move", "rename", "relocate"),
    "create": ("create", "make", "add"),
}.items():
    for _v in _vs:
        _VERB[_v] = _cat
_NOISE = re.compile(r"^(?:the\s+|any\s+|ever\s+|to\s+|these\s+|those\s+|my\s+|"
                    r"files?\s+|folders?\s+|directory\s+|directories\s+|folder\s+)+", re.I)


def _targets(rest):
    rest = _NOISE.sub("", rest.strip())
    parts = re.split(r",|\band\b|\bor\b", rest)
    return [p.strip().strip(".'\"") for p in parts if p.strip().strip(".'\"")]


def set_constraints(constraints, text):
    """If text is a 'don't' command, add (action, target) constraints. Returns the list added."""
    m = re.match(rf"\s*{_PROHIBIT}\s+(.+)$", text.strip().rstrip("."), re.I)
    if not m:
        return []
    words = m.group(1).split()
    action, vi = None, -1
    for i, w in enumerate(words):
        c = _VERB.get(w.lower().rstrip(","))
        if c:
            action, vi = c, i
            break
    if action is None:                                   # "do not do X" -> generic prohibition
        tgts, action = [m.group(1).strip()], "do"
    else:
        tgts = _targets(" ".join(words[vi + 1:]))
    added = []
    for tg in tgts:
        rule = [action, tg]
        if tg and rule not in constraints:
            constraints.append(rule)
            added.append((action, tg))
    return added


def _hits(constraints, action, target):
    tl = str(target).lower().strip()
    tb = os.path.basename(tl)
    for act, tg in constraints:
        if act != action and act != "do":
            continue
        g = tg.lower()
        if g == tl or g == tb or os.path.basename(g) == tb or (len(g) > 2 and g in tl):
            return (act, tg)
    return None


def forbidden(constraints, text):
    """If the user is ASKING the brain to perform a forbidden action, return (action, target)
    to refuse; else None. (The action layer — once the brain can touch files, the same check
    gates the real operation.)"""
    if not constraints:
        return None
    t = text.strip().lower().rstrip(".!?")
    if re.match(rf"\s*{_PROHIBIT}\b", t):                # that's setting a rule, not requesting
        return None
    words = t.split()
    for i, w in enumerate(words):
        act = _VERB.get(w.rstrip(","))
        if act:
            for tg in _targets(" ".join(words[i + 1:])):
                hit = _hits(constraints, act, tg)
                if hit:
                    return hit
    return None


def allowed(constraints, action, target):
    """The hard gate for a REAL action (file op, etc.). Returns (ok: bool, reason)."""
    hit = _hits(constraints, action, target)
    if hit:
        return False, f"blocked by your constraint: do not {hit[0]} {hit[1]}"
    return True, ""


def describe(added):
    by_act = {}
    for act, tg in added:
        by_act.setdefault(act, []).append(tg)
    bits = [f"{act} {', '.join(tg)}" for act, tg in by_act.items()]
    return "; ".join(bits)
