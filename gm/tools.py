"""Hard-coded TOOLS the model calls instead of hallucinating answers.

The model is trained (see prep_tooluse.py) to emit a CALL line; this module RUNS it and
returns a RESULT string, which the model then verbalises. The logic lives here in plain,
inspectable code — the neural net only has to (a) decide which tool to call and (b) turn the
result into natural language. That's the whole point: reliability from code, fluency from the net.

  CALL: store <s> <rel> <o>     -> remember a fact            -> "ok"
  CALL: find <rel> <obj>        -> what has/can <obj>         -> "a, b" | "none"
  CALL: ask <s> <rel> <o>       -> is it true?                -> "yes" | "no"
  CALL: profile <s>             -> what is <s>                -> "isa a,b" | "nothing"
  CALL: reward <goal...>        -> build a reward spec        -> "+term -term ..."
  CALL: calc <expression>       -> evaluate arithmetic        -> the number | "error"

The reward builder is COMPOSITIONAL: it matches a domain template, then layers on modifiers
("safely", "fast", "without spilling"). Unlike the net's nearest-neighbour guess it is exact,
extensible (add a line), and degrades gracefully on an unknown goal instead of guessing wrong.
The calculator is exact arithmetic (a tiny LM cannot do reliable math) via a SAFE evaluator.
"""
import ast
import math
import operator
import re

# ---- calculator: exact arithmetic via a safe AST evaluator (no eval(), no names/attrs) ----
_BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
           ast.FloorDiv: operator.floordiv}
_UNOPS = {ast.USub: operator.neg, ast.UAdd: operator.pos}
_FUNCS = {"sqrt": math.sqrt, "abs": abs, "round": round, "log": math.log,
          "log10": math.log10, "ln": math.log, "exp": math.exp, "floor": math.floor,
          "ceil": math.ceil, "factorial": math.factorial,
          "sin": math.sin, "cos": math.cos, "tan": math.tan}
_CONSTS = {"pi": math.pi, "e": math.e}


def _eval_node(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNOPS:
        return _UNOPS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
        return _FUNCS[node.func.id](*[_eval_node(a) for a in node.args])
    if isinstance(node, ast.Name) and node.id in _CONSTS:
        return _CONSTS[node.id]
    raise ValueError("unsupported")


def _normalize_math(expr):
    """Turn natural-language arithmetic into a Python expression string."""
    e = " " + expr.lower().strip().rstrip("?=. ") + " "   # keep '!' (factorial)
    e = re.sub(r"\b(what'?s|what is|whats|calculate|compute|how much is|how many is|solve|"
               r"the answer to|tell me|please|equals?|equal to)\b", " ", e)
    e = re.sub(r"(\d),(\d)", r"\1\2", e)                       # 1,000 -> 1000
    e = re.sub(r"\bsquare root of\s*", " sqrt ", e)
    e = re.sub(r"\bsqrt\s+(\d)", r" sqrt(\1", e)              # sqrt 16 -> sqrt(16  (close below)
    e = re.sub(r"\b(plus|and added to|added to)\b", "+", e)
    e = re.sub(r"\b(minus|subtract|less)\b", "-", e)
    e = re.sub(r"\b(times|multiplied by)\b", "*", e)
    e = re.sub(r"\b(divided by|over)\b", "/", e)
    e = re.sub(r"\bto the power of\b", "**", e)
    e = re.sub(r"\bmod(ulo)?\b", "%", e)
    e = re.sub(r"\bsquared\b", "**2", e)
    e = re.sub(r"\bcubed\b", "**3", e)
    e = e.replace("^", "**")
    e = re.sub(r"(\d+(?:\.\d+)?)\s*%\s*of\s*", r"(\1/100)*", e)   # 20% of 80
    e = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"(\1/100)", e)            # bare 30%
    e = re.sub(r"(\d)\s*x\s*(\d)", r"\1*\2", e)                  # 3 x 4 -> 3*4
    e = re.sub(r"(\d+)\s*!", r"factorial(\1)", e)               # 5! -> factorial(5)
    if e.count("(") > e.count(")"):                             # close sqrt( etc.
        e += ")" * (e.count("(") - e.count(")"))
    return e.strip()


def calc(expr):
    try:
        val = _eval_node(ast.parse(_normalize_math(expr), mode="eval").body)
        if isinstance(val, float):
            val = int(val) if val.is_integer() else round(val, 6)
        return str(val)
    except Exception:
        return "error"

# domain keyword (matched as a whole word) -> core reward terms. First match wins; order
# matters only where one keyword is a substring concept of another, so list specifics first.
REWARD_LIB = [
    (("jump", "leap", "hop"), "+jump_height +stable_landing +upright -fall"),
    (("run", "sprint", "dash"), "+forward_velocity +upright -fall"),
    (("walk", "step", "stride"), "+forward_velocity +upright -fall -jerk"),
    (("crawl",), "+forward_velocity +stability -fall"),
    (("balance", "steady"), "+upright -lateral_deviation -fall"),
    (("climb", "ascend"), "+height_gained +upright -fall"),
    (("swim", "paddle"), "+forward_velocity -drag"),
    (("stand", "stay still", "hold still"), "+upright -movement -fall"),
    (("move forward", "go forward", "move", "locomot"), "+forward_velocity +upright -fall"),
    (("grab", "grasp", "pick up", "grip"), "+grasp_success -drops -collisions"),
    (("lift", "raise"), "+object_height +grasp_success -drops"),
    (("stack",), "+items_stacked +stability -drops"),
    (("place", "put", "set down"), "+placed_correctly -drops -collisions"),
    (("carry", "transport"), "+distance_carried -spills -tilt"),
    (("pour", "fill"), "+amount_transferred -spilled -overflow"),
    (("open",), "+opened -force_used -collisions"),
    (("organize", "sort", "tidy", "arrange", "file"), "+correctly_placed +tidiness -misplaced"),
    (("alphabetize", "order"), "+correct_order -out_of_order"),
    (("clean", "wash", "scrub", "wipe"), "+area_cleaned -mess_remaining"),
    (("sweep", "vacuum"), "+floor_cleaned -dirt_left"),
    (("reach", "go to", "target", "navigate"), "+reached_target -distance -collisions"),
    (("avoid", "dodge", "evade"), "+obstacles_avoided -collisions"),
    (("follow",), "+on_path -deviation"),
    (("explore", "search"), "+area_explored -repeated_areas"),
    (("find the exit", "escape"), "+reached_exit -distance"),
    (("win", "beat", "defeat"), "+score +wins -losses"),
    (("score", "points"), "+points_scored -turnovers"),
    (("survive",), "+time_alive -damage_taken"),
    (("build", "construct", "assemble"), "+structure_complete +stability -collapses"),
    (("tower",), "+tower_height +stability -blocks_fallen"),
    (("bridge",), "+bridge_stability +span -collapses"),
    (("draw", "sketch"), "+shape_accuracy -deviation"),
    (("paint",), "+coverage +smoothness -drips"),
    (("cook", "fry", "boil", "grill"), "+taste +cooked_through -burnt -undercooked"),
    (("bake",), "+rise +taste -burnt -collapse"),
    (("chop", "slice", "cut"), "+pieces_cut +uniformity -injury"),
    (("season", "flavor"), "+flavor_balance -too_salty -bland"),
    (("park",), "+within_lines +centered -curb_hit -collisions"),
    (("drive", "merge", "steer"), "+distance_covered +lane_keeping -collisions -hard_braking"),
    (("save money", "save",), "+amount_saved +savings_rate -overspending"),
    (("invest",), "+returns +diversification -risk"),
    (("budget", "spend"), "+within_budget -overspending"),
    (("pay off", "debt"), "+debt_reduced -interest_paid"),
    (("grow", "plant", "garden"), "+growth +health -wilting"),
    (("water",), "+soil_moisture -overwatering -underwatering"),
    (("play", "perform", "song"), "+correct_notes +timing -wrong_notes"),
    (("rhythm", "beat", "tempo"), "+on_beat -tempo_drift"),
    (("tune",), "+pitch_accuracy -out_of_tune"),
    (("write", "essay", "report"), "+clarity +structure -typos -filler"),
    (("summarize", "summarise"), "+key_points_covered +brevity -omissions"),
    (("proofread", "edit"), "+errors_found -errors_missed"),
    (("code", "program", "implement"), "+tests_passing +readability -bugs -complexity"),
    (("debug",), "+bugs_fixed -new_bugs"),
    (("optimize", "optimise", "speed up"), "+speedup -memory_used -broken_behavior"),
    (("calm", "soothe", "comfort"), "+calmness +comfort -crying -distress"),
    (("feed",), "+nutrition +finished_meal -mess"),
    (("diagnose",), "+correct_diagnosis -misdiagnosis -missed_symptoms"),
    (("dose", "medicate"), "+correct_dose +on_schedule -overdose"),
    (("converse", "chat", "talk"), "+engagement +relevance +empathy -off_topic"),
    (("persuade", "convince"), "+agreement +clarity -confusion"),
    (("predict", "forecast"), "+forecast_accuracy -false_alarms -missed_events"),
    (("charge",), "+charge_level +battery_health -overcharge -overheating"),
    (("fly", "hover", "drone"), "+stable_hover +altitude_hold -drift -crashes"),
    (("land", "touch down"), "+smooth_touchdown +on_runway -hard_landing"),
    (("take off", "takeoff"), "+altitude_gained +stable_climb -stall"),
    (("type",), "+words_per_minute +accuracy -typos"),
    (("study", "learn", "revise"), "+material_covered +retention -distraction"),
    (("memorize", "memorise", "recall"), "+items_recalled -errors"),
    (("photograph", "photo", "shoot a"), "+focus +composition +exposure -blur"),
    (("help the customer", "customer", "support", "assist"),
     "+issue_resolved +satisfaction -wait_time"),
    (("sell", "close the sale"), "+deals_closed +customer_fit -churn"),
    (("teach", "tutor", "explain"), "+student_understanding +engagement -confusion"),
    (("shoot", "throw"), "+accuracy +made -misses"),
    (("kick", "pass"), "+accuracy +goals -misses"),
    (("serve", "ace the"), "+aces +in_bounds -faults"),
    (("fix", "repair", "patch"), "+fixed -damage -time"),
    (("sleep", "rest"), "+sleep_duration +deep_sleep -wakeups"),
    (("eat", "diet"), "+nutrient_balance +portion_control -overeating"),
    (("manage time", "schedule", "prioritize", "prioritise"),
     "+tasks_completed +on_schedule -procrastination"),
    (("negotiate", "deal"), "+mutual_benefit +agreement -conflict"),
    (("speak", "speech", "present"), "+clarity +engagement +confidence -filler_words"),
    (("juggle",), "+catches +rhythm -drops"),
    (("dance",), "+on_beat +form -stumbles"),
]
# modifier keyword -> extra terms layered onto whatever core matched
MODIFIERS = [
    (("safe", "safely", "without crashing", "without falling"), "-collisions -damage"),
    (("fast", "quick", "rapid"), "-time"),
    (("slow", "careful", "gentle"), "-jerk -abruptness"),
    (("smooth",), "-jerk"),
    (("accurate", "precise", "exact"), "+accuracy -errors"),
    (("efficient", "less energy", "low power"), "-energy_cost"),
    (("stable", "stabl", "without tipping", "without spilling"), "+stability"),
    (("neat", "tidy", "tidily"), "+tidiness"),
]
GENERIC = "+task_progress -time -errors"
INTENSIFY = ("high", "tall", "fast", "strong", "far", "max", "maximum",
             "as much as possible", "as far as possible", "as high as possible")


def _match_word(g, k):
    """Is keyword k present in goal g, tolerant of word-forms? Multiword keys: substring.
    Single keys: a goal word starts with the keyword's stem (so land~landing, save~saving).
    Returns the char position of the match, or None."""
    if " " in k:
        i = g.find(k)
        return i if i != -1 else None
    stem = k[:-1] if k.endswith("e") and len(k) > 3 else k          # save->sav, land->land
    if len(stem) < 3:
        stem = k
    for m in re.finditer(r"[a-z]+", g):
        if m.group(0).startswith(stem):
            return m.start()
    return None


def _pos(g, keys):
    hits = [p for p in (_match_word(g, k) for k in keys) if p is not None]
    return min(hits) if hits else None


def build_reward(goal):
    """Turn a free-text goal into a signed reward spec by matching a domain template and
    layering modifiers. Returns the term string (no 'reward:' prefix)."""
    g = " " + goal.lower().strip().rstrip(".!?") + " "
    # choose the template whose keyword appears EARLIEST in the goal (its leading concept)
    cands = [(p, i, terms) for i, (keys, terms) in enumerate(REWARD_LIB)
             if (p := _pos(g, keys)) is not None]
    if not cands:
        return GENERIC + "  (generic - no specific template for this goal)"
    cands.sort()
    terms = cands[0][2].split()
    # intensify the leading positive term when the goal asks for "high/fast/max/..."
    if any(_match_word(g, w) is not None for w in INTENSIFY) and terms:
        if terms[0].startswith("+") and "(" not in terms[0]:
            terms[0] = terms[0] + "(strong)"
    for keys, extra in MODIFIERS:
        if _pos(g, keys) is not None:
            terms += extra.split()
    # "without X" / "no X" -> penalise X
    for m in re.finditer(r"\b(?:without|no|avoid(?:ing)?)\s+([a-z]+)", g):
        terms.append("-" + m.group(1))
    seen, out = set(), []                       # dedupe, keep order, drop +x if -x present
    for t in terms:
        base = t.lstrip("+-").split("(")[0]
        key = t.lstrip("+-")
        if key not in seen:
            seen.add(key)
            out.append(t)
    negs = {t[1:].split("(")[0] for t in out if t.startswith("-")}
    out = [t for t in out if not (t.startswith("+") and t[1:].split("(")[0] in negs)]
    return " ".join(out)


class Tools:
    """Executes a CALL line against the knowledge engine + reward builder; returns RESULT."""

    def __init__(self, know):
        self.know = know

    def run(self, call):
        call = call.strip()
        parts = call.split()
        if not parts:
            return "error"
        op, args = parts[0].lower(), parts[1:]
        try:
            if op == "store" and len(args) >= 3:
                self.know._add(args[0], args[1], " ".join(args[2:]))
                return "ok"
            if op == "find" and len(args) >= 2:
                subs = self.know.subjects_with(args[0], " ".join(args[1:]))
                return ", ".join(subs) if subs else "none"
            if op == "ask" and len(args) >= 3:
                s, rel, o = args[0], args[1], " ".join(args[2:])
                if rel == "isa":
                    return "yes" if o in self.know.ancestors(s) else "no"
                return "yes" if o in self.know.rel_objs(s, rel) else "no"
            if op == "profile" and args:
                parents = self.know.rel_objs(" ".join(args), "isa")
                return "isa " + ",".join(parents) if parents else "nothing"
            if op == "reward" and args:
                return build_reward(" ".join(args))
            if op == "calc" and args:
                return calc(" ".join(args))
        except Exception:
            return "error"
        return "error"
