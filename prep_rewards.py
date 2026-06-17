#!/usr/bin/env python3
"""Teach LANGUAGE -> REWARD across MANY domains and phrasings, so it learns the OPERATION
("turn a goal into a signed list of relevant terms") rather than memorising a handful of
answers. The wider and more varied the domains, the more it has to generalise instead of
looking up the nearest trained example.

These are EXAMPLES of the breadth, not the limit. Output: data/rewards/chat.txt
"""
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))

# (gerund-form goal, reward spec) across as many distinct domains as possible. Each domain
# uses its OWN vocabulary of terms on purpose — shared terms would let it cheat by lookup.
GOALS = [
    # --- locomotion ---
    ("moving forward efficiently", "+forward_velocity +upright -fall -energy_cost"),
    ("walking smoothly", "+forward_velocity +upright -fall -jerk"),
    ("running fast", "+forward_velocity(strong) +upright -fall"),
    ("jumping well", "+jump_height +stable_landing +upright -fall -energy_cost"),
    ("jumping high", "+jump_height(strong) +stable_landing -fall"),
    ("balancing", "+upright -lateral_deviation -fall -energy_cost"),
    ("climbing stairs", "+height_gained +upright -fall -energy_cost"),
    ("swimming forward", "+forward_velocity -drag -energy_cost"),
    ("standing still", "+upright -movement -fall"),
    # --- manipulation ---
    ("grabbing the object", "+grasp_success -drops -collisions"),
    ("lifting the box", "+box_height +grasp_success -drops"),
    ("stacking the blocks", "+blocks_stacked +stability -drops -collisions"),
    ("placing the cup on the shelf", "+placed_correctly -drops -collisions"),
    ("carrying the tray without spilling", "+distance_carried -spills -tilt"),
    ("opening the door", "+door_opened -collisions -force_used"),
    ("pouring water into the glass", "+water_in_glass -spilled -overflow"),
    # --- organizing / sorting ---
    ("organizing the files", "+files_in_correct_folder +tidiness -misplaced -time"),
    ("sorting the items by color", "+items_correctly_sorted -mistakes -time"),
    ("tidying the desk", "+items_put_away +tidiness -clutter_left"),
    ("alphabetizing the list", "+correct_order -out_of_order -time"),
    ("grouping similar objects", "+correct_groups -wrong_groups"),
    # --- cleaning ---
    ("cleaning the room", "+area_cleaned -mess_remaining -time"),
    ("sweeping the floor", "+floor_cleaned -dirt_left -time"),
    ("washing the dishes", "+dishes_cleaned -dishes_left -water_wasted"),
    # --- navigation ---
    ("reaching the target", "+reached_target -distance -time -collisions"),
    ("avoiding the obstacles", "+obstacles_avoided -collisions"),
    ("following the path", "+on_path -deviation -time"),
    ("exploring the room", "+area_explored -repeated_areas -time"),
    ("finding the exit", "+reached_exit -distance -time"),
    # --- games / objectives ---
    ("winning the game", "+score +wins -losses"),
    ("scoring points", "+points_scored -turnovers"),
    ("beating the opponent", "+score_difference +wins -losses"),
    ("surviving as long as possible", "+time_alive -damage_taken"),
    # --- building / making ---
    ("building a tall tower", "+tower_height +stability -blocks_fallen"),
    ("building a stable bridge", "+bridge_stability +span -collapses -materials_used"),
    ("drawing a circle", "+shape_accuracy -deviation"),
    # --- generic quality / efficiency ---
    ("finishing the task quickly", "+task_progress -time -energy_cost"),
    ("doing it accurately", "+accuracy -errors"),
    ("using less energy", "+task_progress -energy_cost"),
    # --- cooking ---
    ("cooking a good dinner", "+taste +cooked_through +presentation -burnt -undercooked"),
    ("baking a cake", "+rise +moisture +taste -burnt -collapse"),
    ("chopping the vegetables", "+pieces_cut +uniformity -injury -waste"),
    ("seasoning the food", "+flavor_balance -too_salty -bland"),
    # --- driving ---
    ("parking the car", "+within_lines +centered -curb_hit -collisions"),
    ("driving safely", "+distance_covered +lane_keeping -collisions -hard_braking"),
    ("merging onto the highway", "+smooth_merge -collisions -abrupt_speed_change"),
    ("backing up the truck", "+aligned -collisions -overshoot"),
    # --- finance ---
    ("saving money", "+amount_saved +savings_rate -overspending -debt"),
    ("investing wisely", "+returns +diversification -risk -fees"),
    ("staying on budget", "+expenses_tracked +within_budget -overspending"),
    ("paying off debt", "+debt_reduced -interest_paid -missed_payments"),
    # --- gardening ---
    ("growing a healthy plant", "+growth +health -wilting -pests"),
    ("watering the plants", "+soil_moisture -overwatering -underwatering"),
    ("planting the seeds", "+germination +spacing -overcrowding"),
    # --- music ---
    ("playing the song well", "+correct_notes +timing +expression -wrong_notes"),
    ("keeping the rhythm", "+on_beat -tempo_drift"),
    ("tuning the guitar", "+pitch_accuracy -out_of_tune"),
    # --- writing ---
    ("writing a clear essay", "+clarity +structure +relevance -typos -filler"),
    ("summarizing the article", "+key_points_covered +brevity -omissions -inaccuracy"),
    ("proofreading the document", "+errors_found -errors_missed -false_corrections"),
    # --- childcare ---
    ("keeping the baby calm", "+calmness +comfort -crying -distress"),
    ("feeding the child", "+nutrition +finished_meal -mess -refusal"),
    ("getting the kid to sleep", "+asleep +sleep_duration -wakeups"),
    # --- health / medical ---
    ("diagnosing the patient", "+correct_diagnosis -misdiagnosis -missed_symptoms"),
    ("administering the dose", "+correct_dose +on_schedule -overdose -missed_dose"),
    ("monitoring the vital signs", "+stable_vitals -alarms_missed -false_alarms"),
    # --- conversation / social ---
    ("holding a friendly conversation", "+engagement +relevance +empathy -rudeness -off_topic"),
    ("comforting a friend", "+reassurance +empathy -dismissiveness"),
    ("persuading the audience", "+agreement +clarity -confusion"),
    # --- forecasting ---
    ("predicting the weather", "+forecast_accuracy -false_alarms -missed_events"),
    # --- energy management ---
    ("managing the power usage", "+tasks_completed -energy_used -peak_load"),
    ("charging the battery", "+charge_level +battery_health -overcharge -overheating"),
    # --- flight ---
    ("flying the drone steadily", "+stable_hover +altitude_hold -drift -crashes"),
    ("landing the plane safely", "+smooth_touchdown +on_runway -hard_landing -overshoot"),
    ("taking off smoothly", "+altitude_gained +stable_climb -stall"),
    # --- typing / study ---
    ("typing fast without typos", "+words_per_minute +accuracy -typos"),
    ("studying for the exam", "+material_covered +retention -distraction -cramming"),
    ("memorizing the list", "+items_recalled -errors"),
    # --- photography ---
    ("taking a sharp photo", "+focus +composition +exposure -blur -noise"),
    # --- sales / service ---
    ("helping the customer", "+issue_resolved +satisfaction -wait_time -escalations"),
    ("closing the sale", "+deals_closed +customer_fit -pushy -churn"),
    # --- teaching ---
    ("teaching the lesson", "+student_understanding +engagement -confusion -boredom"),
    # --- sports (specific) ---
    ("shooting the basketball", "+baskets_made +arc -airballs -travels"),
    ("kicking the ball into the goal", "+goals +accuracy -misses"),
    ("serving the tennis ball", "+aces +in_bounds -faults"),
    # --- repair / assembly ---
    ("fixing the leak", "+leak_stopped -water_damage -time"),
    ("assembling the furniture", "+parts_assembled +stability -missing_steps -damage"),
    # --- sleep / diet ---
    ("sleeping well", "+sleep_duration +deep_sleep -wakeups -restlessness"),
    ("eating healthy", "+nutrient_balance +portion_control -junk_food -overeating"),
    # --- time management ---
    ("managing your time", "+tasks_completed +on_schedule -procrastination -overrun"),
    ("prioritizing the tasks", "+important_done_first -low_value_time"),
    # --- coding ---
    ("writing correct code", "+tests_passing +readability -bugs -complexity"),
    ("debugging the program", "+bugs_fixed -new_bugs -time"),
    ("optimizing the function", "+speedup -memory_used -broken_behavior"),
    # --- art ---
    ("painting the wall evenly", "+coverage +smoothness -drips -missed_spots"),
    ("drawing a portrait", "+likeness +proportion -smudges"),
    # --- negotiation / speaking ---
    ("reaching a fair deal", "+mutual_benefit +agreement -conflict -walkaways"),
    ("giving a good speech", "+clarity +engagement +confidence -filler_words -nervousness"),
]
REQ = [
    "design a reward system for {g}", "design a reward for {g}",
    "make a reward function for {g}", "make a reward for {g}",
    "give me a reward for {g}", "how should i reward {g}",
    "what's a good reward for {g}", "i need a reward system for {g}",
    "set up a reward for {g}", "reward it for {g}",
]
# direct imperative style too ("make it jump high")
DIRECT = {
    "move forward efficiently": "+forward_velocity +upright -fall -energy_cost",
    "walk smoothly": "+forward_velocity +upright -fall -jerk",
    "jump high": "+jump_height(strong) +stable_landing -fall",
    "run fast": "+forward_velocity(strong) +upright -fall",
    "balance": "+upright -lateral_deviation -fall -energy_cost",
    "stack the blocks": "+blocks_stacked +stability -drops",
    "reach the target": "+reached_target -distance -time -collisions",
    "clean the room": "+area_cleaned -mess_remaining -time",
    "park the car": "+within_lines +centered -curb_hit -collisions",
    "save money": "+amount_saved +savings_rate -overspending -debt",
    "cook a good dinner": "+taste +cooked_through -burnt -undercooked",
    "land the plane safely": "+smooth_touchdown +on_runway -hard_landing -overshoot",
    "write clean code": "+tests_passing +readability -bugs -complexity",
    "keep the baby calm": "+calmness +comfort -crying -distress",
}
VAGUE = [
    ("design a reward system", "for what task? tell me the goal and i'll build the reward."),
    ("make a reward", "a reward for what? give me the goal."),
    ("make it better", "better at what - the goal decides the reward."),
]


def main(n=200, seed=2):
    r = random.Random(seed)
    out = []
    for _ in range(n):
        for g, spec in GOALS:
            out.append(f"USER: {r.choice(REQ).format(g=g)}\nBOT: reward: {spec}\n")
        for g, spec in DIRECT.items():
            verb = r.choice(["make it", "teach it to", "i want it to", "get it to"])
            out.append(f"USER: {verb} {g}\nBOT: reward: {spec}\n")
        for u, b in VAGUE:
            out.append(f"USER: {u}\nBOT: {b}\n")
    r.shuffle(out)
    out_dir = os.path.join(HERE, "data", "rewards")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "chat.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print(f"[rewards] {len(out):,} reward turns ({len(GOALS)} domains x {len(REQ)} phrasings) "
          f"-> data/rewards/chat.txt")


if __name__ == "__main__":
    main()
