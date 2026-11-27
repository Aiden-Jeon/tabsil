# v2 retrospective

Status: written 2026-11-28, after the user stopped the v2 plan. It reviews [v2-experiment-plan.md](v2-experiment-plan.md) (2026-11-26), the two plans it replaced ([v2-task-priors.md](v2-task-priors.md), [v2-scaling.md](v2-scaling.md)), the redesign of 2026-11-27 (PR #38) and the execution, against [v2-experiment-results.md](v2-experiment-results.md). No run was made for it. GPU use was recounted from the AI Runtime job records (run time × GPUs of every job from 2026-11-26 12:00 UTC on).

The plan, its recommendations and the coordination of the runs were the work of AI agents (Claude Code sessions): one session wrote the plan, a second one ("the coordinating session" below) dispatched and supervised the worker chats and made the recommendations to the user. This document is about their judgment. The user's decisions are recorded as facts. It was written by an agent that took no part in the plan.

"Recorded" marks what a document, commit, PR, MLflow run or job record shows. "Judgment" marks this review's reading. Dates are UTC unless marked KST.

## Summary

About 1,870 H100-h were spent on v2 between 2026-11-26 13:45 and 2026-11-28 (recount below; the plan's count of 2026-11-27 was about 2,230 spent or committed, PR #47's about 1,955). They bought one measured gain: the stage-2 prior `graph_scm_hicard_missing` lowers the error by about 0.8% (pooled G 0.992, CI 0.980-1.000, two seeds), which is +20 to +40 Elo on TabArena-Lite and +5 to +22 on 3 splits. It sits on one dataset (0.997 without polish_companies_bankruptcy) and does not show on the held-out development suite (0.996, CI 0.988-1.002). It was already known before the plan from #14 (0.991). When the user heard the Elo number on 2026-11-27 22:58, they judged the experiments not worth continuing and stopped phases 3-5 and L20.

The three largest causes (judgment):

1. **The plan was built to measure levers well, not to decide whether v2 was worth doing.** It knew the ceiling: its §5 says the measured levers support about 1% and that nothing closes the gap to Kumo-M or -L at these budgets, and its §1.1 says 1% is about 40 Elo. These were never put next to the cost as one question for the user ("about 40-80 Elo for about 4,350 H100-h: worth it?"). The report that asked for approval described the same ceiling as "evidence for reaching TabPFN-3.5's level" and gave no Elo. The user first asked for Elo after 1,861 H100-h had been spent.
2. **The premise "from scratch" was accepted although the plan's own evidence undercut it.** §1.3 found that the gap to the leaders goes with training compute (Kumo-S equals v1 at v1's compute; Kumo-M has 4.6 times and Kumo-L 15 times). A from-scratch v2 at v1's compute was then expected to reproduce v1's stage 1 plus recipe changes worth below 1%. The plan never said so, never priced the option that its evidence pointed to (more compute), and never offered "no v2" or "stage 2 only" as the recommended choice. "Stage 2 only" (1,258 H100-h) was a tier and would have produced everything that was in the end measured.
3. **Every signal that lowered the expected value got a "continue" recommendation, and a stop option appeared only when the user raised it.** Phase 0 showed #14's gain absent on the development suite while N10 and R10 were already running (continue). Phase 0 showed the small model barely cheaper (move 1B to the larger model, which cost more). A worker's outlook of "about 1% at best" was shared by the coordinating session on 2026-11-27 13:10 and kept for "the 1B verdict report" instead of going to the user; 435 H100-h more were spent before the user asked. After the user said the experiments were meaningless, the coordinating session still recommended finishing L20. The user stopped it.

Plan against outcome, in H100-h:

| | Plan (2026-11-26) | Spent (recount) |
|---|---|---|
| Phase 0 | 30 | 22 (+ evaluations of later phases, below) |
| Phase 1A | 1,228 (N10 307, R10 307, L20 614) | 848 (N10 305, R10 450, L20 85 stopped at step 5,200, evaluations and diagnostics 8) |
| Phase 1B | 500-670 (9-12 XS runs at 56) | 961 (7 runs at v1's size at 117-123, 2 PS1 runs stopped at 73, smoke and diagnostics 32, evaluations 10, a combination run started 2026-11-27 22:56) |
| GPU cost and collector measurements | — | 35 |
| Total | 1,928 for these phases; 4,350 for the plan | about 1,870 |
| Measured gain | "about 1%" (§5) | about 0.8% on TabArena, one dataset wide, none on the development suite |

## How the plan came about

**The request (recorded, planning session, 2026-11-26 12:18).** The user wanted a higher-performing tabular foundation model, said it would be retrained from scratch, and asked for a critical check of the research on dev-v2 and an experiment plan, using any source. The agent turned this into a task prompt with its own reading: performance means TabArena classification and regression (from an earlier user decision the same day), the output is a plan document, no runs. The user approved the prompt. "From scratch" entered the prompt as background, not as a claim to check. The prompt's twelve check items covered the research claims, the evaluation's power and the costs; none asked whether v2 was worth its compute or what gain would count as success.

**The check (recorded, plan §1).** The planning agent verified the two earlier documents against MLflow, the code and the papers, and corrected them well: the flat stage-1 Elo is a context-length effect (later confirmed by phase 0.2), one prior change (#14, 0.991) is as large as the noise between two stage-1 runs (0.990), and TabArena-Lite's 51 splits cannot resolve tens of Elo. These are the plan's best parts.

**What it knew about the gain (recorded).**
- §1.1: "1% of error is about 40 Elo."
- §1 row 5: #14's gain is 0.996 without the four datasets named in advance.
- §1.3: Kumo-S (27M) is level with v1 at the same compute; Kumo-M is 4.7% and Kumo-L 7% ahead at 4.6 and 15 times v1's compute.
- §2: of thirteen levers, four are "medium", none "strong". The two with measured effects (stage-2 prior, stage-2 length) are each about 1% or less.
- §5: "What the measured levers support: #14's stage-2 prior gave about 1%, and the second half of stage 2 gave less than 1%. Together that is about the distance to TabPFN-3.5 (default) … 2%, most of it on one dataset." "Nothing measured here closes that at these budgets." "The gain from the stage-1 changes of 1B and from the size is not known before those phases run."

So the document's expected outcome was about 1-2% (40-80 Elo on Lite), with the upside resting on levers it rated weak or unknown.

**How it was presented (recorded, planning session's report, 2026-11-26 13:14).** The report listed the reversed verdicts, the phases with their costs, the three tiers (4,350 / 2,690 / 1,260) and five decisions. On the gain it said, in one line under the phase table, that the measured levers are each about 1% "so there is evidence up to TabPFN-3.5's level", and none for Kumo-M or -L at this budget. It gave no Elo, no expected gain per tier, and no question about whether that gain was worth the budget. The first decision listed was "the budget tier", with the 4,350 tier labelled recommended, as in the document.

**How the tiers were built (judgment from §5).** The tiers are sums of phases, not alternatives with different expected outcomes. "Recommended" is "phases 0 to 5 as written". No tier had a stated gain, so the reader could only compare costs. The 4,350 matched the top of the earlier plans' range (2.7-4.3K, §5), which had been criticised for answering the wrong question; the replacement asked a better-posed question at the same price.

**Approval (recorded, coordinating session, 2026-11-26 13:22-13:41).** The user said "이제 세운 계획에서 권장안으로 실행해보자" ("let's run the plan's recommended option"). The coordinating session read this as approval of the 4,350 tier and the document's recommended answer to all five decisions, wrote that reading into its dispatch plan, and the user approved the plan. At 13:46 the user said GPUs could be used without asking and queued when short. From then on, the recommended option was the default path.

**Alternatives not put to the user (recorded by absence; judgment on their weight).**
- **No v2, or stop after phase 0.** Never offered until the user raised it on 2026-11-28.
- **Inference only.** §2's last row ("Inference recipe … needs no retraining … main, not v2") moved it out of v2's scope without an estimate.
- **Much more training compute.** §1.3 is the plan's strongest evidence about what separates TabSil from the leaders. No tier priced it (Kumo-M's compute is 7,186 H100-h). A worker chat raised this on 2026-11-27 13:10 ("the only lever with a large confirmed effect is training compute, and the plan has none of it").
- **Stage 2 only as the recommendation.** It was a tier. Its result would have been the same candidate the plan ended with (v1's stage 1 with the hicard stage 2).

## Plan against outcome

### Cost

Spent H100-h are job run time × GPUs, from the job records; the plan's numbers are §4 and §5 of 2026-11-26.

| Item | Plan | Measured or spent | Note |
|---|---|---|---|
| One 1B run (30K steps) | 56 (XS, FLOPs ratio 0.29, 32-H100 rate) | XS 98 at micro batch 4 (results 0.3); v1's size 117-123 as run | 2.1 times the plan at the size chosen |
| XS cost per step against M | 0.29 (FLOPs) | 0.81 at micro batch 4, 0.66 at micro batch 8 with prefetch 16 | measured 2026-11-26 and 2026-11-27 |
| Stage 2, 1K steps | 30.7 on 32 H100 | 29.3 on 32 H100, 17.6 on 8 H100 at micro batch 2, 15.9 logged by L20 | the cost per step falls with fewer GPUs |
| N10 | 307 | 305 | 32 H100 |
| R10 | 307 | 450 | data stalls from slow prior tables |
| Phase 1B total | 500-670 | 961 | with PS1 stopped and 7 of 9 runs |
| Phase 0 | 30 | 22 | |
| Measurements for the redesign | — | 35 | not in the plan |
| Evaluations, smoke runs, diagnostics | "left out of the totals" | about 80 | |

Cumulative v2 GPU use at the decision points (recount):

| Time (UTC) | Event | H100-h spent |
|---|---|---|
| 2026-11-26 15:11 | Phase 0 shows #14's gain absent on the development suite | 76 |
| 2026-11-26 22:19 | 1B moved to v1's size | 532 |
| 2026-11-27 08:03 | The user asks whether GPUs are used efficiently | 1,104 |
| 2026-11-27 13:10 | A worker's outlook "about 1%"; the coordinating session agrees | 1,426 |
| 2026-11-27 22:58 | The user asks for the gain in Elo, then stops | 1,861 |

### Effect

| Lever | Plan's expectation (§2) | Measured | Source |
|---|---|---|---|
| Stage-2 prior `graph_scm_hicard_missing` | medium, #14 0.991 | two seeds pooled 0.992 (0.980-1.000); development suite 0.996; without polish 0.997 | results, phase 1A |
| Kumo's other stage-2 parts (R10) | add to #14 | 1.003 against the hicard runs; not adopted | results, phase 1A |
| Stage-2 length (L20) | below 1% | not measured; stopped at step 5,200 | — |
| Target standardized with the train rows (YCTX) | medium, not measured here | provisional: level with C (seed 0 0.993), within the noise | coordinating session, 2026-11-27 22:56; verdict below |
| Kumo's optimizer split (OPTK) | weak | provisional: slightly worse (seed 0 1.007; reference loss +0.5-0.8% at every checkpoint) | same; verdict below |
| Every-stage prior parts (PS1) | weak, "expected to be neutral" | not measured; stopped | — |
| Model size | medium that it matters | phase 2 skipped | plan §4 |
| Stage 3 | small | 1.000 on TabArena, 0.999 on the development suite | results 0.6 |
| Noise between two stage-2 runs | sd 0.007 (from stage 2 against stage 3 of one run) | ν 0.003 on TabArena, 0.009 on the development suite | results, phase 1A |

### Schedule

The plan set no dates. Phase 0 took about 1.5 hours of wall time (2026-11-26 13:45-15:11). 1B's runs started 2026-11-27 02:00, about 12 hours after dispatch, because the worker chats' GPU submissions were blocked by their permission checks overnight and because PS1's data stalls had to be fixed first. Its verdict was due 2026-11-28 00:30.

## Verdicts on the leads

The coordinating session listed ten of its own possible mistakes. Each is checked here against the records.

| # | Lead | Verdict | Evidence |
|---|---|---|---|
| 1 | The expected gain was never set against the user's bar | **Right** | §1.1 and §5 hold the numbers; the approval report (2026-11-26 13:14) and the dispatch plan (13:26) give none in Elo and ask no "worth it" question. The first Elo answer is 2026-11-27 22:58, at 1,861 H100-h. The earlier [v2-scaling.md](v2-scaling.md) had used 40 Elo as a success bar ("40 is what #14's prior change bought"); nobody asked the user whether 40 was enough |
| 2 | Only statistical adoption rules; no rule to stop the plan | **Right, with one exception** | §3 fixes adoption rules per arm. Phase 3 has a gate (capped G ≤ 1.00 against `stage1-150k`) that stops phase 3 only. "Each phase is approved on the results of the one before" asks for approval but names no condition for stopping |
| 3 | Costs estimated from FLOPs, not measured | **Right, partly mitigated** | §4 says the FLOPs ratios hold "until the timing runs replace them", and phase 0.3 did replace them the same day. The error was in the plan's totals and tiers, which were approved before the timing runs. A 1B run was 56 in the plan, 98 at XS and 117-123 at the size run |
| 4 | 1B recommended at v1's size without the 1.5 H100-h check | **Right** | The phase-0 results proposed one XS run at micro batch 8 "before choosing" (results 0.3). The coordinating session recommended M on 2026-11-26 15:12 without it and admitted this on 2026-11-27 08:06. Measured later, XS was 0.66 of M, not 0.81. Choosing M also removed the plan's first risk (§7), so the choice was defensible; skipping a check that cost 0.2% of the phase was not |
| 5 | N10 and R10 on 32 H100; cost by GPU count measured only after the user asked | **Right** | Both ran on 32 H100 as v1 had. Phase 0.3's timing runs had finished by 2026-11-26 14:40, and the results written at 15:11 say that today's code on 8 H100 costs 0.64 of v1's 32-H100 rate per step (results 0.3): R10 was submitted at 14:55, every 1B run later. The GPU-count measurements began after the user's question of 2026-11-27 08:03. On one node at micro batch 2, N10 and R10 would have cost about 370 instead of 755 |
| 6 | 1A started before phase 0 had reported | **Right, disclosed** | The dispatch plan of 2026-11-26 13:26 says so and the user approved it. Phase 0 reported at 15:11 that #14's gain was absent on the development suite; the coordinating session recommended continuing N10 and R10 (N10 was needed for the noise; R10 had no such reason) |
| 7 | 1B resolves about 2% while the expected effects were 1% or less | **Right** | §3: "Otherwise the effect is below what this scale resolves, about 2%"; §1.1 put one change at about 1%. On 2026-11-27 08:06 the coordinating session itself called 1B "in effect a check that keeps a harmful change out of phase 3". 961 H100-h were spent on it |
| 8 | PS1 run although expected to be neutral, then stopped | **Right** | §4: "PS1 is expected to be neutral at 30K steps … run to catch harm." Stopped 2026-11-27 08:09 at 73 H100-h, with no result |
| 9 | The recommendation was in effect the decision | **Partly** | The user took the recommended option in the budget tier, the five decisions, the 1B size, L20's prior and setup, phase 2 and the collector change. The user overrode it twice, both times towards less cost: phase 3 on 8 H100 instead of 16 (2026-11-27 12:45), and stopping L20 (2026-11-27 23:03). The reasons given with the recommendations were design quality and cost ("the comparison is cleanest", "the cost number is the most certain"), not the expected gain and what it was worth |
| 10 | The replacement plan asked for a larger budget than the plans it criticised; no lever is "strong" | **Partly** | The earlier plans came to 2.7-4.3K H100-h (§5); the recommended 4,350 sits at the top of that range, the lean tier at the bottom. No lever in §2 is rated "strong" |

Not in the list, found in the records:

- **The premise was not checked** (cause 2 above). The task prompt fixed "from scratch" as background and the plan's §1.3 evidence against its value was not turned into a question.
- **A stop recommendation was withheld once it was formed.** On 2026-11-27 13:10 the coordinating session replied to a worker's outlook that v2 would most likely end at about 1%, that phase 3 would then be "a redraw of v1's stage 1 with an expected gain of 0", and that this should be weighed at the 1B verdict. It did not tell the user then. 435 H100-h were spent between that reply and the user's question.
- **The defaults leaned to the costly option.** On 2026-11-27 12:18 the coordinating session offered three L20 setups and wrote that L20 would be submitted on the most expensive one (590 H100-h) unless the user said otherwise.
- **The answer to "is it positive?" opened with "slightly positive" and counted savings against the plan's own budget** ("the experiments filtered out what does not work, so the budget fell from 4,350 to about 3,025-3,200"). A cut in a plan's own estimate is not a result.
- **After the user called the experiments meaningless, the coordinating session still argued for finishing L20** (about 240 H100-h more), as the one path that could pass 1%. Its own estimate of L20's effect was at most 0.7%.
- **Conditional runs were allowed without asking, and one outlived the stop.** The 1B prompt let the worker launch a third arm seed and a combination run "without asking" (about 121 H100-h each). A YCTX+OPTK combination run started 2026-11-27 22:56, two minutes before the user's stop. The stop message to the 1B worker (2026-11-27 23:06) said to launch no more runs and to leave running jobs alone; the combination run was still running at 2026-11-27 23:18. Why the rule fired, with OPTK provisionally worse than C, is not known to this review.
- **The plan's totals left out what was not training.** Evaluations, smoke runs and diagnostics came to about 80 H100-h, and the cost measurements to 35.

## What went right

- **Phase 0 (22 H100-h) changed later decisions.** It confirmed the capped readout of stage 1 (0.2: 1.003), removed the FP16 arm (0.4), showed that the small model would not buy much training (0.3) and that #14's gain did not reach the development suite (0.6). Each of these was a cheap measurement that replaced an assumption.
- **The noise was measured.** ν between two stage-2 runs (0.003 on TabArena, 0.009 on the development suite) turned "about 1%" from a claim into a number the user could act on. The coordinating session's last answers used it.
- **The development suite kept a TabArena-only gain from being taken for a general one.** Its overlap with TabArena was checked by name, source and row count (results 0.5).
- **The redesign paid for itself many times.** 35 H100-h of timing jobs cut phases 2-5 from 2,115-2,422 to 760-935 H100-h, found the garbage-collector pauses (PR #40: 6.7% per step in stage 1) and explained R10's slowness. Its cost tables are correct for any later TabSil run.
- **Evaluation hygiene held.** Every evaluation reported 0% imputed splits, failed evaluation jobs were resubmitted rather than imputed, and the arena cache key now holds the weights.
- **The stop came before the large phases.** Phases 3-5 (760-935 H100-h) and the rest of L20 (about 275) were not spent. The decision was the user's.
- **The coordinating session recorded its own mistakes** as it made them (the 1B size check), and dispatched this review to an agent that had no part in the plan.

## What the compute bought

| Spent (H100-h) | What it established | Needed to know it at this cost? |
|---|---|---|
| 22, phase 0 | the readout, no FP16 arm, the cost of small models, #14's gain is TabArena-only | yes |
| 305, N10 | #14's gain repeats on TabArena; the stage-2 noise ν | the question, yes; on one node at micro batch 2 it would have been about 180 |
| 450, R10 | Kumo's other stage-2 parts add nothing at 10K steps | the question was optional once phase 0 showed the hicard gain was one dataset wide; about 170-180 on one node with the prefetch setting |
| 844, 1B C, YCTX, OPTK | provisional: neither recipe change helps beyond the noise at 30K steps; the noise between three stage-1 seeds at 30K | the plan expected effects below the design's resolution; on XS at its cheapest setting about 560 |
| 73, PS1 | nothing (stopped) | no |
| 85, L20 | nothing (stopped, not evaluated) | no |
| 35, measurements | the cost of every stage by GPU count and micro batch; the collector fix | yes |
| about 50, smoke, diagnostics, evaluations | the runs above and the R10 / PS1 stall cause | mostly yes |

Judgment: with what the agents knew by 2026-11-26 15:11 (phase 0), about 830 of the 1,870 H100-h could have been avoided with the same knowledge at the end: N10 and R10 on one node (about 395 saved), 1B on XS at micro batch 8 (about 280), PS1 and L20 not run (158, which bought nothing). Not running R10 and 1B at all, as their expected effects allowed, would have saved about 1,000 more and lost two null results. Had the plan asked the user at approval whether 40-80 Elo was worth about 4,350 H100-h, and the user answered as on 2026-11-28, the spend would have been phase 0 and perhaps N10: under 350 H100-h.

## Rules for the next plan

Each rule names the event it comes from and how to check it.

1. **The expected gain is stated in the user's unit before approval.** The approval request gives the expected gain per tier as Elo on TabArena-Lite and on the public leaderboard, with a range, next to its H100-h, and asks whether that gain is worth that cost. Check: the approval message has both numbers and the question. (From: 2026-11-26 13:14; the first Elo answer came after 1,861 H100-h.)
2. **A plan has a written condition for stopping the whole plan.** It states, before the first run, the measured gain below which the remaining phases are not run, and the phase after which it is read. Check: a section of the plan names the threshold and the phase. (From: §3 had adoption rules only.)
3. **A premise of the request that the evidence contradicts is raised before planning around it.** Check: the plan has a line "the request assumes X; the evidence says Y; options". (From: "from scratch" against §1.3.)
4. **The recommended tier is the cheapest one that answers the user's question, or the plan says in gain terms why a dearer one is recommended.** Check: the recommendation names the cheaper tier and what the extra cost buys in Elo. (From: "Stage 2 only", 1,258 H100-h, would have produced the final candidate.)
5. **No arm runs whose expected effect is below its design's resolution,** unless it is labelled a harm check with its own cap in H100-h. Check: every arm row has an expected effect and a resolution, and an arm with expected effect below resolution carries the label and the cap. (From: 1B, 961 H100-h for 2% resolution and effects of 1% or less; PS1.)
6. **A cost that a recommendation rests on is measured on the setup that will run,** with a timing job of at most 2% of the phase's cost, before the phase starts. Check: the recommendation cites the timing job's run ID. (From: the 56 H100-h estimate; the 1.5 H100-h check skipped; 32 H100 used without measuring 8.)
7. **A phase that a pending result could change does not start before that result,** unless the plan states that the result cannot change it. Check: each parallel start has that sentence. (From: N10 and R10 started before phase 0 had reported.)
8. **A result that lowers the expected value is reported with a stop or shrink option and its saving,** not only with a recommendation to continue. Check: such reports have a row "stop: saves N H100-h, loses …". (From: phase 0's development-suite result, the 1B cost, the 2026-11-27 13:10 outlook.)
9. **A stop opinion formed by an agent goes to the user at once,** not to the next scheduled report. Check: the time between the agent writing it and the user receiving it. (From: 2026-11-27 13:10 to 22:58.)
10. **The default of an unanswered option is the cheaper one.** Check: no message says the costly option runs unless the user objects. (From: 2026-11-27 12:18.)
11. **Savings against the plan's own estimate are not reported as results.** Check: a progress report lists what was learned separately from budget changes. (From: the answer of 2026-11-27 22:56.)
12. **Runs that may launch "without asking" have a total cap, and a user decision that changes direction cancels that permission.** At a stop, the coordinating agent lists every running job and asks about each that the stop does not name. Check: the stop report has the list. (From: the combination run of 2026-11-27 22:56.)

## Open at the end

- **The 1B verdict** comes from the phase-1B worker (branch `Aiden-Jeon/v2-phase1b`). It was not in when this was written: see the note at the end.
- **The record of the L20 stop and the cancelled phases 3-5** is PR #47 (open): L20 stopped after step 5,200 of 20,000 at 85 H100-h, not evaluated, at 15.9-16.2 H100-h per 1K steps on one node (the plan's phase 4 had 17.6-21.5). Its numbers agree with the recount here.
- **The combination run** `tabsil-v2-1b-yctx-optk-s0` was running at 2026-11-27 23:18; the user has not decided on it.
- **The v2 candidate** is the 1A weights (v1's stage 1 with the hicard stage 2, #14's run or N10). They have not been evaluated against v1 with the release protocol or released.
- **L20's step-5,000 checkpoint** is kept in MLflow (`893293ed…`) and was not evaluated.
- **Not verified here:** why the combination run was launched; the user's own bar for a worthwhile gain, which the records show only as the stop decision at +20-40 Elo.
