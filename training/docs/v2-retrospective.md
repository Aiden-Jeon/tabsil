# v2 retrospective

Status: written on 2026-11-28, after the user stopped the v2 plan ([v2-experiment-plan.md](v2-experiment-plan.md) §7, decisions 12 and 13). It reviews that plan (2026-11-26), the two documents it replaced ([v2-task-priors.md](v2-task-priors.md), [v2-scaling.md](v2-scaling.md)), the redesign of 2026-11-27 and the execution, against [v2-experiment-results.md](v2-experiment-results.md). Nothing was trained or evaluated for it, and it proposes no new experiment. The phase-1B verdict was not out when this was written ("Open at the end").

The plan, every recommendation and the coordination of the runs were made by AI agents (Claude Code sessions). One session wrote the plan ("the planning session"). A second one dispatched the work to worker chats, supervised them and put the recommendations to the user ("the coordinating session"). This document reviews their judgment. The user's decisions are recorded as facts. It was written from the records by an agent that took no part in the plan.

How to read it:

- **Recorded** is what a document, a commit, a PR, an MLflow run, a job record or a conversation shows. **Judgment** is this review's reading. What could not be checked is marked **not verified**.
- Conversations are cited by session and time, and are paraphrased.
- Times are UTC. The user works in KST (UTC+9), so the stop of 2026-11-27 23:00 is the plan's "decisions of 2026-11-28".
- GPU use was counted again from the AI Runtime job records: run time × GPUs of every job of the plan, from the first one (2026-11-26 13:45) on. G is the paired error ratio of the plan's §3. Below 1 is better.

## Summary

**What happened.** The plan's recommended tier was about 4,350 H100-h. 1,870 were spent in 34 hours. The user stopped the plan when the measured gain was put in Elo: +20 to +40 on TabArena-Lite.

**What the compute changed in the candidate: nothing.** The v2 candidate at the stop is v1's stage 1 with 10K steps of stage 2 on `graph_scm_hicard_missing`. That is #14's run (`277916ec…`), which finished 13 hours before the plan was written. Its gain was in the plan as 0.991 and +42 Elo (§1.1). Phase 0 put it at 1808 against 1768 for v1.1 eight minutes after the plan's first job, for 1 H100-h (results 0.1). The rest of the 1,870 H100-h added a second seed (N10: 1788), the noise between runs, the cost of a run, and results without a gain (R10, stage 3, and provisionally YCTX and OPTK). PS1 and L20 were stopped without a result.

**The three largest causes (judgment).**

1. **The plan was built to measure effects of 1% correctly. It did not ask whether a v2 that ends about 1% better should be made.** Its own check had the answer. The only gain with evidence already existed as weights (#14). The from-scratch branch, 60-71% of the recommended tier, had a gain that §5 calls "not known" and a gate that asks for parity with v1 (phase 3: G ≤ 1.00). The one lever its sources tie to a large gain, several times v1's training compute together with the size (§1.3), was in no tier. The plan's answer to "every effect is as large as the noise" was a better way to measure.
2. **Gain and cost never reached the user as one question, and the option labelled "Recommended" was in effect the decision.** The plan wrote the gain as "about 1%" next to "the distance to TabPFN-3.5" (§5), and the report that asked for approval called it "evidence up to TabPFN-3.5's level". No approval question named a gain until 2026-11-27 23:00. Over all chats of the plan, 17 choice questions carried an option labelled recommended, and the user chose it in 16. Every change of course began with a question from the user. Results that lowered the expected value came with a recommendation to go on, and until 2026-11-27 23:00 none came with the option to stop the plan.
3. **Expensive runs started before the cheap measurements that priced them or could have cancelled them.** Phase 1A started beside phase 0, against the plan's own order. N10 and R10 ran on 32 H100 before the cost per GPU count was measured. R10 was the first run of its prior at scale and had no smoke run. Phase 1B moved to v1's size without a 1.5 H100-h check, and it resolves 2% where the plan expected 1% or less. About 585 H100-h of N10, R10 and 1B were the price of the setup and not of the answers, and about 160 bought no answer (estimates, "What the compute bought").

**Plan against outcome.**

| | Plan (2026-11-26) | Outcome |
|---|---|---|
| Budget | 4,350 H100-h recommended, 2,689 lean, 1,258 for stage 2 only | 1,870 spent; phases 2 to 5 not run |
| Phase 0 | about 30 | 22 |
| Phase 1A | 1,228 | 848: N10 305, R10 450, L20 85 (stopped at step 5,200 of 20,000) |
| Phase 1B | 500-670 | 965: 7 runs of 117-123 at v1's size, 2 PS1 runs stopped |
| Cost measurements | — | 35 |
| Gain | "about 1%" from the stage-2 prior and "less than 1%" from the stage-2 length (§5) | 0.8% from the stage-2 prior (pooled G 0.992, 0.980-1.000), almost all of it on one dataset; the length not measured |
| TabArena-Lite Elo | no number | 1768 → 1788 (N10), 1808 (#14's run) |

## How the plan came about

### The path from the request to the first job

| Time (UTC) | Event | Source |
|---|---|---|
| 2026-11-25 23:53 | #14's stage-2 run on `graph_scm_hicard_missing` ends: TabArena-Lite Elo 1769 against 1727 for `stage2-10k` | PR #14, job record |
| 2026-11-26 05:19, 08:52 | Two v2 plans are written by agents: task priors (PR #23) and scaling (PR #25). No run of either is approved | PR #23, PR #25 |
| 2026-11-26 | The user sets the goal of v2 to TabArena classification and regression accuracy | planning session's task prompt |
| 2026-11-26 12:18 | The user asks for a higher-performing model, says it will be retrained from scratch, and asks for a critical check of the research on dev-v2 and an experiment plan from every source available | planning session |
| 12:22-12:27 | The planning agent turns the request into a task prompt. The user approves it with the option labelled recommended | planning session |
| 12:28-13:08 | Six research agents in parallel, MLflow reads, and the document. Commit `01973a5` at 13:08 | planning session, git |
| 13:14 | The report: reversed verdicts, phases and costs, three tiers, five decisions | planning session |
| 13:17 | The agent had not been able to push its work branch. The user says to commit and push straight to the branch, which the agent takes to be dev-v2. The plan goes onto dev-v2 without a PR | planning session |
| 13:22-13:36 | The user asks to run the plan with the recommended option. The coordinating session reads that as the 4,350 tier and the recommended answer to all five decisions, and adds one change of its own: phase 1A starts beside phase 0. The user approves the dispatch with the option labelled recommended. The other option was phase 0 first | coordinating session |
| 13:44 | The phase-1A worker asks whether N10 and R10 run at the same time (recommended) or in turn. The user chooses the same time | phase-1A worker chat |
| 13:45 | N10 starts on 32 H100, 37 minutes after the plan's commit, in the same minute as phase 0's first jobs | job records |
| 13:46 | The user says GPU jobs need no approval each and are queued when capacity is short | coordinating session |

Three v2 plans were written in eight hours, each by an agent and each checking the one before. The third was the first to be run. Nobody but its author had read it when stage-2 runs planned at 614 H100-h were approved on it.

### What the task prompt asked for

**Recorded.** The prompt listed twelve claims of the earlier documents to verify and seven things the plan had to hold: a verification table, the goal and the evaluation, a ranking of the levers "with the strength of the evidence and the expected effect", an experiment list with arms, steps, H100-h and a decision rule for each, budget tiers, what changes in the earlier plans, and risks and decisions.

**Not in it.** What gain would make v2 worth doing. How much compute was available. Whether to retrain at all: "from scratch" stood in the prompt as background, from the user's request, and not as a claim to check. A condition for stopping.

**Judgment.** The plan answered its prompt. The critical check was pointed at the earlier documents and not at the premise of the request or at the value of the plan itself. The expected effect per lever, which the prompt did ask for, is not in §2. Some rows have a number inside their evidence, the table's column is the strength of that evidence, and no row is "strong".

### What the plan knew about the gain

All recorded in the plan as committed on 2026-11-26:

- **§1.1:** #14's gain is 0.991 (0.982-0.998) and +42 Elo, and "1% of error is about 40 Elo". Two stage-1 runs with the same seed and prior differ by 0.990. Without the four datasets named in advance, #14's gain is 0.996 (§1, row 5).
- **§1.1:** the evidence for a longer stage 2 is 0.993 between step 5,000 and step 10,000 of one 10K schedule, which "overstates the effect of the length". The summary reads it as "stage 2 was still improving at 10K steps". A run improves until its learning rate reaches 0, so this does not show that a longer schedule ends lower (judgment).
- **§1.3:** Kumo-Tabular-S is level with v1 at v1's compute (0.996). Kumo-M is 4.7% ahead with 4.6 times the compute, and Kumo-L 7% with 15 times.
- **§2:** thirteen levers. Four are "medium", none is "strong".
- **§5:** "What the measured levers support: #14's stage-2 prior gave about 1%, and the second half of stage 2 gave less than 1%." "Nothing measured here closes that [the distance to Kumo-M and -L] at these budgets." "The gain from the stage-1 changes of 1B and from the size is not known before those phases run."

In the plan's own unit that is about +40 Elo that existed as weights, under +40 from a lever whose evidence the plan called overstated, and nothing stated for the rest.

### How it was put to the user

**Recorded (planning session, 2026-11-26 13:14).** Under the table of phases the report had one line on the expectation: the measured levers are each about 1%, "so there is evidence up to TabPFN-3.5's level", and none for Kumo-M or -L at this budget. The expectation was in percent and not in Elo, and no tier had a gain. Elsewhere the report said that #14's +42 Elo is an error ratio of 0.991. The five decisions it asked for were the budget tier, phase 1A on v1's weights, changing the recipe and the size, one model for both tasks, and the task priors after the core. Whether the gain was worth the budget was not among them.

### How the tiers were built

**Recorded (§5 of `01973a5`).** The tiers are sums of phases:

| Tier | Sum | H100-h |
|---|---|---|
| Stage 2 only | 30 + 1,228 | 1,258 |
| Lean | 30 + 614 (1A without L20) + about 500 (1B, 9 runs) + 950 + 307 + 288 | 2,689 |
| Recommended | 30 + 1,228 + 670 + 570 + 950 + 614 + 288 | 4,350 |

**Judgment.**

- **"Recommended" is the top of every range.** It holds 1B with every conditional third seed (670) and phase 4 at 20K steps (614), which assumes that L20 passes. The plan gives no reason why this tier and not the lean one is recommended.
- **No tier has a gain.** The 1,661 H100-h between lean and recommended buy L20, phase 2, the second 10K steps of phase 4 and three conditional seeds. What they were expected to add is not stated, so a reader could compare the tiers by cost only.
- **The from-scratch branch is most of the budget and has no stated gain.** Phases 1B to 5 are 2,615-3,092 of the 4,350 H100-h. Phase 3 trains for the compute of v1's stage 1, which is 150K steps at v1's size, and its gate is parity with v1's stage 1. What could make it better than v1 are 1B's recipe changes, each expected at 1% or less (§1.1, §4).
- **Against the plans it replaced.** They came to 2.7-4.3K H100-h (§5). The new recommended tier is at the top of that range.

### Alternatives that were not offered

| Alternative | In the plan | First raised |
|---|---|---|
| Train nothing: evaluate #14's weights (phase 0) and decide on them | No tier. Phase 1A's text calls its best run "a release candidate that does not depend on phases 1B to 3" | 2026-11-27 12:52, by a separate chat the user asked for an outlook, as "finish on the 1A weights" |
| Inference only | §2's last row: "It needs no retraining … main, not v2". No estimate. §1.1 has two inference changes at +35 and +80 Elo, both already in v1.1's 1768 | — |
| Much more training compute | §1.3 is the plan's evidence for it. No tier has it. Kumo-M's compute is 7,186 H100-h. Phase 6 lists "a longer stage 1" among the things without evidence | 2026-11-27 12:52, by the same chat |
| Not making v2 | — | 2026-11-27 23:00, by the user |

## Plan against outcome

### Cost

| Item | Plan | Spent or measured | Note |
|---|---|---|---|
| Phase 0 | about 30 | 22 | 19 jobs |
| N10 | 307 | 305 | 32 H100, 9.54 h |
| R10 | 307 | 450 | 32 H100, 14.07 h: data stalls (results, "Why R10 was slow") |
| L20 | 614 on 32 H100 | 85 | moved to 8 H100 at micro batch 2 (about 360 for 20K steps); stopped at step 5,200 |
| Phase 1A, evaluations and R10's diagnosis | left out of the totals | 8 | |
| One 1B run | 56 (XS, FLOPs ratio 0.29) | XS: 98 at micro batch 4, about 81 at its cheapest setting. v1's size, as run: 117-123 | results 0.3 and "GPU cost" |
| Phase 1B, the 7 runs of C, YCTX and OPTK | 500-670 for 9-12 XS runs | 845 | |
| Phase 1B, PS1 | in the above | 73 | 2 runs, cancelled at steps 10,500 and 5,000 |
| Phase 1B, smoke runs, PS1's speed diagnosis, evaluations, the combination run | left out of the totals | 47 | the combination run was cancelled after 5 |
| GPU cost and collector measurements | — | 35 | not in the plan |
| Phases 2 to 5 | 2,115-2,422 | 0 | redesigned to 760-935, then not run |
| **Total** | **4,350** | **1,870** | |

- The plan's budget table has about 1,955 H100-h for phases 0 to 1B after the stop. It holds 1B at its estimate of 2026-11-27. The job records give 1,870, with two one-GPU evaluations of C's third seed still running.
- Stage 2 costs 29.3 H100-h per 1K steps on 32 H100 (N10), 21.5 on 8 H100 at micro batch 1 and 16.2 at micro batch 2 (L20's 5,200 steps). The plan priced every run at the 32-H100 rate.

What had been spent at the points where the course could change:

| Time (UTC) | Event | H100-h spent |
|---|---|---|
| 2026-11-26 15:11 | Phase 0 is reported: #14's gain is absent on the development suite; XS costs 0.81 of M | 76 |
| 2026-11-26 22:28 | The user approves 1B at v1's size (about 1,090) | 542 |
| 2026-11-27 05:57 | The user approves L20 | 970 |
| 2026-11-27 08:03 | The user asks whether the GPUs are used efficiently | 1,105 |
| 2026-11-27 12:52 | A separate chat tells the user that v2 will most likely end at about 1% | about 1,410 |
| 2026-11-27 13:07 | The coordinating session agrees | 1,424 |
| 2026-11-27 22:57 | The user asks how many Elo "about 1%" is | 1,861 |

96% of the spend came after phase 0 was reported.

### Effect

| Lever | The plan's expectation | Measured | Source |
|---|---|---|---|
| Stage-2 prior `graph_scm_hicard_missing` | 0.991, "medium" | Two seeds against `stage2-10k`, pooled: 0.992 (0.980-1.000). Development suite 0.996 (0.988-1.002). Without polish_companies_bankruptcy 0.997. Elo 1768 → 1788, 1808 on Lite and 1782 → 1787, 1804 on 3 splits | results, phase 1A |
| Kumo's other stage-2 parts (R10) | no number | 1.003 (0.999-1.008) against the two hicard runs. Not adopted | results, phase 1A |
| Stage-2 length (L20) | under 1%, "medium" | not measured | results, "L20" |
| Target standardized with the train rows (YCTX) | "medium, not measured here" | provisional: G 0.997 against C's first two seeds. Reference loss at 30K steps −0.2% against C's mean in both tasks; C's three seeds span 0.4% | phase-1B worker, 2026-11-27 22:55; MLflow, read for this review |
| Kumo's optimizer split (OPTK) | "weak" | provisional: pooled G 1.005 (0.999-1.012). Reference classification loss +0.4%, both seeds above all three C seeds | the same |
| Every-stage prior parts (PS1) | "expected to be neutral" | not measured | — |
| Model size | "medium that it matters, direction unknown" | not run. XS costs 0.66-0.83 of M per step, where the plan had 0.29 | results, "GPU cost" |
| Stage 3 | "small" | 1.000 on TabArena, 0.999 on the development suite, 0.999 on tables above 10K rows | results 0.6, "Stage 3 by table size" |
| Noise between two stage-2 runs | not known; 1.000 with sd 0.007 from stage 2 against stage 3 | ν 0.003 pooled, 0.009 on the development suite | results, "ν" |
| Noise between two stage-1 runs | 0.990 at 25K steps, one pair | provisional: 0.999 (0.991-1.007) between C's seeds 0 and 1, with classification 1.005 and regression 0.988 | phase-1B worker, 2026-11-27 22:08 |
| The capped readout of stage 1 | "within about 1%" | 1.003 (1.000-1.006) | results 0.2 |

### Schedule

| | Plan | What happened |
|---|---|---|
| Order | phase 0, then 1A and 1B, then 2 to 5, "Each phase is approved on the results of the one before it" | phase 0 and 1A started in the same minute; 1B 12 hours later; phases 2 to 5 not run |
| Phase 0 | no time | 2026-11-26 13:45-15:01, reported at 15:11 |
| N10, R10 | 9.6 h each | 9.54 h and 14.07 h. R10 started 70 minutes after N10: the workspace allows 8 nodes, and a request above that fails instead of waiting |
| 1B | no time | 2026-11-27 02:00-23:20. The quota was full until N10 ended, the first worker chat did not accept the GPU rule that reached it through another session, and PS1's stalls were diagnosed before the launch. Verdict due 2026-11-28 00:30 |
| L20 | 19 h on 32 H100 | 44-46 h planned on 8 H100; cancelled after 10.6 h |
| The plan | no dates | stopped 33 hours after its first job |

## What went wrong and why

### Who recommended and who decided

| Time (UTC) | Decision | Recommended by | Decided by | What the user was shown |
|---|---|---|---|---|
| 2026-11-26 12:27 | The task prompt of the plan | planning session | user, the recommended option | the prompt |
| 13:17 | The plan goes onto dev-v2 without a PR | — | user | the report |
| 13:36 | The 4,350 tier; phase 1A beside phase 0 | coordinating session | user, the recommended option | the units and their costs (30; 307 + 307 + 614); no gain |
| 13:44 | N10 and R10 at the same time, 32 H100 each | phase-1A worker | user, the recommended option | 614 H100-h in 10 h, or in 20 h in turn |
| 13:46 | GPU jobs without approval each | — | user | — |
| 15:12 | N10 and R10 go on after phase 0; L20 is not launched by itself | coordinating session | no answer from the user; the runs went on | the development-suite result as G ratios |
| 22:28 | 1B at v1's size: 9 runs, about 1,090 H100-h (plan 500-670) | coordinating session | user, the recommended option | the cost against the plan's; 1B on XS as the other option (880); no gain |
| 2026-11-27 05:57 | L20 on the hicard prior, 614 H100-h | phase-1A worker, coordinating session | user | the G table; three options, one of them no L20; "may end without a conclusion below 0.5%" |
| 08:07 | PS1 dropped; phases 2 to 5 redesigned | coordinating session, after the user's question | user | the options with their savings |
| 12:21 | L20 on 8 H100 at micro batch 2 | coordinating session, unasked | user | three setups with cost, time and what differs from the controls |
| 12:42 | Phase 2 skipped; phase 3 on 8 H100; the collector change | coordinating session (the redesign worker had recommended 16 H100) | user, the four recommended options | costs and wall times |
| 23:00 | Phases 3 to 5 not run | coordinating session, after the user's judgment | user, the recommended option | "no expected gain in accuracy" |
| 23:00 | L20 stopped | coordinating session recommended to finish it | user, against the recommendation | "at most 0.7% (20-30 Elo)", 240 H100-h left |
| 23:28 | The combination run cancelled | coordinating session | user, the recommended option | 115 H100-h saved |

### 1. The plan measured where it had to decide (planning session)

**Recorded.** The summary's "the plan therefore" has five points: the paired ratio on more data, stage-2 changes as branches, stage-1 changes as bundles with seeds, the size at matched compute, and the task priors after the core. All five are about how to measure. The three sentences on the gain are in §5, under the budget table.

**Judgment: what followed from the plan's own text and was not said.**

- **The gain with evidence was in hand.** #14's weights existed. Phase 0.1 was enough to state their gain under the current inference, and it cost 1 H100-h.
- **The from-scratch branch could at best tie v1's stage 1 and then add the same stage 2.** §1.3 shows a model of another size reaching v1's accuracy at v1's compute. Phase 3 has the compute of v1's stage 1. The plan's own gate for it is G ≤ 1.00.
- **A neutral recipe fails that gate about half the time.** The coordinating session noted this on 2026-11-27 13:07. It was true on 2026-11-26.
- **The lever with support in the sources was compute, and the plan had none of it.** §5 says so in one sentence ("nothing measured here closes that at these budgets") and then asks for the budget.

**The way of judging behind it.** A finding was treated as a problem of method. "One prior change is as large as the noise" led to a design that resolves smaller effects. It did not lead to the question whether effects of that size are what the user wants to buy. The check was asked to be critical of the earlier documents, and it was. Nobody was asked to be critical of the plan.

**What should have been different, and when.** In the report of 2026-11-26 13:14, before any approval: the gain that has evidence is about +40 Elo and exists as #14's weights; the from-scratch branch costs 2,600-3,100 H100-h and has no evidence of a gain; the gap to the leaders goes with several times v1's compute. Then the options: evaluate and release #14's weights, fund a much larger training, or stop.

### 2. Gain and cost were never one question (both sessions)

**Recorded.**

- **No approval question named a gain.** Not the dispatch dialog of 2026-11-26 13:27, not the worker's of 13:44, not the 1B dialog of 22:19. The first one that did was that of 2026-11-27 23:00, after the user had asked.
- **The user followed the label.** 17 choice questions with an option labelled "Recommended" were put to the user: 1 by the planning session, 11 by the coordinating session and 5 by worker chats. 16 ended on that option. The exception is the stop of L20.
- **The reasons given were about design and cost.** L20 was recommended on 2026-11-27 05:55 because "the recommended tier has L20, and its design is the soundest of the three". 1B at v1's size because it removes a risk for 1.24 times the cost of XS.
- **Every change of course began with a question from the user.** Whether the GPUs were used efficiently (2026-11-27 08:03) led to dropping PS1 and to the cost measurements. Whether performance would rise at all (12:43, to a separate chat) led to the first outlook: "about 1%, Lite Elo 1768 to 1788-1808". Whether the experiments were positive (22:56) led to the stop.
- **Results that lowered the expected value came with "continue".**

  | Time (UTC) | Result | Recommendation |
  |---|---|---|
  | 2026-11-26 15:12 | #14's gain is 1.000 on the development suite; by the plan's branch rule the prior would not be adopted | N10 and R10 go on. R10 had run for 17 minutes |
  | 2026-11-26 15:12 | A small model saves 19% and not 71% | 1B moves to v1's size, at twice the plan's cost |
  | 2026-11-27 01:28 | N10: the gain sits in a few datasets (0.995 without polish), and the pooled median ratio is 1.000 | "nothing to decide now" |
  | 2026-11-27 05:55 | R10 adds nothing; the hicard prior passes the rule only with polish | L20 on the hicard prior for 614 H100-h |
  | 2026-11-27 13:07 | The coordinating session agrees that v2 ends at about 1% and that phase 3 without a recipe gain has an expected gain of 0 | The options go to the user with the 1B verdict. 1B and L20 run on: 438 H100-h more until the stop |
  | 2026-11-27 23:00 | The user judges +20 to +40 Elo not worth more runs | L20 should finish: "the one path left that could pass 1%" |

- **The status reports had no outlook.** They listed runs, steps, speeds and PRs. Where v2 would end was first said by a chat outside the plan, when the user asked it.
- **The answer to "is it positive?"** (2026-11-27 22:56) began with "slightly positive" and counted the fall of the plan's budget from 4,350 to 3,025-3,200 among what the experiments had achieved.

**Judgment.**

- **The numbers were on the page, in a unit and a frame that did not invite the question.** "About 1%" next to "the distance to TabPFN-3.5" reads as reaching TabPFN-3.5. "+40 Elo, which you have, for 4,350 H100-h" reads differently. The same facts drew "not worth running" from the user once they were in Elo.
- **The facts the user stopped on were reported on 2026-11-26 15:12**, at 76 H100-h: #14's weights at 1808, no gain on the held-out suite, stage 3 without effect, the small model not cheap. They came as G ratios under the heading "results that need a decision", and the decisions asked for were that L20 does not start by itself, and XS or M for 1B. It was night in KST. The user's next message came at 22:13, and N10 and R10 had run on: about 530 H100-h were spent by then.
- **"Recommended" meant "as planned".** The plan was the reference, so each recommendation defended the plan's next step. A cut in the plan's own estimate was reported as a result for the same reason.
- **The outlook of 2026-11-27 13:07 was not held back.** The coordinating session wrote it in its own chat and the other chat passed it on. What was missing is the question that follows from it: whether the two things still spending, 1B and L20, should stop. The user saw the Elo range at 12:52 and stopped at 23:00, which is the next morning in KST.

**What should have been different, and when.** At the approval of 2026-11-26 13:27: the expected gain in Elo next to 4,350 H100-h, and the question whether that is worth it. At 15:12: the same question again with phase 0's numbers and an option to stop. With the plan's order, phase 0 first, 22 H100-h would have been spent while that question waited for the user. At 22:19: "1B now costs twice the plan and resolves 2%; drop it?" as a question. The dialog offered 1B at v1's size, 1B on XS, or another dispatch. In every status report: one line on where v2 would end.

### 3. Expensive runs before cheap measurements (coordinating session and workers)

**Recorded.**

- **Phase 1A beside phase 0.** The plan's order was phase 0 first. The coordinating session wrote that phase 0's steps 0.2 and 0.4 decide only 1B. Steps 0.1 and 0.6 bear on 1A: they measured #14's prior, which N10 repeats and R10 builds on. When they arrived, N10 had spent 46 H100-h and R10 9.
- **R10 before N10's noise.** R10's rule needs ν, which N10 measures. Both were launched together.
- **R10 without a smoke run.** `graph_scm_realism` had never trained at scale (PR #26: "No training run is part of this PR"). R10 ran 4.95 s per step against 3.30 and cost 143 H100-h more. A one-GPU run of 300 steps later showed 16.4% of the time waiting for data, for 0.3 H100-h (results, "Why R10 was slow"). The 1B worker did run smoke jobs first and found the same stalls in PS1 before its runs.
- **32 H100 without the cost per GPU count.** The plan priced every run at v1's 32-H100 rate. Phase 0.3 noted on 2026-11-26 15:11 that stage 1 on 8 H100 costs 0.64 of v1's record per step and that 32 H100 was not measured. Nothing followed until the user's question 17 hours later. The stage-2 measurement then took 3.6 H100-h.
- **1B at v1's size without the 1.5 H100-h check.** Results 0.3 proposed an XS timing run at micro batch 8 "before choosing". The coordinating session's report of 15:12 asked the user for XS or M, recommended M and did not mention the check. It said so itself on 2026-11-27 08:06. Measured later, XS costs 0.66 of M.
- **1B below its own resolution.** §3 says the design resolves about 2%, and §4 says PS1 "is expected to be neutral" and that the scale "cannot attribute a 1% effect". The coordinating session called 1B "in effect a check that keeps a harmful change out of phase 3" (2026-11-27 08:06). The check cost 965 H100-h. Phase 3 was planned at 950 and redesigned to 580-620.
- **A rule that adopts by default.** §3 adopts an arm whose effect is below the resolution if it "removes a train/inference mismatch or copies a published configuration, and G is at most 1.01", and §4 adds a combination run "if two or more arms are adopted". On 2026-11-27 22:56 the 1B worker launched that run, about 119 H100-h, because both arms fell under this clause with G 0.997 and 1.005. C's third seed was not evaluated yet. Two results without an effect led to a further run. It was cancelled after 5 H100-h.
- **L20.** Its expected effect was "0.7% at most", from a number the plan called overstated, against a threshold of 0.56% (0.9944). It was then moved to another GPU count and micro batch than its controls, so a result near the threshold could not have been read. It was stopped at 26% without a result.

**The way of judging behind it.** A parallel start was weighed by the day it saves and not by what the pending result could cancel. A cost was taken as a property of the run and not of its setup. A run was justified by being in the plan, and its expected effect was not set against its resolution when it was launched, although the plan had both numbers.

**What should have been different, and when.** Before the dispatch of 2026-11-26 13:27: timing jobs for every setup that a run above 100 H100-h would use, stage 2 on 8 and 32 H100 among them, and a smoke run of the realism prior. N10 alone after phase 0, R10 only after N10's ν and the development-suite result. On 2026-11-26 22:19: the 1.5 H100-h check before the size of 1B, and the question whether a 2% screen for 1% effects should run at all.

### The ten leads

The coordinating session listed ten of its own possible mistakes. Each was checked against the records.

| # | Lead | Verdict | Evidence |
|---|---|---|---|
| 1 | The expected gain was never set against the user's bar | **Right, with one correction** | §5's sentence is in the plan of 2026-11-26. The report of 13:14 and every approval question until 2026-11-27 23:00 give no gain in Elo and ask no "worth it" (cause 2). The correction: the user first saw v2's end in Elo on 2026-11-27 12:52, from a separate chat, at about 1,410 H100-h. The user first asked the coordinating session for it at 22:57, at 1,861 |
| 2 | The only success criteria were statistical adoption rules; nothing said when to drop the plan | **Right** | §3 has rules per arm with thresholds of 0.5-2%. Phase 3 has a gate for itself. "Each phase is approved on the results of the one before it" names no condition. L20 and phase 3 were in fact held for a decision; no threshold was fixed for either |
| 3 | Costs were estimated from FLOPs and not measured | **Partly** | §4 marks the FLOPs costs as estimates "until the timing runs replace them", and phase 0.3 did that 36 minutes after the first job. Wrong were three things. The tiers the user approved rested on them. §1 row 11 gave the cost estimates the verdict "Confirmed", which held for the arithmetic only. And "56 against 124" holds two changes: XS measured 98, and 117-123 is v1's size, chosen afterwards. The cost by GPU count was not estimated at all |
| 4 | 1B was recommended at v1's size without a check of 1.5 H100-h | **Right** | cause 3 |
| 5 | N10 and R10 ran on 32 H100; the cost by GPU count was measured only after the user asked | **Right** | cause 3. With the recipe of #14's run on one node, 10K steps cost about 223 H100-h (plan, phase 4, option B) against 305 |
| 6 | 1A started without waiting for phase 0 | **Right** | cause 3. The other option, phase 0 first, was shown to the user and not recommended |
| 7 | 1B resolves 2% by design, the expected effects were 1% or less, and about 900 H100-h went into it | **Right: 965** | cause 3 |
| 8 | An arm expected to be neutral (PS1) was run and then stopped | **Right: 73 H100-h, not 64** | job records; plan §4 |
| 9 | The recommendation was in effect the decision | **Right** | 16 of 17 choice questions, and the free-text decisions on L20 followed the recommendation too (cause 2). Phase 3 on 8 H100 was not an exception: the redesign worker had recommended 16 H100, and the coordinating session's dialog labelled 8 H100 as recommended |
| 10 | The plan that criticised and replaced two plans asked for a larger budget; no lever is "strong" | **Partly** | The recommended 4,350 is at the top of the earlier plans' 2.7-4.3K and the lean 2,689 at the bottom, so it is no smaller and not clearly larger. No lever in §2 is "strong" |

### Found beyond the leads

- **The candidate at the end is the candidate from before the plan**, and the facts the user stopped on were reported at 76 H100-h (summary, cause 2).
- **The from-scratch branch had no stated gain and a parity gate**, and the lever the sources support, compute, was in no tier (cause 1).
- **"Recommended" was the top of every range**, without a reason ("How the tiers were built").
- **The plan was not reviewed.** It went onto dev-v2 without a PR, and its first run started 37 minutes after its commit. PR #32, the code of 1B, was given to a chat that had not written it, because that code would spend about 1,100 H100-h (coordinating session, 2026-11-26 22:19). The plan that spent them was not.
- **The evidence for a longer stage 2 was a comparison inside one schedule** ("What the plan knew about the gain"). L20 and the second 10K steps of phase 4, 921 H100-h of the recommended tier, rested on it.
- **The 1B rule adopts by default and starts further runs** (cause 3).
- **R10 was the first run of an untested prior at scale** (cause 3).
- **After the user's judgment, the coordinating session recommended finishing L20** with an argument that was not in the plan: a pass would make 40K-60K steps of stage 2 "a 2% candidate" at about 1,000 H100-h. No measurement supports that extrapolation.
- **The spend that was reported was estimated, not counted.** "About 1,150 so far" on 2026-11-27 07:50 was 1,091 by the job records. The 1B remainder was corrected from 750 to 850 within the hour. The spend at the stop was put at about 2,000, and PS1's two runs at 64 where the records have 73. The errors are small, but no report was a count.

## What went right

- **Phase 0 replaced assumptions for 22 H100-h.** The capped readout holds (1.003), no FP16 arm is needed, the small model is not cheap, #14's gain and the 0.1.1 inference gain add up, and #14's gain is absent on the held-out suite. The plan had put the timing runs first (§1, row 11), and they were first.
- **The plan's check of the earlier documents was right where it was tested.** The flat stage-1 Elo is a context-length effect: phase 0.2 confirmed it for 2 H100-h. It also corrected the Kumo prior comparison and three citations.
- **Branches from one checkpoint are the cheaper design.** ν is 0.003 pooled, so a branch resolves about 0.6% where a screen from scratch resolves 2% (plan, "Later screens").
- **The development suite showed how wide the gain is.** 48 datasets outside TabArena, checked by name, source and row count (results 0.5). Without it, +42 Elo on Lite would have stood as the result.
- **The noise was measured before effects were read**: ν for stage 2, the seed difference on the development suite (0.009), and three C seeds for stage 1.
- **Results without a gain were written down as such.** The results say that #14's prior "would not be adopted on this evidence" by the branch rule, that the hicard prior passes only with one dataset, and that R10 adds nothing.
- **The 1B worker ran smoke jobs first** (12 H100-h). They found PS1's data stalls before nine runs were launched, and the prefetch setting came out of it.
- **R10's slowness was explained for 3 H100-h**: the generation time of rare tables, the loader's order, and why the logged data wait showed 1/32 of it.
- **The cost measurements (35 H100-h) are true for any later run**: the cost by GPU count and micro batch, the collector pauses (PR #40: 6.7% in stage 1), and that a narrower model is not cheaper. They made L20 cost 85 H100-h for its 5,200 steps where 32 H100 would have cost about 155.
- **The coordinating session did three things unasked that lowered the cost or the risk.** It kept L20 from starting by itself after phase 0 (2026-11-26 15:12). It brought up L20's setup again when the measurements came in (2026-11-27 12:18). It wrote on 13:07 that phase 3 should not be launched as a formality.
- **The stop.** The user's two decisions kept about 240 H100-h of L20 and 760-935 of phases 3 to 5 from being spent, and the third the 115 of the combination run.
- **Every evaluation had 0% imputed splits**, and a failed evaluation job was run again and not filled in.
- **The mistakes were recorded as they were found**, and the review was given to an agent outside the plan.

## What the compute bought

"Cheaper" is an estimate from the costs measured on 2026-11-27 (plan, "Cost basis for phases 2 to 5"). The last column is judgment.

| Spent (H100-h) | What is known from it | The same answer for less |
|---|---|---|
| 22, phase 0 | the five things above. This is what the stop rests on | — |
| 305, N10 | #14's gain repeats on TabArena, almost all of it on one dataset; ν | about 223 on one node with the recipe of #14's run |
| 450, R10 | Kumo's other stage-2 parts add nothing at 10K steps | 305 without the stalls; about 223 on one node, if the prefetch setting removes them there too |
| 845, the 7 runs of 1B | provisional: YCTX and OPTK do not help at 30K steps; the noise between three stage-1 seeds | about 570 on XS at its cheapest setting, with the question whether XS stands for v1's size |
| 73, PS1 | nothing | 0 |
| 85, L20 | nothing on its question. The cost of stage 2 on one node over 5,200 steps | 0 |
| 5, the combination run | nothing | 0 |
| 35, cost measurements | the costs above | — |
| 50, smoke runs, diagnoses and evaluations of 1A and 1B | the causes of the stalls; the evaluations of the runs above | — |

- **About 585 of the 1,600 H100-h of N10, R10 and 1B's seven runs were the price of the setup**: 82 + 227 + 275. About 160 bought no answer: PS1, L20 and the combination run.
- **What 1B's answers were for is gone.** They were to choose the recipe of phase 3. Whether phase 3 was worth running could be asked on 2026-11-26 from §1.3 and §5.
- **The stop needed 22 H100-h.** The facts behind it are phase 0's. With the plan's order, phase 0 first, the question of 2026-11-27 22:57 could have been asked at 22 H100-h. With a second seed to make sure, N10, at about 330. As it was run, 76 were spent when phase 0 was reported and about 530 when the user next wrote. Whether the user would have answered the same way then is not verified.

## Rules for the next plan

Each rule names what it comes from and how it is checked.

1. **The approval question holds the expected gain in the user's unit next to the cost.** Elo on TabArena-Lite, as a range, with what part of it already exists. Check: the question the user answers contains both numbers. (From: §5's "about 1%"; the first such question was that of 2026-11-27 23:00.)
2. **The tier table has a tier that trains nothing, and every tier states the gain it adds to the one below.** Check: the first row costs no training; each row has "adds +X Elo, evidence: …". (From: #14's weights; "Recommended" as the top of every range.)
3. **The plan fixes, before its first run, the gain below which the rest is not run, and the points at which it is read.** The user sets the number. Check: the plan names the threshold and the points, and each status report prints the current expected gain next to it. (From: lead 2; the stop came from the user's question.)
4. **A premise of the request that the plan's own evidence speaks against goes back to the user before the plan is built on it.** Check: the plan has a line "the request assumes …; the evidence says …; options". (From: "from scratch" against §1.3.)
5. **Every run has three entries: the expected effect, the resolution of its design, and the decision each outcome leads to.** A run whose resolution is coarser than its expected effect is not in the plan. A result without an effect changes nothing and starts no run. Check: the three entries in the run's row; no rule adopts on "not worse". (From: 1B, PS1, L20, the combination run.)
6. **A cost in the tier table is measured on the setup that will run.** A run above 100 H100-h cites a timing job with its stage, size, prior, GPU count and micro batch. A prior or a setup used at scale for the first time has a smoke run. Check: the run IDs in the plan. (From: 56 against 98 and 117-123; 32 H100; R10.)
7. **A check that costs under 1% of the run it informs is made before the decision.** If it is skipped, the recommendation says so. Check: the recommendation lists the checks proposed and what became of each. (From: the XS timing run of 1.5 H100-h before 1,090.)
8. **A run does not start beside the phase whose results bear on it, unless its launch names the result that cancels it.** When that result arrives, the report says whether the condition is met. Check: the launch message and the report. (From: 1A beside phase 0; R10 before ν.)
9. **A cost that rises by more than 30%, or an expected gain that falls, reopens the decision, with dropping the item as an option.** Check: such a report has the option and what it saves. (From: 1B at 500-670 becoming 1,090; R10 at 4.95 s per step by half-time; the results of 2026-11-26 15:12 and 2026-11-27 01:28.)
10. **"Recommended" is put on an option only with its expected gain, and not on one below the bar the user has stated. "It is in the plan" is not a reason.** Every choice question has an option to stop or skip. Check: the dialogs. (From: 16 of 17; L20 on 2026-11-27 05:55 and 23:00.)
11. **A status report begins with the outlook**: the gain so far in Elo, where the plan is expected to end, what is spent as counted from the job records, and what is left. A lower estimate of the plan's own budget is not listed as a result. Check: the first lines of each report. (From: the outlook came from outside the plan; the answer of 2026-11-27 22:56; the reported spend.)
12. **A plan above 500 H100-h is read, before its first run, by an agent that did not write it, with one question: what is the expected gain, and is the cheapest way to it in the plan?** Check: the review is dated before the first job. (From: 37 minutes from commit to N10; the code of 1B was reviewed, the plan was not.)

## Open at the end

- **The 1B verdict** comes as a PR of the phase-1B worker and was not open at 2026-11-27 23:45. The provisional numbers above are that worker's with two of C's three seeds, and this review's reading of the reference loss.
- **The v2 candidate** is the weights of phase 1A. They have not been evaluated with the release protocol of phase 5, and whether they are released is not decided (plan §4, phase 5).
- **L20's checkpoint of step 5,000** is in its MLflow run (`893293ed…`). No resume is planned.
- **Not verified:** the bar the user would have set on 2026-11-26. The records show it only as the stop at +20 to +40 Elo. Of the worker chats, the questions to the user and the user's messages were read, and the rest through their PRs and their mails to the coordinating session.
