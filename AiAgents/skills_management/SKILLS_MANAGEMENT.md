# Skill Management

## What This Is

A coding agent arrives at a repository knowing how software is written in general and nothing about
how it is written *here*. It knows Django; it does not know that this codebase forbids raw SQL in
views. It knows migrations; it does not know that a migration run without a particular flag records
itself as applied and creates nothing. Left to itself it produces code that is competent, plausible,
and wrong in ways that only a person who has been burned before would recognise.

**Skills are how a repository tells an agent what it learned the hard way.** Each one is a markdown
file holding the rules for one kind of work: the prohibitions, the conventions that are not
discoverable by reading a single file, the mistake that looks correct until production.

The obvious approach - write everything down and load it all - fails immediately. The combined text
outweighs the work itself, and an agent carrying fifty pages of instructions has less room for the
task and less attention for any individual rule. The opposite approach, loading nothing, means the
rules exist and are never read.

So the system is built on **selective loading**. A tiny index is always present, listing what exists
and when each piece applies. The rules themselves stay on disk until the index says they are
relevant, at which point one file is pulled in. Cost scales with what the current task actually needs
rather than with everything the repository has ever learned.

That design produces its own problem, and this document is mostly about that problem.

## Why It Needs Managing

**Skills describe code, and code moves.** Nothing connects the two. When a function is renamed, no
test fails because a skill still names the old one. When a subsystem is deleted, the skill describing
it goes on describing it. When a new constraint is introduced, no skill mentions it until someone
writes it down.

Every other artifact in a repository has a failure signal. Break a test and it goes red. Break a type
and the compiler objects. Break a skill and **everything continues to work perfectly**, while every
agent that reads it complies faithfully with a rule that stopped being true.

That inversion is why this needs a process rather than good intentions:

> **A wrong skill is worse than a missing one.** An absent skill makes an agent think. A wrong skill
> makes it certain.

Skills also decay in a specific direction. They grow, because adding a rule is easy and deleting one
feels like losing information. They accumulate rules that were never needed, written against
imagined failures rather than real ones. They drift toward restating what the agent already knew.
Left alone, a skill set becomes long, vague, partly false, and ignored.

## How It Is Kept True

Two halves, deliberately separated:

**Detection is cheap, mechanical and frequent.** Scripts ask questions with exact answers: does this
change touch a file a skill mentions? does every path a skill cites still exist? No judgement, no
model, so nothing can be invented and nothing can be flattered.

**Editing is judged, rare, and attached to the change that caused it.** An agent proposes, a person
accepts, and the edit ships in the same branch as the code that made it necessary. Never a separate
tidy-up later, which always loses to the next feature.

Keeping these apart is the central decision. Merged - by asking an agent after every commit whether
the skills need updating - the system produces churn: most commits need nothing, and an agent asked a
yes-or-no question hundreds of times says yes wrongly often enough to erode what it was meant to
protect.

In practice this reduces to **six actions**, each a prompt you paste to an agent. Everything after
them explains why they are what they are.

---

## Index

**Practical**

- [Practice - The Six Actions](#practice---the-six-actions)
  - [Action 1 - Check a Branch](#action-1---check-a-branch) - before merging
  - [Action 2 - Find Drift](#action-2---find-drift) - weekly
  - [Action 3 - Write a Rule Down](#action-3---write-a-rule-down) - after a repeat mistake
  - [Action 4 - Look for a Missing Skill](#action-4---look-for-a-missing-skill) - after a new subsystem
  - [Action 5 - Clean House](#action-5---clean-house) - quarterly
  - [Action 6 - Catch Up](#action-6---catch-up) - after a long gap
    - [Phase 1 - Inventory](#phase-1---inventory)
    - [Phase 2 - Audit the Existing Rules](#phase-2---audit-the-existing-rules)
    - [Phase 3 - Apply](#phase-3---apply)
  - [The Skill Writing Contract](#the-skill-writing-contract) - what goes in a skill, what stays out

**Reference**

- [0 - How the Pieces Fit](#0---how-the-pieces-fit) - the layout, the symlinks, what is generated
- [1 - How Skills Load](#1---how-skills-load) - anatomy of an AGENTS.md, and an annotated skill
- [2 - Where a Rule Belongs](#2---where-a-rule-belongs)
- [3 - When To Create a Skill](#3---when-to-create-a-skill)
- [4 - When To Update](#4---when-to-update)
- [5 - The Update Procedure](#5---the-update-procedure)
- [6 - The Cadence](#6---the-cadence)
- [7 - Automation 1 - The Citation Hook](#7---automation-1---the-citation-hook)
- [8 - Automation 2 - The Weekly Drift Audit](#8---automation-2---the-weekly-drift-audit)
- [9 - The Rule of Two](#9---the-rule-of-two)
- [10 - The Actions Explained](#10---the-actions-explained) - why each is shaped that way, worked example
- [11 - Troubleshooting](#11---troubleshooting)
- [12 - Reviewing a Proposed Skill Edit](#12---reviewing-a-proposed-skill-edit)
- [13 - Anti-Patterns](#13---anti-patterns)
- [14 - Quick Reference](#14---quick-reference)

**Setting it up somewhere else**

- [15 - Bootstrapping an Existing Repository](#15---bootstrapping-an-existing-repository) - the only
  prompt outside the six actions: it installs the whole system in a repository that has none

---

# Practice - The Six Actions

**You only ever do six things.** Each one is a prompt you paste to Claude. The Reference half that
follows explains why they are what they are; you do not need it to use them.

| # | Action | When |
| --- | --- | --- |
| 1 | Check a branch | Before merging any pull request |
| 2 | Find drift | Every Monday |
| 3 | Write a rule down | After correcting an agent twice on the same thing |
| 4 | Look for a missing skill | After shipping a new subsystem |
| 5 | Clean house | Every quarter |
| 6 | Catch up | After months of not doing 1 to 5 |

---

## Action 1 - Check a Branch

**When**: before merging any pull request.
**What it does**: finds rules this branch made wrong, and new constraints it imposes on future work.

```
Review this branch for skill impact. Report only, change nothing.

STEP 1 - get the diff. Try in this order and say which you used:
  git diff master...HEAD                     (on a feature branch)
  git diff $(git describe --tags --abbrev=0)..HEAD   (already merged to master)
  git diff HEAD~{N}..HEAD                    (ask me for N if neither works)
If the diff is empty, stop and say so.

STEP 2 - narrow before you read. Pass the SAME range to:
  ./skills/skill-sync/assets/citation-check.sh <range>
Read the skills it names in full, and only skim the rest. Do not read every
skill in the repository. If it names none, say so: the branch may genuinely
touch nothing any skill describes, and that is a valid finding, not a
reason to keep reading.

STEP 3 - report three lists:
  WRONG NOW  - rules in skills/*/SKILL.md or any AGENTS.md that this branch
               made false. Quote the skill line and the code that
               contradicts it.
  NEW RULE   - constraints this branch imposes on future unrelated work
               (a registry to append to, a required decorator, a chokepoint
               that must not be bypassed). Say which file it belongs in:
               AGENTS.md CRITICAL RULES if it breaks silently during
               unrelated work, a skill otherwise.
  DEFERRED   - anything describing code not merged yet.

Only include a rule if it is checkable by reading a diff and you can name
the failure it prevents. All three lists being empty is a normal result.
```

Then, for what you accept:

```
Apply items {N}. Follow "The Skill Writing Contract" (end of the Practice section in
readmes/SKILLS_MANAGEMENT.md) before writing anything into a skill.

- rules going in a skill: edit skills/{name}/SKILL.md, keep the edit under
  15 lines, prefer replacing an existing rule over adding a section, bump
  metadata.version
- rules going in AGENTS.md: add ONE line to the CRITICAL RULES block of the
  right scope. Never touch the auto-invoke tables; they are generated
- if a rule would go in both, it goes in AGENTS.md only, with the detail in
  the skill. Never state the same rule twice

Run ./skills/skill-sync/assets/sync.sh and show me the generated diff.
Commit on this branch as docs(skills): <summary>. No Co-Authored-By trailer.
```

---

## Action 2 - Find Drift

**When**: every Monday.
**What it does**: finds skills pointing at files and symbols that no longer exist.

```
Run ./skills/skill-sync/assets/drift-audit.sh.

Follow "The Skill Writing Contract" (end of the Practice section in
readmes/SKILLS_MANAGEMENT.md) before writing anything into a skill.

The audit covers skills/ AND every AGENTS.md. Treat an AGENTS.md hit as
MORE urgent: those rules are always in context, so a stale one misleads
every agent on every task.

PATH DRIFT - for each STALE line, decide which applies:
  the file moved            -> update the path
  the pattern is gone       -> delete the rule
  the pattern moved         -> rewrite the rule against its new location
Never fix a citation by deleting the link and keeping the prose. A rule
with no verifiable anchor is exactly what this audit exists to prevent.

SYMBOL DRIFT - the pass the script cannot do. Keep it bounded: check the
skills and AGENTS.md files the audit flagged, PLUS anything citing a file
changed this week:
  git diff --name-only HEAD@{7.days.ago}..HEAD
For each, grep for every symbol it names in prose or code blocks (class
names, function names, settings keys, environment variables, management
commands) and report any that no longer exist. Do not sweep all 38 skills;
this runs weekly and must stay cheap.

Report first. Do not edit until I confirm. Clean weeks are the normal
result: say "no drift" and stop.
```

---

## Action 3 - Write a Rule Down

**When**: an agent has gotten the same thing wrong twice.
**What it does**: turns a repeated correction into a rule, in the right file.

```
Agents keep getting {TOPIC} wrong. The failures:
  1. {WHAT IT DID / WHAT IT SHOULD HAVE DONE}
  2. {WHAT IT DID / WHAT IT SHOULD HAVE DONE}

STEP 1 - is the rule ALREADY written? Grep skills/ and every AGENTS.md for
it before assuming it is missing. This is the most common case and it
changes the fix entirely:
  - stated in a skill that never loads for this task -> the TRIGGER is
    wrong, or the rule belongs in AGENTS.md CRITICAL RULES. Move it. Do not
    write it twice
  - stated but not checkable ("handle X properly") -> rewrite it as
    something verifiable from a diff
  - stated, checkable, loaded, and still broken -> it needs a test or a
    registry, not more prose. Say so and stop
If it genuinely does not exist, continue.

STEP 2 - decide where it goes, in two sentences:
  - an existing skill shares this trigger -> extend that one
  - an agent would break it during UNRELATED work -> one line in the
    matching AGENTS.md CRITICAL RULES
  - otherwise -> propose a new skill and justify the distinct trigger

STEP 3 - propose the exact wording and WAIT for my approval before editing.
Follow "The Skill Writing Contract" (end of the Practice section in
readmes/SKILLS_MANAGEMENT.md). In particular: checkable by reading a diff, anchored to a real file, verified
against the code before writing, prohibition first.

After approval: bump metadata.version, run
./skills/skill-sync/assets/sync.sh, show me both diffs. Do not commit yet.
```

---

## Action 4 - Look for a Missing Skill

**When**: you shipped a new subsystem and no skill mentions it.
**What it does**: decides whether it deserves a skill, then writes it.

Two steps on purpose: proposing the trigger and the rule list costs minutes, writing 200 lines
against the wrong trigger costs an hour.

```
We built {SUBSYSTEM} at {PATHS}. No skill covers it. Decide whether one
should exist. Do NOT write it yet.

Answer all six. If any fails, stop and say which, and what to do instead
(a readme, one AGENTS.md line, or nothing):
  1. Will this pattern recur, or was it a one-off?
  2. Does an agent get it wrong by default? Name the actual failure.
  3. Does it deviate from generic best practice?
  4. Is every rule you would write checkable by reading a diff?
  5. Is the trigger distinct and observable, without matching half the
     repository's work?
  6. Is it genuinely uncovered by any skill or AGENTS.md rule? Grep to
     prove it; reject incidental name matches and say which you rejected.

Before proposing a skill, check the other destination: any rule an agent
could break while working on something ELSE, which is invisible from the
file it is editing, belongs as ONE line in that component's AGENTS.md
CRITICAL RULES, not in a skill nobody will load in time. Report those
separately. A new subsystem often yields both: one or two always-loaded
lines, plus a skill for the rest.

If it passes, read the real code, then propose ONLY:
  - the skill name, and why the naming convention picks it
  - the exact "Trigger:" clause, and metadata.scope
  - a numbered list of the rules it would carry, one line each, with the
    file that proves each one was a real failure
Stop there and wait.
```

Then, once you approve the shape:

```
Write skills/{name}/SKILL.md from the approved rule list.

Follow "The Skill Writing Contract" (end of the Practice section in
readmes/SKILLS_MANAGEMENT.md) in full. It governs what goes in, what stays out, and how rules are worded.

Additional rules that apply only when creating a NEW skill:
  - frontmatter: name, description with a literal "Trigger:" clause,
    license, metadata.author, metadata.version, metadata.scope,
    metadata.auto_invoke, allowed-tools
  - body order: When to Use, Critical Rules, Patterns, Commands, Resources
  - anything over ~40 lines of code or config goes in assets/, linked
  - pointers to local docs go in references/
  - under 200 lines

Then: add the row to the skills table in AGENTS.md by hand (sync.sh does
NOT maintain that table), run ./skills/skill-sync/assets/sync.sh, run
./skills/skill-sync/assets/drift-audit.sh --skill {name} and show it clean,
and show me both diffs.
```

---

## Action 5 - Clean House

**When**: every quarter.
**What it does**: finds overlapping, oversized, dead and unrouted skills.

```
Audit the skill set as a whole. Report only, no edits.

For every skill in skills/: line count, metadata.version, last commit date
(git log -1 --format=%ad -- skills/{name}/SKILL.md), and its trigger.

Then report:
  OVERLAP   - pairs whose triggers could match the same task and would give
              different answers
  COLLISION - two skills claiming the same or near-identical auto_invoke
              action. Check the generated tables in every AGENTS.md: an
              action that routes to two skills routes reliably to neither
  DUPLICATE - a rule stated BOTH in a skill and in an AGENTS.md CRITICAL
              RULES block. Two copies drift; say which copy should survive
  MISPLACED - a rule in a skill that breaks silently during unrelated work
              (belongs in AGENTS.md CRITICAL RULES), or an AGENTS.md rule
              that only matters inside one domain (belongs in a skill).
              Always-loaded context is the scarcest resource here
  OVERSIZE  - over 400 lines, with a proposed split ALONG A TRIGGER
              BOUNDARY, never by topic
  UNDERSIZE - under ~20 lines, which should be folded into the skill that
              shares its trigger
  DEAD      - describes something no longer in the codebase. Grep to prove
              it before claiming it
  UNROUTED  - missing metadata.scope or metadata.auto_invoke, therefore
              invisible in every AGENTS.md
  STALE TRIGGER - description names paths, commands or tools that changed

Rank by how likely each is to cause a wrong action, not by untidiness.
A quarter with nothing to report is a good quarter, not a failed audit.
```

---

## Action 6 - Catch Up

**When**: you have not done actions 1 to 5 for months.
**What it does**: three passes, in order. This is the one action that is a project rather than a
single prompt, so it is split into three phases. **Do not skip phase 1.**

**Why the order matters.** After a long gap the dominant problem is usually **absence**, not error:
whole subsystems get built that no skill mentions. Auditing existing rules (phase 2) finds none of that,
so the inventory has to come first. It is also the cheapest pass, and it tells you how much of 6.2
is even worth running.

### Phase 1 - Inventory

What is dead, stale, and uncovered. Run once. Report only. Keep the output; phases 2 and 3 are driven by it.

```
The skills have not been maintained for {PERIOD}. Build the catch-up
worklist. Report only, change nothing.

PASS A - mechanical, no judgement:
  ./skills/skill-sync/assets/drift-audit.sh
  ./skills/skill-sync/assets/sync.sh --dry-run
Report the stale citations, and any skill listed as missing sync metadata.

PASS B - dead premises. For every skill in skills/, check whether the thing
it describes still EXISTS at all. Not whether its rules are right - whether
its subject is there. A skill describing a directory, tool or workflow that
was removed is DEAD, and rewriting its rules is wasted work.

PASS C - uncovered subsystems. This is usually the biggest finding.
  1. List the real subsystems in the codebase: directories under
     each component's source root (its module directories, its background
     job directories, its route or page directories, its top-level
     packages). Derive these from the repository layout; do not assume
     any particular tree.
  2. For each, grep skills/*/SKILL.md for its name.
  3. Report any subsystem with zero real coverage. Ignore incidental
     matches: an SDK check named iam_user_mfa_enabled is not coverage of an
     MFA subsystem. Say which matches you rejected and why.
  4. Give each a size (file count) so I can judge whether it earns a skill.

PASS D - churn classification, to set priority only:
  git log --since="{PERIOD}" --name-only --pretty=format: master \
    | grep -v '^$' | awk -F/ '{if(NF>1) print $1"/"$2; else print $1}' \
    | sort | uniq -c | sort -rn | head -20
Classify each area as MECHANICAL (bulk rename or reformat: breaks citations,
produces no rules), STRUCTURAL (new modules, moved responsibilities), or
BEHAVIOURAL (new conventions, new constraints). Cite a commit for each.
Do not let a large mechanical rename outrank a small behavioural change.

OUTPUT - one worklist table:
  | Item | Type (STALE/DEAD/UNCOVERED/AUDIT) | Scope | Size | Priority | Why |
Ordered by how likely it is to cause a wrong action today, not by size.
```

### Phase 2 - Audit the Existing Rules

One run per scope.

Run once for each scope that phase 1 flagged as worth auditing. Scopes are the components declared in
skill metadata, one per component plus `root`.

```
Audit the skills for scope {SCOPE} against the code as it stands TODAY.
Report only.

Method, and this matters: do NOT read months of diffs. Read each rule and
verify it against the CURRENT code. The diff says what changed; only the
current tree says what is true. A rule can be broken by one commit and
fixed by a later one. Use git log -S"{symbol}" only to find where something
went.

Verify every rule, path, class, function, settings key, environment
variable and command. Grep for each one. A rule you did not grep does not
count as checked.

Classify each finding:
  WRONG       - contradicts current code. Quote the skill line AND the code.
  DEAD        - the subject no longer exists. Propose delete or rewrite.
  STALE PATH  - the rule holds, the citation moved. Give the new path.
  MISSING     - the code imposes a constraint no skill states. Only report
                it if it is checkable by reading a diff and you can name the
                failure it prevents.
  OK          - the count only, not a list.

Report counts first, then details. If a skill is entirely correct, say so
in one line and move on.
```

### Phase 3 - Apply

One category at a time, worst first.

Order: **WRONG**, then **DEAD**, then **STALE PATH**, then **MISSING**. One category per run, so a
bad edit is easy to find and undo.

```
Apply the {CATEGORY} findings for scope {SCOPE}. Only that category.

Follow "The Skill Writing Contract" (end of the Practice section in
readmes/SKILLS_MANAGEMENT.md) before writing anything into a skill.

- edit skills/{name}/SKILL.md only; never .claude/skills, never the
  AGENTS.md auto-invoke tables
- prefer REPLACING an existing rule over adding a section
- for DEAD skills: delete the skill directory and remove its row from the
  skills table in AGENTS.md. Do not "fix" it by softening the rules into
  something vague; a vague skill is a wrong skill that survives review
- bump metadata.version on every touched skill
- touch Trigger and auto_invoke only if routing actually changed

Then:
  ./skills/skill-sync/assets/sync.sh --dry-run    (show me the output)
  ./skills/skill-sync/assets/sync.sh
  ./skills/skill-sync/assets/drift-audit.sh       (must be clean for what
                                                   you touched)

One commit per skill: docs(skills): <what changed>. No Co-Authored-By.
Report what you changed and what you deliberately left alone.
```

**Then, and only then**: take each UNCOVERED subsystem from phase 1 to **Action 4**, one at a time. Most
should fail the six creation tests. Writing skills for the survivors is the real work of a catch-up,
and it is worth more than every fix in phase 3 combined.

---

## The Skill Writing Contract

Every action that **edits** a skill points here rather than restating the rules, so there is one copy
to keep true. Paste this block into any prompt that will write to a skill, or tell the agent to read
this section by name.

A skill is not documentation. Documentation explains a system to someone who wants to understand it.
**A skill constrains an agent that is about to act.** Everything below follows from that difference.

### A Rule Earns Its Place Only If All Four Hold

1. **Checkable by reading a diff.** If compliance cannot be verified from the change itself, it is
   not a rule. "Handle errors properly" is a mood. "Every new view class must appear in
   the route registry in the same commit" is a rule.
2. **It has failed before.** Name the commit, the bug, or the correction. A rule written against an
   imagined failure usually guards the wrong thing, and it is indistinguishable from noise.
3. **The default is wrong.** If a competent engineer would do it anyway, the model already knows.
   Documenting it costs context and returns nothing.
4. **It is stated exactly once.** In this skill, or in an `AGENTS.md`, never both. Two copies drift
   and then contradict each other, and the reader cannot tell which is current.

### Must Be In

- **Prohibitions first.** Lead with what looks right and is wrong. That is where the value is
  concentrated; an agent's default behaviour already covers the rest.
- **Real paths, as links.** Every rule anchored to a file that exists. An unanchored rule cannot be
  audited and cannot be repaired when the code moves.
- **Examples that are copy targets.** The next agent pastes them verbatim. Smallest complete correct
  unit, no ellipses, no `# your logic here`.
- **Tables for decisions.** Which base class, which directory, which of three options.
- **A `Trigger:` clause naming observable conditions** - paths, symbols, commands - not intentions.

### Must Be Out

- **Framework and language defaults.** Document only where this repository deviates.
- **Anything you did not verify against the code.** Not remembered, not inferred from a filename.
- **Rationale and background.** Link it. A skill states what to do; a readme explains why.
- **Troubleshooting sections and Keywords sections.** Routing reads frontmatter, not prose.
- **Web URLs in `references/`.** Local repository paths only.
- **In-flight process.** No branch names, no "we are currently migrating X", no phase numbers.
  Skills are read out of time-context, months later, by an agent that has no idea it finished.
- **Speculation.** "We may later" belongs in an issue.
- **Content duplicated from anywhere else in the repository.** Point at it.

### How to Change One

- **Replace, do not accumulate.** Skills grow by addition and die by bloat. If you are adding a rule
  near one that is now weaker, delete the weaker one.
- **Keep the edit small.** An edit over ~15 lines usually means the skill needed restructuring, not
  extending. Say so instead.
- **Bump `metadata.version`.** Patch for wording, minor for a new rule, major for a reversal or a
  restructure.
- **Touch `Trigger:` and `auto_invoke` only if routing actually changed.** They are not a summary.
- **Never edit the auto-invoke tables in any `AGENTS.md`.** They are generated; run `sync.sh`.
- **Never soften a wrong rule into a vague one.** A vague rule is a wrong rule that survives review.
  Delete it or fix it.
- **Verify before finishing**: `sync.sh` shows table rows only, and `drift-audit.sh` is clean for
  whatever you touched.

### The Test That Matters

Hand the edited skill to a fresh agent with a real task from that domain and watch the diff. Every
time you have to interject with context, that context was missing from the skill. Nothing else -
not how well it reads, not how complete it feels - is evidence that it works.

---

# Reference

Everything below explains why the six actions are what they are. Read it when something does not
behave as expected, when you are deciding where a rule belongs, or when you are setting the
automation up for the first time. You do not need any of it to run the actions above.

---

## 0 - How the Pieces Fit

The introduction covered what skills are and why they need managing. This section covers the
mechanism: where the files live, who owns what, and which part is generated.

### The Layout

```
repository/
│
├── AGENTS.md ...................... ALWAYS loaded, every turn
│   │                                repo-wide rules + skills catalogue
│   │                                + the GENERATED auto-invoke table
│   └── CLAUDE.md -> AGENTS.md ..... symlink. One file, two names, zero drift
│
├── skills/
│   │
│   ├── <skill-name>/
│   │   ├── SKILL.md ............... loaded ONLY when its trigger matches
│   │   ├── assets/ ................ optional: files the agent copies
│   │   └── references/ ............ optional: pointers to local docs
│   │
│   ├── <another-skill>/
│   │   └── SKILL.md
│   │
│   └── skill-sync/assets/
│       ├── sync.sh ................ compiles triggers into the tables
│       ├── citation-check.sh ...... commit-time detection
│       └── drift-audit.sh ......... weekly detection
│
├── <component-a>/
│   ├── AGENTS.md .................. ALWAYS loaded when working in here
│   ├── CLAUDE.md -> AGENTS.md
│   └── src/ ...
│
└── <component-b>/
    ├── AGENTS.md
    ├── CLAUDE.md -> AGENTS.md
    └── src/ ...
```

Two rules the layout encodes. **`AGENTS.md` files are always in context** - one at the root, one per
component - so what they contain is permanently expensive and must be rationed. **Skills are never
in context until selected**, so a skill can be as long as the subject genuinely needs.

`CLAUDE.md` is not a second file. Different agent tools look for different filenames, so each
`AGENTS.md` has a symlink beside it. One copy on disk means the two names cannot disagree:

```
   api/AGENTS.md   <──────┐  the real file
                          │
   api/CLAUDE.md  ────────┘  symlink

   vs. the failure mode this avoids:

   api/AGENTS.md   ......... two real files
   api/CLAUDE.md   ......... two sets of rules, drifting apart, both authoritative
```

### Who Owns the Index Entry

The index physically lives in the `AGENTS.md` files, because those are what is always loaded. But the
knowledge of *when a skill applies* belongs to the skill itself, which is the only thing that knows
what it covers.

That means the same fact must exist in two places. Two copies of anything drift, so the second copy
is **generated, never written**. `sync.sh` reads each skill's declared scope and routes its rows to
the matching file:

```
  SOURCE                              COMPILER            GENERATED OUTPUT
  ────────────────────────────────    ────────────        ────────────────────────

  skills/db-migrations/SKILL.md
    scope: [api]                 ─┐
    auto_invoke:                  │
      "Creating migrations"       │
                                  │
  skills/ui-components/SKILL.md   ├──►  sync.sh  ──┬──►  AGENTS.md          (root)
    scope: [ui]                  ─┤                │       | Action | Skill |
                                  │                │
  skills/commit-style/SKILL.md    │                ├──►  <component-a>/AGENTS.md
    scope: [root]                ─┘                │       | Creating migrations | db-migrations |
                                                   │
                                                   └──►  <component-b>/AGENTS.md
                                                           | ... | ui-components |
```

A skill declares `scope` and `auto_invoke` in its frontmatter; nothing else decides where its rows
land. **The declaration is the source; the table is a build artifact.** You change when a skill
applies by editing the skill and re-running [sync.sh](../../skills/skill-sync/assets/sync.sh), never
by editing the table.

### The Third Tier: Rules That Cannot Wait to Be Selected

Selective loading has one hole. Some rules break during work that has nothing to do with them: an
agent adding an unrelated endpoint cannot be told "load the identity skill first", because it has no
reason to think identity is involved until it has already broken something.

Those few rules live directly in the `AGENTS.md` CRITICAL RULES blocks: always loaded, permanently
paid for, and therefore rationed. Section 2 is the test for which tier a rule belongs to.

### The Moving Parts

| Part | What it is | Who maintains it |
| --- | --- | --- |
| `skills/{name}/SKILL.md` | The rules, and the trigger declaration | You and agents, by hand |
| `AGENTS.md` auto-invoke tables | The generated routing index | `sync.sh`, never by hand |
| `AGENTS.md` CRITICAL RULES | The few always-loaded rules | You, by hand, sparingly |
| `AGENTS.md` skills catalogue table | The human-readable list of skills | You, by hand (`sync.sh` does **not** touch it) |
| `sync.sh` | Compiles declarations into the index | Fixed |
| `citation-check.sh` | Commit-time detection | Fixed, advisory only |
| `drift-audit.sh` | Weekly detection of dead citations | Fixed |
| The six action prompts | How you ask an agent to do any of this | The Practice section |

---

## 1 - How Skills Load

Three layers, and most mistakes come from confusing them.

| Layer | Loaded | Cost | Holds |
| --- | --- | --- | --- |
| One `AGENTS.md` at the repository root, plus one per component | **Always**, every turn | Permanent context | The few rules that must never be missed |
| The `### Auto-invoke Skills` table inside each AGENTS.md | Always | One line per skill | **Routing**: action to skill name |
| `skills/{name}/SKILL.md` body | Only when invoked | Free until used | The payload |

Only the frontmatter `description` and the auto-invoke row sit in context by default. **The
description is a routing decision; the body is the payload.** A well-written skill with a vague
description never loads, which is the most common failure of all.

### Source of Truth

* **Edit `skills/{name}/SKILL.md`.** That is the only copy.
* `.claude/skills` is a **symlink** to `skills/`. Never edit through it.
* The auto-invoke tables in every `AGENTS.md` are **generated output**. Never hand-edit them; they
  are regenerated from skill metadata by [sync.sh](../../skills/skill-sync/assets/sync.sh), which reads
  `$REPO_ROOT/skills` only.

### Why Both `AGENTS.md` and `CLAUDE.md` Exist

Different agent tools look for different filenames. `AGENTS.md` is the cross-vendor convention;
`CLAUDE.md` is what Claude Code looks for. Rather than maintain two files, **`CLAUDE.md` is a symlink
to `AGENTS.md`** in every directory that has one.

One file on disk, several names. Content cannot drift between them, because there is only one copy.
The failure mode to avoid is two *real* files with copied content, which is how most repositories end
up with contradictory instructions. Verify a new symlink resolves: a dangling one is worse than none,
because it looks like the instructions exist.

### Inside an AGENTS.md

Three zones, with different owners. Getting this wrong is the most common way a first installation
breaks:

```
 ┌─ AGENTS.md ────────────────────────────────────────────────────────────┐
 │                                                                        │
 │   # <Component> - Agent Ruleset                                        │
 │                                                                        │
 │   > Skills Reference: ...                            ◄── BY HAND       │
 │                                                          a human list  │
 │                                                                        │
 │   ### Auto-invoke Skills                             ◄── MARKER        │
 │                                                          exact wording │
 │   | Action              | Skill          |           ◄── GENERATED     │
 │   | Creating migrations | db-migrations  |               sync.sh       │
 │   | Writing components  | ui-components  |               overwrites    │
 │                                                          everything    │
 │   ---                                                    from the      │
 │                                                          marker to     │
 │                                                          the next      │
 │                                                          --- or ##     │
 │                                                                        │
 │   ## CRITICAL RULES - NON-NEGOTIABLE                 ◄── BY HAND       │
 │   - NEVER ...                                            ALWAYS in     │
 │   - ALWAYS ...                                           context.      │
 │                                                          Ration it     │
 │   ---                                                                  │
 │                                                                        │
 │   ## TECH STACK / PROJECT STRUCTURE /                ◄── BY HAND       │
 │   ## COMMANDS / QA CHECKLIST                                           │
 │                                                                        │
 └────────────────────────────────────────────────────────────────────────┘
```

Consequences worth internalising:

* **Hand-editing the generated table is pointless** - the next `sync.sh` run erases it. Change
  `auto_invoke` in the skill instead.
* **The `### Auto-invoke Skills` heading is a marker, not a title.** Reword it and the table silently
  stops being maintained.
* **The skills catalogue in the root file is NOT generated.** `sync.sh` never touches it, so a new
  skill has to be added there by hand as well.

### Required Frontmatter

```yaml
---
name: <skill-name>
description: >
  One line on what this skill covers.
  Trigger: the literal conditions under which an agent should load it.
license: <repository license>
metadata:
  author: <org>
  version: "1.0.0"
  scope: [<scope>]                   # one or more component scope names; "root" is repo-wide
  auto_invoke:                       # string or list; becomes the AGENTS.md rows
    - "<action phrase, as it should read in the table>"
allowed-tools: Read, Edit, Write, Glob, Grep, Bash
---
```

`scope` decides **which** AGENTS.md gets the row. `auto_invoke` decides **what the row says**.
A skill missing either is invisible to routing: it exists, and nothing ever loads it.


### A Complete Skill, Annotated

Everything above in one file. This is the shape to copy; a filled-in version lives in
[templates/SKILL.example.md](templates/SKILL.example.md).

```markdown
---
name: acme-identity                   # lowercase-hyphens, matches the directory name
description: >                        # the ONLY part always in context besides the table row
  Session issuing, credential checks and role resolution in the API service.
  Trigger: When working on sign-in, tokens, sessions, or any endpoint under
  the platform namespace.
license: <repository license>
metadata:
  author: <org>
  version: "1.2.0"                    # quoted; unquoted 1.0 is a YAML float
  scope: [api]                        # which AGENTS.md receives the rows
  auto_invoke:                        # each entry becomes one table row
    - "Working on sessions, tokens, or sign-in flows"
    - "Adding endpoints under the platform namespace"
allowed-tools: Read, Edit, Write, Glob, Grep, Bash
---

## When to Use                        <- and what to use INSTEAD for adjacent work

Session and credential work in the API service. For generic framework
patterns use `acme-api`; for data isolation use `acme-tenancy`.

## Critical Rules                     <- prohibitions FIRST; this is the payload

- NEVER call `<low-level issuer>` directly. `<module>:<chokepoint>()` is its
  only caller; it applies the account, membership and policy checks in order.
- NEVER read authorization state from the token except the tenant identifier.
  Role is read from the database on every request.
- ALWAYS register a new view class in [the route-classification test](<path>)
  in the same commit. The sweep fails on anything unclassified.

## Patterns                           <- copy targets: complete, no placeholders

```python
from acme.identity.session import issue_session

tokens = issue_session(user, tenant_id, source="password")
```

## Commands                           <- copy-pasteable

```bash
<the real test command for this area>
```

## Resources                          <- point, never duplicate

- [<local design doc>](<relative path>) - the authentication chapter
- Related skills: `acme-api`, `acme-test-api`
```

Three things to notice. The **`Trigger:` clause names observable conditions** - paths and nouns that
appear in a real request - not "when doing identity work". The **rules are checkable**: a reviewer
can confirm or refute each one by reading a diff. And the **Resources section points rather than
summarises**, so nothing in this file can contradict the document it links to.

---

## 2 - Where a Rule Belongs

Before writing anything, decide which file it goes in. This is the decision people get wrong most
often, and it is not a matter of taste.

| The rule... | Goes in | Because |
| --- | --- | --- |
| Breaks silently during **unrelated** work and is invisible in local code | `AGENTS.md` for that scope | It has to be in context before the agent knows it needs it |
| Only matters once you are already working in that domain | `skills/{name}/SKILL.md` | Loading it on demand costs nothing until it is needed |
| Decides **whether** a skill loads at all | Frontmatter `Trigger:` and `metadata.auto_invoke` | That is the routing surface |
| Is a table mapping actions to skills | Nowhere: run `sync.sh` | It is build output |
| Is process for work currently in flight | A readme, not a skill | Skills are read out of time-context, months later |

Always-loaded context is expensive and permanent. A rule earns a place in `AGENTS.md` only if an
agent could plausibly break it while doing something else entirely. Everything else is a skill.

---

## 3 - When To Create a Skill

Updating an existing skill is cheap. **Creating one is not.** Every new skill adds a row to an
always-loaded auto-invoke table and one more candidate the router has to discriminate between. The
routing table is a shared budget: a skill that does not clearly earn its row makes every other skill
slightly harder to select correctly.

### Create Only When All Six Are True

1. **The pattern recurs.** You will hit it again, in work you cannot yet name. A one-off, however
   painful, is a commit message, not a skill.
2. **An agent gets it wrong by default.** Not "would an agent benefit" but "has an agent, or would
   an agent predictably, produce the wrong diff here". If the model's untrained default is already
   correct, the skill costs context and returns nothing.
3. **It deviates from generic best practice.** Repository-specific, version-specific, or
   convention-specific. Anything a competent engineer would do anyway is already in the model.
4. **The guidance is checkable.** Every rule verifiable by reading a diff. Guidance that cannot be
   checked cannot be followed reliably and cannot be reviewed at all.
5. **The trigger is distinct and observable.** You can state the condition under which it loads
   ("editing under the API's view layer", "adding a compliance JSON") without that condition
   also matching half the repository's work.
6. **It does not already exist.** Check `skills/` **and** the CRITICAL RULES blocks in every
   `AGENTS.md` before writing a line. Duplicated guidance in two places drifts and then contradicts.

### Do Not Create When

| Situation | What to do instead |
| --- | --- |
| Documentation already covers it | Add a `references/` pointer from the relevant skill to the local doc |
| It is a one-off, however painful | Write it in the commit message or a readme |
| It is trivial or self-explanatory | Nothing |
| It is under roughly 20 lines of rules | Add them to an existing skill **with the same trigger** |
| It is process for work in flight | A readme under [readmes/](.) or [my_readmes/](../../my_readmes/) |
| It restates framework or language defaults | Nothing; the model already knows |
| You cannot name a failure it prevents | Nothing yet. Wait until it goes wrong a second time |
| It would break silently in unrelated work | One line in the right `AGENTS.md`, not a skill |

### Split by Trigger, Never by Topic

The most common bad instinct is to split a skill because it covers "two different things". That is a
topic split and it produces two skills that always load together, doubling the routing cost for no
gain.

```
Two bodies of guidance that always load together   -> ONE skill
One skill with two genuinely unrelated triggers    -> TWO skills
A skill over ~400 lines                            -> split, but only along a trigger boundary
A skill under ~20 lines                            -> fold into the skill sharing its trigger
```

The test: write the two `Trigger:` clauses. If you cannot state them so that a task matches exactly
one, it is one skill.

### Extend, Create, or Neither

```
Does an existing skill share the trigger?
  yes -> EXTEND it (bump version, add the rule, done)
  no  -> Would an agent break this rule while doing unrelated work?
           yes -> one line in the matching AGENTS.md CRITICAL RULES
           no  -> Does it satisfy all six creation tests above?
                    yes -> CREATE the skill
                    no  -> readme, commit message, or nothing
```

### Naming and Placement

| Type | Pattern | Examples |
| --- | --- | --- |
| Generic, any project | `{technology}` | `pytest`, `typescript`, `playwright` |
| Project-specific | `{project}-{component}` | `acme-api`, `acme-ui` |
| Testing | `{project}-test-{component}` | `acme-test-api`, `acme-test-sdk` |
| Workflow or action | `{action}-{target}` | `skill-creator`, `skill-sync` |

```
skills/{skill-name}/
├── SKILL.md          Required. Under ~200 lines.
├── assets/           Optional. Things the agent COPIES: templates, schemas, configs, scripts.
└── references/       Optional. Pointers to LOCAL docs the agent should read. Never web URLs.
```

If you are tempted to paste a 200-line config into `SKILL.md`, it belongs in `assets/`. If you are
tempted to summarise an existing document, link it from `references/` instead.

---

## 4 - When To Update

### Immediately, in the Same Pull Request

* A code change made an existing skill **wrong**: a renamed symbol, a moved path, a changed base
  class, a removed helper. The skill is now lying to every agent that loads it.
* You added a **gate that other people's work must satisfy**: a registry that must be appended to, a
  newly required decorator, a constraint test that fails on unclassified additions.
* A **convention decision** landed, for example a display-copy rule or a naming standard.
* **You corrected an agent on the same point twice.** The third time, write it down instead of
  correcting again. This is the cheapest signal you will ever get about what is missing, and it is
  the single best source of skill content.

### On Merge, Not Before

Anything describing code that does not exist yet. A skill documenting a planned endpoint sends
agents looking for a module that is not there, which is worse than saying nothing. Write the skill
edit on the feature branch, merge it with the feature.

### In Batch, at a Milestone

Narrative sections, architecture overviews, operation counts, diagrams. Incremental edits
systematically miss stale prose, so a full re-read at the end of a programme of work is the only
thing that catches it. Use Action 5, widened to the whole area the programme touched.

### Never

In-flight process: branch names, "we are currently on phase 2.5", container policies for one
programme, reporting formats for one series of pull requests. That belongs in a readme under
[readmes/](.) or [my_readmes/](../../my_readmes/), and it expires.

### Delete or Merge When

* The pattern it describes was removed from the codebase.
* Two skills overlap enough that an agent could load either and get different answers.
* Nothing has loaded it in months and you cannot name the task that would.

Deletion is a maintenance action, not a failure. Prune aggressively.

---

## 5 - The Update Procedure

Seven steps. The whole thing takes a few minutes.

```bash
# 1. Edit the source of truth. Only this path.
$EDITOR skills/{name}/SKILL.md

# 2. If the trigger changed, update BOTH routing surfaces:
#      - the frontmatter description's "Trigger:" line
#      - metadata.auto_invoke (and metadata.scope if the audience changed)

# 3. Bump metadata.version. Patch for a wording fix, minor for a new rule,
#    major for a restructure or a reversed rule.

# 4. Regenerate the routing tables. Read the dry run first.
./skills/skill-sync/assets/sync.sh --dry-run
./skills/skill-sync/assets/sync.sh

# 5. Verify the generated diff is table rows and nothing else.
git diff -- $(git ls-files '*AGENTS.md')

# 6. Verify every path the edit cites actually resolves.
./skills/skill-sync/assets/drift-audit.sh --skill {name}

# 7. Commit in the SAME branch as the code change that motivated it.
git add skills/{name}/SKILL.md <the AGENTS.md files sync.sh changed>
git commit -m "docs(skills): {what changed}"
```

Two things worth checking against your own repository's conventions:

* If formatting hooks rewrite staged files in place, **the first `git commit` will abort**. That is
  expected and nothing is committed. Recover by re-staging what the hooks touched and running the
  same commit again. Never bypass the hooks.
* Follow whatever commit-message and trailer conventions the repository already uses. They were
  identified in phase 1 of the bootstrap, if this system was installed that way.

**The rule that makes the whole system work: a skill edit ships in the pull request that caused it.**
A deferred "update the skills" pull request always loses to the next feature, and then the skill is
stale in exactly the window when people are building on the new code.

---

## 6 - The Cadence

The lifecycle at a glance. Detection is cheap, deterministic and frequent. Editing is judged, rare,
and always attached to a branch.

| Trigger | Action | Who | Cost |
| --- | --- | --- | --- |
| Every commit | Citation check prints which skills cite the touched files | Hook, no model | Milliseconds |
| Citation flag fires | Note it, keep working. Do not stop to edit | You | Seconds |
| Pull request opened | An agent reviews the **whole branch diff** and proposes edits | Agent, Action 1 | One call per branch |
| Weekly | Drift audit lists skills citing paths that no longer exist | Scheduled job, no model | Seconds |
| Drift audit non-empty | Repair the listed skills | Agent, Action 2 | Small |
| An agent gets the same thing wrong twice | Write or extend a skill | Agent, Action 3 | Small |
| Milestone or programme end | Full re-read: narrative, counts, diagrams | Agent, Action 5 | Large, rare |
| Quarterly | Consolidation review: overlap, dead skills, oversize | Agent, Action 5 | Medium |

### Why Not "Update Skills After Every Commit"

It is the wrong trigger and it actively degrades the skills. Most commits need no skill change. An
agent asked "should the skills change?" on every commit will answer yes wrongly some fraction of the
time, and you get **churn**: edits invented to justify the question. Skills are scar tissue, and a
mechanism that touches them daily erodes them. You would also pay a model call per commit to answer
"no" almost always.

**Separate detection from editing.** Detect on every commit with a deterministic grep. Edit once per
branch with judgment.

---

## 7 - Automation 1 - The Citation Hook

Every skill cites real paths and symbols. The check is simply: does this commit touch anything a
skill mentions? Deterministic, no model, and advisory only. It never blocks a commit.

The script is [skills/skill-sync/assets/citation-check.sh](../../skills/skill-sync/assets/citation-check.sh).
It takes a changed-file list, skips `skills/` and documentation, greps every `SKILL.md` for each
path and basename, prints what matched, and always exits 0.

It has **two modes, for two different jobs**:

| Invocation | Reads | Used by | Purpose |
| --- | --- | --- | --- |
| `citation-check.sh` | staged changes | the pre-commit hook, automatically | **Warning**: you may have just invalidated a rule |
| `citation-check.sh master...HEAD` | any git range | you, inside Action 1 | **Focusing**: which skills to read at review time |

Same grep, opposite intent. With no argument it reads `git diff --cached`, so running it by hand
outside a commit prints nothing - that is correct behaviour, not a broken script. Pass a range when
you want it at review time.

It is registered in [.pre-commit-config.yaml](../../.pre-commit-config.yaml) under the existing
`repo: local` block:

```yaml
      - id: skill-citation-check
        name: "Skills - citation check (advisory)"
        entry: ./skills/skill-sync/assets/citation-check.sh
        language: system
        pass_filenames: false
        always_run: true
        verbose: true
        priority: 90
```

`verbose: true` is required, otherwise the output is hidden on success. `priority: 90` runs it last,
after the formatters, so the notice is the final thing on screen.

**Expected volume**: a handful of times per week, not per commit. If it fires on nearly every commit
your skills are citing too broadly and should name specific files rather than directories.

### Optional: The Same Signal Inside Claude Code

If you would rather the notice reach the agent than the terminal, the equivalent is a `PostToolUse`
hook in `.claude/settings.json` matching `Bash` calls containing `git commit`. Use the
`update-config` skill to add it. Same script, same advisory contract: it prints, it never blocks.

---

## 8 - Automation 2 - The Weekly Drift Audit

The citation hook catches changes as they happen. The drift audit catches everything that slipped
through, including changes made outside a commit you were watching.

The script is [skills/skill-sync/assets/drift-audit.sh](../../skills/skill-sync/assets/drift-audit.sh).
It extracts every markdown link target and backticked repository path from every `SKILL.md` and
checks that each one still resolves. `--skill <name>` limits it to one skill.

**Why it tries several roots.** Skills cite paths relative to a *component* root, not the repository
root rather than the repository root. A skill may cite `base_views.py` as `api/base_views.py` when
the file actually lives at `api/src/main/api/base_views.py`, because that is how it is referred to
inside that component. A reference counts as live if it resolves under the skill's own directory or
under any configured component root. **The `ROOTS` array at the top of the script is the one thing to edit when installing this
elsewhere.** Without it the audit reports a flood of false positives and stops being worth reading.

Schedule it weekly. Either plain cron:

```cron
0 9 * * 1  cd /path/to/repository && ./skills/skill-sync/assets/drift-audit.sh
```

or, to have an agent both run it and open the repair, use the `schedule` skill to create a Monday
routine whose prompt is **Action 2**.

**Path drift is what this catches. Symbol drift** (a renamed function still named correctly in prose)
needs judgment, so it belongs to Action 2's prompt, not to the script.

---

## 9 - The Rule of Two

The one input that cannot be automated, and the highest-value one: **what agents actually get wrong**,
as opposed to what someone imagined they would.

There is deliberately **no ledger file**. A log you have to stop and fill in mid-task gets three
entries and then dies, and an empty table reads as "nothing goes wrong" when it means "nobody wrote
it down". The trigger is not a file, it is your own irritation:

**When an agent does the same wrong thing a second time, stop and run Action 3 right then.** It takes
about as long as writing a log entry would have, and it produces a rule instead of a row.

One occurrence is an accident. Two is a pattern the skills do not cover.

---

## 10 - The Actions Explained

Each action is shaped the way it is for a reason, and the reasons are not obvious from the prompt
text. This section is what to read when an action does something you did not expect.

| Action | Reports before editing | Why it is shaped that way |
| --- | --- | --- |
| 1 Check a branch | yes | It narrows before it reads. Feeding an agent the whole skill set finds less than feeding it the three skills that cite the changed files |
| 2 Find drift | yes | The script answers the mechanical half exactly; the prompt only handles the half that needs judgement, and stays bounded so a weekly job stays cheap |
| 3 Write a rule down | yes, twice | It checks whether the rule already exists **first**, because that is the common case and it changes the fix completely |
| 4 Look for a missing skill | yes, two stages | Proposing a trigger costs minutes; writing 200 lines against the wrong trigger costs an hour |
| 5 Clean house | report only | It never edits. Consolidation decisions are structural and belong to a person |
| 6 Catch up | yes, per phase | Inventory before audit, because after a long gap the dominant problem is absence, which auditing existing rules cannot find |

**Why they all report first.** An agent asked to "update the skills" produces a large, confident,
unnecessary rewrite - it is trying to be useful, and editing is how it demonstrates that. Splitting
report from apply makes the useful behaviour cheap and the wasteful behaviour visible before it
lands.

**Why none of them run on a schedule you did not choose.** Detection is automated; editing is not.
See section 6.

### A Worked Example: Action 1 End to End

A branch adds an endpoint. Nothing about it looks like skill work.

**1. Run the action.** The agent resolves the range and narrows:

```
$ ./skills/skill-sync/assets/citation-check.sh master...HEAD
  acme-api         serializers.py views.py
  acme-test-api    serializers.py
```

Two skills, not the whole set. It reads those two in full.

**2. The report.** One finding, not a rewrite:

```
WRONG NOW - none.

NEW RULE - the route-classification test now fails on any view class not
listed in one of its sets. Every future endpoint must add itself in the
same commit. This breaks silently during unrelated work and is invisible
from the view file, so it belongs in that component's AGENTS.md CRITICAL
RULES, not in a skill.
  Evidence: the sweep test added alongside the registry.

DEFERRED - none.
```

**3. Accept it.** One line is added to that component's `AGENTS.md`, not a new skill and not a
paragraph:

```markdown
- ALWAYS: register a new view class in the route-classification test in the
  same commit; the sweep fails on anything unclassified.
```

**4. Verify and commit.** `sync.sh` shows no table change - CRITICAL RULES are not generated - and
`drift-audit.sh` confirms the cited path resolves. It ships in **the same branch**, as
`docs(skills): record the route classification requirement`.

Total cost: one report, one line, one commit. That is what a healthy Action 1 looks like. The
failure mode it avoids is the same finding arriving as a new 150-line skill nobody asked for.

---

## 11 - Troubleshooting

Everything here has actually happened. None of it is a bug.

| Symptom | Cause | Fix |
| --- | --- | --- |
| Ran `sync.sh`, the skill is in no table | `metadata.scope` or `metadata.auto_invoke` missing. It is **skipped, not failed** | Add both, rerun. The script lists these under "missing sync metadata" at the end of every run |
| `sync.sh` deleted a table row I wrote by hand | Working as designed. The auto-invoke tables are generated | Put it in `auto_invoke` in the skill and rerun |
| `sync.sh` did not update the skills catalogue | It never does. That table is hand-maintained | Edit the root `AGENTS.md` yourself |
| The commit hook prints nothing | Nothing staged, or the change touches only files no skill cites | Both are correct behaviour |
| The hook prints nothing, ever | Hook manager hides stdout on success | Set the verbose flag. Committing from an IDE sidebar may swallow it entirely; commit from a terminal |
| Hook refuses to run | The hook config file itself is modified but unstaged | Stage or commit the config |
| `citation-check.sh` prints nothing when run by hand | No argument means staged-only, and nothing is staged | Pass a range: `citation-check.sh master...HEAD` |
| Drift audit flags files that plainly exist | `ROOTS` does not include your component roots. Skills cite paths relative to a component, not the repository | Edit the `ROOTS` array at the top of the script |
| Drift audit is clean but a skill is still wrong | It checks paths, not symbols or claims | Symbol drift is Action 2's second pass; wrong claims are Action 6 phase 2 |
| The agent ignored a skill that clearly applies | The `Trigger:` names an intention rather than an observable condition | Rewrite it around paths, symbols and commands; rerun `sync.sh` |
| The agent followed a rule that is no longer true | Exactly the failure this system exists for | Action 2 finds it if a path moved; Action 6.2 if the claim rotted |
| Two skills give contradictory answers | Overlapping triggers, or a rule stated in both a skill and an `AGENTS.md` | Action 5 reports both as OVERLAP and DUPLICATE |
| An action produced a huge rewrite | The report step was skipped | Rerun with the report prompt; never accept an edit you did not see proposed |

**When something is wrong and none of these fit**, check in this order: does the file exist, does
`sync.sh --dry-run` show what you expect, and does the frontmatter parse as YAML. Roughly every
problem so far has been one of those three.

---

## 12 - Reviewing a Proposed Skill Edit

Before accepting an agent's edit, check the following. Most rejected edits fail on the first three.

- [ ] The rule is **checkable by reading a diff**. Not "handle errors properly".
- [ ] The rule **has failed before**, here or elsewhere. If nobody can name the incident, it is
      documentation and does not belong in a skill.
- [ ] It **replaces** rather than accumulates. Skills grow by addition and die by bloat.
- [ ] It **points instead of duplicating**. Duplicated content drifts and then contradicts.
- [ ] Cited paths **resolve** (the drift audit is clean).
- [ ] `metadata.version` is bumped.
- [ ] The `Trigger:` and `auto_invoke` were touched **only if** routing actually changed.
- [ ] `sync.sh` was run and the AGENTS.md diff is table rows only.
- [ ] It contains **no in-flight process**: no branch names, no current-phase references.
- [ ] It does not restate framework defaults the model already knows.
- [ ] It ships in the **same branch** as the code that motivated it.

---

## 13 - Anti-Patterns

| Anti-pattern | Why it fails |
| --- | --- |
| Asking an agent to update skills after every commit | Manufactures churn; most commits need nothing |
| Hand-editing the auto-invoke tables in AGENTS.md | Overwritten on the next `sync.sh` run |
| Editing through `.claude/skills` | It is a symlink; also `sync.sh` reads `skills/` only |
| A separate "update the skills" pull request | Always loses to the next feature |
| Documenting an endpoint before it merges | Sends agents looking for a module that does not exist |
| Copying framework documentation into a skill | Worse than the model's own knowledge, and it drifts |
| Recording current process in a skill | Skills are read out of time-context, months later |
| Leaving a wrong skill in place while you decide | A wrong skill is worse than no skill; delete first, decide later |
| Growing one skill past 400 lines | The agent skims it and the critical rules at the top lose force |
| Creating a skill for a one-off problem | Costs a routing row forever to solve something that happened once |
| Splitting a skill by topic rather than by trigger | Produces two skills that always load together, doubling routing cost |
| Creating a skill before any agent has gotten it wrong | Guidance written against imagined failures rarely matches real ones |
| A "getting started" or "overview" skill | Matches everything, so it either loads constantly or never |

---

## 14 - Quick Reference

```bash
# Regenerate the AGENTS.md auto-invoke tables from skill metadata
./skills/skill-sync/assets/sync.sh --dry-run
./skills/skill-sync/assets/sync.sh
./skills/skill-sync/assets/sync.sh --scope api      # one AGENTS.md only

# Verify every path cited by every skill still resolves
./skills/skill-sync/assets/drift-audit.sh
./skills/skill-sync/assets/drift-audit.sh --skill <skill-name>

# Which skills cite the files I am about to commit
./skills/skill-sync/assets/citation-check.sh

# When was a skill last touched
git log -1 --format='%ad  %s' -- skills/<skill-name>/SKILL.md

# Which skills are unrouted (no scope or no auto_invoke)
grep -L "auto_invoke" skills/*/SKILL.md
```

**Related**: [skills/skill-creator/SKILL.md](../../skills/skill-creator/SKILL.md) for authoring
conventions, [skills/skill-sync/SKILL.md](../../skills/skill-sync/SKILL.md) for the sync mechanism,
[AGENTS.md](../../AGENTS.md) for the repository-wide rules and the full skill index.

---

## 15 - Bootstrapping an Existing Repository

One prompt that installs the whole system into a repository that has none of it.

It ships with **reference files in [templates/](templates/)** - the three working scripts, an
`AGENTS.md` skeleton, an empty skill, a filled-in example skill, and hook snippets per hook manager.
Copy this directory alongside the prompt. The prompt tells the agent to **copy** those files rather
than reconstruct them, because a reconstructed `sync.sh` fails silently: rows in the wrong order, a
skill quietly dropped, a table appended instead of replaced.

It runs in six phases with **stop gates between them**. The gates exist because the two decisions
that matter - what the components are, and which skills are worth having - are cheap to correct
before anything is written and expensive afterwards.

**The discipline that makes or breaks a bootstrap**: create **three to five skills, not twenty**. A
new installation has no history of real failures to draw on, so almost anything written on day one is
guesswork. The system is designed to grow from observed mistakes (Action 3) and from subsystems that
turn out to need coverage (Action 4). Start nearly empty and let it earn its content.

```
Install an agent-skill system in this repository. It currently has none.

<templates> below means the templates/ directory shipped next to this
document. Find it and list it before you start; if it is not there, say so
now rather than improvising later.

Work in six phases. STOP after each phase and wait for my approval. Do not
run ahead. Two decisions are expensive to correct afterwards - the component
list in phase 1 and the skill list in phase 5 - and one is irreversible:
merging into files that already exist, in phase 2.

================================================================
WHAT YOU ARE BUILDING, AND WHY
================================================================

A coding agent knows how software is written in general, and nothing about
how it is written HERE. Skills are how a repository records what it learned
the hard way: the prohibitions, the conventions not discoverable from one
file, the mistake that looks right until production.

Loading all of them at once wastes the context the work needs. Loading none
means they are never read. So the system loads selectively, in three tiers:

  TIER 1  AGENTS.md files          always in context, every turn.
          Only rules an agent could break while doing something ELSE
          entirely, which it cannot discover by reading local code.
          Expensive and permanent: ration them.

  TIER 2  auto-invoke tables       always in context, one line per skill.
          Maps an action to a skill name. GENERATED from skill metadata by
          a script, never written by hand.

  TIER 3  skills/{name}/SKILL.md   loaded only when selected.
          The actual rules. Free until something needs one.

Each skill DECLARES when it applies, in its own frontmatter. A sync script
COMPILES every declaration into the tier-2 tables. The declaration is the
source; the table is a build artifact.

================================================================
PHASE 1 - MAP THE REPOSITORY          (report only, then STOP)
================================================================

Determine, from evidence rather than convention:

  a) The COMPONENTS: independently built or deployed parts, each with its
     own language, build system, and idioms. Look for separate manifests
     (pyproject.toml, package.json, go.mod, Cargo.toml, pom.xml), separate
     Dockerfiles, separate test roots. Report the source root of each.

  b) For each component: language, framework, test runner, package manager,
     lint and format tooling, and how tests are actually run (read the CI
     config and the Makefile or task runner, not the README).

  c) The repository-wide conventions already in force: commit message
     format, branch naming, pull request gates, pre-commit or git hooks.

  d) Any file already doing this job: AGENTS.md, CLAUDE.md, CONTRIBUTING.md,
     .cursorrules, .github/copilot-instructions.md, docs for contributors.
     Quote what they contain. Existing content is migrated, not replaced.

  e) A one-word SCOPE NAME per component, lowercase, for skill metadata
     (for example: root, api, ui, sdk, cli, worker). "root" always exists
     and means repository-wide.

Output: a component table (scope name, source root, language, test command)
and a list of existing instruction files with what they hold.

STOP. I will correct the component list before you create anything.

================================================================
PHASE 2 - CREATE THE STRUCTURE                     (then STOP)
================================================================

Create, using the scope names I approved:

  skills/                          all skills, at the repository root
  AGENTS.md                        repository-wide
  <component-root>/AGENTS.md       one per component

If this repository has ONE component, create only the root AGENTS.md and use
the single scope "root". Per-component files for a single-language repository
are pure overhead.

Copy <templates>/AGENTS.template.md to each location and fill it in from the
phase 1 evidence: TECH STACK, PROJECT STRUCTURE, COMMANDS and QA CHECKLIST
all have answers you already gathered. Do not leave them empty; empty
headings invite padding later.

IF A FILE ALREADY EXISTS at any of those paths, or phase 1 found an
AGENTS.md, CONTRIBUTING.md, .cursorrules or copilot-instructions file:
MERGE, never overwrite. Show me the existing content and your proposed
merge BEFORE writing. Rules already written down were written for a reason,
and losing them is the one irreversible mistake in this whole procedure.

Two things in the template are load-bearing and must not be reworded:

  - the heading "### Auto-invoke Skills", spelled exactly that way. sync.sh
    finds it and replaces everything between it and the next "---" or "##"
  - the table under it is GENERATED. Never hand-edit it

CRITICAL RULES starts EMPTY unless phase 1 found real rules to migrate.
Every line there is in context on every turn forever, so it is the scarcest
space in the repository. A rule belongs there only if an agent could break
it while working on something else entirely and could not discover it from
the file being edited. Everything else is a skill.

Then, next to every AGENTS.md, create a symlink:

    ln -s AGENTS.md CLAUDE.md

Different agent tools look for different filenames. A symlink means one
source of truth under several names, and two copies can never drift apart.
Verify each resolves: a dangling symlink is worse than none, because it
looks like instructions exist. On a platform without symlinks, say so and
ask me rather than creating a second real file.

The root AGENTS.md additionally needs a SKILLS CATALOGUE table, which is
maintained BY HAND and is not touched by sync.sh:

    | Skill | Description | URL |
    | --- | --- | --- |
    | `<name>` | <one line> | [SKILL.md](skills/<name>/SKILL.md) |

STOP. Show me the tree and one AGENTS.md in full.

================================================================
PHASE 3 - INSTALL THE THREE SCRIPTS                (then STOP)
================================================================

COPY them from the templates/ directory shipped with this document. Do NOT
write them from my description: their failure modes are silent, and a
reconstruction that looks right but sorts differently, drops a skill or
appends instead of replacing will not be noticed for months.

    mkdir -p skills/skill-sync/assets
    cp <templates>/sync.sh           skills/skill-sync/assets/
    cp <templates>/citation-check.sh skills/skill-sync/assets/
    cp <templates>/drift-audit.sh    skills/skill-sync/assets/
    chmod +x skills/skill-sync/assets/*.sh

What each one does, so you can tell whether it is behaving:

1) sync.sh - reads name, metadata.scope and metadata.auto_invoke from every
   skills/*/SKILL.md, buckets each (action, skill) pair by scope, maps scope
   to the matching AGENTS.md, sorts rows for stable diffs, and replaces the
   block between "### Auto-invoke Skills" and the next "---" or "##".
   A skill missing scope or auto_invoke is SKIPPED and listed at the end as
   "missing sync metadata" - never silently dropped.
   Flags: --dry-run, --scope <name>.

2) citation-check.sh - advisory, never blocks, always exits 0. With no
   argument it reads the staged file list; with a git range it reads that
   range. It greps every SKILL.md and every AGENTS.md for each changed path
   and basename, and prints what matched.

3) drift-audit.sh - checks that every path cited by every skill AND every
   AGENTS.md still resolves. AGENTS.md hits matter more: those rules are
   always in context, so a stale one misleads every agent on every task.
   Flag: --skill <name>.

THE ONE THING YOU MUST CONFIGURE: the ROOTS array at the top of
drift-audit.sh. Skills cite paths relative to a COMPONENT root, not the
repository root, so set it from the component table in phase 1. Leave it
wrong and the audit reports a flood of false positives and is never read
again.

If the templates directory is genuinely unavailable, say so and write the
scripts - but then prove each behaviour before continuing:
  - a skill with a LIST auto_invoke produces one row per entry
  - a skill missing scope is REPORTED, not dropped
  - running sync.sh twice leaves the file byte-identical
  - drift-audit resolves a path that lives under a component root

Run all three. On an empty install sync.sh reports no skills, drift-audit
reports zero citations, citation-check prints nothing.

STOP.

================================================================
PHASE 4 - REGISTER THE COMMIT HOOK                 (then STOP)
================================================================

Wire citation-check.sh into whatever hook manager this repository already
uses. <templates>/hook-registration.md has a ready snippet for pre-commit,
pre-commit, prek, husky, lefthook and a plain git hook. Requirements:

  - runs on every commit, on the staged set
  - ADVISORY ONLY: it must never block a commit, and must always exit 0
  - its output must be VISIBLE. Many hook managers hide stdout on success;
    if yours has a verbose flag, set it. A hook nobody sees is worthless,
    and nothing is stored anywhere to recover it later
  - runs last, after formatters, so the notice is the final thing on screen
  - COMMITTED to the repository. A hook in .git/hooks/ protects one machine
    and does not survive a clone; use it only if there is no hook manager at
    all, and say so in your report

Verify by staging a file and running the hook. Show me the output.

STOP.

================================================================
PHASE 5 - PROPOSE THE FIRST SKILLS    (report only, then STOP)
================================================================

This is the decision that determines whether the system is useful or
ignored. AIM FOR THREE TO FIVE. Never twenty.

If the evidence supports fewer, install fewer. ZERO IS A VALID RESULT: a
repository can be conventional enough that agents get it right unaided, and
an empty skills/ with working machinery is a correct installation. The
system is designed to grow from observed mistakes, not from a writing
session. Never invent a skill to reach a number.

Find candidates from EVIDENCE, not from topics that seem important:
  - fix and revert commits clustered on the same file or pattern
  - tests that exist only to enforce a convention (registry tests,
    "every X must be registered" tests, constraint-presence tests)
  - registries, allowlists or enum maps that must be appended to when
    something new is added, where forgetting is silent
  - comments containing "do not", "never", "must", "always", "careful"
  - chokepoints: a function everything must route through, where calling
    around it bypasses a control
  - version-specific idioms that contradict the library's own docs
  - conventions only visible after reading several files

Each candidate must pass ALL SIX tests. Report the test number for any
that fails, and drop it:
  1. Will the pattern recur, or was it a one-off?
  2. Does an agent get it wrong BY DEFAULT? Name the actual failure.
  3. Does it deviate from generic best practice? If a competent engineer
     would do it anyway, the model already knows.
  4. Is every rule checkable by reading a diff?
  5. Is the trigger distinct and observable, without matching half the
     repository's work?
  6. Is it genuinely uncovered by an AGENTS.md rule?

Split by TRIGGER, never by topic: guidance that always loads together is
ONE skill; one skill with two unrelated triggers is TWO.

For each survivor report only: proposed name, the exact "Trigger:" clause,
scope, and a numbered list of the rules it would carry with the file or
commit proving each was a real failure.

Also report the REJECTED candidates and which test each failed. That list
is as valuable as the accepted one.

STOP. I will cut this list before you write anything.

================================================================
PHASE 6 - WRITE THE APPROVED SKILLS AND VERIFY
================================================================

For each approved skill, copy <templates>/SKILL.template.md to
skills/{name}/SKILL.md and fill it in. <templates>/SKILL.example.md is a
filled-in skill showing the shape to aim for.

The frontmatter, which decides whether the skill is ever loaded:

    ---
    name: {skill-name}                 # lowercase-hyphens, matches the directory
    description: >
      {what it covers}
      Trigger: {literal observable conditions to load it. Include the
      tokens that appear in a request or a path: real paths, symbol names,
      commands. This string is the routing decision.}
    license: {repository license}
    metadata:
      author: {org}
      version: "1.0.0"                 # quoted: unquoted 1.0 is a YAML float
      scope: [{scope}]                 # which AGENTS.md gets the rows
      auto_invoke:                     # each entry becomes one table row
        - "{action phrase}"
    allowed-tools: Read, Edit, Write, Glob, Grep, Bash
    ---

    ## When to Use          - and what to use INSTEAD for adjacent work
    ## Critical Rules       - ALWAYS / NEVER pairs, prohibitions FIRST
    ## {Patterns}           - minimal complete examples
    ## Commands             - copy-pasteable
    ## Resources            - assets/, references/, related skills

THE WRITING CONTRACT. A skill is not documentation. Documentation explains
a system to someone who wants to understand it; a skill CONSTRAINS AN AGENT
THAT IS ABOUT TO ACT. Everything below follows from that difference.

A rule earns its place only if all four hold:
  1. CHECKABLE by reading a diff. If compliance cannot be verified from the
     change itself it is not a rule. "Handle errors properly" is a mood.
  2. IT HAS FAILED BEFORE. Name the commit, bug or correction. A rule
     written against an imagined failure usually guards the wrong thing.
  3. THE DEFAULT IS WRONG. If a competent engineer would do it anyway, the
     model already knows; documenting it costs context and returns nothing.
  4. STATED EXACTLY ONCE. In this skill or in an AGENTS.md, never both.
     Two copies drift and then contradict, and no reader can tell which is
     current.

Must be IN:
  - prohibitions FIRST: what looks right and is wrong. That is where the
    value is concentrated; the agent's default covers the rest
  - real paths, as links. An unanchored rule cannot be audited or repaired
  - examples as COPY TARGETS: smallest complete correct unit, no ellipses,
    no "your logic here", no pseudo-code
  - tables for decisions: which base class, which directory, which option
  - a "Trigger:" clause naming observable conditions, not intentions
  - verify every rule against the code AS YOU WRITE IT, never from memory

Must be OUT:
  - framework and language defaults. Document only where THIS repository
    deviates from them
  - anything you did not verify. Not remembered, not inferred from a name
  - rationale and background. Link it; a skill states what to do
  - troubleshooting sections and Keywords sections. Routing reads the
    frontmatter, not the body
  - web URLs in references/. Local repository paths only
  - in-flight process: no branch names, no current phase, no migration
    status. Skills are read out of time-context, months later, by an agent
    with no idea it finished
  - speculation. "We may later" belongs in an issue
  - anything duplicated from elsewhere in the repository. Point at it

Size and placement:
  - under 200 lines. Over 400 is a manual and will be skimmed
  - anything over ~40 lines of code or config goes in assets/, linked
  - pointers to local docs go in references/

THE TEST THAT MATTERS: hand the finished skill to a fresh agent with a real
task from that domain and watch the diff. Every time you would have to
interject with context, that context is missing from the skill. How well it
reads is not evidence.

Then:
  - add each skill's row to the SKILLS CATALOGUE in the root AGENTS.md
    (by hand: sync.sh does not maintain that table)
  - run ./skills/skill-sync/assets/sync.sh --dry-run, then for real
  - run ./skills/skill-sync/assets/drift-audit.sh and show it clean
  - VERIFY ROUTING END TO END, which nothing above actually proves:
      grep -h '^| ' $(git ls-files '*AGENTS.md') | sort -u
    every auto_invoke action you wrote must appear, in the AGENTS.md its
    scope points at. A skill missing from that output has no scope or no
    auto_invoke and will never be loaded by anything
  - confirm each SKILL.md frontmatter parses as YAML
  - copy this document AND its templates/ directory into the repository's
    documentation directory, and fix any relative paths inside it for the
    new location
  - COMMIT everything on one branch: the skills, the AGENTS.md files, the
    symlinks, the scripts, the hook registration and the document. Use the
    repository's commit convention, which you identified in phase 1

FINAL REPORT:
  - the component and scope table
  - anything that already existed and how you merged rather than replaced it
  - the tree created, and every symlink verified
  - each skill written, with the evidence for each rule
  - the candidates REJECTED and why
  - generated auto-invoke tables, one diff per AGENTS.md
  - anything you could not verify against the code
  - HOW THIS GROWS, in three lines: Action 1 before every merge, Action 2
    weekly, Action 3 the second time an agent repeats a mistake. Point at
    the Practice section of the document you just installed
```

### After the Bootstrap

The installation is deliberately thin. It fills up through use, not through a second writing session:

| Then | Do |
| --- | --- |
| First pull request | Action 1 |
| First Monday | Action 2 |
| First time an agent repeats a mistake | Action 3 |
| First subsystem nobody documented | Action 4 |
| Three months in | Action 5 |

If after a quarter the skill set has not grown, that is not a failure. It means the repository is
more conventional than it felt, and the agents are getting it right without help.
