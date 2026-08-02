<!--
TEMPLATE - one AGENTS.md per component, plus one at the repository root.

Copy this file to <component-root>/AGENTS.md, fill the angle brackets, delete
this comment, then create the symlink:

    ln -s AGENTS.md CLAUDE.md

Two things in here are load-bearing and must not be reworded:

  1. The heading "### Auto-invoke Skills" is spelled exactly that way. sync.sh
     finds it and replaces everything between it and the next "---" or "##".
     Change the wording and the table stops being generated.

  2. The table under it is GENERATED OUTPUT. Never edit it by hand; edit
     metadata.auto_invoke in the skill and re-run sync.sh.

If this repository already had an AGENTS.md, CONTRIBUTING.md, .cursorrules or
similar, MERGE its real content into the sections below. Do not discard it.
-->

# <Component> - Agent Ruleset

> **Skills Reference**: For detailed patterns, use these skills:
>
> - [`<skill-name>`](../skills/<skill-name>/SKILL.md) - <one line>
>
> Path note: `../skills/...` is correct for a COMPONENT AGENTS.md. In the
> repository-root AGENTS.md the path is `skills/...` with no `../`.

### Auto-invoke Skills

When performing these actions, ALWAYS invoke the corresponding skill FIRST:

| Action | Skill |
| ------ | ----- |

---

## CRITICAL RULES - NON-NEGOTIABLE

<!--
The scarcest resource in this file. Every line here is in context on EVERY
turn, forever, whether or not it is relevant.

A rule belongs here ONLY if an agent could break it while working on something
else entirely, and could not discover it by reading the file it is editing.
Everything else belongs in a skill, which loads on demand and costs nothing
until it does.

Start EMPTY rather than inventing rules. Migrate anything real from the
repository's existing instruction files. Typical members of this section:

  - a registry, allowlist or enum that must be appended to when something is
    added, where forgetting is silent
  - a chokepoint function that must not be called around
  - a command flag whose absence fails silently rather than loudly
  - a naming or placement rule enforced by a test that is not obviously named

Format each as ALWAYS or NEVER, anchored to a real file:

  - NEVER: call `<function>` directly. `<module>:<chokepoint>()` is the only
    caller; it applies <checks> in order.
  - ALWAYS: add a new <thing> to `<registry>` in `<path>` in the same commit.
    `<test name>` fails on anything unregistered.
-->

---

## TECH STACK

<!-- Language, framework, and the versions that actually constrain choices. One line. -->

## PROJECT STRUCTURE

<!-- The tree an agent needs to place a new file correctly. Directories only, annotated. -->

## COMMANDS

<!-- Copy-pasteable. How tests, lint and the dev loop are REALLY run - from CI
     config and the task runner, not from the README. -->

```bash

```

## QA CHECKLIST

<!-- What must be true before a change in this component is proposed. -->

- [ ]
