<!--
TEMPLATE - copy to skills/<skill-name>/SKILL.md, fill it in, delete this comment.

See SKILL.example.md in this directory for a filled-in version.

The frontmatter is not decoration. Three fields decide whether this skill is
ever loaded at all:

  description  the "Trigger:" clause is the routing decision. It, and the
               generated table row, are the ONLY parts of this file in context
               before the skill is selected
  scope        which AGENTS.md receives the generated rows
  auto_invoke  each entry becomes one row in that table

A skill missing `scope` or `auto_invoke` is skipped by sync.sh: it exists, and
nothing ever routes to it.
-->
---
name: <skill-name>
description: >
  <One line: what this skill covers.>
  Trigger: <The literal, observable conditions under which to load this.
  Name paths, symbols and commands that will actually appear in a request or
  a diff. "When working on authentication" is an intention and routes badly;
  "when editing files under <path>, or touching <symbol>" routes well.>
license: <repository license, e.g. Apache-2.0>
metadata:
  author: <org>
  version: "1.0.0"
  scope: [<scope>]
  auto_invoke:
    - "<Action phrase, as it should read in the table: 'Creating X', 'Adding Y'>"
allowed-tools: Read, Edit, Write, Glob, Grep, Bash
---

## When to Use

<!-- When this applies, and just as importantly what to use INSTEAD for
     adjacent work. A skill that does not say what it is not prevents overlap. -->

- <condition>

For <adjacent concern>, use `<other-skill>` instead.

---

## Critical Rules

<!-- The payload. PROHIBITIONS FIRST: the move that looks right and is wrong is
     where all the value is; the agent's default already covers the rest.

     Every rule must satisfy all four:
       1. checkable by reading a diff
       2. it has failed before - name the commit, bug or correction
       3. the default is wrong - if a competent engineer would do it anyway,
          the model already knows
       4. stated exactly once - here or in an AGENTS.md, never both
-->

- **NEVER** <the plausible wrong move>. <What to do instead, with the path.>
- **ALWAYS** <the required move>, because <the failure it prevents>.

---

## <Patterns>

<!-- Copy targets. The next agent pastes these verbatim, so: smallest complete
     correct unit, no ellipses, no "your logic here", no pseudo-code.
     Anything over ~40 lines belongs in assets/ and gets linked instead. -->

```<language>

```

---

## Decision Table

<!-- Optional. Use a table wherever the answer is "one of N": which base class,
     which directory, which of three options. Tables are read; prose is skimmed. -->

| If | Then |
| -- | ---- |
|    |      |

---

## Commands

```bash

```

---

## Resources

<!-- POINT, never duplicate. Anything already written elsewhere in the
     repository gets a link. A summary here is a second copy that will drift
     and then contradict the original. -->

- [<local doc>](<relative path>) - <what it covers>
- Related skills: `<skill>`, `<skill>`
