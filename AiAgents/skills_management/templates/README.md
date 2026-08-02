# Bootstrap Templates

Reference files for [SKILLS_MANAGEMENT.md](../SKILLS_MANAGEMENT.md) section 15, which installs the
skill system into a repository that has none of it.

The three `.sh` files are **symlinks to the live scripts** under
`skills/skill-sync/assets/`, so there is exactly one copy of each in this repository and they cannot
drift. Plain `cp` follows a symlink and copies the content; to copy the whole directory into another
repository use `cp -rL`, which resolves them.

**Copy these. Do not have an agent recreate them from a description.** `sync.sh` in particular is a
few hundred lines of frontmatter parsing and block replacement whose failure modes are all silent:
rows in the wrong order, a skill quietly dropped, a table appended instead of replaced. A
reconstruction that looks right and behaves differently is worse than no script.

| File | What it is | Where it goes |
| --- | --- | --- |
| [sync.sh](sync.sh) | Compiles skill metadata into the auto-invoke tables | `skills/skill-sync/assets/` |
| [citation-check.sh](citation-check.sh) | Commit-time detection, advisory | `skills/skill-sync/assets/` |
| [drift-audit.sh](drift-audit.sh) | Checks every cited path still resolves | `skills/skill-sync/assets/` |
| [AGENTS.template.md](AGENTS.template.md) | One per component, plus the root | `<component>/AGENTS.md` |
| [SKILL.template.md](SKILL.template.md) | An empty skill to fill in | `skills/<name>/SKILL.md` |
| [SKILL.example.md](SKILL.example.md) | A filled-in skill, for shape | reference only |
| [hook-registration.md](hook-registration.md) | Hook snippets per hook manager | reference only |

## The One Thing to Configure

`drift-audit.sh` has a `ROOTS` array near the top:

```bash
ROOTS=(. <component-a>/<its/source/root> <component-b> <component-c>)
```

Skills cite paths relative to a **component** root, not the repository root - `api/base_views.py`
may mean `<component>/<deep/source/root>/base_views.py`. Set this from the component table produced in phase 1
of the bootstrap. Leave it wrong and the audit reports a flood of false positives, and nobody reads
it a second time.

Nothing else in the three scripts is repository-specific.

## Verifying a Fresh Install

```bash
chmod +x skills/skill-sync/assets/*.sh

./skills/skill-sync/assets/sync.sh --dry-run   # lists skills; flags any missing scope/auto_invoke
./skills/skill-sync/assets/drift-audit.sh      # must end "all N citations resolve"
./skills/skill-sync/assets/citation-check.sh   # silent with nothing staged: correct
```

Then the check that actually matters - that routing works end to end:

```bash
# every auto_invoke action must appear in the AGENTS.md its scope points at.
# The awk filter keeps only rows whose Skill column is a `backticked` name,
# which excludes header rows and the hand-maintained skills catalogue.
grep -h '^| ' $(git ls-files '*AGENTS.md') | awk -F'|' '$3 ~ /`/ {print $2"->"$3}' | sort -u
```

If a skill you wrote is missing from that output, it has no `scope` or no `auto_invoke`, and nothing
will ever route to it.
