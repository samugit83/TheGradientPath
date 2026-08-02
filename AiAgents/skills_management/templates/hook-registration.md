# Registering the Citation Check as a Commit Hook

`citation-check.sh` runs on every commit and reports which skills and `AGENTS.md` files cite the
files being committed. Pick the block matching whatever hook manager the repository already uses.

Four requirements, whichever you pick:

1. **Advisory only.** It must never block a commit, and must always exit 0.
2. **Output must be visible.** Most hook managers hide stdout on success. A hook nobody sees is
   worthless, and nothing is stored anywhere to recover it afterwards.
3. **Runs last**, after formatters, so the notice is the final thing on screen.
4. **Committed to the repository.** A hook that lives only in `.git/hooks/` protects one machine.

---

## prek

Add to the existing `repo: local` block in `.pre-commit-config.yaml`, or create one:

```yaml
  - repo: local
    hooks:
      - id: skill-citation-check
        name: "Skills - citation check (advisory)"
        description: "Reports which agent skills cite the files in this commit. Never blocks."
        entry: ./skills/skill-sync/assets/citation-check.sh
        language: system
        pass_filenames: false
        always_run: true
        verbose: true
        priority: 90
```

`verbose: true` is **required** - without it the output is suppressed on success, which is every
time. `priority: 90` runs it after the formatters.

## pre-commit (upstream)

The same block **without `priority`**, which is a prek extension that upstream pre-commit rejects as
an unexpected key. Ordering there comes from position in the file, so put this hook last:

```yaml
  - repo: local
    hooks:
      - id: skill-citation-check
        name: "Skills - citation check (advisory)"
        entry: ./skills/skill-sync/assets/citation-check.sh
        language: system
        pass_filenames: false
        always_run: true
        verbose: true
```

---

## husky

`.husky/pre-commit`:

```bash
#!/usr/bin/env sh
# ... existing hooks first ...
./skills/skill-sync/assets/citation-check.sh
exit 0
```

The explicit `exit 0` matters: husky aborts the commit on a non-zero status from any line.

---

## lefthook

`lefthook.yml`:

```yaml
pre-commit:
  follow: true
  commands:
    skill-citation-check:
      run: ./skills/skill-sync/assets/citation-check.sh
      priority: 90
```

---

## Plain git hook (fallback)

`.git/hooks/pre-commit`, `chmod +x`:

```bash
#!/usr/bin/env bash
./skills/skill-sync/assets/citation-check.sh
exit 0
```

**This is local-only.** `.git/hooks/` is not committed, so nobody else on the team gets it and it
does not survive a fresh clone. Use it only when the repository has no hook manager at all, and say
so when you report.

---

## Verifying It

Staging must be non-empty, and the staged files must be ones a skill actually mentions:

```bash
git add <a file some skill cites>
<hook manager> run skill-citation-check     # or: git commit --dry-run
```

Expected output:

```
--------------------------------------------------------------------
Skill citation check: this commit touches files cited by:
  api/AGENTS.md                 models.py
  example-api                   models.py
--------------------------------------------------------------------
```

Printing nothing is correct when nothing is staged, or when the staged files are not cited anywhere.
Printing nothing **on every commit, forever** means the output is being suppressed - check the
verbose flag, and try committing from a terminal rather than an IDE sidebar.
