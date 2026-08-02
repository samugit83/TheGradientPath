<!--
EXAMPLE - a filled-in skill, for reference. Do not copy this file directly;
copy SKILL.template.md and write your own against your own code.

Read this one for the SHAPE of a good skill:

  - every rule names a real file and a real symbol
  - every rule is refutable by reading a diff
  - the trigger names paths and nouns that appear in real requests
  - the examples are complete enough to paste
  - Resources points at documents instead of summarising them
  - it is short. This is a well-covered subsystem in ~90 lines

Paths here are inline code rather than links, because this file is
illustrative and the targets do not exist. In a REAL skill every one of them
is a working markdown link - that is what lets drift-audit.sh catch it when
the code moves.
-->
---
name: example-identity
description: >
  Session minting, tenant domain verification and platform roles in the API.
  Trigger: When working on sign-in, tokens, sessions, SSO, MFA, tenant
  domains, or any endpoint under /api/v1/platform/.
license: Apache-2.0
metadata:
  author: example-org
  version: "1.2.0"
  scope: [api]
  auto_invoke:
    - "Working on sessions, tokens, or sign-in flows"
    - "Adding endpoints under /api/v1/platform/"
    - "Adding or changing MFA behaviour"
allowed-tools: Read, Edit, Write, Glob, Grep, Bash
---

## When to Use

- Anything that issues, refreshes or revokes a session
- Tenant domain claiming and verification
- Platform-role endpoints and permissions

For generic REST framework patterns use `example-drf`; for tenant isolation
and row-level security use `example-api`. This skill covers only the identity
plane.

---

## Critical Rules

- **NEVER** call `generate_tokens` directly.
  `api/identity/session.py` exposes
  `mint_session()`, which is its only caller: it checks account state,
  membership, SSO enforcement and MFA policy in that order. A direct call
  skips all four and issues a session that should not exist.

- **NEVER** read authorization state from `request.auth` other than
  `tenant_id`. `platform_role` is read from the database on every request. A
  token claim is attacker-influenced input, not a permission.

- **ALWAYS** add a new view class to `SESSION_ONLY_VIEWS` or
  `API_KEY_ALLOWED_VIEWS` in
  `api/tests/test_api_key_plane_isolation.py`
  **in the same commit**. `test_every_route_is_classified` sweeps the URL
  configuration and fails on anything unclassified. This is the design
  working, not a nuisance test.

- **ALWAYS** filter `TenantDomain` queries in an auth path by
  `state="verified"`. A pending claim must never route a login.

---

## Patterns

Minting a session, the only supported way:

```python
from api.identity.session import mint_session

tokens = mint_session(user, tenant_id, source="password")
```

Refreshing, which re-reads the database rather than trusting the old token:

```python
from api.identity.session import refresh_session

tokens = refresh_session(refresh_token)
```

---

## Decision Table

| If the endpoint... | Then it inherits |
| ------------------ | ---------------- |
| reads or writes tenant security data | `APIKeyAllowedMixin` first, then `BaseRLSViewSet` |
| touches identity, authority or keys | `BaseRLSViewSet` alone - session only |
| is platform-scoped | `BaseViewSet` behind `IsPlatformUser` |

---

## Commands

```bash
cd api && <test runner> tests/test_identity_session.py
cd api && <the command that regenerates the API schema>
```

---

## Resources

- `docs/api-design.md` - the authentication chapter
- Related skills: `example-api`, `example-test-api`
