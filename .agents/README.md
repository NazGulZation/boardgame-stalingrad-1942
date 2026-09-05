# Agent Skills Index

This workspace uses the shared **Agent Skills** format (a `SKILL.md` per
skill directory with YAML frontmatter) so any standards-compatible coding
agent — Claude Code, Cline, Cursor, Windsurf, and others — can load it.

## Available skills

| Skill | Purpose |
|---|---|
| `skills/stalingrad-1942/` | Extend, refactor, and test this Stalingrad 1942 turn-based board wargame (Python + Flask): engine conventions, hard rules, API contract, and validation. |

## How agents find skills here

- The canonical location is `.agents/skills/<skill-name>/SKILL.md`.
- Each skill keeps deep references under `references/` and deterministic
  validation scripts under `scripts/`.
- If your agent reads skills from a different directory (for example
  `.claude/skills/`, `.cline/skills/`, or `.cursor/skills/`), point its
  skills setting at this folder, or add a tiny pointer file next to it that
  reads: `Canonical skill: see .agents/skills/stalingrad-1942/SKILL.md`.

See `skills/stalingrad-1942/SKILL.md` for the full guidance on working in
this codebase.