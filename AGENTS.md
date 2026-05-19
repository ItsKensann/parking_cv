# SwiftPark Agent Operating Rules

## Authority
- Agents may create branches, issues, worktrees, pull requests, test reports, and draft messages.
- Agents may not merge to main without explicit human approval.
- Agents may not deploy production without explicit human approval.
- Agents may not send client emails without explicit human approval.
- Agents may not submit or edit YC application materials without explicit human approval.
- Agents may not update visual snapshot baselines without explicit human approval.

## Workflow
Every feature must use:
1. GitHub Issue
2. Branch or worktree
3. Pull Request
4. QA report
5. Visual screenshots in Discord
6. Human approval before merge/deploy

## Agent Roles
- PM Agent: clarify goal, write implementation brief, define acceptance criteria.
- Backend Agent: API, database, auth, server behavior, migrations.
- Frontend Agent: UI, visual polish, layout, responsive behavior.
- QA Agent: independent testing, Playwright, screenshots, regression checks.
- Ops Agent: drafts only for email, YC, Reddit, CRM, and client updates.

## Visual QA
- Every feature must run visual tests before approval.
- Screenshots must be posted to Discord.
- Mobile and desktop views must both be checked.
- Do not hide, ignore, or delete failing screenshots.
- Do not update baselines unless the human explicitly approves the visual change.

## Safety
- Never print secrets, tokens, cookies, or private keys.
- Never commit .env files.
- Never bypass branch protections.
- Never use --yolo / dangerous approval bypasses outside a fully isolated sandbox.
- Ask for approval before touching billing, auth, production data, email sending, YC, or deployment.
