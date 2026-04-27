# Plugin Slash Commands

Exported Wed Apr  8 13:17:31 PDT 2026

## /test-browser

_Source: `cache/every-marketplace/compound-engineering/2.31.0/commands/test-browser.md`_

---
name: test-browser
description: Run browser tests on pages affected by current PR or branch
argument-hint: "[PR number, branch name, or 'current' for current branch]"
---

# Browser Test Command

<command_purpose>Run end-to-end browser tests on pages affected by a PR or branch changes using agent-browser CLI.</command_purpose>

## CRITICAL: Use agent-browser CLI Only

**DO NOT use Chrome MCP tools (mcp__claude-in-chrome__*).**

This command uses the `agent-browser` CLI exclusively. The agent-browser CLI is a Bash-based tool from Vercel that runs headless Chromium. It is NOT the same as Chrome browser automation via MCP.

If you find yourself calling `mcp__claude-in-chrome__*` tools, STOP. Use `agent-browser` Bash commands instead.

## Introduction

<role>QA Engineer specializing in browser-based end-to-end testing</role>

This command tests affected pages in a real browser, catching issues that unit tests miss:
- JavaScript integration bugs
- CSS/layout regressions
- User workflow breakages
- Console errors

## Prerequisites

<requirements>
- Local development server running (e.g., `bin/dev`, `rails server`, `npm run dev`)
- agent-browser CLI installed (see Setup below)
- Git repository with changes to test
</requirements>

## Setup

**Check installation:**
```bash
command -v agent-browser >/dev/null 2>&1 && echo "Installed" || echo "NOT INSTALLED"
```

**Install if needed:**
```bash
npm install -g agent-browser
agent-browser install  # Downloads Chromium (~160MB)
```

See the `agent-browser` skill for detailed usage.

## Main Tasks

### 0. Verify agent-browser Installation

Before starting ANY browser testing, verify agent-browser is installed:

```bash
command -v agent-browser >/dev/null 2>&1 && echo "Ready" || (echo "Installing..." && npm install -g agent-browser && agent-browser install)
```

If installation fails, inform the user and stop.

### 1. Ask Browser Mode

<ask_browser_mode>

Before starting tests, ask user if they want to watch the browser:

Use AskUserQuestion with:
- Question: "Do you want to watch the browser tests run?"
- Options:
  1. **Headed (watch)** - Opens visible browser window so you can see tests run
  2. **Headless (faster)** - Runs in background, faster but invisible

Store the choice and use `--headed` flag when user selects "Headed".

</ask_browser_mode>

### 2. Determine Test Scope

<test_target> $ARGUMENTS </test_target>

<determine_scope>

**If PR number provided:**
```bash
gh pr view [number] --json files -q '.files[].path'
```

**If 'current' or empty:**
```bash
git diff --name-only main...HEAD
```

**If branch name provided:**
```bash
git diff --name-only main...[branch]
```

</determine_scope>

### 3. Map Files to Routes

<file_to_route_mapping>

Map changed files to testable routes:

| File Pattern | Route(s) |
|-------------|----------|
| `app/views/users/*` | `/users`, `/users/:id`, `/users/new` |
| `app/controllers/settings_controller.rb` | `/settings` |
| `app/javascript/controllers/*_controller.js` | Pages using that Stimulus controller |
| `app/components/*_component.rb` | Pages rendering that component |
| `app/views/layouts/*` | All pages (test homepage at minimum) |
| `app/assets/stylesheets/*` | Visual regression on key pages |
| `app/helpers/*_helper.rb` | Pages using that helper |
| `src/app/*` (Next.js) | Corresponding routes |
| `src/components/*` | Pages using those components |

Build a list of URLs to test based on the mapping.

</file_to_route_mapping>

### 4. Verify Server is Running

<check_server>

Before testing, verify the local server is accessible:

```bash
agent-browser open http://localhost:3000
agent-browser snapshot -i
```

If server is not running, inform user:
```markdown
**Server not running**

Please start your development server:
- Rails: `bin/dev` or `rails server`
- Node/Next.js: `npm run dev`

Then run `/test-browser` again.
```

</check_server>

### 5. Test Each Affected Page

<test_pages>

For each affected route, use agent-browser CLI commands (NOT Chrome MCP):

**Step 1: Navigate and capture snapshot**
```bash
agent-browser open "http://localhost:3000/[route]"
agent-browser snapshot -i
```

**Step 2: For headed mode (visual debugging)**
```bash
agent-browser --headed open "http://localhost:3000/[route]"
agent-browser --headed snapshot -i
```

**Step 3: Verify key elements**
- Use `agent-browser snapshot -i` to get interactive elements with refs
- Page title/heading present
- Primary content rendered
- No error messages visible
- Forms have expected fields

**Step 4: Test critical interactions**
```bash
agent-browser click @e1  # Use ref from snapshot
agent-browser snapshot -i
```

**Step 5: Take screenshots**
```bash
agent-browser screenshot page-name.png
agent-browser screenshot --full page-name-full.png  # Full page
```

</test_pages>

### 6. Human Verification (When Required)

<human_verification>

Pause for human input when testing touches:

| Flow Type | What to Ask |
|-----------|-------------|
| OAuth | "Please sign in with [provider] and confirm it works" |
| Email | "Check your inbox for the test email and confirm receipt" |
| Payments | "Complete a test purchase in sandbox mode" |
| SMS | "Verify you received the SMS code" |
| External APIs | "Confirm the [service] integration is working" |

Use AskUserQuestion:
```markdown
**Human Verification Needed**

This test touches the [flow type]. Please:
1. [Action to take]
2. [What to verify]

Did it work correctly?
1. Yes - continue testing
2. No - describe the issue
```

</human_verification>

### 7. Handle Failures

<failure_handling>

When a test fails:

1. **Document the failure:**
   - Screenshot the error state: `agent-browser screenshot error.png`
   - Note the exact reproduction steps

2. **Ask user how to proceed:**
   ```markdown
   **Test Failed: [route]**

   Issue: [description]
   Console errors: [if any]

   How to proceed?
   1. Fix now - I'll help debug and fix
   2. Create todo - Add to todos/ for later
   3. Skip - Continue testing other pages
   ```

3. **If "Fix now":**
   - Investigate the issue
   - Propose a fix
   - Apply fix
   - Re-run the failing test

4. **If "Create todo":**
   - Create `{id}-pending-p1-browser-test-{description}.md`
   - Continue testing

5. **If "Skip":**
   - Log as skipped
   - Continue testing

</failure_handling>

### 8. Test Summary

<test_summary>

After all tests complete, present summary:

```markdown
## Browser Test Results

**Test Scope:** PR #[number] / [branch name]
**Server:** http://localhost:3000

### Pages Tested: [count]

| Route | Status | Notes |
|-------|--------|-------|
| `/users` | Pass | |
| `/settings` | Pass | |
| `/dashboard` | Fail | Console error: [msg] |
| `/checkout` | Skip | Requires payment credentials |

### Console Errors: [count]
- [List any errors found]

### Human Verifications: [count]
- OAuth flow: Confirmed
- Email delivery: Confirmed

### Failures: [count]
- `/dashboard` - [issue description]

### Created Todos: [count]
- `005-pending-p1-browser-test-dashboard-error.md`

### Result: [PASS / FAIL / PARTIAL]
```

</test_summary>

## Quick Usage Examples

```bash
# Test current branch changes
/test-browser

# Test specific PR
/test-browser 847

# Test specific branch
/test-browser feature/new-dashboard
```

## agent-browser CLI Reference

**ALWAYS use these Bash commands. NEVER use mcp__claude-in-chrome__* tools.**

```bash
# Navigation
agent-browser open <url>           # Navigate to URL
agent-browser back                 # Go back
agent-browser close                # Close browser

# Snapshots (get element refs)
agent-browser snapshot -i          # Interactive elements with refs (@e1, @e2, etc.)
agent-browser snapshot -i --json   # JSON output

# Interactions (use refs from snapshot)
agent-browser click @e1            # Click element
agent-browser fill @e1 "text"      # Fill input
agent-browser type @e1 "text"      # Type without clearing
agent-browser press Enter          # Press key

# Screenshots
agent-browser screenshot out.png       # Viewport screenshot
agent-browser screenshot --full out.png # Full page screenshot

# Headed mode (visible browser)
agent-browser --headed open <url>      # Open with visible browser
agent-browser --headed click @e1       # Click in visible browser

# Wait
agent-browser wait @e1             # Wait for element
agent-browser wait 2000            # Wait milliseconds
```

---

## /resolve_parallel

_Source: `cache/every-marketplace/compound-engineering/2.31.0/commands/resolve_parallel.md`_

---
name: resolve_parallel
description: Resolve all TODO comments using parallel processing
argument-hint: "[optional: specific TODO pattern or file]"
disable-model-invocation: true
---

Resolve all TODO comments using parallel processing.

## Workflow

### 1. Analyze

Gather the things todo from above.

### 2. Plan

Create a TodoWrite list of all unresolved items grouped by type.Make sure to look at dependencies that might occur and prioritize the ones needed by others. For example, if you need to change a name, you must wait to do the others. Output a mermaid flow diagram showing how we can do this. Can we do everything in parallel? Do we need to do one first that leads to others in parallel? I'll put the to-dos in the mermaid diagram flow‑wise so the agent knows how to proceed in order.

### 3. Implement (PARALLEL)

Spawn a pr-comment-resolver agent for each unresolved item in parallel.

So if there are 3 comments, it will spawn 3 pr-comment-resolver agents in parallel. liek this

1. Task pr-comment-resolver(comment1)
2. Task pr-comment-resolver(comment2)
3. Task pr-comment-resolver(comment3)

Always run all in parallel subagents/Tasks for each Todo item.

### 4. Commit & Resolve

- Commit changes
- Push to remote

---

## /reproduce-bug

_Source: `cache/every-marketplace/compound-engineering/2.31.0/commands/reproduce-bug.md`_

---
name: reproduce-bug
description: Reproduce and investigate a bug using logs, console inspection, and browser screenshots
argument-hint: "[GitHub issue number]"
disable-model-invocation: true
---

# Reproduce Bug Command

Look at github issue #$ARGUMENTS and read the issue description and comments.

## Phase 1: Log Investigation

Run the following agents in parallel to investigate the bug:

1. Task rails-console-explorer(issue_description)
2. Task appsignal-log-investigator(issue_description)

Think about the places it could go wrong looking at the codebase. Look for logging output we can look for.

Run the agents again to find any logs that could help us reproduce the bug.

Keep running these agents until you have a good idea of what is going on.

## Phase 2: Visual Reproduction with Playwright

If the bug is UI-related or involves user flows, use Playwright to visually reproduce it:

### Step 1: Verify Server is Running

```
mcp__plugin_compound-engineering_pw__browser_navigate({ url: "http://localhost:3000" })
mcp__plugin_compound-engineering_pw__browser_snapshot({})
```

If server not running, inform user to start `bin/dev`.

### Step 2: Navigate to Affected Area

Based on the issue description, navigate to the relevant page:

```
mcp__plugin_compound-engineering_pw__browser_navigate({ url: "http://localhost:3000/[affected_route]" })
mcp__plugin_compound-engineering_pw__browser_snapshot({})
```

### Step 3: Capture Screenshots

Take screenshots at each step of reproducing the bug:

```
mcp__plugin_compound-engineering_pw__browser_take_screenshot({ filename: "bug-[issue]-step-1.png" })
```

### Step 4: Follow User Flow

Reproduce the exact steps from the issue:

1. **Read the issue's reproduction steps**
2. **Execute each step using Playwright:**
   - `browser_click` for clicking elements
   - `browser_type` for filling forms
   - `browser_snapshot` to see the current state
   - `browser_take_screenshot` to capture evidence

3. **Check for console errors:**
   ```
   mcp__plugin_compound-engineering_pw__browser_console_messages({ level: "error" })
   ```

### Step 5: Capture Bug State

When you reproduce the bug:

1. Take a screenshot of the bug state
2. Capture console errors
3. Document the exact steps that triggered it

```
mcp__plugin_compound-engineering_pw__browser_take_screenshot({ filename: "bug-[issue]-reproduced.png" })
```

## Phase 3: Document Findings

**Reference Collection:**

- [ ] Document all research findings with specific file paths (e.g., `app/services/example_service.rb:42`)
- [ ] Include screenshots showing the bug reproduction
- [ ] List console errors if any
- [ ] Document the exact reproduction steps

## Phase 4: Report Back

Add a comment to the issue with:

1. **Findings** - What you discovered about the cause
2. **Reproduction Steps** - Exact steps to reproduce (verified)
3. **Screenshots** - Visual evidence of the bug (upload captured screenshots)
4. **Relevant Code** - File paths and line numbers
5. **Suggested Fix** - If you have one

---

## /generate_command

_Source: `cache/every-marketplace/compound-engineering/2.31.0/commands/generate_command.md`_

---
name: generate_command
description: Create a new custom slash command following conventions and best practices
argument-hint: "[command purpose and requirements]"
disable-model-invocation: true
---

# Create a Custom Claude Code Command

Create a new slash command in `.claude/commands/` for the requested task.

## Goal

#$ARGUMENTS

## Key Capabilities to Leverage

**File Operations:**
- Read, Edit, Write - modify files precisely
- Glob, Grep - search codebase
- MultiEdit - atomic multi-part changes

**Development:**
- Bash - run commands (git, tests, linters)
- Task - launch specialized agents for complex tasks
- TodoWrite - track progress with todo lists

**Web & APIs:**
- WebFetch, WebSearch - research documentation
- GitHub (gh cli) - PRs, issues, reviews
- Playwright - browser automation, screenshots

**Integrations:**
- AppSignal - logs and monitoring
- Context7 - framework docs
- Stripe, Todoist, Featurebase (if relevant)

## Best Practices

1. **Be specific and clear** - detailed instructions yield better results
2. **Break down complex tasks** - use step-by-step plans
3. **Use examples** - reference existing code patterns
4. **Include success criteria** - tests pass, linting clean, etc.
5. **Think first** - use "think hard" or "plan" keywords for complex problems
6. **Iterate** - guide the process step by step

## Required: YAML Frontmatter

**EVERY command MUST start with YAML frontmatter:**

```yaml
---
name: command-name
description: Brief description of what this command does (max 100 chars)
argument-hint: "[what arguments the command accepts]"
---
```

**Fields:**
- `name`: Lowercase command identifier (used internally)
- `description`: Clear, concise summary of command purpose
- `argument-hint`: Shows user what arguments are expected (e.g., `[file path]`, `[PR number]`, `[optional: format]`)

## Structure Your Command

```markdown
# [Command Name]

[Brief description of what this command does]

## Steps

1. [First step with specific details]
   - Include file paths, patterns, or constraints
   - Reference existing code if applicable

2. [Second step]
   - Use parallel tool calls when possible
   - Check/verify results

3. [Final steps]
   - Run tests
   - Lint code
   - Commit changes (if appropriate)

## Success Criteria

- [ ] Tests pass
- [ ] Code follows style guide
- [ ] Documentation updated (if needed)
```

## Tips for Effective Commands

- **Use $ARGUMENTS** placeholder for dynamic inputs
- **Reference CLAUDE.md** patterns and conventions
- **Include verification steps** - tests, linting, visual checks
- **Be explicit about constraints** - don't modify X, use pattern Y
- **Use XML tags** for structured prompts: `<task>`, `<requirements>`, `<constraints>`

## Example Pattern

```markdown
Implement #$ARGUMENTS following these steps:

1. Research existing patterns
   - Search for similar code using Grep
   - Read relevant files to understand approach

2. Plan the implementation
   - Think through edge cases and requirements
   - Consider test cases needed

3. Implement
   - Follow existing code patterns (reference specific files)
   - Write tests first if doing TDD
   - Ensure code follows CLAUDE.md conventions

4. Verify
   - Run tests: `bin/rails test`
   - Run linter: `bundle exec standardrb`
   - Check changes with git diff

5. Commit (optional)
   - Stage changes
   - Write clear commit message
```

## Creating the Command File

1. **Create the file** at `.claude/commands/[name].md` (subdirectories like `workflows/` supported)
2. **Start with YAML frontmatter** (see section above)
3. **Structure the command** using the template above
4. **Test the command** by using it with appropriate arguments

## Command File Template

```markdown
---
name: command-name
description: What this command does
argument-hint: "[expected arguments]"
---

# Command Title

Brief introduction of what the command does and when to use it.

## Workflow

### Step 1: [First Major Step]

Details about what to do.

### Step 2: [Second Major Step]

Details about what to do.

## Success Criteria

- [ ] Expected outcome 1
- [ ] Expected outcome 2
```

---

## /changelog

_Source: `cache/every-marketplace/compound-engineering/2.31.0/commands/changelog.md`_

---
name: changelog
description: Create engaging changelogs for recent merges to main branch
argument-hint: "[optional: daily|weekly, or time period in days]"
disable-model-invocation: true
---

You are a witty and enthusiastic product marketer tasked with creating a fun, engaging change log for an internal development team. Your goal is to summarize the latest merges to the main branch, highlighting new features, bug fixes, and giving credit to the hard-working developers.

## Time Period

- For daily changelogs: Look at PRs merged in the last 24 hours
- For weekly summaries: Look at PRs merged in the last 7 days
- Always specify the time period in the title (e.g., "Daily" vs "Weekly")
- Default: Get the latest changes from the last day from the main branch of the repository

## PR Analysis

Analyze the provided GitHub changes and related issues. Look for:

1. New features that have been added
2. Bug fixes that have been implemented
3. Any other significant changes or improvements
4. References to specific issues and their details
5. Names of contributors who made the changes
6. Use gh cli to lookup the PRs as well and the description of the PRs
7. Check PR labels to identify feature type (feature, bug, chore, etc.)
8. Look for breaking changes and highlight them prominently
9. Include PR numbers for traceability
10. Check if PRs are linked to issues and include issue context

## Content Priorities

1. Breaking changes (if any) - MUST be at the top
2. User-facing features
3. Critical bug fixes
4. Performance improvements
5. Developer experience improvements
6. Documentation updates

## Formatting Guidelines

Now, create a change log summary with the following guidelines:

1. Keep it concise and to the point
2. Highlight the most important changes first
3. Group similar changes together (e.g., all new features, all bug fixes)
4. Include issue references where applicable
5. Mention the names of contributors, giving them credit for their work
6. Add a touch of humor or playfulness to make it engaging
7. Use emojis sparingly to add visual interest
8. Keep total message under 2000 characters for Discord
9. Use consistent emoji for each section
10. Format code/technical terms in backticks
11. Include PR numbers in parentheses (e.g., "Fixed login bug (#123)")

## Deployment Notes

When relevant, include:

- Database migrations required
- Environment variable updates needed
- Manual intervention steps post-deploy
- Dependencies that need updating

Your final output should be formatted as follows:

<change_log>

# 🚀 [Daily/Weekly] Change Log: [Current Date]

## 🚨 Breaking Changes (if any)

[List any breaking changes that require immediate attention]

## 🌟 New Features

[List new features here with PR numbers]

## 🐛 Bug Fixes

[List bug fixes here with PR numbers]

## 🛠️ Other Improvements

[List other significant changes or improvements]

## 🙌 Shoutouts

[Mention contributors and their contributions]

## 🎉 Fun Fact of the Day

[Include a brief, work-related fun fact or joke]

</change_log>

## Style Guide Review

Now review the changelog using the EVERY_WRITE_STYLE.md file and go one by one to make sure you are following the style guide. Use multiple agents, run in parallel to make it faster.

Remember, your final output should only include the content within the <change_log> tags. Do not include any of your thought process or the original data in the output.

## Discord Posting (Optional)

You can post changelogs to Discord by adding your own webhook URL:

```
# Set your Discord webhook URL
DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/YOUR_WEBHOOK_ID/YOUR_WEBHOOK_TOKEN"

# Post using curl
curl -H "Content-Type: application/json" \
  -d "{\"content\": \"{{CHANGELOG}}\"}" \
  $DISCORD_WEBHOOK_URL
```

To get a webhook URL, go to your Discord server → Server Settings → Integrations → Webhooks → New Webhook.

## Error Handling

- If no changes in the time period, post a "quiet day" message: "🌤️ Quiet day! No new changes merged."
- If unable to fetch PR details, list the PR numbers for manual review
- Always validate message length before posting to Discord (max 2000 chars)

## Schedule Recommendations

- Run daily at 6 AM NY time for previous day's changes
- Run weekly summary on Mondays for the previous week
- Special runs after major releases or deployments

## Audience Considerations

Adjust the tone and detail level based on the channel:

- **Dev team channels**: Include technical details, performance metrics, code snippets
- **Product team channels**: Focus on user-facing changes and business impact
- **Leadership channels**: Highlight progress on key initiatives and blockers

---

## /deploy-docs

_Source: `cache/every-marketplace/compound-engineering/2.31.0/commands/deploy-docs.md`_

---
name: deploy-docs
description: Validate and prepare documentation for GitHub Pages deployment
disable-model-invocation: true
---

# Deploy Documentation Command

Validate the documentation site and prepare it for GitHub Pages deployment.

## Step 1: Validate Documentation

Run these checks:

```bash
# Count components
echo "Agents: $(ls plugins/compound-engineering/agents/*.md | wc -l)"
echo "Commands: $(ls plugins/compound-engineering/commands/*.md | wc -l)"
echo "Skills: $(ls -d plugins/compound-engineering/skills/*/ 2>/dev/null | wc -l)"

# Validate JSON
cat .claude-plugin/marketplace.json | jq . > /dev/null && echo "✓ marketplace.json valid"
cat plugins/compound-engineering/.claude-plugin/plugin.json | jq . > /dev/null && echo "✓ plugin.json valid"

# Check all HTML files exist
for page in index agents commands skills mcp-servers changelog getting-started; do
  if [ -f "plugins/compound-engineering/docs/pages/${page}.html" ] || [ -f "plugins/compound-engineering/docs/${page}.html" ]; then
    echo "✓ ${page}.html exists"
  else
    echo "✗ ${page}.html MISSING"
  fi
done
```

## Step 2: Check for Uncommitted Changes

```bash
git status --porcelain plugins/compound-engineering/docs/
```

If there are uncommitted changes, warn the user to commit first.

## Step 3: Deployment Instructions

Since GitHub Pages deployment requires a workflow file with special permissions, provide these instructions:

### First-time Setup

1. Create `.github/workflows/deploy-docs.yml` with the GitHub Pages workflow
2. Go to repository Settings > Pages
3. Set Source to "GitHub Actions"

### Deploying

After merging to `main`, the docs will auto-deploy. Or:

1. Go to Actions tab
2. Select "Deploy Documentation to GitHub Pages"
3. Click "Run workflow"

### Workflow File Content

```yaml
name: Deploy Documentation to GitHub Pages

on:
  push:
    branches: [main]
    paths:
      - 'plugins/compound-engineering/docs/**'
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: "pages"
  cancel-in-progress: false

jobs:
  deploy:
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/configure-pages@v4
      - uses: actions/upload-pages-artifact@v3
        with:
          path: 'plugins/compound-engineering/docs'
      - uses: actions/deploy-pages@v4
```

## Step 4: Report Status

Provide a summary:

```
## Deployment Readiness

✓ All HTML pages present
✓ JSON files valid
✓ Component counts match

### Next Steps
- [ ] Commit any pending changes
- [ ] Push to main branch
- [ ] Verify GitHub Pages workflow exists
- [ ] Check deployment at https://everyinc.github.io/every-marketplace/
```

---

## /triage

_Source: `cache/every-marketplace/compound-engineering/2.31.0/commands/triage.md`_

---
name: triage
description: Triage and categorize findings for the CLI todo system
argument-hint: "[findings list or source type]"
disable-model-invocation: true
---

- First set the /model to Haiku
- Then read all pending todos in the todos/ directory

Present all findings, decisions, or issues here one by one for triage. The goal is to go through each item and decide whether to add it to the CLI todo system.

**IMPORTANT: DO NOT CODE ANYTHING DURING TRIAGE!**

This command is for:

- Triaging code review findings
- Processing security audit results
- Reviewing performance analysis
- Handling any other categorized findings that need tracking

## Workflow

### Step 1: Present Each Finding

For each finding, present in this format:

```
---
Issue #X: [Brief Title]

Severity: 🔴 P1 (CRITICAL) / 🟡 P2 (IMPORTANT) / 🔵 P3 (NICE-TO-HAVE)

Category: [Security/Performance/Architecture/Bug/Feature/etc.]

Description:
[Detailed explanation of the issue or improvement]

Location: [file_path:line_number]

Problem Scenario:
[Step by step what's wrong or could happen]

Proposed Solution:
[How to fix it]

Estimated Effort: [Small (< 2 hours) / Medium (2-8 hours) / Large (> 8 hours)]

---
Do you want to add this to the todo list?
1. yes - create todo file
2. next - skip this item
3. custom - modify before creating
```

### Step 2: Handle User Decision

**When user says "yes":**

1. **Update existing todo file** (if it exists) or **Create new filename:**

   If todo already exists (from code review):

   - Rename file from `{id}-pending-{priority}-{desc}.md` → `{id}-ready-{priority}-{desc}.md`
   - Update YAML frontmatter: `status: pending` → `status: ready`
   - Keep issue_id, priority, and description unchanged

   If creating new todo:

   ```
   {next_id}-ready-{priority}-{brief-description}.md
   ```

   Priority mapping:

   - 🔴 P1 (CRITICAL) → `p1`
   - 🟡 P2 (IMPORTANT) → `p2`
   - 🔵 P3 (NICE-TO-HAVE) → `p3`

   Example: `042-ready-p1-transaction-boundaries.md`

2. **Update YAML frontmatter:**

   ```yaml
   ---
   status: ready # IMPORTANT: Change from "pending" to "ready"
   priority: p1 # or p2, p3 based on severity
   issue_id: "042"
   tags: [category, relevant-tags]
   dependencies: []
   ---
   ```

3. **Populate or update the file:**

   ```yaml
   # [Issue Title]

   ## Problem Statement
   [Description from finding]

   ## Findings
   - [Key discoveries]
   - Location: [file_path:line_number]
   - [Scenario details]

   ## Proposed Solutions

   ### Option 1: [Primary solution]
   - **Pros**: [Benefits]
   - **Cons**: [Drawbacks if any]
   - **Effort**: [Small/Medium/Large]
   - **Risk**: [Low/Medium/High]

   ## Recommended Action
   [Filled during triage - specific action plan]

   ## Technical Details
   - **Affected Files**: [List files]
   - **Related Components**: [Components affected]
   - **Database Changes**: [Yes/No - describe if yes]

   ## Resources
   - Original finding: [Source of this issue]
   - Related issues: [If any]

   ## Acceptance Criteria
   - [ ] [Specific success criteria]
   - [ ] Tests pass
   - [ ] Code reviewed

   ## Work Log

   ### {date} - Approved for Work
   **By:** Claude Triage System
   **Actions:**
   - Issue approved during triage session
   - Status changed from pending → ready
   - Ready to be picked up and worked on

   **Learnings:**
   - [Context and insights]

   ## Notes
   Source: Triage session on {date}
   ```

4. **Confirm approval:** "✅ Approved: `{new_filename}` (Issue #{issue_id}) - Status: **ready** → Ready to work on"

**When user says "next":**

- **Delete the todo file** - Remove it from todos/ directory since it's not relevant
- Skip to the next item
- Track skipped items for summary

**When user says "custom":**

- Ask what to modify (priority, description, details)
- Update the information
- Present revised version
- Ask again: yes/next/custom

### Step 3: Continue Until All Processed

- Process all items one by one
- Track using TodoWrite for visibility
- Don't wait for approval between items - keep moving

### Step 4: Final Summary

After all items processed:

````markdown
## Triage Complete

**Total Items:** [X] **Todos Approved (ready):** [Y] **Skipped:** [Z]

### Approved Todos (Ready for Work):

- `042-ready-p1-transaction-boundaries.md` - Transaction boundary issue
- `043-ready-p2-cache-optimization.md` - Cache performance improvement ...

### Skipped Items (Deleted):

- Item #5: [reason] - Removed from todos/
- Item #12: [reason] - Removed from todos/

### Summary of Changes Made:

During triage, the following status updates occurred:

- **Pending → Ready:** Filenames and frontmatter updated to reflect approved status
- **Deleted:** Todo files for skipped findings removed from todos/ directory
- Each approved file now has `status: ready` in YAML frontmatter

### Next Steps:

1. View approved todos ready for work:
   ```bash
   ls todos/*-ready-*.md
   ```
````

2. Start work on approved items:

   ```bash
   /resolve_todo_parallel  # Work on multiple approved items efficiently
   ```

3. Or pick individual items to work on

4. As you work, update todo status:
   - Ready → In Progress (in your local context as you work)
   - In Progress → Complete (rename file: ready → complete, update frontmatter)

```

## Example Response Format

```

---

Issue #5: Missing Transaction Boundaries for Multi-Step Operations

Severity: 🔴 P1 (CRITICAL)

Category: Data Integrity / Security

Description: The google_oauth2_connected callback in GoogleOauthCallbacks concern performs multiple database operations without transaction protection. If any step fails midway, the database is left in an inconsistent state.

Location: app/controllers/concerns/google_oauth_callbacks.rb:13-50

Problem Scenario:

1. User.update succeeds (email changed)
2. Account.save! fails (validation error)
3. Result: User has changed email but no associated Account
4. Next login attempt fails completely

Operations Without Transaction:

- User confirmation (line 13)
- Waitlist removal (line 14)
- User profile update (line 21-23)
- Account creation (line 28-37)
- Avatar attachment (line 39-45)
- Journey creation (line 47)

Proposed Solution: Wrap all operations in ApplicationRecord.transaction do ... end block

Estimated Effort: Small (30 minutes)

---

Do you want to add this to the todo list?

1. yes - create todo file
2. next - skip this item
3. custom - modify before creating

```

## Important Implementation Details

### Status Transitions During Triage

**When "yes" is selected:**
1. Rename file: `{id}-pending-{priority}-{desc}.md` → `{id}-ready-{priority}-{desc}.md`
2. Update YAML frontmatter: `status: pending` → `status: ready`
3. Update Work Log with triage approval entry
4. Confirm: "✅ Approved: `{filename}` (Issue #{issue_id}) - Status: **ready**"

**When "next" is selected:**
1. Delete the todo file from todos/ directory
2. Skip to next item
3. No file remains in the system

### Progress Tracking

Every time you present a todo as a header, include:
- **Progress:** X/Y completed (e.g., "3/10 completed")
- **Estimated time remaining:** Based on how quickly you're progressing
- **Pacing:** Monitor time per finding and adjust estimate accordingly

Example:
```

Progress: 3/10 completed | Estimated time: ~2 minutes remaining

```

### Do Not Code During Triage

- ✅ Present findings
- ✅ Make yes/next/custom decisions
- ✅ Update todo files (rename, frontmatter, work log)
- ❌ Do NOT implement fixes or write code
- ❌ Do NOT add detailed implementation details
- ❌ That's for /resolve_todo_parallel phase
```

When done give these options

```markdown
What would you like to do next?

1. run /resolve_todo_parallel to resolve the todos
2. commit the todos
3. nothing, go chill
```

---

## /deepen-plan

_Source: `cache/every-marketplace/compound-engineering/2.31.0/commands/deepen-plan.md`_

---
name: deepen-plan
description: Enhance a plan with parallel research agents for each section to add depth, best practices, and implementation details
argument-hint: "[path to plan file]"
---

# Deepen Plan - Power Enhancement Mode

## Introduction

**Note: The current year is 2026.** Use this when searching for recent documentation and best practices.

This command takes an existing plan (from `/workflows:plan`) and enhances each section with parallel research agents. Each major element gets its own dedicated research sub-agent to find:
- Best practices and industry patterns
- Performance optimizations
- UI/UX improvements (if applicable)
- Quality enhancements and edge cases
- Real-world implementation examples

The result is a deeply grounded, production-ready plan with concrete implementation details.

## Plan File

<plan_path> #$ARGUMENTS </plan_path>

**If the plan path above is empty:**
1. Check for recent plans: `ls -la docs/plans/`
2. Ask the user: "Which plan would you like to deepen? Please provide the path (e.g., `docs/plans/2026-01-15-feat-my-feature-plan.md`)."

Do not proceed until you have a valid plan file path.

## Main Tasks

### 1. Parse and Analyze Plan Structure

<thinking>
First, read and parse the plan to identify each major section that can be enhanced with research.
</thinking>

**Read the plan file and extract:**
- [ ] Overview/Problem Statement
- [ ] Proposed Solution sections
- [ ] Technical Approach/Architecture
- [ ] Implementation phases/steps
- [ ] Code examples and file references
- [ ] Acceptance criteria
- [ ] Any UI/UX components mentioned
- [ ] Technologies/frameworks mentioned (Rails, React, Python, TypeScript, etc.)
- [ ] Domain areas (data models, APIs, UI, security, performance, etc.)

**Create a section manifest:**
```
Section 1: [Title] - [Brief description of what to research]
Section 2: [Title] - [Brief description of what to research]
...
```

### 2. Discover and Apply Available Skills

<thinking>
Dynamically discover all available skills and match them to plan sections. Don't assume what skills exist - discover them at runtime.
</thinking>

**Step 1: Discover ALL available skills from ALL sources**

```bash
# 1. Project-local skills (highest priority - project-specific)
ls .claude/skills/

# 2. User's global skills (~/.claude/)
ls ~/.claude/skills/

# 3. compound-engineering plugin skills
ls ~/.claude/plugins/cache/*/compound-engineering/*/skills/

# 4. ALL other installed plugins - check every plugin for skills
find ~/.claude/plugins/cache -type d -name "skills" 2>/dev/null

# 5. Also check installed_plugins.json for all plugin locations
cat ~/.claude/plugins/installed_plugins.json
```

**Important:** Check EVERY source. Don't assume compound-engineering is the only plugin. Use skills from ANY installed plugin that's relevant.

**Step 2: For each discovered skill, read its SKILL.md to understand what it does**

```bash
# For each skill directory found, read its documentation
cat [skill-path]/SKILL.md
```

**Step 3: Match skills to plan content**

For each skill discovered:
- Read its SKILL.md description
- Check if any plan sections match the skill's domain
- If there's a match, spawn a sub-agent to apply that skill's knowledge

**Step 4: Spawn a sub-agent for EVERY matched skill**

**CRITICAL: For EACH skill that matches, spawn a separate sub-agent and instruct it to USE that skill.**

For each matched skill:
```
Task general-purpose: "You have the [skill-name] skill available at [skill-path].

YOUR JOB: Use this skill on the plan.

1. Read the skill: cat [skill-path]/SKILL.md
2. Follow the skill's instructions exactly
3. Apply the skill to this content:

[relevant plan section or full plan]

4. Return the skill's full output

The skill tells you what to do - follow it. Execute the skill completely."
```

**Spawn ALL skill sub-agents in PARALLEL:**
- 1 sub-agent per matched skill
- Each sub-agent reads and uses its assigned skill
- All run simultaneously
- 10, 20, 30 skill sub-agents is fine

**Each sub-agent:**
1. Reads its skill's SKILL.md
2. Follows the skill's workflow/instructions
3. Applies the skill to the plan
4. Returns whatever the skill produces (code, recommendations, patterns, reviews, etc.)

**Example spawns:**
```
Task general-purpose: "Use the dhh-rails-style skill at ~/.claude/plugins/.../dhh-rails-style. Read SKILL.md and apply it to: [Rails sections of plan]"

Task general-purpose: "Use the frontend-design skill at ~/.claude/plugins/.../frontend-design. Read SKILL.md and apply it to: [UI sections of plan]"

Task general-purpose: "Use the agent-native-architecture skill at ~/.claude/plugins/.../agent-native-architecture. Read SKILL.md and apply it to: [agent/tool sections of plan]"

Task general-purpose: "Use the security-patterns skill at ~/.claude/skills/security-patterns. Read SKILL.md and apply it to: [full plan]"
```

**No limit on skill sub-agents. Spawn one for every skill that could possibly be relevant.**

### 3. Discover and Apply Learnings/Solutions

<thinking>
Check for documented learnings from /workflows:compound. These are solved problems stored as markdown files. Spawn a sub-agent for each learning to check if it's relevant.
</thinking>

**LEARNINGS LOCATION - Check these exact folders:**

```
docs/solutions/           <-- PRIMARY: Project-level learnings (created by /workflows:compound)
├── performance-issues/
│   └── *.md
├── debugging-patterns/
│   └── *.md
├── configuration-fixes/
│   └── *.md
├── integration-issues/
│   └── *.md
├── deployment-issues/
│   └── *.md
└── [other-categories]/
    └── *.md
```

**Step 1: Find ALL learning markdown files**

Run these commands to get every learning file:

```bash
# PRIMARY LOCATION - Project learnings
find docs/solutions -name "*.md" -type f 2>/dev/null

# If docs/solutions doesn't exist, check alternate locations:
find .claude/docs -name "*.md" -type f 2>/dev/null
find ~/.claude/docs -name "*.md" -type f 2>/dev/null
```

**Step 2: Read frontmatter of each learning to filter**

Each learning file has YAML frontmatter with metadata. Read the first ~20 lines of each file to get:

```yaml
---
title: "N+1 Query Fix for Briefs"
category: performance-issues
tags: [activerecord, n-plus-one, includes, eager-loading]
module: Briefs
symptom: "Slow page load, multiple queries in logs"
root_cause: "Missing includes on association"
---
```

**For each .md file, quickly scan its frontmatter:**

```bash
# Read first 20 lines of each learning (frontmatter + summary)
head -20 docs/solutions/**/*.md
```

**Step 3: Filter - only spawn sub-agents for LIKELY relevant learnings**

Compare each learning's frontmatter against the plan:
- `tags:` - Do any tags match technologies/patterns in the plan?
- `category:` - Is this category relevant? (e.g., skip deployment-issues if plan is UI-only)
- `module:` - Does the plan touch this module?
- `symptom:` / `root_cause:` - Could this problem occur with the plan?

**SKIP learnings that are clearly not applicable:**
- Plan is frontend-only → skip `database-migrations/` learnings
- Plan is Python → skip `rails-specific/` learnings
- Plan has no auth → skip `authentication-issues/` learnings

**SPAWN sub-agents for learnings that MIGHT apply:**
- Any tag overlap with plan technologies
- Same category as plan domain
- Similar patterns or concerns

**Step 4: Spawn sub-agents for filtered learnings**

For each learning that passes the filter:

```
Task general-purpose: "
LEARNING FILE: [full path to .md file]

1. Read this learning file completely
2. This learning documents a previously solved problem

Check if this learning applies to this plan:

---
[full plan content]
---

If relevant:
- Explain specifically how it applies
- Quote the key insight or solution
- Suggest where/how to incorporate it

If NOT relevant after deeper analysis:
- Say 'Not applicable: [reason]'
"
```

**Example filtering:**
```
# Found 15 learning files, plan is about "Rails API caching"

# SPAWN (likely relevant):
docs/solutions/performance-issues/n-plus-one-queries.md      # tags: [activerecord] ✓
docs/solutions/performance-issues/redis-cache-stampede.md    # tags: [caching, redis] ✓
docs/solutions/configuration-fixes/redis-connection-pool.md  # tags: [redis] ✓

# SKIP (clearly not applicable):
docs/solutions/deployment-issues/heroku-memory-quota.md      # not about caching
docs/solutions/frontend-issues/stimulus-race-condition.md    # plan is API, not frontend
docs/solutions/authentication-issues/jwt-expiry.md           # plan has no auth
```

**Spawn sub-agents in PARALLEL for all filtered learnings.**

**These learnings are institutional knowledge - applying them prevents repeating past mistakes.**

### 4. Launch Per-Section Research Agents

<thinking>
For each major section in the plan, spawn dedicated sub-agents to research improvements. Use the Explore agent type for open-ended research.
</thinking>

**For each identified section, launch parallel research:**

```
Task Explore: "Research best practices, patterns, and real-world examples for: [section topic].
Find:
- Industry standards and conventions
- Performance considerations
- Common pitfalls and how to avoid them
- Documentation and tutorials
Return concrete, actionable recommendations."
```

**Also use Context7 MCP for framework documentation:**

For any technologies/frameworks mentioned in the plan, query Context7:
```
mcp__plugin_compound-engineering_context7__resolve-library-id: Find library ID for [framework]
mcp__plugin_compound-engineering_context7__query-docs: Query documentation for specific patterns
```

**Use WebSearch for current best practices:**

Search for recent (2024-2026) articles, blog posts, and documentation on topics in the plan.

### 5. Discover and Run ALL Review Agents

<thinking>
Dynamically discover every available agent and run them ALL against the plan. Don't filter, don't skip, don't assume relevance. 40+ parallel agents is fine. Use everything available.
</thinking>

**Step 1: Discover ALL available agents from ALL sources**

```bash
# 1. Project-local agents (highest priority - project-specific)
find .claude/agents -name "*.md" 2>/dev/null

# 2. User's global agents (~/.claude/)
find ~/.claude/agents -name "*.md" 2>/dev/null

# 3. compound-engineering plugin agents (all subdirectories)
find ~/.claude/plugins/cache/*/compound-engineering/*/agents -name "*.md" 2>/dev/null

# 4. ALL other installed plugins - check every plugin for agents
find ~/.claude/plugins/cache -path "*/agents/*.md" 2>/dev/null

# 5. Check installed_plugins.json to find all plugin locations
cat ~/.claude/plugins/installed_plugins.json

# 6. For local plugins (isLocal: true), check their source directories
# Parse installed_plugins.json and find local plugin paths
```

**Important:** Check EVERY source. Include agents from:
- Project `.claude/agents/`
- User's `~/.claude/agents/`
- compound-engineering plugin (but SKIP workflow/ agents - only use review/, research/, design/, docs/)
- ALL other installed plugins (agent-sdk-dev, frontend-design, etc.)
- Any local plugins

**For compound-engineering plugin specifically:**
- USE: `agents/review/*` (all reviewers)
- USE: `agents/research/*` (all researchers)
- USE: `agents/design/*` (design agents)
- USE: `agents/docs/*` (documentation agents)
- SKIP: `agents/workflow/*` (these are workflow orchestrators, not reviewers)

**Step 2: For each discovered agent, read its description**

Read the first few lines of each agent file to understand what it reviews/analyzes.

**Step 3: Launch ALL agents in parallel**

For EVERY agent discovered, launch a Task in parallel:

```
Task [agent-name]: "Review this plan using your expertise. Apply all your checks and patterns. Plan content: [full plan content]"
```

**CRITICAL RULES:**
- Do NOT filter agents by "relevance" - run them ALL
- Do NOT skip agents because they "might not apply" - let them decide
- Launch ALL agents in a SINGLE message with multiple Task tool calls
- 20, 30, 40 parallel agents is fine - use everything
- Each agent may catch something others miss
- The goal is MAXIMUM coverage, not efficiency

**Step 4: Also discover and run research agents**

Research agents (like `best-practices-researcher`, `framework-docs-researcher`, `git-history-analyzer`, `repo-research-analyst`) should also be run for relevant plan sections.

### 6. Wait for ALL Agents and Synthesize Everything

<thinking>
Wait for ALL parallel agents to complete - skills, research agents, review agents, everything. Then synthesize all findings into a comprehensive enhancement.
</thinking>

**Collect outputs from ALL sources:**

1. **Skill-based sub-agents** - Each skill's full output (code examples, patterns, recommendations)
2. **Learnings/Solutions sub-agents** - Relevant documented learnings from /workflows:compound
3. **Research agents** - Best practices, documentation, real-world examples
4. **Review agents** - All feedback from every reviewer (architecture, security, performance, simplicity, etc.)
5. **Context7 queries** - Framework documentation and patterns
6. **Web searches** - Current best practices and articles

**For each agent's findings, extract:**
- [ ] Concrete recommendations (actionable items)
- [ ] Code patterns and examples (copy-paste ready)
- [ ] Anti-patterns to avoid (warnings)
- [ ] Performance considerations (metrics, benchmarks)
- [ ] Security considerations (vulnerabilities, mitigations)
- [ ] Edge cases discovered (handling strategies)
- [ ] Documentation links (references)
- [ ] Skill-specific patterns (from matched skills)
- [ ] Relevant learnings (past solutions that apply - prevent repeating mistakes)

**Deduplicate and prioritize:**
- Merge similar recommendations from multiple agents
- Prioritize by impact (high-value improvements first)
- Flag conflicting advice for human review
- Group by plan section

### 7. Enhance Plan Sections

<thinking>
Merge research findings back into the plan, adding depth without changing the original structure.
</thinking>

**Enhancement format for each section:**

```markdown
## [Original Section Title]

[Original content preserved]

### Research Insights

**Best Practices:**
- [Concrete recommendation 1]
- [Concrete recommendation 2]

**Performance Considerations:**
- [Optimization opportunity]
- [Benchmark or metric to target]

**Implementation Details:**
```[language]
// Concrete code example from research
```

**Edge Cases:**
- [Edge case 1 and how to handle]
- [Edge case 2 and how to handle]

**References:**
- [Documentation URL 1]
- [Documentation URL 2]
```

### 8. Add Enhancement Summary

At the top of the plan, add a summary section:

```markdown
## Enhancement Summary

**Deepened on:** [Date]
**Sections enhanced:** [Count]
**Research agents used:** [List]

### Key Improvements
1. [Major improvement 1]
2. [Major improvement 2]
3. [Major improvement 3]

### New Considerations Discovered
- [Important finding 1]
- [Important finding 2]
```

### 9. Update Plan File

**Write the enhanced plan:**
- Preserve original filename
- Add `-deepened` suffix if user prefers a new file
- Update any timestamps or metadata

## Output Format

Update the plan file in place (or if user requests a separate file, append `-deepened` after `-plan`, e.g., `2026-01-15-feat-auth-plan-deepened.md`).

## Quality Checks

Before finalizing:
- [ ] All original content preserved
- [ ] Research insights clearly marked and attributed
- [ ] Code examples are syntactically correct
- [ ] Links are valid and relevant
- [ ] No contradictions between sections
- [ ] Enhancement summary accurately reflects changes

## Post-Enhancement Options

After writing the enhanced plan, use the **AskUserQuestion tool** to present these options:

**Question:** "Plan deepened at `[plan_path]`. What would you like to do next?"

**Options:**
1. **View diff** - Show what was added/changed
2. **Run `/technical_review`** - Get feedback from reviewers on enhanced plan
3. **Start `/workflows:work`** - Begin implementing this enhanced plan
4. **Deepen further** - Run another round of research on specific sections
5. **Revert** - Restore original plan (if backup exists)

Based on selection:
- **View diff** → Run `git diff [plan_path]` or show before/after
- **`/technical_review`** → Call the /technical_review command with the plan file path
- **`/workflows:work`** → Call the /workflows:work command with the plan file path
- **Deepen further** → Ask which sections need more research, then re-run those agents
- **Revert** → Restore from git or backup

## Example Enhancement

**Before (from /workflows:plan):**
```markdown
## Technical Approach

Use React Query for data fetching with optimistic updates.
```

**After (from /workflows:deepen-plan):**
```markdown
## Technical Approach

Use React Query for data fetching with optimistic updates.

### Research Insights

**Best Practices:**
- Configure `staleTime` and `cacheTime` based on data freshness requirements
- Use `queryKey` factories for consistent cache invalidation
- Implement error boundaries around query-dependent components

**Performance Considerations:**
- Enable `refetchOnWindowFocus: false` for stable data to reduce unnecessary requests
- Use `select` option to transform and memoize data at query level
- Consider `placeholderData` for instant perceived loading

**Implementation Details:**
```typescript
// Recommended query configuration
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5 * 60 * 1000, // 5 minutes
      retry: 2,
      refetchOnWindowFocus: false,
    },
  },
});
```

**Edge Cases:**
- Handle race conditions with `cancelQueries` on component unmount
- Implement retry logic for transient network failures
- Consider offline support with `persistQueryClient`

**References:**
- https://tanstack.com/query/latest/docs/react/guides/optimistic-updates
- https://tkdodo.eu/blog/practical-react-query
```

NEVER CODE! Just research and enhance the plan.

---

## /report-bug

_Source: `cache/every-marketplace/compound-engineering/2.31.0/commands/report-bug.md`_

---
name: report-bug
description: Report a bug in the compound-engineering plugin
argument-hint: "[optional: brief description of the bug]"
disable-model-invocation: true
---

# Report a Compounding Engineering Plugin Bug

Report bugs encountered while using the compound-engineering plugin. This command gathers structured information and creates a GitHub issue for the maintainer.

## Step 1: Gather Bug Information

Use the AskUserQuestion tool to collect the following information:

**Question 1: Bug Category**
- What type of issue are you experiencing?
- Options: Agent not working, Command not working, Skill not working, MCP server issue, Installation problem, Other

**Question 2: Specific Component**
- Which specific component is affected?
- Ask for the name of the agent, command, skill, or MCP server

**Question 3: What Happened (Actual Behavior)**
- Ask: "What happened when you used this component?"
- Get a clear description of the actual behavior

**Question 4: What Should Have Happened (Expected Behavior)**
- Ask: "What did you expect to happen instead?"
- Get a clear description of expected behavior

**Question 5: Steps to Reproduce**
- Ask: "What steps did you take before the bug occurred?"
- Get reproduction steps

**Question 6: Error Messages**
- Ask: "Did you see any error messages? If so, please share them."
- Capture any error output

## Step 2: Collect Environment Information

Automatically gather:
```bash
# Get plugin version
cat ~/.claude/plugins/installed_plugins.json 2>/dev/null | grep -A5 "compound-engineering" | head -10 || echo "Plugin info not found"

# Get Claude Code version
claude --version 2>/dev/null || echo "Claude CLI version unknown"

# Get OS info
uname -a
```

## Step 3: Format the Bug Report

Create a well-structured bug report with:

```markdown
## Bug Description

**Component:** [Type] - [Name]
**Summary:** [Brief description from argument or collected info]

## Environment

- **Plugin Version:** [from installed_plugins.json]
- **Claude Code Version:** [from claude --version]
- **OS:** [from uname]

## What Happened

[Actual behavior description]

## Expected Behavior

[Expected behavior description]

## Steps to Reproduce

1. [Step 1]
2. [Step 2]
3. [Step 3]

## Error Messages

```
[Any error output]
```

## Additional Context

[Any other relevant information]

---
*Reported via `/report-bug` command*
```

## Step 4: Create GitHub Issue

Use the GitHub CLI to create the issue:

```bash
gh issue create \
  --repo EveryInc/compound-engineering-plugin \
  --title "[compound-engineering] Bug: [Brief description]" \
  --body "[Formatted bug report from Step 3]" \
  --label "bug,compound-engineering"
```

**Note:** If labels don't exist, create without labels:
```bash
gh issue create \
  --repo EveryInc/compound-engineering-plugin \
  --title "[compound-engineering] Bug: [Brief description]" \
  --body "[Formatted bug report]"
```

## Step 5: Confirm Submission

After the issue is created:
1. Display the issue URL to the user
2. Thank them for reporting the bug
3. Let them know the maintainer (Kieran Klaassen) will be notified

## Output Format

```
✅ Bug report submitted successfully!

Issue: https://github.com/EveryInc/compound-engineering-plugin/issues/[NUMBER]
Title: [compound-engineering] Bug: [description]

Thank you for helping improve the compound-engineering plugin!
The maintainer will review your report and respond as soon as possible.
```

## Error Handling

- If `gh` CLI is not authenticated: Prompt user to run `gh auth login` first
- If issue creation fails: Display the formatted report so user can manually create the issue
- If required information is missing: Re-prompt for that specific field

## Privacy Notice

This command does NOT collect:
- Personal information
- API keys or credentials
- Private code from your projects
- File paths beyond basic OS info

Only technical information about the bug is included in the report.

---

## /lfg

_Source: `cache/every-marketplace/compound-engineering/2.31.0/commands/lfg.md`_

---
name: lfg
description: Full autonomous engineering workflow
argument-hint: "[feature description]"
disable-model-invocation: true
---

Run these slash commands in order. Do not do anything else.

1. `/ralph-wiggum:ralph-loop "finish all slash commands" --completion-promise "DONE"`
2. `/workflows:plan $ARGUMENTS`
3. `/compound-engineering:deepen-plan`
4. `/workflows:work`
5. `/workflows:review`
6. `/compound-engineering:resolve_todo_parallel`
7. `/compound-engineering:test-browser`
8. `/compound-engineering:feature-video`
9. Output `<promise>DONE</promise>` when video is in PR

Start with step 1 now.

---

## /work

_Source: `cache/every-marketplace/compound-engineering/2.31.0/commands/workflows/work.md`_

---
name: workflows:work
description: Execute work plans efficiently while maintaining quality and finishing features
argument-hint: "[plan file, specification, or todo file path]"
---

# Work Plan Execution Command

Execute a work plan efficiently while maintaining quality and finishing features.

## Introduction

This command takes a work document (plan, specification, or todo file) and executes it systematically. The focus is on **shipping complete features** by understanding requirements quickly, following existing patterns, and maintaining quality throughout.

## Input Document

<input_document> #$ARGUMENTS </input_document>

## Execution Workflow

### Phase 1: Quick Start

1. **Read Plan and Clarify**

   - Read the work document completely
   - Review any references or links provided in the plan
   - If anything is unclear or ambiguous, ask clarifying questions now
   - Get user approval to proceed
   - **Do not skip this** - better to ask questions now than build the wrong thing

2. **Setup Environment**

   First, check the current branch:

   ```bash
   current_branch=$(git branch --show-current)
   default_branch=$(git symbolic-ref refs/remotes/origin/HEAD 2>/dev/null | sed 's@^refs/remotes/origin/@@')

   # Fallback if remote HEAD isn't set
   if [ -z "$default_branch" ]; then
     default_branch=$(git rev-parse --verify origin/main >/dev/null 2>&1 && echo "main" || echo "master")
   fi
   ```

   **If already on a feature branch** (not the default branch):
   - Ask: "Continue working on `[current_branch]`, or create a new branch?"
   - If continuing, proceed to step 3
   - If creating new, follow Option A or B below

   **If on the default branch**, choose how to proceed:

   **Option A: Create a new branch**
   ```bash
   git pull origin [default_branch]
   git checkout -b feature-branch-name
   ```
   Use a meaningful name based on the work (e.g., `feat/user-authentication`, `fix/email-validation`).

   **Option B: Use a worktree (recommended for parallel development)**
   ```bash
   skill: git-worktree
   # The skill will create a new branch from the default branch in an isolated worktree
   ```

   **Option C: Continue on the default branch**
   - Requires explicit user confirmation
   - Only proceed after user explicitly says "yes, commit to [default_branch]"
   - Never commit directly to the default branch without explicit permission

   **Recommendation**: Use worktree if:
   - You want to work on multiple features simultaneously
   - You want to keep the default branch clean while experimenting
   - You plan to switch between branches frequently

3. **Create Todo List**
   - Use TodoWrite to break plan into actionable tasks
   - Include dependencies between tasks
   - Prioritize based on what needs to be done first
   - Include testing and quality check tasks
   - Keep tasks specific and completable

### Phase 2: Execute

1. **Task Execution Loop**

   For each task in priority order:

   ```
   while (tasks remain):
     - Mark task as in_progress in TodoWrite
     - Read any referenced files from the plan
     - Look for similar patterns in codebase
     - Implement following existing conventions
     - Write tests for new functionality
     - Run tests after changes
     - Mark task as completed in TodoWrite
     - Mark off the corresponding checkbox in the plan file ([ ] → [x])
     - Evaluate for incremental commit (see below)
   ```

   **IMPORTANT**: Always update the original plan document by checking off completed items. Use the Edit tool to change `- [ ]` to `- [x]` for each task you finish. This keeps the plan as a living document showing progress and ensures no checkboxes are left unchecked.

2. **Incremental Commits**

   After completing each task, evaluate whether to create an incremental commit:

   | Commit when... | Don't commit when... |
   |----------------|---------------------|
   | Logical unit complete (model, service, component) | Small part of a larger unit |
   | Tests pass + meaningful progress | Tests failing |
   | About to switch contexts (backend → frontend) | Purely scaffolding with no behavior |
   | About to attempt risky/uncertain changes | Would need a "WIP" commit message |

   **Heuristic:** "Can I write a commit message that describes a complete, valuable change? If yes, commit. If the message would be 'WIP' or 'partial X', wait."

   **Commit workflow:**
   ```bash
   # 1. Verify tests pass (use project's test command)
   # Examples: bin/rails test, npm test, pytest, go test, etc.

   # 2. Stage only files related to this logical unit (not `git add .`)
   git add <files related to this logical unit>

   # 3. Commit with conventional message
   git commit -m "feat(scope): description of this unit"
   ```

   **Handling merge conflicts:** If conflicts arise during rebasing or merging, resolve them immediately. Incremental commits make conflict resolution easier since each commit is small and focused.

   **Note:** Incremental commits use clean conventional messages without attribution footers. The final Phase 4 commit/PR includes the full attribution.

3. **Follow Existing Patterns**

   - The plan should reference similar code - read those files first
   - Match naming conventions exactly
   - Reuse existing components where possible
   - Follow project coding standards (see CLAUDE.md)
   - When in doubt, grep for similar implementations

4. **Test Continuously**

   - Run relevant tests after each significant change
   - Don't wait until the end to test
   - Fix failures immediately
   - Add new tests for new functionality

5. **Figma Design Sync** (if applicable)

   For UI work with Figma designs:

   - Implement components following design specs
   - Use figma-design-sync agent iteratively to compare
   - Fix visual differences identified
   - Repeat until implementation matches design

6. **Track Progress**
   - Keep TodoWrite updated as you complete tasks
   - Note any blockers or unexpected discoveries
   - Create new tasks if scope expands
   - Keep user informed of major milestones

### Phase 3: Quality Check

1. **Run Core Quality Checks**

   Always run before submitting:

   ```bash
   # Run full test suite (use project's test command)
   # Examples: bin/rails test, npm test, pytest, go test, etc.

   # Run linting (per CLAUDE.md)
   # Use linting-agent before pushing to origin
   ```

2. **Consider Reviewer Agents** (Optional)

   Use for complex, risky, or large changes:

   - **code-simplicity-reviewer**: Check for unnecessary complexity
   - **kieran-rails-reviewer**: Verify Rails conventions (Rails projects)
   - **performance-oracle**: Check for performance issues
   - **security-sentinel**: Scan for security vulnerabilities
   - **cora-test-reviewer**: Review test quality (Rails projects with comprehensive test coverage)

   Run reviewers in parallel with Task tool:

   ```
   Task(code-simplicity-reviewer): "Review changes for simplicity"
   Task(kieran-rails-reviewer): "Check Rails conventions"
   ```

   Present findings to user and address critical issues.

3. **Final Validation**
   - All TodoWrite tasks marked completed
   - All tests pass
   - Linting passes
   - Code follows existing patterns
   - Figma designs match (if applicable)
   - No console errors or warnings

### Phase 4: Ship It

1. **Create Commit**

   ```bash
   git add .
   git status  # Review what's being committed
   git diff --staged  # Check the changes

   # Commit with conventional format
   git commit -m "$(cat <<'EOF'
   feat(scope): description of what and why

   Brief explanation if needed.

   🤖 Generated with [Claude Code](https://claude.com/claude-code)

   Co-Authored-By: Claude <noreply@anthropic.com>
   EOF
   )"
   ```

2. **Capture and Upload Screenshots for UI Changes** (REQUIRED for any UI work)

   For **any** design changes, new views, or UI modifications, you MUST capture and upload screenshots:

   **Step 1: Start dev server** (if not running)
   ```bash
   bin/dev  # Run in background
   ```

   **Step 2: Capture screenshots with agent-browser CLI**
   ```bash
   agent-browser open http://localhost:3000/[route]
   agent-browser snapshot -i
   agent-browser screenshot output.png
   ```
   See the `agent-browser` skill for detailed usage.

   **Step 3: Upload using imgup skill**
   ```bash
   skill: imgup
   # Then upload each screenshot:
   imgup -h pixhost screenshot.png  # pixhost works without API key
   # Alternative hosts: catbox, imagebin, beeimg
   ```

   **What to capture:**
   - **New screens**: Screenshot of the new UI
   - **Modified screens**: Before AND after screenshots
   - **Design implementation**: Screenshot showing Figma design match

   **IMPORTANT**: Always include uploaded image URLs in PR description. This provides visual context for reviewers and documents the change.

3. **Create Pull Request**

   ```bash
   git push -u origin feature-branch-name

   gh pr create --title "Feature: [Description]" --body "$(cat <<'EOF'
   ## Summary
   - What was built
   - Why it was needed
   - Key decisions made

   ## Testing
   - Tests added/modified
   - Manual testing performed

   ## Before / After Screenshots
   | Before | After |
   |--------|-------|
   | ![before](URL) | ![after](URL) |

   ## Figma Design
   [Link if applicable]

   ---

   [![Compound Engineered](https://img.shields.io/badge/Compound-Engineered-6366f1)](https://github.com/EveryInc/compound-engineering-plugin) 🤖 Generated with [Claude Code](https://claude.com/claude-code)
   EOF
   )"
   ```

4. **Notify User**
   - Summarize what was completed
   - Link to PR
   - Note any follow-up work needed
   - Suggest next steps if applicable

---

## Swarm Mode (Optional)

For complex plans with multiple independent workstreams, enable swarm mode for parallel execution with coordinated agents.

### When to Use Swarm Mode

| Use Swarm Mode when... | Use Standard Mode when... |
|------------------------|---------------------------|
| Plan has 5+ independent tasks | Plan is linear/sequential |
| Multiple specialists needed (review + test + implement) | Single-focus work |
| Want maximum parallelism | Simpler mental model preferred |
| Large feature with clear phases | Small feature or bug fix |

### Enabling Swarm Mode

To trigger swarm execution, say:

> "Make a Task list and launch an army of agent swarm subagents to build the plan"

Or explicitly request: "Use swarm mode for this work"

### Swarm Workflow

When swarm mode is enabled, the workflow changes:

1. **Create Team**
   ```
   Teammate({ operation: "spawnTeam", team_name: "work-{timestamp}" })
   ```

2. **Create Task List with Dependencies**
   - Parse plan into TaskCreate items
   - Set up blockedBy relationships for sequential dependencies
   - Independent tasks have no blockers (can run in parallel)

3. **Spawn Specialized Teammates**
   ```
   Task({
     team_name: "work-{timestamp}",
     name: "implementer",
     subagent_type: "general-purpose",
     prompt: "Claim implementation tasks, execute, mark complete",
     run_in_background: true
   })

   Task({
     team_name: "work-{timestamp}",
     name: "tester",
     subagent_type: "general-purpose",
     prompt: "Claim testing tasks, run tests, mark complete",
     run_in_background: true
   })
   ```

4. **Coordinate and Monitor**
   - Team lead monitors task completion
   - Spawn additional workers as phases unblock
   - Handle plan approval if required

5. **Cleanup**
   ```
   Teammate({ operation: "requestShutdown", target_agent_id: "implementer" })
   Teammate({ operation: "requestShutdown", target_agent_id: "tester" })
   Teammate({ operation: "cleanup" })
   ```

See the `orchestrating-swarms` skill for detailed swarm patterns and best practices.

---

## Key Principles

### Start Fast, Execute Faster

- Get clarification once at the start, then execute
- Don't wait for perfect understanding - ask questions and move
- The goal is to **finish the feature**, not create perfect process

### The Plan is Your Guide

- Work documents should reference similar code and patterns
- Load those references and follow them
- Don't reinvent - match what exists

### Test As You Go

- Run tests after each change, not at the end
- Fix failures immediately
- Continuous testing prevents big surprises

### Quality is Built In

- Follow existing patterns
- Write tests for new code
- Run linting before pushing
- Use reviewer agents for complex/risky changes only

### Ship Complete Features

- Mark all tasks completed before moving on
- Don't leave features 80% done
- A finished feature that ships beats a perfect feature that doesn't

## Quality Checklist

Before creating PR, verify:

- [ ] All clarifying questions asked and answered
- [ ] All TodoWrite tasks marked completed
- [ ] Tests pass (run project's test command)
- [ ] Linting passes (use linting-agent)
- [ ] Code follows existing patterns
- [ ] Figma designs match implementation (if applicable)
- [ ] Before/after screenshots captured and uploaded (for UI changes)
- [ ] Commit messages follow conventional format
- [ ] PR description includes summary, testing notes, and screenshots
- [ ] PR description includes Compound Engineered badge

## When to Use Reviewer Agents

**Don't use by default.** Use reviewer agents only when:

- Large refactor affecting many files (10+)
- Security-sensitive changes (authentication, permissions, data access)
- Performance-critical code paths
- Complex algorithms or business logic
- User explicitly requests thorough review

For most features: tests + linting + following patterns is sufficient.

## Common Pitfalls to Avoid

- **Analysis paralysis** - Don't overthink, read the plan and execute
- **Skipping clarifying questions** - Ask now, not after building wrong thing
- **Ignoring plan references** - The plan has links for a reason
- **Testing at the end** - Test continuously or suffer later
- **Forgetting TodoWrite** - Track progress or lose track of what's done
- **80% done syndrome** - Finish the feature, don't move on early
- **Over-reviewing simple changes** - Save reviewer agents for complex work

---

## /compound

_Source: `cache/every-marketplace/compound-engineering/2.31.0/commands/workflows/compound.md`_

---
name: workflows:compound
description: Document a recently solved problem to compound your team's knowledge
argument-hint: "[optional: brief context about the fix]"
---

# /compound

Coordinate multiple subagents working in parallel to document a recently solved problem.

## Purpose

Captures problem solutions while context is fresh, creating structured documentation in `docs/solutions/` with YAML frontmatter for searchability and future reference. Uses parallel subagents for maximum efficiency.

**Why "compound"?** Each documented solution compounds your team's knowledge. The first time you solve a problem takes research. Document it, and the next occurrence takes minutes. Knowledge compounds.

## Usage

```bash
/workflows:compound                    # Document the most recent fix
/workflows:compound [brief context]    # Provide additional context hint
```

## Execution Strategy: Two-Phase Orchestration

<critical_requirement>
**Only ONE file gets written - the final documentation.**

Phase 1 subagents return TEXT DATA to the orchestrator. They must NOT use Write, Edit, or create any files. Only the orchestrator (Phase 2) writes the final documentation file.
</critical_requirement>

### Phase 1: Parallel Research

<parallel_tasks>

Launch these subagents IN PARALLEL. Each returns text data to the orchestrator.

#### 1. **Context Analyzer**
   - Extracts conversation history
   - Identifies problem type, component, symptoms
   - Validates against schema
   - Returns: YAML frontmatter skeleton

#### 2. **Solution Extractor**
   - Analyzes all investigation steps
   - Identifies root cause
   - Extracts working solution with code examples
   - Returns: Solution content block

#### 3. **Related Docs Finder**
   - Searches `docs/solutions/` for related documentation
   - Identifies cross-references and links
   - Finds related GitHub issues
   - Returns: Links and relationships

#### 4. **Prevention Strategist**
   - Develops prevention strategies
   - Creates best practices guidance
   - Generates test cases if applicable
   - Returns: Prevention/testing content

#### 5. **Category Classifier**
   - Determines optimal `docs/solutions/` category
   - Validates category against schema
   - Suggests filename based on slug
   - Returns: Final path and filename

</parallel_tasks>

### Phase 2: Assembly & Write

<sequential_tasks>

**WAIT for all Phase 1 subagents to complete before proceeding.**

The orchestrating agent (main conversation) performs these steps:

1. Collect all text results from Phase 1 subagents
2. Assemble complete markdown file from the collected pieces
3. Validate YAML frontmatter against schema
4. Create directory if needed: `mkdir -p docs/solutions/[category]/`
5. Write the SINGLE final file: `docs/solutions/[category]/[filename].md`

</sequential_tasks>

### Phase 3: Optional Enhancement

**WAIT for Phase 2 to complete before proceeding.**

<parallel_tasks>

Based on problem type, optionally invoke specialized agents to review the documentation:

- **performance_issue** → `performance-oracle`
- **security_issue** → `security-sentinel`
- **database_issue** → `data-integrity-guardian`
- **test_failure** → `cora-test-reviewer`
- Any code-heavy issue → `kieran-rails-reviewer` + `code-simplicity-reviewer`

</parallel_tasks>

## What It Captures

- **Problem symptom**: Exact error messages, observable behavior
- **Investigation steps tried**: What didn't work and why
- **Root cause analysis**: Technical explanation
- **Working solution**: Step-by-step fix with code examples
- **Prevention strategies**: How to avoid in future
- **Cross-references**: Links to related issues and docs

## Preconditions

<preconditions enforcement="advisory">
  <check condition="problem_solved">
    Problem has been solved (not in-progress)
  </check>
  <check condition="solution_verified">
    Solution has been verified working
  </check>
  <check condition="non_trivial">
    Non-trivial problem (not simple typo or obvious error)
  </check>
</preconditions>

## What It Creates

**Organized documentation:**

- File: `docs/solutions/[category]/[filename].md`

**Categories auto-detected from problem:**

- build-errors/
- test-failures/
- runtime-errors/
- performance-issues/
- database-issues/
- security-issues/
- ui-bugs/
- integration-issues/
- logic-errors/

## Common Mistakes to Avoid

| ❌ Wrong | ✅ Correct |
|----------|-----------|
| Subagents write files like `context-analysis.md`, `solution-draft.md` | Subagents return text data; orchestrator writes one final file |
| Research and assembly run in parallel | Research completes → then assembly runs |
| Multiple files created during workflow | Single file: `docs/solutions/[category]/[filename].md` |

## Success Output

```
✓ Documentation complete

Subagent Results:
  ✓ Context Analyzer: Identified performance_issue in brief_system
  ✓ Solution Extractor: 3 code fixes
  ✓ Related Docs Finder: 2 related issues
  ✓ Prevention Strategist: Prevention strategies, test suggestions
  ✓ Category Classifier: `performance-issues`

Specialized Agent Reviews (Auto-Triggered):
  ✓ performance-oracle: Validated query optimization approach
  ✓ kieran-rails-reviewer: Code examples meet Rails standards
  ✓ code-simplicity-reviewer: Solution is appropriately minimal
  ✓ every-style-editor: Documentation style verified

File created:
- docs/solutions/performance-issues/n-plus-one-brief-generation.md

This documentation will be searchable for future reference when similar
issues occur in the Email Processing or Brief System modules.

What's next?
1. Continue workflow (recommended)
2. Link related documentation
3. Update other references
4. View documentation
5. Other
```

## The Compounding Philosophy

This creates a compounding knowledge system:

1. First time you solve "N+1 query in brief generation" → Research (30 min)
2. Document the solution → docs/solutions/performance-issues/n-plus-one-briefs.md (5 min)
3. Next time similar issue occurs → Quick lookup (2 min)
4. Knowledge compounds → Team gets smarter

The feedback loop:

```
Build → Test → Find Issue → Research → Improve → Document → Validate → Deploy
    ↑                                                                      ↓
    └──────────────────────────────────────────────────────────────────────┘
```

**Each unit of engineering work should make subsequent units of work easier—not harder.**

## Auto-Invoke

<auto_invoke> <trigger_phrases> - "that worked" - "it's fixed" - "working now" - "problem solved" </trigger_phrases>

<manual_override> Use /workflows:compound [context] to document immediately without waiting for auto-detection. </manual_override> </auto_invoke>

## Routes To

`compound-docs` skill

## Applicable Specialized Agents

Based on problem type, these agents can enhance documentation:

### Code Quality & Review
- **kieran-rails-reviewer**: Reviews code examples for Rails best practices
- **code-simplicity-reviewer**: Ensures solution code is minimal and clear
- **pattern-recognition-specialist**: Identifies anti-patterns or repeating issues

### Specific Domain Experts
- **performance-oracle**: Analyzes performance_issue category solutions
- **security-sentinel**: Reviews security_issue solutions for vulnerabilities
- **cora-test-reviewer**: Creates test cases for prevention strategies
- **data-integrity-guardian**: Reviews database_issue migrations and queries

### Enhancement & Documentation
- **best-practices-researcher**: Enriches solution with industry best practices
- **every-style-editor**: Reviews documentation style and clarity
- **framework-docs-researcher**: Links to Rails/gem documentation references

### When to Invoke
- **Auto-triggered** (optional): Agents can run post-documentation for enhancement
- **Manual trigger**: User can invoke agents after /workflows:compound completes for deeper review

## Related Commands

- `/research [topic]` - Deep investigation (searches docs/solutions/ for patterns)
- `/workflows:plan` - Planning workflow (references documented solutions)

---

## /plan

_Source: `cache/every-marketplace/compound-engineering/2.31.0/commands/workflows/plan.md`_

---
name: workflows:plan
description: Transform feature descriptions into well-structured project plans following conventions
argument-hint: "[feature description, bug report, or improvement idea]"
---

# Create a plan for a new feature or bug fix

## Introduction

**Note: The current year is 2026.** Use this when dating plans and searching for recent documentation.

Transform feature descriptions, bug reports, or improvement ideas into well-structured markdown files issues that follow project conventions and best practices. This command provides flexible detail levels to match your needs.

## Feature Description

<feature_description> #$ARGUMENTS </feature_description>

**If the feature description above is empty, ask the user:** "What would you like to plan? Please describe the feature, bug fix, or improvement you have in mind."

Do not proceed until you have a clear feature description from the user.

### 0. Idea Refinement

**Check for brainstorm output first:**

Before asking questions, look for recent brainstorm documents in `docs/brainstorms/` that match this feature:

```bash
ls -la docs/brainstorms/*.md 2>/dev/null | head -10
```

**Relevance criteria:** A brainstorm is relevant if:
- The topic (from filename or YAML frontmatter) semantically matches the feature description
- Created within the last 14 days
- If multiple candidates match, use the most recent one

**If a relevant brainstorm exists:**
1. Read the brainstorm document
2. Announce: "Found brainstorm from [date]: [topic]. Using as context for planning."
3. Extract key decisions, chosen approach, and open questions
4. **Skip the idea refinement questions below** - the brainstorm already answered WHAT to build
5. Use brainstorm decisions as input to the research phase

**If multiple brainstorms could match:**
Use **AskUserQuestion tool** to ask which brainstorm to use, or whether to proceed without one.

**If no brainstorm found (or not relevant), run idea refinement:**

Refine the idea through collaborative dialogue using the **AskUserQuestion tool**:

- Ask questions one at a time to understand the idea fully
- Prefer multiple choice questions when natural options exist
- Focus on understanding: purpose, constraints and success criteria
- Continue until the idea is clear OR user says "proceed"

**Gather signals for research decision.** During refinement, note:

- **User's familiarity**: Do they know the codebase patterns? Are they pointing to examples?
- **User's intent**: Speed vs thoroughness? Exploration vs execution?
- **Topic risk**: Security, payments, external APIs warrant more caution
- **Uncertainty level**: Is the approach clear or open-ended?

**Skip option:** If the feature description is already detailed, offer:
"Your description is clear. Should I proceed with research, or would you like to refine it further?"

## Main Tasks

### 1. Local Research (Always Runs - Parallel)

<thinking>
First, I need to understand the project's conventions, existing patterns, and any documented learnings. This is fast and local - it informs whether external research is needed.
</thinking>

Run these agents **in parallel** to gather local context:

- Task repo-research-analyst(feature_description)
- Task learnings-researcher(feature_description)

**What to look for:**
- **Repo research:** existing patterns, CLAUDE.md guidance, technology familiarity, pattern consistency
- **Learnings:** documented solutions in `docs/solutions/` that might apply (gotchas, patterns, lessons learned)

These findings inform the next step.

### 1.5. Research Decision

Based on signals from Step 0 and findings from Step 1, decide on external research.

**High-risk topics → always research.** Security, payments, external APIs, data privacy. The cost of missing something is too high. This takes precedence over speed signals.

**Strong local context → skip external research.** Codebase has good patterns, CLAUDE.md has guidance, user knows what they want. External research adds little value.

**Uncertainty or unfamiliar territory → research.** User is exploring, codebase has no examples, new technology. External perspective is valuable.

**Announce the decision and proceed.** Brief explanation, then continue. User can redirect if needed.

Examples:
- "Your codebase has solid patterns for this. Proceeding without external research."
- "This involves payment processing, so I'll research current best practices first."

### 1.5b. External Research (Conditional)

**Only run if Step 1.5 indicates external research is valuable.**

Run these agents in parallel:

- Task best-practices-researcher(feature_description)
- Task framework-docs-researcher(feature_description)

### 1.6. Consolidate Research

After all research steps complete, consolidate findings:

- Document relevant file paths from repo research (e.g., `app/services/example_service.rb:42`)
- **Include relevant institutional learnings** from `docs/solutions/` (key insights, gotchas to avoid)
- Note external documentation URLs and best practices (if external research was done)
- List related issues or PRs discovered
- Capture CLAUDE.md conventions

**Optional validation:** Briefly summarize findings and ask if anything looks off or missing before proceeding to planning.

### 2. Issue Planning & Structure

<thinking>
Think like a product manager - what would make this issue clear and actionable? Consider multiple perspectives
</thinking>

**Title & Categorization:**

- [ ] Draft clear, searchable issue title using conventional format (e.g., `feat: Add user authentication`, `fix: Cart total calculation`)
- [ ] Determine issue type: enhancement, bug, refactor
- [ ] Convert title to filename: add today's date prefix, strip prefix colon, kebab-case, add `-plan` suffix
  - Example: `feat: Add User Authentication` → `2026-01-21-feat-add-user-authentication-plan.md`
  - Keep it descriptive (3-5 words after prefix) so plans are findable by context

**Stakeholder Analysis:**

- [ ] Identify who will be affected by this issue (end users, developers, operations)
- [ ] Consider implementation complexity and required expertise

**Content Planning:**

- [ ] Choose appropriate detail level based on issue complexity and audience
- [ ] List all necessary sections for the chosen template
- [ ] Gather supporting materials (error logs, screenshots, design mockups)
- [ ] Prepare code examples or reproduction steps if applicable, name the mock filenames in the lists

### 3. SpecFlow Analysis

After planning the issue structure, run SpecFlow Analyzer to validate and refine the feature specification:

- Task spec-flow-analyzer(feature_description, research_findings)

**SpecFlow Analyzer Output:**

- [ ] Review SpecFlow analysis results
- [ ] Incorporate any identified gaps or edge cases into the issue
- [ ] Update acceptance criteria based on SpecFlow findings

### 4. Choose Implementation Detail Level

Select how comprehensive you want the issue to be, simpler is mostly better.

#### 📄 MINIMAL (Quick Issue)

**Best for:** Simple bugs, small improvements, clear features

**Includes:**

- Problem statement or feature description
- Basic acceptance criteria
- Essential context only

**Structure:**

````markdown
---
title: [Issue Title]
type: [feat|fix|refactor]
date: YYYY-MM-DD
---

# [Issue Title]

[Brief problem/feature description]

## Acceptance Criteria

- [ ] Core requirement 1
- [ ] Core requirement 2

## Context

[Any critical information]

## MVP

### test.rb

```ruby
class Test
  def initialize
    @name = "test"
  end
end
```

## References

- Related issue: #[issue_number]
- Documentation: [relevant_docs_url]
````

#### 📋 MORE (Standard Issue)

**Best for:** Most features, complex bugs, team collaboration

**Includes everything from MINIMAL plus:**

- Detailed background and motivation
- Technical considerations
- Success metrics
- Dependencies and risks
- Basic implementation suggestions

**Structure:**

```markdown
---
title: [Issue Title]
type: [feat|fix|refactor]
date: YYYY-MM-DD
---

# [Issue Title]

## Overview

[Comprehensive description]

## Problem Statement / Motivation

[Why this matters]

## Proposed Solution

[High-level approach]

## Technical Considerations

- Architecture impacts
- Performance implications
- Security considerations

## Acceptance Criteria

- [ ] Detailed requirement 1
- [ ] Detailed requirement 2
- [ ] Testing requirements

## Success Metrics

[How we measure success]

## Dependencies & Risks

[What could block or complicate this]

## References & Research

- Similar implementations: [file_path:line_number]
- Best practices: [documentation_url]
- Related PRs: #[pr_number]
```

#### 📚 A LOT (Comprehensive Issue)

**Best for:** Major features, architectural changes, complex integrations

**Includes everything from MORE plus:**

- Detailed implementation plan with phases
- Alternative approaches considered
- Extensive technical specifications
- Resource requirements and timeline
- Future considerations and extensibility
- Risk mitigation strategies
- Documentation requirements

**Structure:**

```markdown
---
title: [Issue Title]
type: [feat|fix|refactor]
date: YYYY-MM-DD
---

# [Issue Title]

## Overview

[Executive summary]

## Problem Statement

[Detailed problem analysis]

## Proposed Solution

[Comprehensive solution design]

## Technical Approach

### Architecture

[Detailed technical design]

### Implementation Phases

#### Phase 1: [Foundation]

- Tasks and deliverables
- Success criteria
- Estimated effort

#### Phase 2: [Core Implementation]

- Tasks and deliverables
- Success criteria
- Estimated effort

#### Phase 3: [Polish & Optimization]

- Tasks and deliverables
- Success criteria
- Estimated effort

## Alternative Approaches Considered

[Other solutions evaluated and why rejected]

## Acceptance Criteria

### Functional Requirements

- [ ] Detailed functional criteria

### Non-Functional Requirements

- [ ] Performance targets
- [ ] Security requirements
- [ ] Accessibility standards

### Quality Gates

- [ ] Test coverage requirements
- [ ] Documentation completeness
- [ ] Code review approval

## Success Metrics

[Detailed KPIs and measurement methods]

## Dependencies & Prerequisites

[Detailed dependency analysis]

## Risk Analysis & Mitigation

[Comprehensive risk assessment]

## Resource Requirements

[Team, time, infrastructure needs]

## Future Considerations

[Extensibility and long-term vision]

## Documentation Plan

[What docs need updating]

## References & Research

### Internal References

- Architecture decisions: [file_path:line_number]
- Similar features: [file_path:line_number]
- Configuration: [file_path:line_number]

### External References

- Framework documentation: [url]
- Best practices guide: [url]
- Industry standards: [url]

### Related Work

- Previous PRs: #[pr_numbers]
- Related issues: #[issue_numbers]
- Design documents: [links]
```

### 5. Issue Creation & Formatting

<thinking>
Apply best practices for clarity and actionability, making the issue easy to scan and understand
</thinking>

**Content Formatting:**

- [ ] Use clear, descriptive headings with proper hierarchy (##, ###)
- [ ] Include code examples in triple backticks with language syntax highlighting
- [ ] Add screenshots/mockups if UI-related (drag & drop or use image hosting)
- [ ] Use task lists (- [ ]) for trackable items that can be checked off
- [ ] Add collapsible sections for lengthy logs or optional details using `<details>` tags
- [ ] Apply appropriate emoji for visual scanning (🐛 bug, ✨ feature, 📚 docs, ♻️ refactor)

**Cross-Referencing:**

- [ ] Link to related issues/PRs using #number format
- [ ] Reference specific commits with SHA hashes when relevant
- [ ] Link to code using GitHub's permalink feature (press 'y' for permanent link)
- [ ] Mention relevant team members with @username if needed
- [ ] Add links to external resources with descriptive text

**Code & Examples:**

````markdown
# Good example with syntax highlighting and line references


```ruby
# app/services/user_service.rb:42
def process_user(user)

# Implementation here

end
```

# Collapsible error logs

<details>
<summary>Full error stacktrace</summary>

`Error details here...`

</details>
````

**AI-Era Considerations:**

- [ ] Account for accelerated development with AI pair programming
- [ ] Include prompts or instructions that worked well during research
- [ ] Note which AI tools were used for initial exploration (Claude, Copilot, etc.)
- [ ] Emphasize comprehensive testing given rapid implementation
- [ ] Document any AI-generated code that needs human review

### 6. Final Review & Submission

**Pre-submission Checklist:**

- [ ] Title is searchable and descriptive
- [ ] Labels accurately categorize the issue
- [ ] All template sections are complete
- [ ] Links and references are working
- [ ] Acceptance criteria are measurable
- [ ] Add names of files in pseudo code examples and todo lists
- [ ] Add an ERD mermaid diagram if applicable for new model changes

## Output Format

**Filename:** Use the date and kebab-case filename from Step 2 Title & Categorization.

```
docs/plans/YYYY-MM-DD-<type>-<descriptive-name>-plan.md
```

Examples:
- ✅ `docs/plans/2026-01-15-feat-user-authentication-flow-plan.md`
- ✅ `docs/plans/2026-02-03-fix-checkout-race-condition-plan.md`
- ✅ `docs/plans/2026-03-10-refactor-api-client-extraction-plan.md`
- ❌ `docs/plans/2026-01-15-feat-thing-plan.md` (not descriptive - what "thing"?)
- ❌ `docs/plans/2026-01-15-feat-new-feature-plan.md` (too vague - what feature?)
- ❌ `docs/plans/2026-01-15-feat: user auth-plan.md` (invalid characters - colon and space)
- ❌ `docs/plans/feat-user-auth-plan.md` (missing date prefix)

## Post-Generation Options

After writing the plan file, use the **AskUserQuestion tool** to present these options:

**Question:** "Plan ready at `docs/plans/YYYY-MM-DD-<type>-<name>-plan.md`. What would you like to do next?"

**Options:**
1. **Open plan in editor** - Open the plan file for review
2. **Run `/deepen-plan`** - Enhance each section with parallel research agents (best practices, performance, UI)
3. **Run `/technical_review`** - Technical feedback from code-focused reviewers (DHH, Kieran, Simplicity)
4. **Review and refine** - Improve the document through structured self-review
5. **Start `/workflows:work`** - Begin implementing this plan locally
6. **Start `/workflows:work` on remote** - Begin implementing in Claude Code on the web (use `&` to run in background)
7. **Create Issue** - Create issue in project tracker (GitHub/Linear)

Based on selection:
- **Open plan in editor** → Run `open docs/plans/<plan_filename>.md` to open the file in the user's default editor
- **`/deepen-plan`** → Call the /deepen-plan command with the plan file path to enhance with research
- **`/technical_review`** → Call the /technical_review command with the plan file path
- **Review and refine** → Load `document-review` skill.
- **`/workflows:work`** → Call the /workflows:work command with the plan file path
- **`/workflows:work` on remote** → Run `/workflows:work docs/plans/<plan_filename>.md &` to start work in background for Claude Code web
- **Create Issue** → See "Issue Creation" section below
- **Other** (automatically provided) → Accept free text for rework or specific changes

**Note:** If running `/workflows:plan` with ultrathink enabled, automatically run `/deepen-plan` after plan creation for maximum depth and grounding.

Loop back to options after Simplify or Other changes until user selects `/workflows:work` or `/technical_review`.

## Issue Creation

When user selects "Create Issue", detect their project tracker from CLAUDE.md:

1. **Check for tracker preference** in user's CLAUDE.md (global or project):
   - Look for `project_tracker: github` or `project_tracker: linear`
   - Or look for mentions of "GitHub Issues" or "Linear" in their workflow section

2. **If GitHub:**

   Use the title and type from Step 2 (already in context - no need to re-read the file):

   ```bash
   gh issue create --title "<type>: <title>" --body-file <plan_path>
   ```

3. **If Linear:**

   ```bash
   linear issue create --title "<title>" --description "$(cat <plan_path>)"
   ```

4. **If no tracker configured:**
   Ask user: "Which project tracker do you use? (GitHub/Linear/Other)"
   - Suggest adding `project_tracker: github` or `project_tracker: linear` to their CLAUDE.md

5. **After creation:**
   - Display the issue URL
   - Ask if they want to proceed to `/workflows:work` or `/technical_review`

NEVER CODE! Just research and write the plan.

---

## /review

_Source: `cache/every-marketplace/compound-engineering/2.31.0/commands/workflows/review.md`_

---
name: workflows:review
description: Perform exhaustive code reviews using multi-agent analysis, ultra-thinking, and worktrees
argument-hint: "[PR number, GitHub URL, branch name, or latest]"
---

# Review Command

<command_purpose> Perform exhaustive code reviews using multi-agent analysis, ultra-thinking, and Git worktrees for deep local inspection. </command_purpose>

## Introduction

<role>Senior Code Review Architect with expertise in security, performance, architecture, and quality assurance</role>

## Prerequisites

<requirements>
- Git repository with GitHub CLI (`gh`) installed and authenticated
- Clean main/master branch
- Proper permissions to create worktrees and access the repository
- For document reviews: Path to a markdown file or document
</requirements>

## Main Tasks

### 1. Determine Review Target & Setup (ALWAYS FIRST)

<review_target> #$ARGUMENTS </review_target>

<thinking>
First, I need to determine the review target type and set up the code for analysis.
</thinking>

#### Immediate Actions:

<task_list>

- [ ] Determine review type: PR number (numeric), GitHub URL, file path (.md), or empty (current branch)
- [ ] Check current git branch
- [ ] If ALREADY on the target branch (PR branch, requested branch name, or the branch already checked out for review) → proceed with analysis on current branch
- [ ] If DIFFERENT branch than the review target → offer to use worktree: "Use git-worktree skill for isolated Call `skill: git-worktree` with branch name
- [ ] Fetch PR metadata using `gh pr view --json` for title, body, files, linked issues
- [ ] Set up language-specific analysis tools
- [ ] Prepare security scanning environment
- [ ] Make sure we are on the branch we are reviewing. Use gh pr checkout to switch to the branch or manually checkout the branch.

Ensure that the code is ready for analysis (either in worktree or on current branch). ONLY then proceed to the next step.

</task_list>

#### Protected Artifacts

<protected_artifacts>
The following paths are compound-engineering pipeline artifacts and must never be flagged for deletion, removal, or gitignore by any review agent:

- `docs/plans/*.md` — Plan files created by `/workflows:plan`. These are living documents that track implementation progress (checkboxes are checked off by `/workflows:work`).
- `docs/solutions/*.md` — Solution documents created during the pipeline.

If a review agent flags any file in these directories for cleanup or removal, discard that finding during synthesis. Do not create a todo for it.
</protected_artifacts>

#### Parallel Agents to review the PR:

<parallel_tasks>

Run ALL or most of these agents at the same time:

1. Task kieran-rails-reviewer(PR content)
2. Task dhh-rails-reviewer(PR title)
3. If turbo is used: Task rails-turbo-expert(PR content)
4. Task git-history-analyzer(PR content)
5. Task dependency-detective(PR content)
6. Task pattern-recognition-specialist(PR content)
7. Task architecture-strategist(PR content)
8. Task code-philosopher(PR content)
9. Task security-sentinel(PR content)
10. Task performance-oracle(PR content)
11. Task devops-harmony-analyst(PR content)
12. Task data-integrity-guardian(PR content)
13. Task agent-native-reviewer(PR content) - Verify new features are agent-accessible

</parallel_tasks>

#### Conditional Agents (Run if applicable):

<conditional_agents>

These agents are run ONLY when the PR matches specific criteria. Check the PR files list to determine if they apply:

**If PR contains database migrations (db/migrate/*.rb files) or data backfills:**

14. Task data-migration-expert(PR content) - Validates ID mappings match production, checks for swapped values, verifies rollback safety
15. Task deployment-verification-agent(PR content) - Creates Go/No-Go deployment checklist with SQL verification queries

**When to run migration agents:**
- PR includes files matching `db/migrate/*.rb`
- PR modifies columns that store IDs, enums, or mappings
- PR includes data backfill scripts or rake tasks
- PR changes how data is read/written (e.g., changing from FK to string column)
- PR title/body mentions: migration, backfill, data transformation, ID mapping

**What these agents check:**
- `data-migration-expert`: Verifies hard-coded mappings match production reality (prevents swapped IDs), checks for orphaned associations, validates dual-write patterns
- `deployment-verification-agent`: Produces executable pre/post-deploy checklists with SQL queries, rollback procedures, and monitoring plans

</conditional_agents>

### 4. Ultra-Thinking Deep Dive Phases

<ultrathink_instruction> For each phase below, spend maximum cognitive effort. Think step by step. Consider all angles. Question assumptions. And bring all reviews in a synthesis to the user.</ultrathink_instruction>

<deliverable>
Complete system context map with component interactions
</deliverable>

#### Phase 3: Stakeholder Perspective Analysis

<thinking_prompt> ULTRA-THINK: Put yourself in each stakeholder's shoes. What matters to them? What are their pain points? </thinking_prompt>

<stakeholder_perspectives>

1. **Developer Perspective** <questions>

   - How easy is this to understand and modify?
   - Are the APIs intuitive?
   - Is debugging straightforward?
   - Can I test this easily? </questions>

2. **Operations Perspective** <questions>

   - How do I deploy this safely?
   - What metrics and logs are available?
   - How do I troubleshoot issues?
   - What are the resource requirements? </questions>

3. **End User Perspective** <questions>

   - Is the feature intuitive?
   - Are error messages helpful?
   - Is performance acceptable?
   - Does it solve my problem? </questions>

4. **Security Team Perspective** <questions>

   - What's the attack surface?
   - Are there compliance requirements?
   - How is data protected?
   - What are the audit capabilities? </questions>

5. **Business Perspective** <questions>
   - What's the ROI?
   - Are there legal/compliance risks?
   - How does this affect time-to-market?
   - What's the total cost of ownership? </questions> </stakeholder_perspectives>

#### Phase 4: Scenario Exploration

<thinking_prompt> ULTRA-THINK: Explore edge cases and failure scenarios. What could go wrong? How does the system behave under stress? </thinking_prompt>

<scenario_checklist>

- [ ] **Happy Path**: Normal operation with valid inputs
- [ ] **Invalid Inputs**: Null, empty, malformed data
- [ ] **Boundary Conditions**: Min/max values, empty collections
- [ ] **Concurrent Access**: Race conditions, deadlocks
- [ ] **Scale Testing**: 10x, 100x, 1000x normal load
- [ ] **Network Issues**: Timeouts, partial failures
- [ ] **Resource Exhaustion**: Memory, disk, connections
- [ ] **Security Attacks**: Injection, overflow, DoS
- [ ] **Data Corruption**: Partial writes, inconsistency
- [ ] **Cascading Failures**: Downstream service issues </scenario_checklist>

### 6. Multi-Angle Review Perspectives

#### Technical Excellence Angle

- Code craftsmanship evaluation
- Engineering best practices
- Technical documentation quality
- Tooling and automation assessment

#### Business Value Angle

- Feature completeness validation
- Performance impact on users
- Cost-benefit analysis
- Time-to-market considerations

#### Risk Management Angle

- Security risk assessment
- Operational risk evaluation
- Compliance risk verification
- Technical debt accumulation

#### Team Dynamics Angle

- Code review etiquette
- Knowledge sharing effectiveness
- Collaboration patterns
- Mentoring opportunities

### 4. Simplification and Minimalism Review

Run the Task code-simplicity-reviewer() to see if we can simplify the code.

### 5. Findings Synthesis and Todo Creation Using file-todos Skill

<critical_requirement> ALL findings MUST be stored in the todos/ directory using the file-todos skill. Create todo files immediately after synthesis - do NOT present findings for user approval first. Use the skill for structured todo management. </critical_requirement>

#### Step 1: Synthesize All Findings

<thinking>
Consolidate all agent reports into a categorized list of findings.
Remove duplicates, prioritize by severity and impact.
</thinking>

<synthesis_tasks>

- [ ] Collect findings from all parallel agents
- [ ] Discard any findings that recommend deleting or gitignoring files in `docs/plans/` or `docs/solutions/` (see Protected Artifacts above)
- [ ] Categorize by type: security, performance, architecture, quality, etc.
- [ ] Assign severity levels: 🔴 CRITICAL (P1), 🟡 IMPORTANT (P2), 🔵 NICE-TO-HAVE (P3)
- [ ] Remove duplicate or overlapping findings
- [ ] Estimate effort for each finding (Small/Medium/Large)

</synthesis_tasks>

#### Step 2: Create Todo Files Using file-todos Skill

<critical_instruction> Use the file-todos skill to create todo files for ALL findings immediately. Do NOT present findings one-by-one asking for user approval. Create all todo files in parallel using the skill, then summarize results to user. </critical_instruction>

**Implementation Options:**

**Option A: Direct File Creation (Fast)**

- Create todo files directly using Write tool
- All findings in parallel for speed
- Use standard template from `.claude/skills/file-todos/assets/todo-template.md`
- Follow naming convention: `{issue_id}-pending-{priority}-{description}.md`

**Option B: Sub-Agents in Parallel (Recommended for Scale)** For large PRs with 15+ findings, use sub-agents to create finding files in parallel:

```bash
# Launch multiple finding-creator agents in parallel
Task() - Create todos for first finding
Task() - Create todos for second finding
Task() - Create todos for third finding
etc. for each finding.
```

Sub-agents can:

- Process multiple findings simultaneously
- Write detailed todo files with all sections filled
- Organize findings by severity
- Create comprehensive Proposed Solutions
- Add acceptance criteria and work logs
- Complete much faster than sequential processing

**Execution Strategy:**

1. Synthesize all findings into categories (P1/P2/P3)
2. Group findings by severity
3. Launch 3 parallel sub-agents (one per severity level)
4. Each sub-agent creates its batch of todos using the file-todos skill
5. Consolidate results and present summary

**Process (Using file-todos Skill):**

1. For each finding:

   - Determine severity (P1/P2/P3)
   - Write detailed Problem Statement and Findings
   - Create 2-3 Proposed Solutions with pros/cons/effort/risk
   - Estimate effort (Small/Medium/Large)
   - Add acceptance criteria and work log

2. Use file-todos skill for structured todo management:

   ```bash
   skill: file-todos
   ```

   The skill provides:

   - Template location: `.claude/skills/file-todos/assets/todo-template.md`
   - Naming convention: `{issue_id}-{status}-{priority}-{description}.md`
   - YAML frontmatter structure: status, priority, issue_id, tags, dependencies
   - All required sections: Problem Statement, Findings, Solutions, etc.

3. Create todo files in parallel:

   ```bash
   {next_id}-pending-{priority}-{description}.md
   ```

4. Examples:

   ```
   001-pending-p1-path-traversal-vulnerability.md
   002-pending-p1-api-response-validation.md
   003-pending-p2-concurrency-limit.md
   004-pending-p3-unused-parameter.md
   ```

5. Follow template structure from file-todos skill: `.claude/skills/file-todos/assets/todo-template.md`

**Todo File Structure (from template):**

Each todo must include:

- **YAML frontmatter**: status, priority, issue_id, tags, dependencies
- **Problem Statement**: What's broken/missing, why it matters
- **Findings**: Discoveries from agents with evidence/location
- **Proposed Solutions**: 2-3 options, each with pros/cons/effort/risk
- **Recommended Action**: (Filled during triage, leave blank initially)
- **Technical Details**: Affected files, components, database changes
- **Acceptance Criteria**: Testable checklist items
- **Work Log**: Dated record with actions and learnings
- **Resources**: Links to PR, issues, documentation, similar patterns

**File naming convention:**

```
{issue_id}-{status}-{priority}-{description}.md

Examples:
- 001-pending-p1-security-vulnerability.md
- 002-pending-p2-performance-optimization.md
- 003-pending-p3-code-cleanup.md
```

**Status values:**

- `pending` - New findings, needs triage/decision
- `ready` - Approved by manager, ready to work
- `complete` - Work finished

**Priority values:**

- `p1` - Critical (blocks merge, security/data issues)
- `p2` - Important (should fix, architectural/performance)
- `p3` - Nice-to-have (enhancements, cleanup)

**Tagging:** Always add `code-review` tag, plus: `security`, `performance`, `architecture`, `rails`, `quality`, etc.

#### Step 3: Summary Report

After creating all todo files, present comprehensive summary:

````markdown
## ✅ Code Review Complete

**Review Target:** PR #XXXX - [PR Title] **Branch:** [branch-name]

### Findings Summary:

- **Total Findings:** [X]
- **🔴 CRITICAL (P1):** [count] - BLOCKS MERGE
- **🟡 IMPORTANT (P2):** [count] - Should Fix
- **🔵 NICE-TO-HAVE (P3):** [count] - Enhancements

### Created Todo Files:

**P1 - Critical (BLOCKS MERGE):**

- `001-pending-p1-{finding}.md` - {description}
- `002-pending-p1-{finding}.md` - {description}

**P2 - Important:**

- `003-pending-p2-{finding}.md` - {description}
- `004-pending-p2-{finding}.md` - {description}

**P3 - Nice-to-Have:**

- `005-pending-p3-{finding}.md` - {description}

### Review Agents Used:

- kieran-rails-reviewer
- security-sentinel
- performance-oracle
- architecture-strategist
- agent-native-reviewer
- [other agents]

### Next Steps:

1. **Address P1 Findings**: CRITICAL - must be fixed before merge

   - Review each P1 todo in detail
   - Implement fixes or request exemption
   - Verify fixes before merging PR

2. **Triage All Todos**:
   ```bash
   ls todos/*-pending-*.md  # View all pending todos
   /triage                  # Use slash command for interactive triage
   ```
````

3. **Work on Approved Todos**:

   ```bash
   /resolve_todo_parallel  # Fix all approved items efficiently
   ```

4. **Track Progress**:
   - Rename file when status changes: pending → ready → complete
   - Update Work Log as you work
   - Commit todos: `git add todos/ && git commit -m "refactor: add code review findings"`

### Severity Breakdown:

**🔴 P1 (Critical - Blocks Merge):**

- Security vulnerabilities
- Data corruption risks
- Breaking changes
- Critical architectural issues

**🟡 P2 (Important - Should Fix):**

- Performance issues
- Significant architectural concerns
- Major code quality problems
- Reliability issues

**🔵 P3 (Nice-to-Have):**

- Minor improvements
- Code cleanup
- Optimization opportunities
- Documentation updates

```

### 7. End-to-End Testing (Optional)

<detect_project_type>

**First, detect the project type from PR files:**

| Indicator | Project Type |
|-----------|--------------|
| `*.xcodeproj`, `*.xcworkspace`, `Package.swift` (iOS) | iOS/macOS |
| `Gemfile`, `package.json`, `app/views/*`, `*.html.*` | Web |
| Both iOS files AND web files | Hybrid (test both) |

</detect_project_type>

<offer_testing>

After presenting the Summary Report, offer appropriate testing based on project type:

**For Web Projects:**
```markdown
**"Want to run browser tests on the affected pages?"**
1. Yes - run `/test-browser`
2. No - skip
```

**For iOS Projects:**
```markdown
**"Want to run Xcode simulator tests on the app?"**
1. Yes - run `/xcode-test`
2. No - skip
```

**For Hybrid Projects (e.g., Rails + Hotwire Native):**
```markdown
**"Want to run end-to-end tests?"**
1. Web only - run `/test-browser`
2. iOS only - run `/xcode-test`
3. Both - run both commands
4. No - skip
```

</offer_testing>

#### If User Accepts Web Testing:

Spawn a subagent to run browser tests (preserves main context):

```
Task general-purpose("Run /test-browser for PR #[number]. Test all affected pages, check for console errors, handle failures by creating todos and fixing.")
```

The subagent will:
1. Identify pages affected by the PR
2. Navigate to each page and capture snapshots (using Playwright MCP or agent-browser CLI)
3. Check for console errors
4. Test critical interactions
5. Pause for human verification on OAuth/email/payment flows
6. Create P1 todos for any failures
7. Fix and retry until all tests pass

**Standalone:** `/test-browser [PR number]`

#### If User Accepts iOS Testing:

Spawn a subagent to run Xcode tests (preserves main context):

```
Task general-purpose("Run /xcode-test for scheme [name]. Build for simulator, install, launch, take screenshots, check for crashes.")
```

The subagent will:
1. Verify XcodeBuildMCP is installed
2. Discover project and schemes
3. Build for iOS Simulator
4. Install and launch app
5. Take screenshots of key screens
6. Capture console logs for errors
7. Pause for human verification (Sign in with Apple, push, IAP)
8. Create P1 todos for any failures
9. Fix and retry until all tests pass

**Standalone:** `/xcode-test [scheme]`

### Important: P1 Findings Block Merge

Any **🔴 P1 (CRITICAL)** findings must be addressed before merging the PR. Present these prominently and ensure they're resolved before accepting the PR.
```

---

## /brainstorm

_Source: `cache/every-marketplace/compound-engineering/2.31.0/commands/workflows/brainstorm.md`_

---
name: workflows:brainstorm
description: Explore requirements and approaches through collaborative dialogue before planning implementation
argument-hint: "[feature idea or problem to explore]"
---

# Brainstorm a Feature or Improvement

**Note: The current year is 2026.** Use this when dating brainstorm documents.

Brainstorming helps answer **WHAT** to build through collaborative dialogue. It precedes `/workflows:plan`, which answers **HOW** to build it.

**Process knowledge:** Load the `brainstorming` skill for detailed question techniques, approach exploration patterns, and YAGNI principles.

## Feature Description

<feature_description> #$ARGUMENTS </feature_description>

**If the feature description above is empty, ask the user:** "What would you like to explore? Please describe the feature, problem, or improvement you're thinking about."

Do not proceed until you have a feature description from the user.

## Execution Flow

### Phase 0: Assess Requirements Clarity

Evaluate whether brainstorming is needed based on the feature description.

**Clear requirements indicators:**
- Specific acceptance criteria provided
- Referenced existing patterns to follow
- Described exact expected behavior
- Constrained, well-defined scope

**If requirements are already clear:**
Use **AskUserQuestion tool** to suggest: "Your requirements seem detailed enough to proceed directly to planning. Should I run `/workflows:plan` instead, or would you like to explore the idea further?"

### Phase 1: Understand the Idea

#### 1.1 Repository Research (Lightweight)

Run a quick repo scan to understand existing patterns:

- Task repo-research-analyst("Understand existing patterns related to: <feature_description>")

Focus on: similar features, established patterns, CLAUDE.md guidance.

#### 1.2 Collaborative Dialogue

Use the **AskUserQuestion tool** to ask questions **one at a time**.

**Guidelines (see `brainstorming` skill for detailed techniques):**
- Prefer multiple choice when natural options exist
- Start broad (purpose, users) then narrow (constraints, edge cases)
- Validate assumptions explicitly
- Ask about success criteria

**Exit condition:** Continue until the idea is clear OR user says "proceed"

### Phase 2: Explore Approaches

Propose **2-3 concrete approaches** based on research and conversation.

For each approach, provide:
- Brief description (2-3 sentences)
- Pros and cons
- When it's best suited

Lead with your recommendation and explain why. Apply YAGNI—prefer simpler solutions.

Use **AskUserQuestion tool** to ask which approach the user prefers.

### Phase 3: Capture the Design

Write a brainstorm document to `docs/brainstorms/YYYY-MM-DD-<topic>-brainstorm.md`.

**Document structure:** See the `brainstorming` skill for the template format. Key sections: What We're Building, Why This Approach, Key Decisions, Open Questions.

Ensure `docs/brainstorms/` directory exists before writing.

### Phase 4: Handoff

Use **AskUserQuestion tool** to present next steps:

**Question:** "Brainstorm captured. What would you like to do next?"

**Options:**
1. **Review and refine** - Improve the document through structured self-review
2. **Proceed to planning** - Run `/workflows:plan` (will auto-detect this brainstorm)
3. **Done for now** - Return later

**If user selects "Review and refine":**

Load the `document-review` skill and apply it to the brainstorm document.

When document-review returns "Review complete", present next steps:

1. **Move to planning** - Continue to `/workflows:plan` with this document
2. **Done for now** - Brainstorming complete. To start planning later: `/workflows:plan [document-path]`

## Output Summary

When complete, display:

```
Brainstorm complete!

Document: docs/brainstorms/YYYY-MM-DD-<topic>-brainstorm.md

Key decisions:
- [Decision 1]
- [Decision 2]

Next: Run `/workflows:plan` when ready to implement.
```

## Important Guidelines

- **Stay focused on WHAT, not HOW** - Implementation details belong in the plan
- **Ask one question at a time** - Don't overwhelm
- **Apply YAGNI** - Prefer simpler approaches
- **Keep outputs concise** - 200-300 words per section max

NEVER CODE! Just explore and document decisions.

---

## /create-agent-skill

_Source: `cache/every-marketplace/compound-engineering/2.31.0/commands/create-agent-skill.md`_

---
name: create-agent-skill
description: Create or edit Claude Code skills with expert guidance on structure and best practices
allowed-tools: Skill(create-agent-skills)
argument-hint: [skill description or requirements]
disable-model-invocation: true
---

Invoke the create-agent-skills skill for: $ARGUMENTS

---

## /slfg

_Source: `cache/every-marketplace/compound-engineering/2.31.0/commands/slfg.md`_

---
name: slfg
description: Full autonomous engineering workflow using swarm mode for parallel execution
argument-hint: "[feature description]"
disable-model-invocation: true
---

Swarm-enabled LFG. Run these steps in order, parallelizing where indicated.

## Sequential Phase

1. `/ralph-wiggum:ralph-loop "finish all slash commands" --completion-promise "DONE"`
2. `/workflows:plan $ARGUMENTS`
3. `/compound-engineering:deepen-plan`
4. `/workflows:work` — **Use swarm mode**: Make a Task list and launch an army of agent swarm subagents to build the plan

## Parallel Phase

After work completes, launch steps 5 and 6 as **parallel swarm agents** (both only need code to be written):

5. `/workflows:review` — spawn as background Task agent
6. `/compound-engineering:test-browser` — spawn as background Task agent

Wait for both to complete before continuing.

## Finalize Phase

7. `/compound-engineering:resolve_todo_parallel` — resolve any findings from the review
8. `/compound-engineering:feature-video` — record the final walkthrough and add to PR
9. Output `<promise>DONE</promise>` when video is in PR

Start with step 1 now.

---

## /test-xcode

_Source: `cache/every-marketplace/compound-engineering/2.31.0/commands/test-xcode.md`_

---
name: test-xcode
description: Build and test iOS apps on simulator using XcodeBuildMCP
argument-hint: "[scheme name or 'current' to use default]"
disable-model-invocation: true
---

# Xcode Test Command

<command_purpose>Build, install, and test iOS apps on the simulator using XcodeBuildMCP. Captures screenshots, logs, and verifies app behavior.</command_purpose>

## Introduction

<role>iOS QA Engineer specializing in simulator-based testing</role>

This command tests iOS/macOS apps by:
- Building for simulator
- Installing and launching the app
- Taking screenshots of key screens
- Capturing console logs for errors
- Supporting human verification for external flows

## Prerequisites

<requirements>
- Xcode installed with command-line tools
- XcodeBuildMCP server connected
- Valid Xcode project or workspace
- At least one iOS Simulator available
</requirements>

## Main Tasks

### 0. Verify XcodeBuildMCP is Installed

<check_mcp_installed>

**First, check if XcodeBuildMCP tools are available.**

Try calling:
```
mcp__xcodebuildmcp__list_simulators({})
```

**If the tool is not found or errors:**

Tell the user:
```markdown
**XcodeBuildMCP not installed**

Please install the XcodeBuildMCP server first:

\`\`\`bash
claude mcp add XcodeBuildMCP -- npx xcodebuildmcp@latest
\`\`\`

Then restart Claude Code and run `/xcode-test` again.
```

**Do NOT proceed** until XcodeBuildMCP is confirmed working.

</check_mcp_installed>

### 1. Discover Project and Scheme

<discover_project>

**Find available projects:**
```
mcp__xcodebuildmcp__discover_projs({})
```

**List schemes for the project:**
```
mcp__xcodebuildmcp__list_schemes({ project_path: "/path/to/Project.xcodeproj" })
```

**If argument provided:**
- Use the specified scheme name
- Or "current" to use the default/last-used scheme

</discover_project>

### 2. Boot Simulator

<boot_simulator>

**List available simulators:**
```
mcp__xcodebuildmcp__list_simulators({})
```

**Boot preferred simulator (iPhone 15 Pro recommended):**
```
mcp__xcodebuildmcp__boot_simulator({ simulator_id: "[uuid]" })
```

**Wait for simulator to be ready:**
Check simulator state before proceeding with installation.

</boot_simulator>

### 3. Build the App

<build_app>

**Build for iOS Simulator:**
```
mcp__xcodebuildmcp__build_ios_sim_app({
  project_path: "/path/to/Project.xcodeproj",
  scheme: "[scheme_name]"
})
```

**Handle build failures:**
- Capture build errors
- Create P1 todo for each build error
- Report to user with specific error details

**On success:**
- Note the built app path for installation
- Proceed to installation step

</build_app>

### 4. Install and Launch

<install_launch>

**Install app on simulator:**
```
mcp__xcodebuildmcp__install_app_on_simulator({
  app_path: "/path/to/built/App.app",
  simulator_id: "[uuid]"
})
```

**Launch the app:**
```
mcp__xcodebuildmcp__launch_app_on_simulator({
  bundle_id: "[app.bundle.id]",
  simulator_id: "[uuid]"
})
```

**Start capturing logs:**
```
mcp__xcodebuildmcp__capture_sim_logs({
  simulator_id: "[uuid]",
  bundle_id: "[app.bundle.id]"
})
```

</install_launch>

### 5. Test Key Screens

<test_screens>

For each key screen in the app:

**Take screenshot:**
```
mcp__xcodebuildmcp__take_screenshot({
  simulator_id: "[uuid]",
  filename: "screen-[name].png"
})
```

**Review screenshot for:**
- UI elements rendered correctly
- No error messages visible
- Expected content displayed
- Layout looks correct

**Check logs for errors:**
```
mcp__xcodebuildmcp__get_sim_logs({ simulator_id: "[uuid]" })
```

Look for:
- Crashes
- Exceptions
- Error-level log messages
- Failed network requests

</test_screens>

### 6. Human Verification (When Required)

<human_verification>

Pause for human input when testing touches:

| Flow Type | What to Ask |
|-----------|-------------|
| Sign in with Apple | "Please complete Sign in with Apple on the simulator" |
| Push notifications | "Send a test push and confirm it appears" |
| In-app purchases | "Complete a sandbox purchase" |
| Camera/Photos | "Grant permissions and verify camera works" |
| Location | "Allow location access and verify map updates" |

Use AskUserQuestion:
```markdown
**Human Verification Needed**

This test requires [flow type]. Please:
1. [Action to take on simulator]
2. [What to verify]

Did it work correctly?
1. Yes - continue testing
2. No - describe the issue
```

</human_verification>

### 7. Handle Failures

<failure_handling>

When a test fails:

1. **Document the failure:**
   - Take screenshot of error state
   - Capture console logs
   - Note reproduction steps

2. **Ask user how to proceed:**
   ```markdown
   **Test Failed: [screen/feature]**

   Issue: [description]
   Logs: [relevant error messages]

   How to proceed?
   1. Fix now - I'll help debug and fix
   2. Create todo - Add to todos/ for later
   3. Skip - Continue testing other screens
   ```

3. **If "Fix now":**
   - Investigate the issue in code
   - Propose a fix
   - Rebuild and retest

4. **If "Create todo":**
   - Create `{id}-pending-p1-xcode-{description}.md`
   - Continue testing

</failure_handling>

### 8. Test Summary

<test_summary>

After all tests complete, present summary:

```markdown
## 📱 Xcode Test Results

**Project:** [project name]
**Scheme:** [scheme name]
**Simulator:** [simulator name]

### Build: ✅ Success / ❌ Failed

### Screens Tested: [count]

| Screen | Status | Notes |
|--------|--------|-------|
| Launch | ✅ Pass | |
| Home | ✅ Pass | |
| Settings | ❌ Fail | Crash on tap |
| Profile | ⏭️ Skip | Requires login |

### Console Errors: [count]
- [List any errors found]

### Human Verifications: [count]
- Sign in with Apple: ✅ Confirmed
- Push notifications: ✅ Confirmed

### Failures: [count]
- Settings screen - crash on navigation

### Created Todos: [count]
- `006-pending-p1-xcode-settings-crash.md`

### Result: [PASS / FAIL / PARTIAL]
```

</test_summary>

### 9. Cleanup

<cleanup>

After testing:

**Stop log capture:**
```
mcp__xcodebuildmcp__stop_log_capture({ simulator_id: "[uuid]" })
```

**Optionally shut down simulator:**
```
mcp__xcodebuildmcp__shutdown_simulator({ simulator_id: "[uuid]" })
```

</cleanup>

## Quick Usage Examples

```bash
# Test with default scheme
/xcode-test

# Test specific scheme
/xcode-test MyApp-Debug

# Test after making changes
/xcode-test current
```

## Integration with /workflows:review

When reviewing PRs that touch iOS code, the `/workflows:review` command can spawn this as a subagent:

```
Task general-purpose("Run /xcode-test for scheme [name]. Build, install on simulator, test key screens, check for crashes.")
```

---

## /agent-native-audit

_Source: `cache/every-marketplace/compound-engineering/2.31.0/commands/agent-native-audit.md`_

---
name: agent-native-audit
description: Run comprehensive agent-native architecture review with scored principles
argument-hint: "[optional: specific principle to audit]"
disable-model-invocation: true
---

# Agent-Native Architecture Audit

Conduct a comprehensive review of the codebase against agent-native architecture principles, launching parallel sub-agents for each principle and producing a scored report.

## Core Principles to Audit

1. **Action Parity** - "Whatever the user can do, the agent can do"
2. **Tools as Primitives** - "Tools provide capability, not behavior"
3. **Context Injection** - "System prompt includes dynamic context about app state"
4. **Shared Workspace** - "Agent and user work in the same data space"
5. **CRUD Completeness** - "Every entity has full CRUD (Create, Read, Update, Delete)"
6. **UI Integration** - "Agent actions immediately reflected in UI"
7. **Capability Discovery** - "Users can discover what the agent can do"
8. **Prompt-Native Features** - "Features are prompts defining outcomes, not code"

## Workflow

### Step 1: Load the Agent-Native Skill

First, invoke the agent-native-architecture skill to understand all principles:

```
/compound-engineering:agent-native-architecture
```

Select option 7 (action parity) to load the full reference material.

### Step 2: Launch Parallel Sub-Agents

Launch 8 parallel sub-agents using the Task tool with `subagent_type: Explore`, one for each principle. Each agent should:

1. Enumerate ALL instances in the codebase (user actions, tools, contexts, data stores, etc.)
2. Check compliance against the principle
3. Provide a SPECIFIC SCORE like "X out of Y (percentage%)"
4. List specific gaps and recommendations

<sub-agents>

**Agent 1: Action Parity**
```
Audit for ACTION PARITY - "Whatever the user can do, the agent can do."

Tasks:
1. Enumerate ALL user actions in frontend (API calls, button clicks, form submissions)
   - Search for API service files, fetch calls, form handlers
   - Check routes and components for user interactions
2. Check which have corresponding agent tools
   - Search for agent tool definitions
   - Map user actions to agent capabilities
3. Score: "Agent can do X out of Y user actions"

Format:
## Action Parity Audit
### User Actions Found
| Action | Location | Agent Tool | Status |
### Score: X/Y (percentage%)
### Missing Agent Tools
### Recommendations
```

**Agent 2: Tools as Primitives**
```
Audit for TOOLS AS PRIMITIVES - "Tools provide capability, not behavior."

Tasks:
1. Find and read ALL agent tool files
2. Classify each as:
   - PRIMITIVE (good): read, write, store, list - enables capability without business logic
   - WORKFLOW (bad): encodes business logic, makes decisions, orchestrates steps
3. Score: "X out of Y tools are proper primitives"

Format:
## Tools as Primitives Audit
### Tool Analysis
| Tool | File | Type | Reasoning |
### Score: X/Y (percentage%)
### Problematic Tools (workflows that should be primitives)
### Recommendations
```

**Agent 3: Context Injection**
```
Audit for CONTEXT INJECTION - "System prompt includes dynamic context about app state"

Tasks:
1. Find context injection code (search for "context", "system prompt", "inject")
2. Read agent prompts and system messages
3. Enumerate what IS injected vs what SHOULD be:
   - Available resources (files, drafts, documents)
   - User preferences/settings
   - Recent activity
   - Available capabilities listed
   - Session history
   - Workspace state

Format:
## Context Injection Audit
### Context Types Analysis
| Context Type | Injected? | Location | Notes |
### Score: X/Y (percentage%)
### Missing Context
### Recommendations
```

**Agent 4: Shared Workspace**
```
Audit for SHARED WORKSPACE - "Agent and user work in the same data space"

Tasks:
1. Identify all data stores/tables/models
2. Check if agents read/write to SAME tables or separate ones
3. Look for sandbox isolation anti-pattern (agent has separate data space)

Format:
## Shared Workspace Audit
### Data Store Analysis
| Data Store | User Access | Agent Access | Shared? |
### Score: X/Y (percentage%)
### Isolated Data (anti-pattern)
### Recommendations
```

**Agent 5: CRUD Completeness**
```
Audit for CRUD COMPLETENESS - "Every entity has full CRUD"

Tasks:
1. Identify all entities/models in the codebase
2. For each entity, check if agent tools exist for:
   - Create
   - Read
   - Update
   - Delete
3. Score per entity and overall

Format:
## CRUD Completeness Audit
### Entity CRUD Analysis
| Entity | Create | Read | Update | Delete | Score |
### Overall Score: X/Y entities with full CRUD (percentage%)
### Incomplete Entities (list missing operations)
### Recommendations
```

**Agent 6: UI Integration**
```
Audit for UI INTEGRATION - "Agent actions immediately reflected in UI"

Tasks:
1. Check how agent writes/changes propagate to frontend
2. Look for:
   - Streaming updates (SSE, WebSocket)
   - Polling mechanisms
   - Shared state/services
   - Event buses
   - File watching
3. Identify "silent actions" anti-pattern (agent changes state but UI doesn't update)

Format:
## UI Integration Audit
### Agent Action → UI Update Analysis
| Agent Action | UI Mechanism | Immediate? | Notes |
### Score: X/Y (percentage%)
### Silent Actions (anti-pattern)
### Recommendations
```

**Agent 7: Capability Discovery**
```
Audit for CAPABILITY DISCOVERY - "Users can discover what the agent can do"

Tasks:
1. Check for these 7 discovery mechanisms:
   - Onboarding flow showing agent capabilities
   - Help documentation
   - Capability hints in UI
   - Agent self-describes in responses
   - Suggested prompts/actions
   - Empty state guidance
   - Slash commands (/help, /tools)
2. Score against 7 mechanisms

Format:
## Capability Discovery Audit
### Discovery Mechanism Analysis
| Mechanism | Exists? | Location | Quality |
### Score: X/7 (percentage%)
### Missing Discovery
### Recommendations
```

**Agent 8: Prompt-Native Features**
```
Audit for PROMPT-NATIVE FEATURES - "Features are prompts defining outcomes, not code"

Tasks:
1. Read all agent prompts
2. Classify each feature/behavior as defined in:
   - PROMPT (good): outcomes defined in natural language
   - CODE (bad): business logic hardcoded
3. Check if behavior changes require prompt edit vs code change

Format:
## Prompt-Native Features Audit
### Feature Definition Analysis
| Feature | Defined In | Type | Notes |
### Score: X/Y (percentage%)
### Code-Defined Features (anti-pattern)
### Recommendations
```

</sub-agents>

### Step 3: Compile Summary Report

After all agents complete, compile a summary with:

```markdown
## Agent-Native Architecture Review: [Project Name]

### Overall Score Summary

| Core Principle | Score | Percentage | Status |
|----------------|-------|------------|--------|
| Action Parity | X/Y | Z% | ✅/⚠️/❌ |
| Tools as Primitives | X/Y | Z% | ✅/⚠️/❌ |
| Context Injection | X/Y | Z% | ✅/⚠️/❌ |
| Shared Workspace | X/Y | Z% | ✅/⚠️/❌ |
| CRUD Completeness | X/Y | Z% | ✅/⚠️/❌ |
| UI Integration | X/Y | Z% | ✅/⚠️/❌ |
| Capability Discovery | X/Y | Z% | ✅/⚠️/❌ |
| Prompt-Native Features | X/Y | Z% | ✅/⚠️/❌ |

**Overall Agent-Native Score: X%**

### Status Legend
- ✅ Excellent (80%+)
- ⚠️ Partial (50-79%)
- ❌ Needs Work (<50%)

### Top 10 Recommendations by Impact

| Priority | Action | Principle | Effort |
|----------|--------|-----------|--------|

### What's Working Excellently

[List top 5 strengths]
```

## Success Criteria

- [ ] All 8 sub-agents complete their audits
- [ ] Each principle has a specific numeric score (X/Y format)
- [ ] Summary table shows all scores and status indicators
- [ ] Top 10 recommendations are prioritized by impact
- [ ] Report identifies both strengths and gaps

## Optional: Single Principle Audit

If $ARGUMENTS specifies a single principle (e.g., "action parity"), only run that sub-agent and provide detailed findings for that principle alone.

Valid arguments:
- `action parity` or `1`
- `tools` or `primitives` or `2`
- `context` or `injection` or `3`
- `shared` or `workspace` or `4`
- `crud` or `5`
- `ui` or `integration` or `6`
- `discovery` or `7`
- `prompt` or `features` or `8`

---

## /resolve_todo_parallel

_Source: `cache/every-marketplace/compound-engineering/2.31.0/commands/resolve_todo_parallel.md`_

---
name: resolve_todo_parallel
description: Resolve all pending CLI todos using parallel processing
argument-hint: "[optional: specific todo ID or pattern]"
---

Resolve all TODO comments using parallel processing.

## Workflow

### 1. Analyze

Get all unresolved TODOs from the /todos/\*.md directory

If any todo recommends deleting, removing, or gitignoring files in `docs/plans/` or `docs/solutions/`, skip it and mark it as `wont_fix`. These are compound-engineering pipeline artifacts that are intentional and permanent.

### 2. Plan

Create a TodoWrite list of all unresolved items grouped by type.Make sure to look at dependencies that might occur and prioritize the ones needed by others. For example, if you need to change a name, you must wait to do the others. Output a mermaid flow diagram showing how we can do this. Can we do everything in parallel? Do we need to do one first that leads to others in parallel? I'll put the to-dos in the mermaid diagram flow‑wise so the agent knows how to proceed in order.

### 3. Implement (PARALLEL)

Spawn a pr-comment-resolver agent for each unresolved item in parallel.

So if there are 3 comments, it will spawn 3 pr-comment-resolver agents in parallel. liek this

1. Task pr-comment-resolver(comment1)
2. Task pr-comment-resolver(comment2)
3. Task pr-comment-resolver(comment3)

Always run all in parallel subagents/Tasks for each Todo item.

### 4. Commit & Resolve

- Commit changes
- Remove the TODO from the file, and mark it as resolved.
- Push to remote

---

## /release-docs

_Source: `cache/every-marketplace/compound-engineering/2.31.0/commands/release-docs.md`_

---
name: release-docs
description: Build and update the documentation site with current plugin components
argument-hint: "[optional: --dry-run to preview changes without writing]"
disable-model-invocation: true
---

# Release Documentation Command

You are a documentation generator for the compound-engineering plugin. Your job is to ensure the documentation site at `plugins/compound-engineering/docs/` is always up-to-date with the actual plugin components.

## Overview

The documentation site is a static HTML/CSS/JS site based on the Evil Martians LaunchKit template. It needs to be regenerated whenever:

- Agents are added, removed, or modified
- Commands are added, removed, or modified
- Skills are added, removed, or modified
- MCP servers are added, removed, or modified

## Step 1: Inventory Current Components

First, count and list all current components:

```bash
# Count agents
ls plugins/compound-engineering/agents/*.md | wc -l

# Count commands
ls plugins/compound-engineering/commands/*.md | wc -l

# Count skills
ls -d plugins/compound-engineering/skills/*/ 2>/dev/null | wc -l

# Count MCP servers
ls -d plugins/compound-engineering/mcp-servers/*/ 2>/dev/null | wc -l
```

Read all component files to get their metadata:

### Agents
For each agent file in `plugins/compound-engineering/agents/*.md`:
- Extract the frontmatter (name, description)
- Note the category (Review, Research, Workflow, Design, Docs)
- Get key responsibilities from the content

### Commands
For each command file in `plugins/compound-engineering/commands/*.md`:
- Extract the frontmatter (name, description, argument-hint)
- Categorize as Workflow or Utility command

### Skills
For each skill directory in `plugins/compound-engineering/skills/*/`:
- Read the SKILL.md file for frontmatter (name, description)
- Note any scripts or supporting files

### MCP Servers
For each MCP server in `plugins/compound-engineering/mcp-servers/*/`:
- Read the configuration and README
- List the tools provided

## Step 2: Update Documentation Pages

### 2a. Update `docs/index.html`

Update the stats section with accurate counts:
```html
<div class="stats-grid">
  <div class="stat-card">
    <span class="stat-number">[AGENT_COUNT]</span>
    <span class="stat-label">Specialized Agents</span>
  </div>
  <!-- Update all stat cards -->
</div>
```

Ensure the component summary sections list key components accurately.

### 2b. Update `docs/pages/agents.html`

Regenerate the complete agents reference page:
- Group agents by category (Review, Research, Workflow, Design, Docs)
- Include for each agent:
  - Name and description
  - Key responsibilities (bullet list)
  - Usage example: `claude agent [agent-name] "your message"`
  - Use cases

### 2c. Update `docs/pages/commands.html`

Regenerate the complete commands reference page:
- Group commands by type (Workflow, Utility)
- Include for each command:
  - Name and description
  - Arguments (if any)
  - Process/workflow steps
  - Example usage

### 2d. Update `docs/pages/skills.html`

Regenerate the complete skills reference page:
- Group skills by category (Development Tools, Content & Workflow, Image Generation)
- Include for each skill:
  - Name and description
  - Usage: `claude skill [skill-name]`
  - Features and capabilities

### 2e. Update `docs/pages/mcp-servers.html`

Regenerate the MCP servers reference page:
- For each server:
  - Name and purpose
  - Tools provided
  - Configuration details
  - Supported frameworks/services

## Step 3: Update Metadata Files

Ensure counts are consistent across:

1. **`plugins/compound-engineering/.claude-plugin/plugin.json`**
   - Update `description` with correct counts
   - Update `components` object with counts
   - Update `agents`, `commands` arrays with current items

2. **`.claude-plugin/marketplace.json`**
   - Update plugin `description` with correct counts

3. **`plugins/compound-engineering/README.md`**
   - Update intro paragraph with counts
   - Update component lists

## Step 4: Validate

Run validation checks:

```bash
# Validate JSON files
cat .claude-plugin/marketplace.json | jq .
cat plugins/compound-engineering/.claude-plugin/plugin.json | jq .

# Verify counts match
echo "Agents in files: $(ls plugins/compound-engineering/agents/*.md | wc -l)"
grep -o "[0-9]* specialized agents" plugins/compound-engineering/docs/index.html

echo "Commands in files: $(ls plugins/compound-engineering/commands/*.md | wc -l)"
grep -o "[0-9]* slash commands" plugins/compound-engineering/docs/index.html
```

## Step 5: Report Changes

Provide a summary of what was updated:

```
## Documentation Release Summary

### Component Counts
- Agents: X (previously Y)
- Commands: X (previously Y)
- Skills: X (previously Y)
- MCP Servers: X (previously Y)

### Files Updated
- docs/index.html - Updated stats and component summaries
- docs/pages/agents.html - Regenerated with X agents
- docs/pages/commands.html - Regenerated with X commands
- docs/pages/skills.html - Regenerated with X skills
- docs/pages/mcp-servers.html - Regenerated with X servers
- plugin.json - Updated counts and component lists
- marketplace.json - Updated description
- README.md - Updated component lists

### New Components Added
- [List any new agents/commands/skills]

### Components Removed
- [List any removed agents/commands/skills]
```

## Dry Run Mode

If `--dry-run` is specified:
- Perform all inventory and validation steps
- Report what WOULD be updated
- Do NOT write any files
- Show diff previews of proposed changes

## Error Handling

- If component files have invalid frontmatter, report the error and skip
- If JSON validation fails, report and abort
- Always maintain a valid state - don't partially update

## Post-Release

After successful release:
1. Suggest updating CHANGELOG.md with documentation changes
2. Remind to commit with message: `docs: Update documentation site to match plugin components`
3. Remind to push changes

## Usage Examples

```bash
# Full documentation release
claude /release-docs

# Preview changes without writing
claude /release-docs --dry-run

# After adding new agents
claude /release-docs
```

---

## /technical_review

_Source: `cache/every-marketplace/compound-engineering/2.31.0/commands/technical_review.md`_

---
name: technical_review
description: Have multiple specialized agents review the technical approach and architecture of a plan in parallel
argument-hint: "[plan file path or plan content]"
disable-model-invocation: true
---

Have @agent-dhh-rails-reviewer @agent-kieran-rails-reviewer @agent-code-simplicity-reviewer review the technical approach in this plan in parallel.

---

## /feature-video

_Source: `cache/every-marketplace/compound-engineering/2.31.0/commands/feature-video.md`_

---
name: feature-video
description: Record a video walkthrough of a feature and add it to the PR description
argument-hint: "[PR number or 'current'] [optional: base URL, default localhost:3000]"
---

# Feature Video Walkthrough

<command_purpose>Record a video walkthrough demonstrating a feature, upload it, and add it to the PR description.</command_purpose>

## Introduction

<role>Developer Relations Engineer creating feature demo videos</role>

This command creates professional video walkthroughs of features for PR documentation:
- Records browser interactions using agent-browser CLI
- Demonstrates the complete user flow
- Uploads the video for easy sharing
- Updates the PR description with an embedded video

## Prerequisites

<requirements>
- Local development server running (e.g., `bin/dev`, `rails server`)
- agent-browser CLI installed
- Git repository with a PR to document
- `ffmpeg` installed (for video conversion)
- `rclone` configured (optional, for cloud upload - see rclone skill)
</requirements>

## Setup

**Check installation:**
```bash
command -v agent-browser >/dev/null 2>&1 && echo "Installed" || echo "NOT INSTALLED"
```

**Install if needed:**
```bash
npm install -g agent-browser && agent-browser install
```

See the `agent-browser` skill for detailed usage.

## Main Tasks

### 1. Parse Arguments

<parse_args>

**Arguments:** $ARGUMENTS

Parse the input:
- First argument: PR number or "current" (defaults to current branch's PR)
- Second argument: Base URL (defaults to `http://localhost:3000`)

```bash
# Get PR number for current branch if needed
gh pr view --json number -q '.number'
```

</parse_args>

### 2. Gather Feature Context

<gather_context>

**Get PR details:**
```bash
gh pr view [number] --json title,body,files,headRefName -q '.'
```

**Get changed files:**
```bash
gh pr view [number] --json files -q '.files[].path'
```

**Map files to testable routes** (same as playwright-test):

| File Pattern | Route(s) |
|-------------|----------|
| `app/views/users/*` | `/users`, `/users/:id`, `/users/new` |
| `app/controllers/settings_controller.rb` | `/settings` |
| `app/javascript/controllers/*_controller.js` | Pages using that Stimulus controller |
| `app/components/*_component.rb` | Pages rendering that component |

</gather_context>

### 3. Plan the Video Flow

<plan_flow>

Before recording, create a shot list:

1. **Opening shot**: Homepage or starting point (2-3 seconds)
2. **Navigation**: How user gets to the feature
3. **Feature demonstration**: Core functionality (main focus)
4. **Edge cases**: Error states, validation, etc. (if applicable)
5. **Success state**: Completed action/result

Ask user to confirm or adjust the flow:

```markdown
**Proposed Video Flow**

Based on PR #[number]: [title]

1. Start at: /[starting-route]
2. Navigate to: /[feature-route]
3. Demonstrate:
   - [Action 1]
   - [Action 2]
   - [Action 3]
4. Show result: [success state]

Estimated duration: ~[X] seconds

Does this look right?
1. Yes, start recording
2. Modify the flow (describe changes)
3. Add specific interactions to demonstrate
```

</plan_flow>

### 4. Setup Video Recording

<setup_recording>

**Create videos directory:**
```bash
mkdir -p tmp/videos
```

**Recording approach: Use browser screenshots as frames**

agent-browser captures screenshots at key moments, then combine into video using ffmpeg:

```bash
ffmpeg -framerate 2 -pattern_type glob -i 'tmp/screenshots/*.png' -vf "scale=1280:-1" tmp/videos/feature-demo.gif
```

</setup_recording>

### 5. Record the Walkthrough

<record_walkthrough>

Execute the planned flow, capturing each step:

**Step 1: Navigate to starting point**
```bash
agent-browser open "[base-url]/[start-route]"
agent-browser wait 2000
agent-browser screenshot tmp/screenshots/01-start.png
```

**Step 2: Perform navigation/interactions**
```bash
agent-browser snapshot -i  # Get refs
agent-browser click @e1    # Click navigation element
agent-browser wait 1000
agent-browser screenshot tmp/screenshots/02-navigate.png
```

**Step 3: Demonstrate feature**
```bash
agent-browser snapshot -i  # Get refs for feature elements
agent-browser click @e2    # Click feature element
agent-browser wait 1000
agent-browser screenshot tmp/screenshots/03-feature.png
```

**Step 4: Capture result**
```bash
agent-browser wait 2000
agent-browser screenshot tmp/screenshots/04-result.png
```

**Create video/GIF from screenshots:**

```bash
# Create directories
mkdir -p tmp/videos tmp/screenshots

# Create MP4 video (RECOMMENDED - better quality, smaller size)
# -framerate 0.5 = 2 seconds per frame (slower playback)
# -framerate 1 = 1 second per frame
ffmpeg -y -framerate 0.5 -pattern_type glob -i 'tmp/screenshots/*.png' \
  -c:v libx264 -pix_fmt yuv420p -vf "scale=1280:-2" \
  tmp/videos/feature-demo.mp4

# Create low-quality GIF for preview (small file, for GitHub embed)
ffmpeg -y -framerate 0.5 -pattern_type glob -i 'tmp/screenshots/*.png' \
  -vf "scale=640:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse" \
  -loop 0 tmp/videos/feature-demo-preview.gif
```

**Note:**
- The `-2` in MP4 scale ensures height is divisible by 2 (required for H.264)
- Preview GIF uses 640px width and 128 colors to keep file size small (~100-200KB)

</record_walkthrough>

### 6. Upload the Video

<upload_video>

**Upload with rclone:**

```bash
# Check rclone is configured
rclone listremotes

# Upload video, preview GIF, and screenshots to cloud storage
# Use --s3-no-check-bucket to avoid permission errors
rclone copy tmp/videos/ r2:kieran-claude/pr-videos/pr-[number]/ --s3-no-check-bucket --progress
rclone copy tmp/screenshots/ r2:kieran-claude/pr-videos/pr-[number]/screenshots/ --s3-no-check-bucket --progress

# List uploaded files
rclone ls r2:kieran-claude/pr-videos/pr-[number]/
```

Public URLs (R2 with public access):
```
Video: https://pub-4047722ebb1b4b09853f24d3b61467f1.r2.dev/pr-videos/pr-[number]/feature-demo.mp4
Preview: https://pub-4047722ebb1b4b09853f24d3b61467f1.r2.dev/pr-videos/pr-[number]/feature-demo-preview.gif
```

</upload_video>

### 7. Update PR Description

<update_pr>

**Get current PR body:**
```bash
gh pr view [number] --json body -q '.body'
```

**Add video section to PR description:**

If the PR already has a video section, replace it. Otherwise, append:

**IMPORTANT:** GitHub cannot embed external MP4s directly. Use a clickable GIF that links to the video:

```markdown
## Demo

[![Feature Demo]([preview-gif-url])]([video-mp4-url])

*Click to view full video*
```

Example:
```markdown
[![Feature Demo](https://pub-4047722ebb1b4b09853f24d3b61467f1.r2.dev/pr-videos/pr-137/feature-demo-preview.gif)](https://pub-4047722ebb1b4b09853f24d3b61467f1.r2.dev/pr-videos/pr-137/feature-demo.mp4)
```

**Update the PR:**
```bash
gh pr edit [number] --body "[updated body with video section]"
```

**Or add as a comment if preferred:**
```bash
gh pr comment [number] --body "## Feature Demo

![Demo]([video-url])

_Automated walkthrough of the changes in this PR_"
```

</update_pr>

### 8. Cleanup

<cleanup>

```bash
# Optional: Clean up screenshots
rm -rf tmp/screenshots

# Keep videos for reference
echo "Video retained at: tmp/videos/feature-demo.gif"
```

</cleanup>

### 9. Summary

<summary>

Present completion summary:

```markdown
## Feature Video Complete

**PR:** #[number] - [title]
**Video:** [url or local path]
**Duration:** ~[X] seconds
**Format:** [GIF/MP4]

### Shots Captured
1. [Starting point] - [description]
2. [Navigation] - [description]
3. [Feature demo] - [description]
4. [Result] - [description]

### PR Updated
- [x] Video section added to PR description
- [ ] Ready for review

**Next steps:**
- Review the video to ensure it accurately demonstrates the feature
- Share with reviewers for context
```

</summary>

## Quick Usage Examples

```bash
# Record video for current branch's PR
/feature-video

# Record video for specific PR
/feature-video 847

# Record with custom base URL
/feature-video 847 http://localhost:5000

# Record for staging environment
/feature-video current https://staging.example.com
```

## Tips

- **Keep it short**: 10-30 seconds is ideal for PR demos
- **Focus on the change**: Don't include unrelated UI
- **Show before/after**: If fixing a bug, show the broken state first (if possible)
- **Annotate if needed**: Add text overlays for complex features

---

## /heal-skill

_Source: `cache/every-marketplace/compound-engineering/2.31.0/commands/heal-skill.md`_

---
name: heal-skill
description: Fix incorrect SKILL.md files when a skill has wrong instructions or outdated API references
argument-hint: [optional: specific issue to fix]
allowed-tools: [Read, Edit, Bash(ls:*), Bash(git:*)]
disable-model-invocation: true
---

<objective>
Update a skill's SKILL.md and related files based on corrections discovered during execution.

Analyze the conversation to detect which skill is running, reflect on what went wrong, propose specific fixes, get user approval, then apply changes with optional commit.
</objective>

<context>
Skill detection: !`ls -1 ./skills/*/SKILL.md | head -5`
</context>

<quick_start>
<workflow>
1. **Detect skill** from conversation context (invocation messages, recent SKILL.md references)
2. **Reflect** on what went wrong and how you discovered the fix
3. **Present** proposed changes with before/after diffs
4. **Get approval** before making any edits
5. **Apply** changes and optionally commit
</workflow>
</quick_start>

<process>
<step_1 name="detect_skill">
Identify the skill from conversation context:

- Look for skill invocation messages
- Check which SKILL.md was recently referenced
- Examine current task context

Set: `SKILL_NAME=[skill-name]` and `SKILL_DIR=./skills/$SKILL_NAME`

If unclear, ask the user.
</step_1>

<step_2 name="reflection_and_analysis">
Focus on $ARGUMENTS if provided, otherwise analyze broader context.

Determine:
- **What was wrong**: Quote specific sections from SKILL.md that are incorrect
- **Discovery method**: Context7, error messages, trial and error, documentation lookup
- **Root cause**: Outdated API, incorrect parameters, wrong endpoint, missing context
- **Scope of impact**: Single section or multiple? Related files affected?
- **Proposed fix**: Which files, which sections, before/after for each
</step_2>

<step_3 name="scan_affected_files">
```bash
ls -la $SKILL_DIR/
ls -la $SKILL_DIR/references/ 2>/dev/null
ls -la $SKILL_DIR/scripts/ 2>/dev/null
```
</step_3>

<step_4 name="present_proposed_changes">
Present changes in this format:

```
**Skill being healed:** [skill-name]
**Issue discovered:** [1-2 sentence summary]
**Root cause:** [brief explanation]

**Files to be modified:**
- [ ] SKILL.md
- [ ] references/[file].md
- [ ] scripts/[file].py

**Proposed changes:**

### Change 1: SKILL.md - [Section name]
**Location:** Line [X] in SKILL.md

**Current (incorrect):**
```
[exact text from current file]
```

**Corrected:**
```
[new text]
```

**Reason:** [why this fixes the issue]

[repeat for each change across all files]

**Impact assessment:**
- Affects: [authentication/API endpoints/parameters/examples/etc.]

**Verification:**
These changes will prevent: [specific error that prompted this]
```
</step_4>

<step_5 name="request_approval">
```
Should I apply these changes?

1. Yes, apply and commit all changes
2. Apply but don't commit (let me review first)
3. Revise the changes (I'll provide feedback)
4. Cancel (don't make changes)

Choose (1-4):
```

**Wait for user response. Do not proceed without approval.**
</step_5>

<step_6 name="apply_changes">
Only after approval (option 1 or 2):

1. Use Edit tool for each correction across all files
2. Read back modified sections to verify
3. If option 1, commit with structured message showing what was healed
4. Confirm completion with file list
</step_6>
</process>

<success_criteria>
- Skill correctly detected from conversation context
- All incorrect sections identified with before/after
- User approved changes before application
- All edits applied across SKILL.md and related files
- Changes verified by reading back
- Commit created if user chose option 1
- Completion confirmed with file list
</success_criteria>

<verification>
Before completing:

- Read back each modified section to confirm changes applied
- Ensure cross-file consistency (SKILL.md examples match references/)
- Verify git commit created if option 1 was selected
- Check no unintended files were modified
</verification>

---

## /marketplace

_Source: `cache/claude-plugins-official/vercel/b95178c7d8df/commands/marketplace.md`_

---
description: Discover and install Vercel Marketplace integrations. Use to find databases, CMS, auth providers, and other services available on the Vercel Marketplace.
---

# Vercel Marketplace

Discover, install, and apply Vercel Marketplace integrations with guided setup and local verification.

## Preflight

1. **Project linked?** — Check for `.vercel/project.json` in the current directory or nearest parent.
   - If not found: run `vercel link` interactively, then re-run `/marketplace`.
   - Do not attempt provisioning until the project is linked.
2. **CLI available?** — Confirm `vercel` is on PATH.
   - If missing: `npm i -g vercel` (or `pnpm add -g vercel` / `bun add -g vercel`).
3. **Repo state** — Note uncommitted changes so the user can diff integration-related code changes later.
4. **Scope** — Detect monorepo (`turbo.json` or `pnpm-workspace.yaml`). If detected, confirm which package is targeted.

## Plan

The marketplace command follows an **apply-guide loop**:

1. **Discover** — Search the Marketplace catalog via `vercel integration discover`.
2. **Select** — User picks an integration (or specifies one in "$ARGUMENTS").
3. **Guide** — Fetch the agent-friendly setup guide via `vercel integration guide <name> --framework <fw>`.
4. **Install** — Run `vercel integration add <name>` for the automated happy path.
5. **Confirm env vars provisioned** — Explicitly verify required environment variables are set after provisioning.
6. **Apply code changes** — Install SDK packages and scaffold configuration code.
7. **Verify drain** — For observability integrations, confirm drain was auto-created and data is flowing.
8. **Run local health check** — Verify the integration works locally before deploying.

If "$ARGUMENTS" specifies an integration name, skip directly to the **Guide** step.

For observability integrations (Datadog, Sentry, Axiom, etc.), the flow extends with drain verification — see Step 7.

No destructive operations unless the user explicitly confirms. Package installs and code scaffolding are additive.

## Commands

### 1. Discover — Search the Marketplace Catalog

```bash
# Search all available integrations
vercel integration discover

# Filter by category
vercel integration discover --category databases
vercel integration discover --category monitoring
vercel integration discover --category auth

# List integrations already installed on this project
vercel integration list
```

Present integrations organized by category:

| Category   | Examples                          |
| ---------- | --------------------------------- |
| Database   | Neon, PlanetScale, Supabase       |
| Cache      | Upstash Redis, Upstash KV         |
| Auth       | Clerk, Auth0                      |
| CMS        | Sanity, Contentful, Prismic       |
| Payments   | Stripe, LemonSqueezy              |
| Monitoring | Datadog, Sentry, Axiom, Honeycomb |

Common replacements for sunset packages:

- **Neon** — Serverless Postgres (replaces `@vercel/postgres`)
- **Upstash** — Serverless Redis (replaces `@vercel/kv`)

### 2. Select — User Picks an Integration

If the user hasn't specified one in "$ARGUMENTS", ask which integration to set up. Accept the integration name or slug.

### 3. Guide — Fetch Setup Steps

```bash
# Get agent-friendly setup guide
vercel integration guide <name> --framework <fw>
```

Use framework-specific guides by default when framework is known. If unknown, infer from the repo and confirm with the user.

The guide returns structured setup steps: required env vars, SDK packages, code snippets, and framework-specific notes. Present these to the user.

### 4. Install — Add the Integration

```bash
vercel integration add <name>
```

`vercel integration add` is the primary scripted/AI flow. It installs against the linked project, auto-connects the integration, and auto-runs local env sync unless disabled.

If the CLI bounces to the dashboard for provider-specific completion, treat it as fallback and open the integration page directly:

```bash
vercel integration open <name>
```

Complete the web step, then continue with env verification.

After installation, the integration auto-provisions environment variables. For observability vendors (Datadog, Sentry, Axiom), this also auto-creates **log and trace drains**.

### 5. Confirm Env Vars Provisioned

Before applying code changes, verify the integration's required environment variables exist:

```bash
vercel env ls
```

Check that each variable name listed in the guide appears in the output. **Never echo variable values — check names only.**

- If local env sync was disabled or `.env.local` is stale, run:

```bash
vercel env pull .env.local --yes
```

- **All present** → Proceed to code changes.
- **Missing vars** → Tell the user which variables are missing. The integration install via `vercel integration add <name>` typically provisions these automatically. Guide the user to provision them before continuing.

### 6. Apply Code Changes

Install the SDK package:

```bash
npm install <sdk-package>   # or pnpm add / bun add
```

Then apply the code scaffolding from the guide:

- Create or update configuration files (e.g., `db.ts`, `redis.ts`, `auth.ts`)
- Add initialization code following the guide's patterns
- Respect the project's existing conventions (TypeScript vs JavaScript, import style, directory structure)

Ask the user for confirmation before writing files.

### 7. Verify Drain (Observability Integrations)

<!-- Sourced from marketplace skill: Observability Integration Path -->
Marketplace observability integrations (Datadog, Sentry, Axiom, Honeycomb, etc.) connect to Vercel's **Drains** system to receive telemetry. Understanding the data-type split is critical for correct setup.

### Data-Type Split

| Data Type          | Delivery Mechanism                                    | Integration Setup                                                                                                      |
| ------------------ | ----------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| **Logs**           | Native drain (auto-configured by Marketplace install) | `vercel integration add <vendor>` auto-creates drain                                                                   |
| **Traces**         | Native drain (OpenTelemetry-compatible)               | Same — auto-configured on install                                                                                      |
| **Speed Insights** | Custom drain endpoint only                            | Requires manual drain creation via REST API or Dashboard (`https://vercel.com/dashboard/{team}/~/settings/log-drains`) |
| **Web Analytics**  | Custom drain endpoint only                            | Requires manual drain creation via REST API or Dashboard (`https://vercel.com/dashboard/{team}/~/settings/log-drains`) |

> **Key distinction:** When you install an observability vendor via the Marketplace, it auto-configures drains for **logs and traces** only. Speed Insights and Web Analytics data require a separate, manually configured drain pointing to a custom endpoint. See `⤳ skill: observability` for drain setup details.

### Agentic Flow: Observability Vendor Setup

Follow this sequence when setting up an observability integration:

#### 1. Pick Vendor

```bash
# Discover observability integrations
vercel integration discover --category monitoring

# Get setup guide for chosen vendor
vercel integration guide datadog
```

#### 2. Install Integration

```bash
# Install — auto-provisions env vars and creates log/trace drains
vercel integration add datadog
```

#### 3. Verify Drain Created

```bash
# Confirm drain was auto-configured
curl -s -H "Authorization: Bearer $VERCEL_TOKEN" \
  "https://api.vercel.com/v1/drains?teamId=$TEAM_ID" | jq '.[] | {id, url, type, sources}'
```

Check the response for a drain pointing to the vendor's ingestion endpoint. If no drain appears, the integration may need manual drain setup — see `⤳ skill: observability` for REST API drain creation.

#### 4. Validate Endpoint

```bash
# Send a test payload to the drain
curl -X POST -H "Authorization: Bearer $VERCEL_TOKEN" \
  "https://api.vercel.com/v1/drains/<drain-id>/test?teamId=$TEAM_ID"
```

Confirm the vendor dashboard shows the test event arriving.

#### 5. Smoke Log Check

```bash
# Trigger a deployment and check logs flow through
vercel logs <deployment-url> --follow --since 5m

# Check integration balance to confirm data is flowing
vercel integration balance datadog
```

Verify that logs appear both in Vercel's runtime logs and in the vendor's dashboard.

> **For drain payload formats and signature verification**, see `⤳ skill: observability` — the Drains section covers JSON/NDJSON schemas and `x-vercel-signature` HMAC-SHA1 verification.

### Speed Insights + Web Analytics Drains

For observability vendors that also want Speed Insights or Web Analytics data, configure a separate drain manually:

```bash
# Create a drain for Speed Insights + Web Analytics
curl -X POST -H "Authorization: Bearer $VERCEL_TOKEN" \
  -H "Content-Type: application/json" \
  "https://api.vercel.com/v1/drains?teamId=$TEAM_ID" \
  -d '{
    "url": "https://your-vendor-endpoint.example.com/vercel-analytics",
    "type": "json",
    "sources": ["lambda"],
    "environments": ["production"]
  }'
```

> **Payload schema reference:** See `⤳ skill: observability` for Web Analytics drain payload formats (JSON array of `{type, url, referrer, timestamp, geo, device}` events).

- **Drain present** → Proceed to health check.
- **No drain found** → Integration may not auto-configure drains. Create one manually via Dashboard or REST API.
- **Drain errored** → Check the drain status in the Vercel Dashboard. Common fixes: endpoint URL typo, auth header missing, endpoint not accepting POST.

### 8. Run Local Health Check

Verify the integration works locally:

```bash
vercel dev
```

Or run the project's dev server and test the integration endpoint/connection:

- **Database** → Confirm connection by running a simple query (e.g., `SELECT 1`)
- **Auth** → Confirm the auth provider redirects correctly
- **Cache** → Confirm a set/get round-trip succeeds
- **CMS** → Confirm content fetch returns data
- **Observability** → Run `vercel logs <deployment-url> --follow --since 5m` and confirm logs appear in the vendor dashboard

If the health check fails, review the error output and guide the user through fixes (common: missing env vars in `.env.local`, wrong SDK version, network issues).

## Verification

After completing the apply-guide loop, confirm:

- [ ] Integration guide was retrieved via `vercel integration guide <name> --framework <fw>`
- [ ] Project was linked before provisioning started
- [ ] All required environment variables are provisioned on Vercel
- [ ] Local env sync is up to date (auto-sync succeeded or `vercel env pull .env.local --yes` ran)
- [ ] SDK package installed without errors
- [ ] Code changes applied and match the guide's patterns
- [ ] For observability integrations: drain verified and test payload received
- [ ] Local health check passed (dev server starts, integration responds)

If any step fails, report the specific error and suggest remediation before continuing.

## Summary

Present a structured result block:

```
## Marketplace Result
- **Integration**: <name>
- **Status**: installed | partially configured | failed
- **Package**: <sdk-package>@<version>
- **Env Vars**: <count> provisioned / <count> required
- **Drain**: configured | not applicable | manual setup needed
- **Health Check**: passed | failed | skipped
- **Files Changed**: <list of created/modified files>
```

## Next Steps

Based on the outcome:

- **Installed successfully** → "Run `/deploy` to deploy with the new integration. Your environment variables are already configured on Vercel."
- **Env vars missing** → "Provision the missing variables via the Vercel dashboard or `vercel integration add <name>`, then re-run `/marketplace <name>` to continue setup."
- **CLI handed off to dashboard** → "Run `vercel integration open <name>` to complete the provider web step, then resume from env verification."
- **Drain not auto-created** → "Create a drain manually via the REST API (`POST /v1/drains`) or the Dashboard."
- **Need Speed Insights / Web Analytics export** → "These data types require manual drain setup — they are not auto-configured by vendor installs. Configure via Dashboard or REST API."
- **Health check failed** → "Review the error above. Common fixes: copy env vars to `.env.local` with `vercel env pull`, check SDK version compatibility, verify network access."
- **Want another integration?** → "Run `/marketplace` again to browse available integrations."
- **Review changes** → "Run `git diff` to review all integration-related code changes before committing."

---

## /_conventions

_Source: `cache/claude-plugins-official/vercel/b95178c7d8df/commands/_conventions.md`_

# Command Conventions

Every slash command in this plugin follows a consistent structure so that the AI agent produces reliable, verifiable results. When authoring or updating a command file, include **all** of the sections below.

## Required Sections

### 1. Preflight

Check prerequisites before doing any work:

- **Project linked?** — Verify `.vercel/project.json` exists. If not, guide the user through `vercel link`.
- **CLI available?** — Confirm `vercel` CLI is on PATH.
- **Repo state** — Note uncommitted changes, dirty working tree, or detached HEAD when relevant.
- **Scope** — Detect monorepo (e.g., `turbo.json` or `pnpm-workspace.yaml`) and confirm which package is targeted.

Preflight failures should produce clear, actionable guidance — never silently skip.

### 2. Plan

Before executing, state what will happen:

- List the commands or MCP calls that will run.
- Flag destructive or production-impacting operations and require explicit user confirmation.
- If multiple strategies exist (MCP-first vs CLI-fallback), state which path was chosen and why.

### 3. Commands

The operational core. Follow these conventions:

- **MCP-first, CLI-fallback** — Prefer the Vercel MCP server for reads; use CLI for writes or when MCP lacks coverage.
- **Structured output** — Request `--format=json` where available; parse and present results in a readable summary.
- **No secrets in output** — Never echo environment variable values. Show names and metadata only.
- **Confirmation for destructive ops** — Production deploys, env removal, cache purge, and domain changes require an explicit "yes" from the user.

### 4. Verification

After execution, confirm the outcome:

- Re-read state (e.g., `vercel ls`, `vercel env ls`) to confirm the operation took effect.
- Compare before/after where possible (e.g., deployment count, env var count).
- Surface errors or warnings from command output.

### 5. Summary

Present a concise result block:

```
## Result
- **Action**: what was done
- **Status**: success | partial | failed
- **Details**: key output (URL, counts, config changes)
```

### 6. Next Steps

Suggest logical follow-ups:

- After deploy → check logs, inspect build, verify preview URL.
- After env change → pull to local, redeploy if production.
- After status → fix any flagged issues, run deploy if stale.

## File Naming

- Command files live in `commands/` and end in `.md`.
- Files prefixed with `_` (like this one) are meta-documents, not slash commands. They are excluded from `plugin.json` enumeration and not presented as user-invocable commands.

## Frontmatter

Every command file must include YAML frontmatter with at least a `description` field:

```yaml
---
description: One-line summary of what the command does.
---
```

## Validation

`scripts/validate.ts` enforces these conventions. Every non-underscore command file is checked for the required sections: Preflight, Plan, Commands, Verification, Summary, Next Steps.

---

## /status

_Source: `cache/claude-plugins-official/vercel/b95178c7d8df/commands/status.md`_

---
description: Show the status of the current Vercel project — recent deployments, linked project info, and environment overview.
---

# Vercel Project Status (Doctor)

Comprehensive project health check. Diagnoses deployment state, environment configuration, domains, and build status.

## Preflight

1. Check for `.vercel/project.json` in the current directory (or nearest parent).
   - **If found**: read `projectId` and `orgId` to confirm linkage. Print project name.
   - **If not found**: print a clear message:
     > This project is not linked to Vercel. Run `vercel link` to connect it, then re-run `/status`.
     Stop here — remaining steps require a linked project. **Do NOT fall back to GitHub Actions (`gh run list`) or any other CI system.** The Vercel CLI is the only authoritative source for this command.
2. Verify `vercel` CLI is available on PATH. If missing, suggest `npm i -g vercel`.
3. Detect monorepo markers (`turbo.json`, `pnpm-workspace.yaml`). If present, note which package scope is active.

## Plan

Gather project diagnostics using MCP reads where available, CLI as fallback:

1. Fetch recent deployments (last 5).
2. Inspect the latest deployment for build status and metadata.
3. List environment variables per environment (counts only — never print values).
4. Check domain configuration and status.
5. Read `vercel.json` for configuration highlights.

No destructive operations — this command is read-only.

## Commands

### 1. Recent Deployments

```
vercel ls
```

> **Note:** `vercel ls` does not support `--limit`. It returns recent deployments by default.

Extract: deployment URL, state (READY / ERROR / BUILDING), target (production / preview), created timestamp.

### 2. Latest Deployment Inspection

```
vercel inspect <latest-deployment-url>
```

Extract: build duration, function count, region, framework detected, Node.js version.

### 3. Environment Variable Counts

```
vercel env ls
```

Count variables per environment (Production, Preview, Development). **Never echo variable values.**

Present as:

| Environment | Count |
|-------------|-------|
| Production  | N     |
| Preview     | N     |
| Development | N     |

### 4. Domain Status

```
vercel domains ls
```

For each domain: name, DNS configured (yes/no), SSL valid (yes/no).

### 5. Configuration Highlights

Read `vercel.json` (if present) and summarize:

- Framework preset
- Build command overrides
- Function configuration (runtime, memory, duration)
- Rewrites / redirects count
- Cron jobs defined
- Headers or middleware config

If `vercel.json` does not exist, note "No vercel.json found — using framework defaults."

### 6. Observability Diagnostics

Check the project's observability posture — drains, error monitoring, analytics instrumentation, and drain security.

#### 6a. Drains Configured?

Use MCP `list_drains` if available, or the REST API:

```bash
curl -s -H "Authorization: Bearer $VERCEL_TOKEN" \
  "https://api.vercel.com/v1/drains?teamId=$TEAM_ID" | jq '.drains | length'
```

- **If drains exist**: list drain count, types (JSON/NDJSON/Syslog), and statuses.
- **If zero drains**: note "No drains configured" and flag as a gap for production observability.

#### 6b. Errored Drains?

For each drain returned, check the status field. If any drain shows an error or disabled state:

```
⚠️ Drain "<drain-url>" is in error state.
Remediation:
  1. Verify the endpoint URL is reachable and returns 2xx.
  2. Check that the endpoint accepts the configured format (JSON/NDJSON/Syslog).
  3. Test the drain: POST /v1/drains/<drain-id>/test
  4. If unrecoverable, delete and recreate the drain.
```

#### 6c. Analytics Instrumentation Present?

Scan the project source for `@vercel/analytics` and `@vercel/speed-insights` imports:

- Check `package.json` dependencies for `@vercel/analytics` and `@vercel/speed-insights`.
- If missing either package, flag:
  > Analytics/Speed Insights not detected. Install `@vercel/analytics` and `@vercel/speed-insights` and add the components to your root layout.

#### 6d. Drain Signature Verification

Vercel signs every drain payload with an HMAC-SHA1 signature in the `x-vercel-signature` header. **Always verify signatures in production** to prevent spoofed data.

> **Critical:** You must verify against the **raw request body** (not a parsed/re-serialized version). JSON parsing and re-stringifying can change key order or whitespace, breaking the signature match.

```ts
import { createHmac, timingSafeEqual } from 'crypto'

function verifyDrainSignature(rawBody: string, signature: string, secret: string): boolean {
  const expected = createHmac('sha1', secret).update(rawBody).digest('hex')
  // Use timing-safe comparison to prevent timing attacks
  if (expected.length !== signature.length) return false
  return timingSafeEqual(Buffer.from(expected), Buffer.from(signature))
}
```

Usage in a drain endpoint:

```ts
// app/api/drain/route.ts
export async function POST(req: Request) {
  const rawBody = await req.text()
  const signature = req.headers.get('x-vercel-signature')
  const secret = process.env.DRAIN_SECRET!

  if (!signature || !verifyDrainSignature(rawBody, signature, secret)) {
    return new Response('Unauthorized', { status: 401 })
  }

  const events = JSON.parse(rawBody)
  // Process verified events...
  return new Response('OK', { status: 200 })
}
```

> **Secret management:** The drain signing secret is shown once when you create the drain. Store it in an environment variable (e.g., `DRAIN_SECRET`). If lost, delete and recreate the drain.

Check whether the project has a `DRAIN_SECRET` env var set via `vercel env ls`. If drains are configured but no signature secret is found, flag as a security gap.

#### 6e. Fallback Guidance (No Drains)

If drains are unavailable (Hobby plan or not yet configured), use these alternatives:

| Need | Alternative | How |
|------|-------------|-----|
| View runtime logs | **Vercel Dashboard** | `https://vercel.com/{team}/{project}/deployments` → select deployment → Logs tab |
| Stream logs from terminal | **Vercel CLI** | `vercel logs <deployment-url> --follow` |
| Query logs programmatically | **MCP / REST API** | `get_runtime_logs` tool or `/v3/deployments/:id/events` |
| Monitor errors post-deploy | **CLI** | `vercel logs <url> --level error --since 1h` |
| Web Analytics data | **Dashboard only** | `https://vercel.com/{team}/{project}/analytics` |
| Performance metrics | **Dashboard only** | `https://vercel.com/{team}/{project}/speed-insights` |

> **Upgrade path:** When ready for centralized observability, upgrade to Pro and configure drains at `https://vercel.com/dashboard/{team}/~/settings/log-drains` or via REST API. The drain setup is typically < 5 minutes.

#### Observability Decision Matrix

| Need | Use | Why |
|------|-----|-----|
| Page views, traffic sources | Web Analytics | First-party, privacy-friendly |
| Business event tracking | Web Analytics custom events | Track conversions, feature usage |
| Core Web Vitals monitoring | Speed Insights | Real user data per route |
| Function debugging | Runtime Logs (CLI `vercel logs` / Dashboard (`https://vercel.com/{team}/{project}/logs`) / REST) | Real-time, per-invocation logs |
| Export logs to external platform | Drains (JSON/NDJSON/Syslog) | Centralize observability (Pro+) |
| Export analytics data | Drains (Web Analytics type) | Warehouse pageviews + custom events (Pro+) |
| OpenTelemetry traces | Drains (OTel-compatible endpoint) | Standards-based distributed tracing (Pro+) |
| Post-response telemetry | `waitUntil` + custom reporting | Non-blocking metrics |
| Server-side event tracking | `@vercel/analytics/server` | Track API-triggered events |
| Hobby plan log access | CLI `vercel logs` + Dashboard (`https://vercel.com/{team}/{project}/logs`) | No drains needed |

## Verification

Confirm each data source returned successfully:

- [ ] Deployment list retrieved (count > 0 or "no deployments yet")
- [ ] Latest deployment inspected (or skipped if no deployments)
- [ ] Environment variable counts retrieved per environment
- [ ] Domain list retrieved (or "no custom domains")
- [ ] vercel.json parsed (or "not present")
- [ ] Drain status checked (count, errored drains identified)
- [ ] Analytics/Speed Insights instrumentation detected (or gap flagged)
- [ ] Drain signature secret checked (if drains configured)

If any check fails, report the specific error and continue with remaining checks.

## Summary

Present the diagnostic report:

```
## Vercel Doctor — Project Status

**Project**: <name> (<org>)
**Latest Deployment**: <url> — <state> (<target>, <timestamp>)
**Build**: <duration>, <framework>, Node <version>

### Deployments (last 5)
| URL | State | Target | Created |
|-----|-------|--------|---------|
| ... | ...   | ...    | ...     |

### Environment Variables
| Environment | Count |
|-------------|-------|
| Production  | N     |
| Preview     | N     |
| Development | N     |

### Domains
| Domain | DNS | SSL |
|--------|-----|-----|
| ...    | ... | ... |

### Config Highlights
- <key settings from vercel.json>

### Observability
| Check | Status |
|-------|--------|
| Drains configured | N configured (N healthy, N errored) |
| Analytics (`@vercel/analytics`) | ✓ installed / ✗ not found |
| Speed Insights (`@vercel/speed-insights`) | ✓ installed / ✗ not found |
| Drain signature secret | ✓ DRAIN_SECRET set / ⚠ missing |
```

## Next Steps

Based on the diagnostic results, suggest relevant actions:

- **Build errors** → "Run `/deploy` to trigger a fresh build, or check `vercel logs <url>` for details."
- **Missing env vars** → "Run `/env list` to review, or `/env pull` to sync locally."
- **DNS not configured** → "Update your domain DNS records. See the Vercel domains dashboard."
- **No deployments** → "Run `/deploy` to create your first deployment."
- **Stale deployment** → "Your latest deployment is over 7 days old. Consider redeploying."
- **No vercel.json** → "Add a `vercel.json` if you need custom build, function, or routing configuration."
- **No drains (Hobby)** → "View logs via Dashboard or `vercel logs <url> --follow`. Upgrade to Pro for drain-based forwarding."
- **No drains (Pro+)** → "Configure drains for centralized observability via Dashboard or REST API."
- **Errored drain** → "Test the endpoint: `POST /v1/drains/<id>/test`. Check URL reachability and format compatibility."
- **Missing analytics** → "Install `@vercel/analytics` and add `<Analytics />` to your root layout."
- **Missing speed insights** → "Install `@vercel/speed-insights` and add `<SpeedInsights />` to your root layout."
- **Missing drain secret** → "Set `DRAIN_SECRET` env var. Without it, drain endpoints can't verify payload authenticity."

---

## /deploy

_Source: `cache/claude-plugins-official/vercel/b95178c7d8df/commands/deploy.md`_

---
description: Deploy the current project to Vercel. Pass "prod" or "production" as argument to deploy to production. Default is preview deployment.
---

# Deploy to Vercel

Deploy the current project to Vercel using the CLI, with preflight safety checks, explicit production confirmation, and post-deploy verification.

## Preflight

Run these checks before any deployment. Stop on failure and print actionable guidance.

1. **CLI available?** — Confirm `vercel` is on PATH.
   - If missing: `npm i -g vercel` (or `pnpm add -g vercel` / `bun add -g vercel`).
2. **Project linked?** — Check for `.vercel/project.json` in the current directory or nearest parent.
   - If not found: run `vercel link` interactively, then re-run `/deploy`.
3. **Monorepo detection** — Look for `turbo.json` or `pnpm-workspace.yaml` at the repo root.
   - If detected: confirm which package is targeted. If ambiguous, ask the user before proceeding.
4. **Uncommitted changes** — Run `git status --porcelain`.
   - If output is non-empty: warn the user that uncommitted changes will **not** be included in the deploy. Ask whether to continue or commit first.
   - If not a git repo, skip this check.
5. **Observability preflight** (production deploys only) —

Before promoting to production, verify observability readiness:

- **Drains check**: Query configured drains via MCP `list_drains` or REST API. If no drains are configured on a Pro/Enterprise plan, warn:
  > ⚠️ No drains configured. Production errors won't be forwarded to external monitoring.
  > Configure drains via Dashboard or REST API before promoting.
- **Errored drains**: If any drain is in error state, warn and suggest remediation before deploying:
  > ⚠️ Drain "<url>" is errored. Fix or recreate before production deploy to avoid monitoring gaps.
- **Error monitoring**: Check that at least one of these is in place: configured drains, an error tracking integration (e.g., Sentry, Datadog via `vercel integration ls`), or `@vercel/analytics` in the project.
- These are warnings, not blockers — the user may proceed after acknowledgment.

## Plan

State the intended action before executing:

- **Preview deploy** (default): `vercel` — creates a preview deployment on a unique URL.
- **Production deploy**: `vercel --prod` — deploys to production domains.

If "$ARGUMENTS" contains "prod" or "production":

> ⚠️ **Production deployment requested.**
> This will deploy to your live production URL and affect real users.
> **Ask the user for explicit confirmation before proceeding.** Do not deploy to production without a clear "yes."

If "$ARGUMENTS" does not indicate production:

> Deploying a **preview** build. This creates an isolated URL and does not affect production.

## Commands

### Preview Deployment

<!-- Sourced from deployments-cicd skill: Deployment Commands > Preview Deployment -->
```bash
# Deploy from project root (creates preview URL)
vercel

# Equivalent explicit form
vercel deploy
```

Preview deployments are created automatically for every push to a non-production branch when using Git integration. They provide a unique URL for testing.

### Production Deployment (requires confirmation)

<!-- Sourced from deployments-cicd skill: Deployment Commands > Production Deployment -->
```bash
# Deploy directly to production
vercel --prod
vercel deploy --prod

# Force a new deployment (skip cache)
vercel --prod --force
```

Capture the full command output. Extract the **deployment URL** from the output (the line containing the `.vercel.app` URL or custom domain URL).

Record the current git commit SHA for the summary:

```bash
git rev-parse --short HEAD
```

## Verification

After deployment completes, verify the result:

### 1. Inspect the Deployment

<!-- Sourced from deployments-cicd skill: Deployment Commands > Inspect Deployments -->
```bash
# View deployment details (build info, functions, metadata)
vercel inspect <deployment-url>

# List recent deployments
vercel ls

# View logs for a deployment
vercel logs <deployment-url>
vercel logs <deployment-url> --follow
```

Extract: deployment state (READY / ERROR / QUEUED / BUILDING), build duration, framework, Node.js version, function count.

### 2. On Failure — Fetch Build Logs

If the deployment state is **ERROR** or the deploy command exited non-zero:

```bash
vercel logs <deployment-url>
```

Present the last 50 lines of build logs to help diagnose the failure. Highlight any lines containing `error`, `Error`, `ERR!`, or `FATAL`.

### 3. On Success — Quick Smoke Check

If the deployment state is **READY**, note the URL is live and accessible.

### 4. Post-Deploy Error Scan (production deploys)

For production deployments, wait 60 seconds after READY state, then scan for early runtime errors:

```bash
vercel logs <deployment-url> --level error --since 1h
```

Or via MCP if available: use `get_runtime_logs` with level filter `error`.

**Interpret results:**

| Finding | Action |
|---------|--------|
| No errors | ✓ Clean deploy — no runtime errors in first hour |
| Errors detected | List error count and first 5 unique error messages. Suggest: check drain payloads for correlated traces, review function logs in Dashboard |
| 500 status codes in logs | Correlate timestamps with drain data (if configured) or `vercel logs <url> --json` for structured output. Flag for immediate investigation |
| Timeout errors | Check function duration limits in `vercel.json` or project settings. Consider increasing `maxDuration` |

**Fallback (no drains):**

If no drains are configured, the error scan relies on CLI and Dashboard:

```bash
# Stream live errors
vercel logs <deployment-url> --level error --follow

# JSON output for parsing
vercel logs <deployment-url> --level error --since 1h --json
```

> For richer post-deploy monitoring, configure drains to forward logs/traces to an external platform via Dashboard or REST API.

## Summary

<!-- Sourced from deployments-cicd skill: Deploy Summary Format -->
Present a structured deploy result block:

```
## Deploy Result
- **URL**: <deployment-url>
- **Target**: production | preview
- **Status**: READY | ERROR | BUILDING | QUEUED
- **Commit**: <short-sha>
- **Framework**: <detected-framework>
- **Build Duration**: <duration>
```

If the deployment failed, append:

```
- **Error**: <summary of failure from logs>
```

For production deploys, also include:

```
### Post-Deploy Observability
- **Error scan**: <N errors found / clean> (scanned via vercel logs --level error --since 1h)
- **Drains**: <N configured / none>
- **Monitoring**: <active / gaps identified>
```

## Next Steps

<!-- Sourced from deployments-cicd skill: Deploy Next Steps -->
Based on the deployment outcome:

- **Success (preview)** → "Visit the preview URL to verify. When ready, run `/deploy prod` to promote to production."
- **Success (production)** → "Your production site is live. Run `/status` to see the full project overview."
- **Build error** → "Check the build logs above. Common fixes: verify `build` script in package.json, check for missing env vars with `/env list`, ensure dependencies are installed."
- **Missing env vars** → "Run `/env pull` to sync environment variables locally, or `/env list` to review what's configured on Vercel."
- **Monorepo issues** → "Ensure the correct project root is configured in Vercel project settings. Check `vercel.json` for `rootDirectory`."
- **Post-deploy errors detected** → "Review errors above. Check `vercel logs <url> --level error` for details. If drains are configured, correlate with external monitoring."
- **No monitoring configured** → "Set up drains or install an error tracking integration before the next production deploy. Run `/status` for a full observability diagnostic."

---

## /env

_Source: `cache/claude-plugins-official/vercel/b95178c7d8df/commands/env.md`_

---
description: Manage Vercel environment variables. Commands include list, pull, add, remove, and diff. Use to sync environment variables between Vercel and your local development environment.
---

# Vercel Environment Variables

Manage environment variables for the current Vercel project with safety rails to prevent secret leakage.

> **🔒 Never-Echo-Secrets Rule**: Environment variable **values** must never appear in command output, summaries, or conversation text. Only show variable **names**, **environments**, and **metadata** (created date, type). This rule applies to all subcommands.

## Preflight

1. **CLI available?** — Confirm `vercel` is on PATH.
   - If missing: `npm i -g vercel` (or `pnpm add -g vercel` / `bun add -g vercel`).
2. **Project linked?** — Check for `.vercel/project.json` in the current directory or nearest parent.
   - If not found: run `vercel link` interactively, then re-run `/env`.
3. **Detect environment files** — Check for `.env`, `.env.local`, `.env.production.local`, `.env.development.local` in the project root. Note which exist for the diff subcommand.

## Plan

Based on "$ARGUMENTS", determine the action:

| Argument | Action | Destructive? |
|----------|--------|-------------|
| `list` / `ls` / _(none)_ | List env var names per environment | No |
| `pull` | Download env vars to local `.env.local` | No (overwrites local file) |
| `add <NAME>` | Add a new env var | Yes (if production) |
| `rm <NAME>` / `remove <NAME>` | Remove an env var | **Yes** |
| `diff` | Compare local vs Vercel key names | No |

For any operation that mutates **production** environment variables (`add` or `rm` targeting production):

> ⚠️ **Production environment mutation requested.**
> This will change environment variables on your live production deployment.
> **Ask the user for explicit confirmation before proceeding.** Do not mutate production env vars without a clear "yes."

## Commands

### "list" or "ls" or no arguments

<!-- Sourced from env-vars skill: vercel env CLI > List Environment Variables -->
```bash
# List all environment variables
vercel env ls

# Filter by environment
vercel env ls production
```

Present results as a table of variable **names only** grouped by environment. **Never print values.**

| Name | Production | Preview | Development |
|------|-----------|---------|-------------|
| DATABASE_URL | ✓ | ✓ | ✓ |
| API_KEY | ✓ | ✓ | — |

### "pull"

<!-- Sourced from env-vars skill: vercel env CLI > Pull Environment Variables -->
```bash
# Pull all env vars for the current environment into .env.local
vercel env pull .env.local

# Pull for a specific environment
vercel env pull .env.local --environment=production
vercel env pull .env.local --environment=preview
vercel env pull .env.local --environment=development

# Overwrite existing file without prompting
vercel env pull .env.local --yes

# Pull to a custom file
vercel env pull .env.production.local --environment=production
```

After pulling, remind the user:

> Ensure `.env*.local` is in your `.gitignore` to avoid committing secrets.

### "add \<NAME\>"

1. Ask the user which environments to target: production, preview, development (can select multiple).
2. If **production** is selected, require explicit confirmation (see Plan section).
3. Run the add command:

<!-- Sourced from env-vars skill: vercel env CLI > Add Environment Variables -->
```bash
# Interactive — prompts for value and environments
vercel env add MY_SECRET

# Non-interactive
echo "secret-value" | vercel env add MY_SECRET production

# Add to multiple environments
echo "secret-value" | vercel env add MY_SECRET production preview development

# Add a sensitive variable (encrypted, not shown in logs)
vercel env add MY_SECRET --sensitive
```

The CLI will prompt for the value interactively — **do not pass the value as a CLI argument or echo it**.

### "rm \<NAME\>" or "remove \<NAME\>"

1. If the variable exists in **production**, require explicit confirmation.
2. Run the remove command:

<!-- Sourced from env-vars skill: vercel env CLI > Remove Environment Variables -->
```bash
# Remove from specific environment
vercel env rm MY_SECRET production

# Remove from all environments
vercel env rm MY_SECRET
```

Confirm the target environment(s) with the user before executing.

### "diff"

Compare local environment file key names against Vercel-configured key names. **Only compare names — never read or display values.**

1. Read local env file keys (from `.env.local` or `.env` — whichever exists):

```bash
grep -v '^#' .env.local | grep '=' | cut -d'=' -f1 | sort
```

2. Fetch Vercel env var names:

```bash
vercel env ls
```

Parse the output to extract variable names for the target environment (default: development).

3. Present the diff:

```
## Env Diff — Local vs Vercel (development)

### In local but NOT on Vercel:
- EXTRA_LOCAL_VAR
- DEBUG_MODE

### On Vercel but NOT in local:
- ANALYTICS_KEY
- SENTRY_DSN

### In both:
- DATABASE_URL
- API_KEY
- NEXT_PUBLIC_APP_URL
```

If all keys match, report: "Local and Vercel environment keys are in sync."

## Environment-Specific Configuration Reference

<!-- Sourced from env-vars skill: Environment-Specific Configuration -->
### Vercel Dashboard vs .env Files

| Use Case | Where to Set |
|----------|-------------|
| Secrets (API keys, tokens) | Vercel Dashboard (`https://vercel.com/{team}/{project}/settings/environment-variables`) or `vercel env add` |
| Public config (site URL, feature flags) | `.env` or `.env.[environment]` files |
| Local-only overrides | `.env.local` |
| CI/CD secrets | Vercel Dashboard (`https://vercel.com/{team}/{project}/settings/environment-variables`) with environment scoping |

### Environment Scoping on Vercel

Variables set in the Vercel Dashboard at `https://vercel.com/{team}/{project}/settings/environment-variables` can be scoped to:

- **Production** — only `vercel.app` production deployments
- **Preview** — branch/PR deployments
- **Development** — `vercel dev` and `vercel env pull`

A variable can be assigned to one, two, or all three environments.

### Git Branch Overrides

Preview environment variables can be scoped to specific Git branches:

```bash
# Add a variable only for the "staging" branch
echo "staging-value" | vercel env add DATABASE_URL preview --git-branch=staging
```

## Common Gotchas

<!-- Sourced from env-vars skill: Gotchas -->
### `vercel env pull` Overwrites Custom Variables

`vercel env pull .env.local` **replaces the entire file** — any manually added variables (custom secrets, local overrides, debug flags) are lost. Always back up or re-add custom vars after pulling:

```bash
# Save custom vars before pulling
grep -v '^#' .env.local | grep -v '^VERCEL_\|^POSTGRES_\|^NEXT_PUBLIC_' > .env.custom.bak
vercel env pull .env.local --yes
cat .env.custom.bak >> .env.local  # Re-append custom vars
```

Or maintain custom vars in a separate `.env.development.local` file (loaded after `.env.local` by Next.js).

### Scripts Don't Auto-Load `.env.local`

Only Next.js auto-loads `.env.local`. Standalone scripts (`drizzle-kit`, `tsx`, custom Node scripts) need explicit loading:

```bash
# Use dotenv-cli
npm install -D dotenv-cli
npx dotenv -e .env.local -- npx drizzle-kit push
npx dotenv -e .env.local -- npx tsx scripts/seed.ts

# Or source manually
source <(grep -v '^#' .env.local | sed 's/^/export /') && node scripts/migrate.js
```

## Verification

After any mutation (`add` or `rm`), verify the change took effect:

```bash
vercel env ls
```

Re-list environment variables and confirm:

- For `add`: the new variable name appears in the expected environment(s).
- For `rm`: the variable name no longer appears in the target environment(s).

If verification fails (variable still present after remove, or missing after add), report the discrepancy and suggest retrying.

## Summary

Present a structured result block:

```
## Env Result
- **Action**: list | pull | add | remove | diff
- **Status**: success | failed
- **Variable**: <NAME> (for add/remove)
- **Environments**: production, preview, development (as applicable)
- **Details**: <key outcome>
```

For `diff`, include counts:

```
- **Local only**: N keys
- **Vercel only**: N keys
- **Shared**: N keys
```

## Next Steps

Based on the action performed:

- **After list** → "Run `/env pull` to sync to local, or `/env diff` to compare local vs Vercel."
- **After pull** → "Restart your dev server to pick up the new variables. Run `/env diff` to verify sync."
- **After add** → "Run `/deploy` to make the new variable available in your next deployment. For production, the variable is available immediately on the next request."
- **After remove** → "The variable is removed. If your app depends on it, update your code or add a replacement. Consider redeploying with `/deploy`."
- **After diff** → "Add missing variables with `/env add <NAME>`, or pull from Vercel with `/env pull` to sync."

---

## /bootstrap

_Source: `cache/claude-plugins-official/vercel/b95178c7d8df/commands/bootstrap.md`_

---
description: Bootstrap a repository with Vercel-linked resources by running preflight checks, provisioning integrations, verifying env keys, and then executing db/dev startup commands safely.
---

# Vercel Project Bootstrap

Run a deterministic bootstrap flow for new or partially configured repositories.

## Preflight

<!-- Sourced from bootstrap skill: Preflight -->
1. Confirm Vercel CLI is installed and authenticated.

```bash
vercel --version
vercel whoami
```

2. Confirm repo linkage by checking `.vercel/project.json`.
3. If not linked, inspect available teams/projects before asking the user to choose:

```bash
vercel teams ls
vercel projects ls --scope <team>
vercel link --yes --scope <team> --project <project>
```

4. Find the env template in priority order: `.env.example`, `.env.sample`, `.env.template`.
5. Create local env file if missing:

```bash
cp .env.example .env.local
```

6. Detect package manager and available scripts (`db:push`, `db:seed`, `db:migrate`, `db:generate`, `dev`) from `package.json`.
7. Inspect auth/database signals (`prisma/schema.prisma`, `drizzle.config.*`, `auth.*`, `src/**/auth.*`) to scope bootstrap details.

Stop with clear guidance if CLI auth or linkage fails.

## Plan

Execute in this order:

1. Preflight validation and project linking.
2. Resource provisioning (prefer Vercel-managed Neon integration).
3. Secret/bootstrap env setup (`AUTH_SECRET`, env pull, key verification).
4. Application bootstrap (`db:*` then `dev`) only after env checks pass.

<!-- Sourced from bootstrap skill: Rules -->
- Do not run `db:push`, `db:migrate`, `db:seed`, or `dev` until Vercel linking is complete and env keys are verified.
- Prefer Vercel-managed provisioning (`vercel integration ...`) for shared resources.
- Use provider CLIs only as fallback when Vercel integration flow is unavailable.
- Never echo secret values in terminal output, logs, or summaries.

## Commands

### 1. Link + local env template

Copy the first matching template file only if `.env.local` does not exist:

```bash
cp .env.example .env.local
```

If `.env.example` is absent, use `.env.sample` or `.env.template`.

### 2. Provision Postgres

<!-- Sourced from bootstrap skill: Resource Setup: Postgres -->
### Preferred path (Vercel-managed Neon)

1. Read integration setup guidance:

```bash
vercel integration guide neon
```

2. Add Neon integration to the Vercel scope:

```bash
vercel integration add neon --scope <team>
```

3. Verify expected environment variable names exist in Vercel and pull locally:

```bash
vercel env ls
vercel env pull .env.local --yes
```

### Fallback path 1 (Dashboard)

1. Provision Neon through the Vercel dashboard integration UI.
2. Re-run `vercel env pull .env.local --yes`.

### Fallback path 2 (Neon CLI)

Use Neon CLI only when Vercel-managed provisioning is unavailable. After creating resources, add required env vars in Vercel and pull again.

### 3. Generate and store `AUTH_SECRET`

<!-- Sourced from bootstrap skill: AUTH_SECRET Generation -->
Generate a high-entropy secret without printing it, then store it in Vercel and refresh local env:

```bash
AUTH_SECRET="$(node -e "console.log(require('node:crypto').randomBytes(32).toString('base64url'))")"
printf "%s" "$AUTH_SECRET" | vercel env add AUTH_SECRET development preview production
unset AUTH_SECRET
vercel env pull .env.local --yes
```

### 4. Verify required env keys

<!-- Sourced from bootstrap skill: Env Verification -->
Compare required keys from template file against `.env.local` keys (names only, never values):

```bash
template_file=""
for candidate in .env.example .env.sample .env.template; do
  if [ -f "$candidate" ]; then
    template_file="$candidate"
    break
  fi
done

comm -23 \
  <(grep -E '^[A-Za-z_][A-Za-z0-9_]*=' "$template_file" | cut -d '=' -f 1 | sort -u) \
  <(grep -E '^[A-Za-z_][A-Za-z0-9_]*=' .env.local | cut -d '=' -f 1 | sort -u)
```

Proceed only when missing key list is empty.

Do not continue if any required keys are missing.

### 5. Run app bootstrap commands (after verification)

<!-- Sourced from bootstrap skill: App Setup -->
After linkage + env verification:

```bash
npm run db:push
npm run db:seed
npm run dev
```

Use the repository package manager (`npm`, `pnpm`, `bun`, or `yarn`) and run only scripts that exist in `package.json`.

## Verification

<!-- Sourced from bootstrap skill: Bootstrap Verification -->
Confirm each checkpoint:

- `vercel whoami` succeeds.
- `.vercel/project.json` exists and matches chosen project.
- Postgres integration path completed (Vercel integration, dashboard, or provider CLI fallback).
- `vercel env pull .env.local --yes` succeeds.
- Required env key diff is empty.
- Database command status is recorded (`db:push`, `db:seed`, `db:migrate`, `db:generate` as applicable).
- `dev` command starts without immediate config/auth/env failure.

If verification fails, stop and report exact failing step plus remediation.

## Summary

<!-- Sourced from bootstrap skill: Summary Format -->
Return a final bootstrap summary in this format:

```md
## Bootstrap Result
- **Linked Project**: <team>/<project>
- **Resource Path**: vercel-integration-neon | dashboard-neon | neon-cli
- **Env Keys**: <count> required, <count> present, <count> missing
- **Secrets**: AUTH_SECRET set in Vercel (value never shown)
- **Migration Status**: not-run | success | failed (<step>)
- **Dev Result**: not-run | started | failed
```

## Next Steps

<!-- Sourced from bootstrap skill: Bootstrap Next Steps -->
- If env keys are still missing, add the missing keys in Vercel and re-run `vercel env pull .env.local --yes`.
- If DB commands fail, fix connectivity/schema issues and re-run only the failed db step.
- If `dev` fails, resolve runtime errors, then restart with your package manager's `run dev`.

---

## /test-browser

_Source: `marketplaces/every-marketplace/plugins/compound-engineering/commands/test-browser.md`_

---
name: test-browser
description: Run browser tests on pages affected by current PR or branch
argument-hint: "[PR number, branch name, or 'current' for current branch]"
---

# Browser Test Command

<command_purpose>Run end-to-end browser tests on pages affected by a PR or branch changes using agent-browser CLI.</command_purpose>

## CRITICAL: Use agent-browser CLI Only

**DO NOT use Chrome MCP tools (mcp__claude-in-chrome__*).**

This command uses the `agent-browser` CLI exclusively. The agent-browser CLI is a Bash-based tool from Vercel that runs headless Chromium. It is NOT the same as Chrome browser automation via MCP.

If you find yourself calling `mcp__claude-in-chrome__*` tools, STOP. Use `agent-browser` Bash commands instead.

## Introduction

<role>QA Engineer specializing in browser-based end-to-end testing</role>

This command tests affected pages in a real browser, catching issues that unit tests miss:
- JavaScript integration bugs
- CSS/layout regressions
- User workflow breakages
- Console errors

## Prerequisites

<requirements>
- Local development server running (e.g., `bin/dev`, `rails server`, `npm run dev`)
- agent-browser CLI installed (see Setup below)
- Git repository with changes to test
</requirements>

## Setup

**Check installation:**
```bash
command -v agent-browser >/dev/null 2>&1 && echo "Installed" || echo "NOT INSTALLED"
```

**Install if needed:**
```bash
npm install -g agent-browser
agent-browser install  # Downloads Chromium (~160MB)
```

See the `agent-browser` skill for detailed usage.

## Main Tasks

### 0. Verify agent-browser Installation

Before starting ANY browser testing, verify agent-browser is installed:

```bash
command -v agent-browser >/dev/null 2>&1 && echo "Ready" || (echo "Installing..." && npm install -g agent-browser && agent-browser install)
```

If installation fails, inform the user and stop.

### 1. Ask Browser Mode

<ask_browser_mode>

Before starting tests, ask user if they want to watch the browser:

Use AskUserQuestion with:
- Question: "Do you want to watch the browser tests run?"
- Options:
  1. **Headed (watch)** - Opens visible browser window so you can see tests run
  2. **Headless (faster)** - Runs in background, faster but invisible

Store the choice and use `--headed` flag when user selects "Headed".

</ask_browser_mode>

### 2. Determine Test Scope

<test_target> $ARGUMENTS </test_target>

<determine_scope>

**If PR number provided:**
```bash
gh pr view [number] --json files -q '.files[].path'
```

**If 'current' or empty:**
```bash
git diff --name-only main...HEAD
```

**If branch name provided:**
```bash
git diff --name-only main...[branch]
```

</determine_scope>

### 3. Map Files to Routes

<file_to_route_mapping>

Map changed files to testable routes:

| File Pattern | Route(s) |
|-------------|----------|
| `app/views/users/*` | `/users`, `/users/:id`, `/users/new` |
| `app/controllers/settings_controller.rb` | `/settings` |
| `app/javascript/controllers/*_controller.js` | Pages using that Stimulus controller |
| `app/components/*_component.rb` | Pages rendering that component |
| `app/views/layouts/*` | All pages (test homepage at minimum) |
| `app/assets/stylesheets/*` | Visual regression on key pages |
| `app/helpers/*_helper.rb` | Pages using that helper |
| `src/app/*` (Next.js) | Corresponding routes |
| `src/components/*` | Pages using those components |

Build a list of URLs to test based on the mapping.

</file_to_route_mapping>

### 4. Verify Server is Running

<check_server>

Before testing, verify the local server is accessible:

```bash
agent-browser open http://localhost:3000
agent-browser snapshot -i
```

If server is not running, inform user:
```markdown
**Server not running**

Please start your development server:
- Rails: `bin/dev` or `rails server`
- Node/Next.js: `npm run dev`

Then run `/test-browser` again.
```

</check_server>

### 5. Test Each Affected Page

<test_pages>

For each affected route, use agent-browser CLI commands (NOT Chrome MCP):

**Step 1: Navigate and capture snapshot**
```bash
agent-browser open "http://localhost:3000/[route]"
agent-browser snapshot -i
```

**Step 2: For headed mode (visual debugging)**
```bash
agent-browser --headed open "http://localhost:3000/[route]"
agent-browser --headed snapshot -i
```

**Step 3: Verify key elements**
- Use `agent-browser snapshot -i` to get interactive elements with refs
- Page title/heading present
- Primary content rendered
- No error messages visible
- Forms have expected fields

**Step 4: Test critical interactions**
```bash
agent-browser click @e1  # Use ref from snapshot
agent-browser snapshot -i
```

**Step 5: Take screenshots**
```bash
agent-browser screenshot page-name.png
agent-browser screenshot --full page-name-full.png  # Full page
```

</test_pages>

### 6. Human Verification (When Required)

<human_verification>

Pause for human input when testing touches:

| Flow Type | What to Ask |
|-----------|-------------|
| OAuth | "Please sign in with [provider] and confirm it works" |
| Email | "Check your inbox for the test email and confirm receipt" |
| Payments | "Complete a test purchase in sandbox mode" |
| SMS | "Verify you received the SMS code" |
| External APIs | "Confirm the [service] integration is working" |

Use AskUserQuestion:
```markdown
**Human Verification Needed**

This test touches the [flow type]. Please:
1. [Action to take]
2. [What to verify]

Did it work correctly?
1. Yes - continue testing
2. No - describe the issue
```

</human_verification>

### 7. Handle Failures

<failure_handling>

When a test fails:

1. **Document the failure:**
   - Screenshot the error state: `agent-browser screenshot error.png`
   - Note the exact reproduction steps

2. **Ask user how to proceed:**
   ```markdown
   **Test Failed: [route]**

   Issue: [description]
   Console errors: [if any]

   How to proceed?
   1. Fix now - I'll help debug and fix
   2. Create todo - Add to todos/ for later
   3. Skip - Continue testing other pages
   ```

3. **If "Fix now":**
   - Investigate the issue
   - Propose a fix
   - Apply fix
   - Re-run the failing test

4. **If "Create todo":**
   - Create `{id}-pending-p1-browser-test-{description}.md`
   - Continue testing

5. **If "Skip":**
   - Log as skipped
   - Continue testing

</failure_handling>

### 8. Test Summary

<test_summary>

After all tests complete, present summary:

```markdown
## Browser Test Results

**Test Scope:** PR #[number] / [branch name]
**Server:** http://localhost:3000

### Pages Tested: [count]

| Route | Status | Notes |
|-------|--------|-------|
| `/users` | Pass | |
| `/settings` | Pass | |
| `/dashboard` | Fail | Console error: [msg] |
| `/checkout` | Skip | Requires payment credentials |

### Console Errors: [count]
- [List any errors found]

### Human Verifications: [count]
- OAuth flow: Confirmed
- Email delivery: Confirmed

### Failures: [count]
- `/dashboard` - [issue description]

### Created Todos: [count]
- `005-pending-p1-browser-test-dashboard-error.md`

### Result: [PASS / FAIL / PARTIAL]
```

</test_summary>

## Quick Usage Examples

```bash
# Test current branch changes
/test-browser

# Test specific PR
/test-browser 847

# Test specific branch
/test-browser feature/new-dashboard
```

## agent-browser CLI Reference

**ALWAYS use these Bash commands. NEVER use mcp__claude-in-chrome__* tools.**

```bash
# Navigation
agent-browser open <url>           # Navigate to URL
agent-browser back                 # Go back
agent-browser close                # Close browser

# Snapshots (get element refs)
agent-browser snapshot -i          # Interactive elements with refs (@e1, @e2, etc.)
agent-browser snapshot -i --json   # JSON output

# Interactions (use refs from snapshot)
agent-browser click @e1            # Click element
agent-browser fill @e1 "text"      # Fill input
agent-browser type @e1 "text"      # Type without clearing
agent-browser press Enter          # Press key

# Screenshots
agent-browser screenshot out.png       # Viewport screenshot
agent-browser screenshot --full out.png # Full page screenshot

# Headed mode (visible browser)
agent-browser --headed open <url>      # Open with visible browser
agent-browser --headed click @e1       # Click in visible browser

# Wait
agent-browser wait @e1             # Wait for element
agent-browser wait 2000            # Wait milliseconds
```

---

## /resolve_parallel

_Source: `marketplaces/every-marketplace/plugins/compound-engineering/commands/resolve_parallel.md`_

---
name: resolve_parallel
description: Resolve all TODO comments using parallel processing
argument-hint: "[optional: specific TODO pattern or file]"
disable-model-invocation: true
---

Resolve all TODO comments using parallel processing.

## Workflow

### 1. Analyze

Gather the things todo from above.

### 2. Plan

Create a TodoWrite list of all unresolved items grouped by type.Make sure to look at dependencies that might occur and prioritize the ones needed by others. For example, if you need to change a name, you must wait to do the others. Output a mermaid flow diagram showing how we can do this. Can we do everything in parallel? Do we need to do one first that leads to others in parallel? I'll put the to-dos in the mermaid diagram flow‑wise so the agent knows how to proceed in order.

### 3. Implement (PARALLEL)

Spawn a pr-comment-resolver agent for each unresolved item in parallel.

So if there are 3 comments, it will spawn 3 pr-comment-resolver agents in parallel. liek this

1. Task pr-comment-resolver(comment1)
2. Task pr-comment-resolver(comment2)
3. Task pr-comment-resolver(comment3)

Always run all in parallel subagents/Tasks for each Todo item.

### 4. Commit & Resolve

- Commit changes
- Push to remote

---

## /reproduce-bug

_Source: `marketplaces/every-marketplace/plugins/compound-engineering/commands/reproduce-bug.md`_

---
name: reproduce-bug
description: Reproduce and investigate a bug using logs, console inspection, and browser screenshots
argument-hint: "[GitHub issue number]"
disable-model-invocation: true
---

# Reproduce Bug Command

Look at github issue #$ARGUMENTS and read the issue description and comments.

## Phase 1: Log Investigation

Run the following agents in parallel to investigate the bug:

1. Task rails-console-explorer(issue_description)
2. Task appsignal-log-investigator(issue_description)

Think about the places it could go wrong looking at the codebase. Look for logging output we can look for.

Run the agents again to find any logs that could help us reproduce the bug.

Keep running these agents until you have a good idea of what is going on.

## Phase 2: Visual Reproduction with Playwright

If the bug is UI-related or involves user flows, use Playwright to visually reproduce it:

### Step 1: Verify Server is Running

```
mcp__plugin_compound-engineering_pw__browser_navigate({ url: "http://localhost:3000" })
mcp__plugin_compound-engineering_pw__browser_snapshot({})
```

If server not running, inform user to start `bin/dev`.

### Step 2: Navigate to Affected Area

Based on the issue description, navigate to the relevant page:

```
mcp__plugin_compound-engineering_pw__browser_navigate({ url: "http://localhost:3000/[affected_route]" })
mcp__plugin_compound-engineering_pw__browser_snapshot({})
```

### Step 3: Capture Screenshots

Take screenshots at each step of reproducing the bug:

```
mcp__plugin_compound-engineering_pw__browser_take_screenshot({ filename: "bug-[issue]-step-1.png" })
```

### Step 4: Follow User Flow

Reproduce the exact steps from the issue:

1. **Read the issue's reproduction steps**
2. **Execute each step using Playwright:**
   - `browser_click` for clicking elements
   - `browser_type` for filling forms
   - `browser_snapshot` to see the current state
   - `browser_take_screenshot` to capture evidence

3. **Check for console errors:**
   ```
   mcp__plugin_compound-engineering_pw__browser_console_messages({ level: "error" })
   ```

### Step 5: Capture Bug State

When you reproduce the bug:

1. Take a screenshot of the bug state
2. Capture console errors
3. Document the exact steps that triggered it

```
mcp__plugin_compound-engineering_pw__browser_take_screenshot({ filename: "bug-[issue]-reproduced.png" })
```

## Phase 3: Document Findings

**Reference Collection:**

- [ ] Document all research findings with specific file paths (e.g., `app/services/example_service.rb:42`)
- [ ] Include screenshots showing the bug reproduction
- [ ] List console errors if any
- [ ] Document the exact reproduction steps

## Phase 4: Report Back

Add a comment to the issue with:

1. **Findings** - What you discovered about the cause
2. **Reproduction Steps** - Exact steps to reproduce (verified)
3. **Screenshots** - Visual evidence of the bug (upload captured screenshots)
4. **Relevant Code** - File paths and line numbers
5. **Suggested Fix** - If you have one

---

## /generate_command

_Source: `marketplaces/every-marketplace/plugins/compound-engineering/commands/generate_command.md`_

---
name: generate_command
description: Create a new custom slash command following conventions and best practices
argument-hint: "[command purpose and requirements]"
disable-model-invocation: true
---

# Create a Custom Claude Code Command

Create a new slash command in `.claude/commands/` for the requested task.

## Goal

#$ARGUMENTS

## Key Capabilities to Leverage

**File Operations:**
- Read, Edit, Write - modify files precisely
- Glob, Grep - search codebase
- MultiEdit - atomic multi-part changes

**Development:**
- Bash - run commands (git, tests, linters)
- Task - launch specialized agents for complex tasks
- TodoWrite - track progress with todo lists

**Web & APIs:**
- WebFetch, WebSearch - research documentation
- GitHub (gh cli) - PRs, issues, reviews
- Playwright - browser automation, screenshots

**Integrations:**
- AppSignal - logs and monitoring
- Context7 - framework docs
- Stripe, Todoist, Featurebase (if relevant)

## Best Practices

1. **Be specific and clear** - detailed instructions yield better results
2. **Break down complex tasks** - use step-by-step plans
3. **Use examples** - reference existing code patterns
4. **Include success criteria** - tests pass, linting clean, etc.
5. **Think first** - use "think hard" or "plan" keywords for complex problems
6. **Iterate** - guide the process step by step

## Required: YAML Frontmatter

**EVERY command MUST start with YAML frontmatter:**

```yaml
---
name: command-name
description: Brief description of what this command does (max 100 chars)
argument-hint: "[what arguments the command accepts]"
---
```

**Fields:**
- `name`: Lowercase command identifier (used internally)
- `description`: Clear, concise summary of command purpose
- `argument-hint`: Shows user what arguments are expected (e.g., `[file path]`, `[PR number]`, `[optional: format]`)

## Structure Your Command

```markdown
# [Command Name]

[Brief description of what this command does]

## Steps

1. [First step with specific details]
   - Include file paths, patterns, or constraints
   - Reference existing code if applicable

2. [Second step]
   - Use parallel tool calls when possible
   - Check/verify results

3. [Final steps]
   - Run tests
   - Lint code
   - Commit changes (if appropriate)

## Success Criteria

- [ ] Tests pass
- [ ] Code follows style guide
- [ ] Documentation updated (if needed)
```

## Tips for Effective Commands

- **Use $ARGUMENTS** placeholder for dynamic inputs
- **Reference CLAUDE.md** patterns and conventions
- **Include verification steps** - tests, linting, visual checks
- **Be explicit about constraints** - don't modify X, use pattern Y
- **Use XML tags** for structured prompts: `<task>`, `<requirements>`, `<constraints>`

## Example Pattern

```markdown
Implement #$ARGUMENTS following these steps:

1. Research existing patterns
   - Search for similar code using Grep
   - Read relevant files to understand approach

2. Plan the implementation
   - Think through edge cases and requirements
   - Consider test cases needed

3. Implement
   - Follow existing code patterns (reference specific files)
   - Write tests first if doing TDD
   - Ensure code follows CLAUDE.md conventions

4. Verify
   - Run tests: `bin/rails test`
   - Run linter: `bundle exec standardrb`
   - Check changes with git diff

5. Commit (optional)
   - Stage changes
   - Write clear commit message
```

## Creating the Command File

1. **Create the file** at `.claude/commands/[name].md` (subdirectories like `workflows/` supported)
2. **Start with YAML frontmatter** (see section above)
3. **Structure the command** using the template above
4. **Test the command** by using it with appropriate arguments

## Command File Template

```markdown
---
name: command-name
description: What this command does
argument-hint: "[expected arguments]"
---

# Command Title

Brief introduction of what the command does and when to use it.

## Workflow

### Step 1: [First Major Step]

Details about what to do.

### Step 2: [Second Major Step]

Details about what to do.

## Success Criteria

- [ ] Expected outcome 1
- [ ] Expected outcome 2
```

---

## /changelog

_Source: `marketplaces/every-marketplace/plugins/compound-engineering/commands/changelog.md`_

---
name: changelog
description: Create engaging changelogs for recent merges to main branch
argument-hint: "[optional: daily|weekly, or time period in days]"
disable-model-invocation: true
---

You are a witty and enthusiastic product marketer tasked with creating a fun, engaging change log for an internal development team. Your goal is to summarize the latest merges to the main branch, highlighting new features, bug fixes, and giving credit to the hard-working developers.

## Time Period

- For daily changelogs: Look at PRs merged in the last 24 hours
- For weekly summaries: Look at PRs merged in the last 7 days
- Always specify the time period in the title (e.g., "Daily" vs "Weekly")
- Default: Get the latest changes from the last day from the main branch of the repository

## PR Analysis

Analyze the provided GitHub changes and related issues. Look for:

1. New features that have been added
2. Bug fixes that have been implemented
3. Any other significant changes or improvements
4. References to specific issues and their details
5. Names of contributors who made the changes
6. Use gh cli to lookup the PRs as well and the description of the PRs
7. Check PR labels to identify feature type (feature, bug, chore, etc.)
8. Look for breaking changes and highlight them prominently
9. Include PR numbers for traceability
10. Check if PRs are linked to issues and include issue context

## Content Priorities

1. Breaking changes (if any) - MUST be at the top
2. User-facing features
3. Critical bug fixes
4. Performance improvements
5. Developer experience improvements
6. Documentation updates

## Formatting Guidelines

Now, create a change log summary with the following guidelines:

1. Keep it concise and to the point
2. Highlight the most important changes first
3. Group similar changes together (e.g., all new features, all bug fixes)
4. Include issue references where applicable
5. Mention the names of contributors, giving them credit for their work
6. Add a touch of humor or playfulness to make it engaging
7. Use emojis sparingly to add visual interest
8. Keep total message under 2000 characters for Discord
9. Use consistent emoji for each section
10. Format code/technical terms in backticks
11. Include PR numbers in parentheses (e.g., "Fixed login bug (#123)")

## Deployment Notes

When relevant, include:

- Database migrations required
- Environment variable updates needed
- Manual intervention steps post-deploy
- Dependencies that need updating

Your final output should be formatted as follows:

<change_log>

# 🚀 [Daily/Weekly] Change Log: [Current Date]

## 🚨 Breaking Changes (if any)

[List any breaking changes that require immediate attention]

## 🌟 New Features

[List new features here with PR numbers]

## 🐛 Bug Fixes

[List bug fixes here with PR numbers]

## 🛠️ Other Improvements

[List other significant changes or improvements]

## 🙌 Shoutouts

[Mention contributors and their contributions]

## 🎉 Fun Fact of the Day

[Include a brief, work-related fun fact or joke]

</change_log>

## Style Guide Review

Now review the changelog using the EVERY_WRITE_STYLE.md file and go one by one to make sure you are following the style guide. Use multiple agents, run in parallel to make it faster.

Remember, your final output should only include the content within the <change_log> tags. Do not include any of your thought process or the original data in the output.

## Discord Posting (Optional)

You can post changelogs to Discord by adding your own webhook URL:

```
# Set your Discord webhook URL
DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/YOUR_WEBHOOK_ID/YOUR_WEBHOOK_TOKEN"

# Post using curl
curl -H "Content-Type: application/json" \
  -d "{\"content\": \"{{CHANGELOG}}\"}" \
  $DISCORD_WEBHOOK_URL
```

To get a webhook URL, go to your Discord server → Server Settings → Integrations → Webhooks → New Webhook.

## Error Handling

- If no changes in the time period, post a "quiet day" message: "🌤️ Quiet day! No new changes merged."
- If unable to fetch PR details, list the PR numbers for manual review
- Always validate message length before posting to Discord (max 2000 chars)

## Schedule Recommendations

- Run daily at 6 AM NY time for previous day's changes
- Run weekly summary on Mondays for the previous week
- Special runs after major releases or deployments

## Audience Considerations

Adjust the tone and detail level based on the channel:

- **Dev team channels**: Include technical details, performance metrics, code snippets
- **Product team channels**: Focus on user-facing changes and business impact
- **Leadership channels**: Highlight progress on key initiatives and blockers

---

## /deploy-docs

_Source: `marketplaces/every-marketplace/plugins/compound-engineering/commands/deploy-docs.md`_

---
name: deploy-docs
description: Validate and prepare documentation for GitHub Pages deployment
disable-model-invocation: true
---

# Deploy Documentation Command

Validate the documentation site and prepare it for GitHub Pages deployment.

## Step 1: Validate Documentation

Run these checks:

```bash
# Count components
echo "Agents: $(ls plugins/compound-engineering/agents/*.md | wc -l)"
echo "Commands: $(ls plugins/compound-engineering/commands/*.md | wc -l)"
echo "Skills: $(ls -d plugins/compound-engineering/skills/*/ 2>/dev/null | wc -l)"

# Validate JSON
cat .claude-plugin/marketplace.json | jq . > /dev/null && echo "✓ marketplace.json valid"
cat plugins/compound-engineering/.claude-plugin/plugin.json | jq . > /dev/null && echo "✓ plugin.json valid"

# Check all HTML files exist
for page in index agents commands skills mcp-servers changelog getting-started; do
  if [ -f "plugins/compound-engineering/docs/pages/${page}.html" ] || [ -f "plugins/compound-engineering/docs/${page}.html" ]; then
    echo "✓ ${page}.html exists"
  else
    echo "✗ ${page}.html MISSING"
  fi
done
```

## Step 2: Check for Uncommitted Changes

```bash
git status --porcelain plugins/compound-engineering/docs/
```

If there are uncommitted changes, warn the user to commit first.

## Step 3: Deployment Instructions

Since GitHub Pages deployment requires a workflow file with special permissions, provide these instructions:

### First-time Setup

1. Create `.github/workflows/deploy-docs.yml` with the GitHub Pages workflow
2. Go to repository Settings > Pages
3. Set Source to "GitHub Actions"

### Deploying

After merging to `main`, the docs will auto-deploy. Or:

1. Go to Actions tab
2. Select "Deploy Documentation to GitHub Pages"
3. Click "Run workflow"

### Workflow File Content

```yaml
name: Deploy Documentation to GitHub Pages

on:
  push:
    branches: [main]
    paths:
      - 'plugins/compound-engineering/docs/**'
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: "pages"
  cancel-in-progress: false

jobs:
  deploy:
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/configure-pages@v4
      - uses: actions/upload-pages-artifact@v3
        with:
          path: 'plugins/compound-engineering/docs'
      - uses: actions/deploy-pages@v4
```

## Step 4: Report Status

Provide a summary:

```
## Deployment Readiness

✓ All HTML pages present
✓ JSON files valid
✓ Component counts match

### Next Steps
- [ ] Commit any pending changes
- [ ] Push to main branch
- [ ] Verify GitHub Pages workflow exists
- [ ] Check deployment at https://everyinc.github.io/every-marketplace/
```

---

## /triage

_Source: `marketplaces/every-marketplace/plugins/compound-engineering/commands/triage.md`_

---
name: triage
description: Triage and categorize findings for the CLI todo system
argument-hint: "[findings list or source type]"
disable-model-invocation: true
---

- First set the /model to Haiku
- Then read all pending todos in the todos/ directory

Present all findings, decisions, or issues here one by one for triage. The goal is to go through each item and decide whether to add it to the CLI todo system.

**IMPORTANT: DO NOT CODE ANYTHING DURING TRIAGE!**

This command is for:

- Triaging code review findings
- Processing security audit results
- Reviewing performance analysis
- Handling any other categorized findings that need tracking

## Workflow

### Step 1: Present Each Finding

For each finding, present in this format:

```
---
Issue #X: [Brief Title]

Severity: 🔴 P1 (CRITICAL) / 🟡 P2 (IMPORTANT) / 🔵 P3 (NICE-TO-HAVE)

Category: [Security/Performance/Architecture/Bug/Feature/etc.]

Description:
[Detailed explanation of the issue or improvement]

Location: [file_path:line_number]

Problem Scenario:
[Step by step what's wrong or could happen]

Proposed Solution:
[How to fix it]

Estimated Effort: [Small (< 2 hours) / Medium (2-8 hours) / Large (> 8 hours)]

---
Do you want to add this to the todo list?
1. yes - create todo file
2. next - skip this item
3. custom - modify before creating
```

### Step 2: Handle User Decision

**When user says "yes":**

1. **Update existing todo file** (if it exists) or **Create new filename:**

   If todo already exists (from code review):

   - Rename file from `{id}-pending-{priority}-{desc}.md` → `{id}-ready-{priority}-{desc}.md`
   - Update YAML frontmatter: `status: pending` → `status: ready`
   - Keep issue_id, priority, and description unchanged

   If creating new todo:

   ```
   {next_id}-ready-{priority}-{brief-description}.md
   ```

   Priority mapping:

   - 🔴 P1 (CRITICAL) → `p1`
   - 🟡 P2 (IMPORTANT) → `p2`
   - 🔵 P3 (NICE-TO-HAVE) → `p3`

   Example: `042-ready-p1-transaction-boundaries.md`

2. **Update YAML frontmatter:**

   ```yaml
   ---
   status: ready # IMPORTANT: Change from "pending" to "ready"
   priority: p1 # or p2, p3 based on severity
   issue_id: "042"
   tags: [category, relevant-tags]
   dependencies: []
   ---
   ```

3. **Populate or update the file:**

   ```yaml
   # [Issue Title]

   ## Problem Statement
   [Description from finding]

   ## Findings
   - [Key discoveries]
   - Location: [file_path:line_number]
   - [Scenario details]

   ## Proposed Solutions

   ### Option 1: [Primary solution]
   - **Pros**: [Benefits]
   - **Cons**: [Drawbacks if any]
   - **Effort**: [Small/Medium/Large]
   - **Risk**: [Low/Medium/High]

   ## Recommended Action
   [Filled during triage - specific action plan]

   ## Technical Details
   - **Affected Files**: [List files]
   - **Related Components**: [Components affected]
   - **Database Changes**: [Yes/No - describe if yes]

   ## Resources
   - Original finding: [Source of this issue]
   - Related issues: [If any]

   ## Acceptance Criteria
   - [ ] [Specific success criteria]
   - [ ] Tests pass
   - [ ] Code reviewed

   ## Work Log

   ### {date} - Approved for Work
   **By:** Claude Triage System
   **Actions:**
   - Issue approved during triage session
   - Status changed from pending → ready
   - Ready to be picked up and worked on

   **Learnings:**
   - [Context and insights]

   ## Notes
   Source: Triage session on {date}
   ```

4. **Confirm approval:** "✅ Approved: `{new_filename}` (Issue #{issue_id}) - Status: **ready** → Ready to work on"

**When user says "next":**

- **Delete the todo file** - Remove it from todos/ directory since it's not relevant
- Skip to the next item
- Track skipped items for summary

**When user says "custom":**

- Ask what to modify (priority, description, details)
- Update the information
- Present revised version
- Ask again: yes/next/custom

### Step 3: Continue Until All Processed

- Process all items one by one
- Track using TodoWrite for visibility
- Don't wait for approval between items - keep moving

### Step 4: Final Summary

After all items processed:

````markdown
## Triage Complete

**Total Items:** [X] **Todos Approved (ready):** [Y] **Skipped:** [Z]

### Approved Todos (Ready for Work):

- `042-ready-p1-transaction-boundaries.md` - Transaction boundary issue
- `043-ready-p2-cache-optimization.md` - Cache performance improvement ...

### Skipped Items (Deleted):

- Item #5: [reason] - Removed from todos/
- Item #12: [reason] - Removed from todos/

### Summary of Changes Made:

During triage, the following status updates occurred:

- **Pending → Ready:** Filenames and frontmatter updated to reflect approved status
- **Deleted:** Todo files for skipped findings removed from todos/ directory
- Each approved file now has `status: ready` in YAML frontmatter

### Next Steps:

1. View approved todos ready for work:
   ```bash
   ls todos/*-ready-*.md
   ```
````

2. Start work on approved items:

   ```bash
   /resolve_todo_parallel  # Work on multiple approved items efficiently
   ```

3. Or pick individual items to work on

4. As you work, update todo status:
   - Ready → In Progress (in your local context as you work)
   - In Progress → Complete (rename file: ready → complete, update frontmatter)

```

## Example Response Format

```

---

Issue #5: Missing Transaction Boundaries for Multi-Step Operations

Severity: 🔴 P1 (CRITICAL)

Category: Data Integrity / Security

Description: The google_oauth2_connected callback in GoogleOauthCallbacks concern performs multiple database operations without transaction protection. If any step fails midway, the database is left in an inconsistent state.

Location: app/controllers/concerns/google_oauth_callbacks.rb:13-50

Problem Scenario:

1. User.update succeeds (email changed)
2. Account.save! fails (validation error)
3. Result: User has changed email but no associated Account
4. Next login attempt fails completely

Operations Without Transaction:

- User confirmation (line 13)
- Waitlist removal (line 14)
- User profile update (line 21-23)
- Account creation (line 28-37)
- Avatar attachment (line 39-45)
- Journey creation (line 47)

Proposed Solution: Wrap all operations in ApplicationRecord.transaction do ... end block

Estimated Effort: Small (30 minutes)

---

Do you want to add this to the todo list?

1. yes - create todo file
2. next - skip this item
3. custom - modify before creating

```

## Important Implementation Details

### Status Transitions During Triage

**When "yes" is selected:**
1. Rename file: `{id}-pending-{priority}-{desc}.md` → `{id}-ready-{priority}-{desc}.md`
2. Update YAML frontmatter: `status: pending` → `status: ready`
3. Update Work Log with triage approval entry
4. Confirm: "✅ Approved: `{filename}` (Issue #{issue_id}) - Status: **ready**"

**When "next" is selected:**
1. Delete the todo file from todos/ directory
2. Skip to next item
3. No file remains in the system

### Progress Tracking

Every time you present a todo as a header, include:
- **Progress:** X/Y completed (e.g., "3/10 completed")
- **Estimated time remaining:** Based on how quickly you're progressing
- **Pacing:** Monitor time per finding and adjust estimate accordingly

Example:
```

Progress: 3/10 completed | Estimated time: ~2 minutes remaining

```

### Do Not Code During Triage

- ✅ Present findings
- ✅ Make yes/next/custom decisions
- ✅ Update todo files (rename, frontmatter, work log)
- ❌ Do NOT implement fixes or write code
- ❌ Do NOT add detailed implementation details
- ❌ That's for /resolve_todo_parallel phase
```

When done give these options

```markdown
What would you like to do next?

1. run /resolve_todo_parallel to resolve the todos
2. commit the todos
3. nothing, go chill
```

---

## /deepen-plan

_Source: `marketplaces/every-marketplace/plugins/compound-engineering/commands/deepen-plan.md`_

---
name: deepen-plan
description: Enhance a plan with parallel research agents for each section to add depth, best practices, and implementation details
argument-hint: "[path to plan file]"
---

# Deepen Plan - Power Enhancement Mode

## Introduction

**Note: The current year is 2026.** Use this when searching for recent documentation and best practices.

This command takes an existing plan (from `/workflows:plan`) and enhances each section with parallel research agents. Each major element gets its own dedicated research sub-agent to find:
- Best practices and industry patterns
- Performance optimizations
- UI/UX improvements (if applicable)
- Quality enhancements and edge cases
- Real-world implementation examples

The result is a deeply grounded, production-ready plan with concrete implementation details.

## Plan File

<plan_path> #$ARGUMENTS </plan_path>

**If the plan path above is empty:**
1. Check for recent plans: `ls -la docs/plans/`
2. Ask the user: "Which plan would you like to deepen? Please provide the path (e.g., `docs/plans/2026-01-15-feat-my-feature-plan.md`)."

Do not proceed until you have a valid plan file path.

## Main Tasks

### 1. Parse and Analyze Plan Structure

<thinking>
First, read and parse the plan to identify each major section that can be enhanced with research.
</thinking>

**Read the plan file and extract:**
- [ ] Overview/Problem Statement
- [ ] Proposed Solution sections
- [ ] Technical Approach/Architecture
- [ ] Implementation phases/steps
- [ ] Code examples and file references
- [ ] Acceptance criteria
- [ ] Any UI/UX components mentioned
- [ ] Technologies/frameworks mentioned (Rails, React, Python, TypeScript, etc.)
- [ ] Domain areas (data models, APIs, UI, security, performance, etc.)

**Create a section manifest:**
```
Section 1: [Title] - [Brief description of what to research]
Section 2: [Title] - [Brief description of what to research]
...
```

### 2. Discover and Apply Available Skills

<thinking>
Dynamically discover all available skills and match them to plan sections. Don't assume what skills exist - discover them at runtime.
</thinking>

**Step 1: Discover ALL available skills from ALL sources**

```bash
# 1. Project-local skills (highest priority - project-specific)
ls .claude/skills/

# 2. User's global skills (~/.claude/)
ls ~/.claude/skills/

# 3. compound-engineering plugin skills
ls ~/.claude/plugins/cache/*/compound-engineering/*/skills/

# 4. ALL other installed plugins - check every plugin for skills
find ~/.claude/plugins/cache -type d -name "skills" 2>/dev/null

# 5. Also check installed_plugins.json for all plugin locations
cat ~/.claude/plugins/installed_plugins.json
```

**Important:** Check EVERY source. Don't assume compound-engineering is the only plugin. Use skills from ANY installed plugin that's relevant.

**Step 2: For each discovered skill, read its SKILL.md to understand what it does**

```bash
# For each skill directory found, read its documentation
cat [skill-path]/SKILL.md
```

**Step 3: Match skills to plan content**

For each skill discovered:
- Read its SKILL.md description
- Check if any plan sections match the skill's domain
- If there's a match, spawn a sub-agent to apply that skill's knowledge

**Step 4: Spawn a sub-agent for EVERY matched skill**

**CRITICAL: For EACH skill that matches, spawn a separate sub-agent and instruct it to USE that skill.**

For each matched skill:
```
Task general-purpose: "You have the [skill-name] skill available at [skill-path].

YOUR JOB: Use this skill on the plan.

1. Read the skill: cat [skill-path]/SKILL.md
2. Follow the skill's instructions exactly
3. Apply the skill to this content:

[relevant plan section or full plan]

4. Return the skill's full output

The skill tells you what to do - follow it. Execute the skill completely."
```

**Spawn ALL skill sub-agents in PARALLEL:**
- 1 sub-agent per matched skill
- Each sub-agent reads and uses its assigned skill
- All run simultaneously
- 10, 20, 30 skill sub-agents is fine

**Each sub-agent:**
1. Reads its skill's SKILL.md
2. Follows the skill's workflow/instructions
3. Applies the skill to the plan
4. Returns whatever the skill produces (code, recommendations, patterns, reviews, etc.)

**Example spawns:**
```
Task general-purpose: "Use the dhh-rails-style skill at ~/.claude/plugins/.../dhh-rails-style. Read SKILL.md and apply it to: [Rails sections of plan]"

Task general-purpose: "Use the frontend-design skill at ~/.claude/plugins/.../frontend-design. Read SKILL.md and apply it to: [UI sections of plan]"

Task general-purpose: "Use the agent-native-architecture skill at ~/.claude/plugins/.../agent-native-architecture. Read SKILL.md and apply it to: [agent/tool sections of plan]"

Task general-purpose: "Use the security-patterns skill at ~/.claude/skills/security-patterns. Read SKILL.md and apply it to: [full plan]"
```

**No limit on skill sub-agents. Spawn one for every skill that could possibly be relevant.**

### 3. Discover and Apply Learnings/Solutions

<thinking>
Check for documented learnings from /workflows:compound. These are solved problems stored as markdown files. Spawn a sub-agent for each learning to check if it's relevant.
</thinking>

**LEARNINGS LOCATION - Check these exact folders:**

```
docs/solutions/           <-- PRIMARY: Project-level learnings (created by /workflows:compound)
├── performance-issues/
│   └── *.md
├── debugging-patterns/
│   └── *.md
├── configuration-fixes/
│   └── *.md
├── integration-issues/
│   └── *.md
├── deployment-issues/
│   └── *.md
└── [other-categories]/
    └── *.md
```

**Step 1: Find ALL learning markdown files**

Run these commands to get every learning file:

```bash
# PRIMARY LOCATION - Project learnings
find docs/solutions -name "*.md" -type f 2>/dev/null

# If docs/solutions doesn't exist, check alternate locations:
find .claude/docs -name "*.md" -type f 2>/dev/null
find ~/.claude/docs -name "*.md" -type f 2>/dev/null
```

**Step 2: Read frontmatter of each learning to filter**

Each learning file has YAML frontmatter with metadata. Read the first ~20 lines of each file to get:

```yaml
---
title: "N+1 Query Fix for Briefs"
category: performance-issues
tags: [activerecord, n-plus-one, includes, eager-loading]
module: Briefs
symptom: "Slow page load, multiple queries in logs"
root_cause: "Missing includes on association"
---
```

**For each .md file, quickly scan its frontmatter:**

```bash
# Read first 20 lines of each learning (frontmatter + summary)
head -20 docs/solutions/**/*.md
```

**Step 3: Filter - only spawn sub-agents for LIKELY relevant learnings**

Compare each learning's frontmatter against the plan:
- `tags:` - Do any tags match technologies/patterns in the plan?
- `category:` - Is this category relevant? (e.g., skip deployment-issues if plan is UI-only)
- `module:` - Does the plan touch this module?
- `symptom:` / `root_cause:` - Could this problem occur with the plan?

**SKIP learnings that are clearly not applicable:**
- Plan is frontend-only → skip `database-migrations/` learnings
- Plan is Python → skip `rails-specific/` learnings
- Plan has no auth → skip `authentication-issues/` learnings

**SPAWN sub-agents for learnings that MIGHT apply:**
- Any tag overlap with plan technologies
- Same category as plan domain
- Similar patterns or concerns

**Step 4: Spawn sub-agents for filtered learnings**

For each learning that passes the filter:

```
Task general-purpose: "
LEARNING FILE: [full path to .md file]

1. Read this learning file completely
2. This learning documents a previously solved problem

Check if this learning applies to this plan:

---
[full plan content]
---

If relevant:
- Explain specifically how it applies
- Quote the key insight or solution
- Suggest where/how to incorporate it

If NOT relevant after deeper analysis:
- Say 'Not applicable: [reason]'
"
```

**Example filtering:**
```
# Found 15 learning files, plan is about "Rails API caching"

# SPAWN (likely relevant):
docs/solutions/performance-issues/n-plus-one-queries.md      # tags: [activerecord] ✓
docs/solutions/performance-issues/redis-cache-stampede.md    # tags: [caching, redis] ✓
docs/solutions/configuration-fixes/redis-connection-pool.md  # tags: [redis] ✓

# SKIP (clearly not applicable):
docs/solutions/deployment-issues/heroku-memory-quota.md      # not about caching
docs/solutions/frontend-issues/stimulus-race-condition.md    # plan is API, not frontend
docs/solutions/authentication-issues/jwt-expiry.md           # plan has no auth
```

**Spawn sub-agents in PARALLEL for all filtered learnings.**

**These learnings are institutional knowledge - applying them prevents repeating past mistakes.**

### 4. Launch Per-Section Research Agents

<thinking>
For each major section in the plan, spawn dedicated sub-agents to research improvements. Use the Explore agent type for open-ended research.
</thinking>

**For each identified section, launch parallel research:**

```
Task Explore: "Research best practices, patterns, and real-world examples for: [section topic].
Find:
- Industry standards and conventions
- Performance considerations
- Common pitfalls and how to avoid them
- Documentation and tutorials
Return concrete, actionable recommendations."
```

**Also use Context7 MCP for framework documentation:**

For any technologies/frameworks mentioned in the plan, query Context7:
```
mcp__plugin_compound-engineering_context7__resolve-library-id: Find library ID for [framework]
mcp__plugin_compound-engineering_context7__query-docs: Query documentation for specific patterns
```

**Use WebSearch for current best practices:**

Search for recent (2024-2026) articles, blog posts, and documentation on topics in the plan.

### 5. Discover and Run ALL Review Agents

<thinking>
Dynamically discover every available agent and run them ALL against the plan. Don't filter, don't skip, don't assume relevance. 40+ parallel agents is fine. Use everything available.
</thinking>

**Step 1: Discover ALL available agents from ALL sources**

```bash
# 1. Project-local agents (highest priority - project-specific)
find .claude/agents -name "*.md" 2>/dev/null

# 2. User's global agents (~/.claude/)
find ~/.claude/agents -name "*.md" 2>/dev/null

# 3. compound-engineering plugin agents (all subdirectories)
find ~/.claude/plugins/cache/*/compound-engineering/*/agents -name "*.md" 2>/dev/null

# 4. ALL other installed plugins - check every plugin for agents
find ~/.claude/plugins/cache -path "*/agents/*.md" 2>/dev/null

# 5. Check installed_plugins.json to find all plugin locations
cat ~/.claude/plugins/installed_plugins.json

# 6. For local plugins (isLocal: true), check their source directories
# Parse installed_plugins.json and find local plugin paths
```

**Important:** Check EVERY source. Include agents from:
- Project `.claude/agents/`
- User's `~/.claude/agents/`
- compound-engineering plugin (but SKIP workflow/ agents - only use review/, research/, design/, docs/)
- ALL other installed plugins (agent-sdk-dev, frontend-design, etc.)
- Any local plugins

**For compound-engineering plugin specifically:**
- USE: `agents/review/*` (all reviewers)
- USE: `agents/research/*` (all researchers)
- USE: `agents/design/*` (design agents)
- USE: `agents/docs/*` (documentation agents)
- SKIP: `agents/workflow/*` (these are workflow orchestrators, not reviewers)

**Step 2: For each discovered agent, read its description**

Read the first few lines of each agent file to understand what it reviews/analyzes.

**Step 3: Launch ALL agents in parallel**

For EVERY agent discovered, launch a Task in parallel:

```
Task [agent-name]: "Review this plan using your expertise. Apply all your checks and patterns. Plan content: [full plan content]"
```

**CRITICAL RULES:**
- Do NOT filter agents by "relevance" - run them ALL
- Do NOT skip agents because they "might not apply" - let them decide
- Launch ALL agents in a SINGLE message with multiple Task tool calls
- 20, 30, 40 parallel agents is fine - use everything
- Each agent may catch something others miss
- The goal is MAXIMUM coverage, not efficiency

**Step 4: Also discover and run research agents**

Research agents (like `best-practices-researcher`, `framework-docs-researcher`, `git-history-analyzer`, `repo-research-analyst`) should also be run for relevant plan sections.

### 6. Wait for ALL Agents and Synthesize Everything

<thinking>
Wait for ALL parallel agents to complete - skills, research agents, review agents, everything. Then synthesize all findings into a comprehensive enhancement.
</thinking>

**Collect outputs from ALL sources:**

1. **Skill-based sub-agents** - Each skill's full output (code examples, patterns, recommendations)
2. **Learnings/Solutions sub-agents** - Relevant documented learnings from /workflows:compound
3. **Research agents** - Best practices, documentation, real-world examples
4. **Review agents** - All feedback from every reviewer (architecture, security, performance, simplicity, etc.)
5. **Context7 queries** - Framework documentation and patterns
6. **Web searches** - Current best practices and articles

**For each agent's findings, extract:**
- [ ] Concrete recommendations (actionable items)
- [ ] Code patterns and examples (copy-paste ready)
- [ ] Anti-patterns to avoid (warnings)
- [ ] Performance considerations (metrics, benchmarks)
- [ ] Security considerations (vulnerabilities, mitigations)
- [ ] Edge cases discovered (handling strategies)
- [ ] Documentation links (references)
- [ ] Skill-specific patterns (from matched skills)
- [ ] Relevant learnings (past solutions that apply - prevent repeating mistakes)

**Deduplicate and prioritize:**
- Merge similar recommendations from multiple agents
- Prioritize by impact (high-value improvements first)
- Flag conflicting advice for human review
- Group by plan section

### 7. Enhance Plan Sections

<thinking>
Merge research findings back into the plan, adding depth without changing the original structure.
</thinking>

**Enhancement format for each section:**

```markdown
## [Original Section Title]

[Original content preserved]

### Research Insights

**Best Practices:**
- [Concrete recommendation 1]
- [Concrete recommendation 2]

**Performance Considerations:**
- [Optimization opportunity]
- [Benchmark or metric to target]

**Implementation Details:**
```[language]
// Concrete code example from research
```

**Edge Cases:**
- [Edge case 1 and how to handle]
- [Edge case 2 and how to handle]

**References:**
- [Documentation URL 1]
- [Documentation URL 2]
```

### 8. Add Enhancement Summary

At the top of the plan, add a summary section:

```markdown
## Enhancement Summary

**Deepened on:** [Date]
**Sections enhanced:** [Count]
**Research agents used:** [List]

### Key Improvements
1. [Major improvement 1]
2. [Major improvement 2]
3. [Major improvement 3]

### New Considerations Discovered
- [Important finding 1]
- [Important finding 2]
```

### 9. Update Plan File

**Write the enhanced plan:**
- Preserve original filename
- Add `-deepened` suffix if user prefers a new file
- Update any timestamps or metadata

## Output Format

Update the plan file in place (or if user requests a separate file, append `-deepened` after `-plan`, e.g., `2026-01-15-feat-auth-plan-deepened.md`).

## Quality Checks

Before finalizing:
- [ ] All original content preserved
- [ ] Research insights clearly marked and attributed
- [ ] Code examples are syntactically correct
- [ ] Links are valid and relevant
- [ ] No contradictions between sections
- [ ] Enhancement summary accurately reflects changes

## Post-Enhancement Options

After writing the enhanced plan, use the **AskUserQuestion tool** to present these options:

**Question:** "Plan deepened at `[plan_path]`. What would you like to do next?"

**Options:**
1. **View diff** - Show what was added/changed
2. **Run `/technical_review`** - Get feedback from reviewers on enhanced plan
3. **Start `/workflows:work`** - Begin implementing this enhanced plan
4. **Deepen further** - Run another round of research on specific sections
5. **Revert** - Restore original plan (if backup exists)

Based on selection:
- **View diff** → Run `git diff [plan_path]` or show before/after
- **`/technical_review`** → Call the /technical_review command with the plan file path
- **`/workflows:work`** → Call the /workflows:work command with the plan file path
- **Deepen further** → Ask which sections need more research, then re-run those agents
- **Revert** → Restore from git or backup

## Example Enhancement

**Before (from /workflows:plan):**
```markdown
## Technical Approach

Use React Query for data fetching with optimistic updates.
```

**After (from /workflows:deepen-plan):**
```markdown
## Technical Approach

Use React Query for data fetching with optimistic updates.

### Research Insights

**Best Practices:**
- Configure `staleTime` and `cacheTime` based on data freshness requirements
- Use `queryKey` factories for consistent cache invalidation
- Implement error boundaries around query-dependent components

**Performance Considerations:**
- Enable `refetchOnWindowFocus: false` for stable data to reduce unnecessary requests
- Use `select` option to transform and memoize data at query level
- Consider `placeholderData` for instant perceived loading

**Implementation Details:**
```typescript
// Recommended query configuration
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5 * 60 * 1000, // 5 minutes
      retry: 2,
      refetchOnWindowFocus: false,
    },
  },
});
```

**Edge Cases:**
- Handle race conditions with `cancelQueries` on component unmount
- Implement retry logic for transient network failures
- Consider offline support with `persistQueryClient`

**References:**
- https://tanstack.com/query/latest/docs/react/guides/optimistic-updates
- https://tkdodo.eu/blog/practical-react-query
```

NEVER CODE! Just research and enhance the plan.

---

## /report-bug

_Source: `marketplaces/every-marketplace/plugins/compound-engineering/commands/report-bug.md`_

---
name: report-bug
description: Report a bug in the compound-engineering plugin
argument-hint: "[optional: brief description of the bug]"
disable-model-invocation: true
---

# Report a Compounding Engineering Plugin Bug

Report bugs encountered while using the compound-engineering plugin. This command gathers structured information and creates a GitHub issue for the maintainer.

## Step 1: Gather Bug Information

Use the AskUserQuestion tool to collect the following information:

**Question 1: Bug Category**
- What type of issue are you experiencing?
- Options: Agent not working, Command not working, Skill not working, MCP server issue, Installation problem, Other

**Question 2: Specific Component**
- Which specific component is affected?
- Ask for the name of the agent, command, skill, or MCP server

**Question 3: What Happened (Actual Behavior)**
- Ask: "What happened when you used this component?"
- Get a clear description of the actual behavior

**Question 4: What Should Have Happened (Expected Behavior)**
- Ask: "What did you expect to happen instead?"
- Get a clear description of expected behavior

**Question 5: Steps to Reproduce**
- Ask: "What steps did you take before the bug occurred?"
- Get reproduction steps

**Question 6: Error Messages**
- Ask: "Did you see any error messages? If so, please share them."
- Capture any error output

## Step 2: Collect Environment Information

Automatically gather:
```bash
# Get plugin version
cat ~/.claude/plugins/installed_plugins.json 2>/dev/null | grep -A5 "compound-engineering" | head -10 || echo "Plugin info not found"

# Get Claude Code version
claude --version 2>/dev/null || echo "Claude CLI version unknown"

# Get OS info
uname -a
```

## Step 3: Format the Bug Report

Create a well-structured bug report with:

```markdown
## Bug Description

**Component:** [Type] - [Name]
**Summary:** [Brief description from argument or collected info]

## Environment

- **Plugin Version:** [from installed_plugins.json]
- **Claude Code Version:** [from claude --version]
- **OS:** [from uname]

## What Happened

[Actual behavior description]

## Expected Behavior

[Expected behavior description]

## Steps to Reproduce

1. [Step 1]
2. [Step 2]
3. [Step 3]

## Error Messages

```
[Any error output]
```

## Additional Context

[Any other relevant information]

---
*Reported via `/report-bug` command*
```

## Step 4: Create GitHub Issue

Use the GitHub CLI to create the issue:

```bash
gh issue create \
  --repo EveryInc/compound-engineering-plugin \
  --title "[compound-engineering] Bug: [Brief description]" \
  --body "[Formatted bug report from Step 3]" \
  --label "bug,compound-engineering"
```

**Note:** If labels don't exist, create without labels:
```bash
gh issue create \
  --repo EveryInc/compound-engineering-plugin \
  --title "[compound-engineering] Bug: [Brief description]" \
  --body "[Formatted bug report]"
```

## Step 5: Confirm Submission

After the issue is created:
1. Display the issue URL to the user
2. Thank them for reporting the bug
3. Let them know the maintainer (Kieran Klaassen) will be notified

## Output Format

```
✅ Bug report submitted successfully!

Issue: https://github.com/EveryInc/compound-engineering-plugin/issues/[NUMBER]
Title: [compound-engineering] Bug: [description]

Thank you for helping improve the compound-engineering plugin!
The maintainer will review your report and respond as soon as possible.
```

## Error Handling

- If `gh` CLI is not authenticated: Prompt user to run `gh auth login` first
- If issue creation fails: Display the formatted report so user can manually create the issue
- If required information is missing: Re-prompt for that specific field

## Privacy Notice

This command does NOT collect:
- Personal information
- API keys or credentials
- Private code from your projects
- File paths beyond basic OS info

Only technical information about the bug is included in the report.

---

## /lfg

_Source: `marketplaces/every-marketplace/plugins/compound-engineering/commands/lfg.md`_

---
name: lfg
description: Full autonomous engineering workflow
argument-hint: "[feature description]"
disable-model-invocation: true
---

Run these slash commands in order. Do not do anything else.

1. `/ralph-wiggum:ralph-loop "finish all slash commands" --completion-promise "DONE"`
2. `/workflows:plan $ARGUMENTS`
3. `/compound-engineering:deepen-plan`
4. `/workflows:work`
5. `/workflows:review`
6. `/compound-engineering:resolve_todo_parallel`
7. `/compound-engineering:test-browser`
8. `/compound-engineering:feature-video`
9. Output `<promise>DONE</promise>` when video is in PR

Start with step 1 now.

---

## /work

_Source: `marketplaces/every-marketplace/plugins/compound-engineering/commands/workflows/work.md`_

---
name: workflows:work
description: Execute work plans efficiently while maintaining quality and finishing features
argument-hint: "[plan file, specification, or todo file path]"
---

# Work Plan Execution Command

Execute a work plan efficiently while maintaining quality and finishing features.

## Introduction

This command takes a work document (plan, specification, or todo file) and executes it systematically. The focus is on **shipping complete features** by understanding requirements quickly, following existing patterns, and maintaining quality throughout.

## Input Document

<input_document> #$ARGUMENTS </input_document>

## Execution Workflow

### Phase 1: Quick Start

1. **Read Plan and Clarify**

   - Read the work document completely
   - Review any references or links provided in the plan
   - If anything is unclear or ambiguous, ask clarifying questions now
   - Get user approval to proceed
   - **Do not skip this** - better to ask questions now than build the wrong thing

2. **Setup Environment**

   First, check the current branch:

   ```bash
   current_branch=$(git branch --show-current)
   default_branch=$(git symbolic-ref refs/remotes/origin/HEAD 2>/dev/null | sed 's@^refs/remotes/origin/@@')

   # Fallback if remote HEAD isn't set
   if [ -z "$default_branch" ]; then
     default_branch=$(git rev-parse --verify origin/main >/dev/null 2>&1 && echo "main" || echo "master")
   fi
   ```

   **If already on a feature branch** (not the default branch):
   - Ask: "Continue working on `[current_branch]`, or create a new branch?"
   - If continuing, proceed to step 3
   - If creating new, follow Option A or B below

   **If on the default branch**, choose how to proceed:

   **Option A: Create a new branch**
   ```bash
   git pull origin [default_branch]
   git checkout -b feature-branch-name
   ```
   Use a meaningful name based on the work (e.g., `feat/user-authentication`, `fix/email-validation`).

   **Option B: Use a worktree (recommended for parallel development)**
   ```bash
   skill: git-worktree
   # The skill will create a new branch from the default branch in an isolated worktree
   ```

   **Option C: Continue on the default branch**
   - Requires explicit user confirmation
   - Only proceed after user explicitly says "yes, commit to [default_branch]"
   - Never commit directly to the default branch without explicit permission

   **Recommendation**: Use worktree if:
   - You want to work on multiple features simultaneously
   - You want to keep the default branch clean while experimenting
   - You plan to switch between branches frequently

3. **Create Todo List**
   - Use TodoWrite to break plan into actionable tasks
   - Include dependencies between tasks
   - Prioritize based on what needs to be done first
   - Include testing and quality check tasks
   - Keep tasks specific and completable

### Phase 2: Execute

1. **Task Execution Loop**

   For each task in priority order:

   ```
   while (tasks remain):
     - Mark task as in_progress in TodoWrite
     - Read any referenced files from the plan
     - Look for similar patterns in codebase
     - Implement following existing conventions
     - Write tests for new functionality
     - Run tests after changes
     - Mark task as completed in TodoWrite
     - Mark off the corresponding checkbox in the plan file ([ ] → [x])
     - Evaluate for incremental commit (see below)
   ```

   **IMPORTANT**: Always update the original plan document by checking off completed items. Use the Edit tool to change `- [ ]` to `- [x]` for each task you finish. This keeps the plan as a living document showing progress and ensures no checkboxes are left unchecked.

2. **Incremental Commits**

   After completing each task, evaluate whether to create an incremental commit:

   | Commit when... | Don't commit when... |
   |----------------|---------------------|
   | Logical unit complete (model, service, component) | Small part of a larger unit |
   | Tests pass + meaningful progress | Tests failing |
   | About to switch contexts (backend → frontend) | Purely scaffolding with no behavior |
   | About to attempt risky/uncertain changes | Would need a "WIP" commit message |

   **Heuristic:** "Can I write a commit message that describes a complete, valuable change? If yes, commit. If the message would be 'WIP' or 'partial X', wait."

   **Commit workflow:**
   ```bash
   # 1. Verify tests pass (use project's test command)
   # Examples: bin/rails test, npm test, pytest, go test, etc.

   # 2. Stage only files related to this logical unit (not `git add .`)
   git add <files related to this logical unit>

   # 3. Commit with conventional message
   git commit -m "feat(scope): description of this unit"
   ```

   **Handling merge conflicts:** If conflicts arise during rebasing or merging, resolve them immediately. Incremental commits make conflict resolution easier since each commit is small and focused.

   **Note:** Incremental commits use clean conventional messages without attribution footers. The final Phase 4 commit/PR includes the full attribution.

3. **Follow Existing Patterns**

   - The plan should reference similar code - read those files first
   - Match naming conventions exactly
   - Reuse existing components where possible
   - Follow project coding standards (see CLAUDE.md)
   - When in doubt, grep for similar implementations

4. **Test Continuously**

   - Run relevant tests after each significant change
   - Don't wait until the end to test
   - Fix failures immediately
   - Add new tests for new functionality

5. **Figma Design Sync** (if applicable)

   For UI work with Figma designs:

   - Implement components following design specs
   - Use figma-design-sync agent iteratively to compare
   - Fix visual differences identified
   - Repeat until implementation matches design

6. **Track Progress**
   - Keep TodoWrite updated as you complete tasks
   - Note any blockers or unexpected discoveries
   - Create new tasks if scope expands
   - Keep user informed of major milestones

### Phase 3: Quality Check

1. **Run Core Quality Checks**

   Always run before submitting:

   ```bash
   # Run full test suite (use project's test command)
   # Examples: bin/rails test, npm test, pytest, go test, etc.

   # Run linting (per CLAUDE.md)
   # Use linting-agent before pushing to origin
   ```

2. **Consider Reviewer Agents** (Optional)

   Use for complex, risky, or large changes:

   - **code-simplicity-reviewer**: Check for unnecessary complexity
   - **kieran-rails-reviewer**: Verify Rails conventions (Rails projects)
   - **performance-oracle**: Check for performance issues
   - **security-sentinel**: Scan for security vulnerabilities
   - **cora-test-reviewer**: Review test quality (Rails projects with comprehensive test coverage)

   Run reviewers in parallel with Task tool:

   ```
   Task(code-simplicity-reviewer): "Review changes for simplicity"
   Task(kieran-rails-reviewer): "Check Rails conventions"
   ```

   Present findings to user and address critical issues.

3. **Final Validation**
   - All TodoWrite tasks marked completed
   - All tests pass
   - Linting passes
   - Code follows existing patterns
   - Figma designs match (if applicable)
   - No console errors or warnings

### Phase 4: Ship It

1. **Create Commit**

   ```bash
   git add .
   git status  # Review what's being committed
   git diff --staged  # Check the changes

   # Commit with conventional format
   git commit -m "$(cat <<'EOF'
   feat(scope): description of what and why

   Brief explanation if needed.

   🤖 Generated with [Claude Code](https://claude.com/claude-code)

   Co-Authored-By: Claude <noreply@anthropic.com>
   EOF
   )"
   ```

2. **Capture and Upload Screenshots for UI Changes** (REQUIRED for any UI work)

   For **any** design changes, new views, or UI modifications, you MUST capture and upload screenshots:

   **Step 1: Start dev server** (if not running)
   ```bash
   bin/dev  # Run in background
   ```

   **Step 2: Capture screenshots with agent-browser CLI**
   ```bash
   agent-browser open http://localhost:3000/[route]
   agent-browser snapshot -i
   agent-browser screenshot output.png
   ```
   See the `agent-browser` skill for detailed usage.

   **Step 3: Upload using imgup skill**
   ```bash
   skill: imgup
   # Then upload each screenshot:
   imgup -h pixhost screenshot.png  # pixhost works without API key
   # Alternative hosts: catbox, imagebin, beeimg
   ```

   **What to capture:**
   - **New screens**: Screenshot of the new UI
   - **Modified screens**: Before AND after screenshots
   - **Design implementation**: Screenshot showing Figma design match

   **IMPORTANT**: Always include uploaded image URLs in PR description. This provides visual context for reviewers and documents the change.

3. **Create Pull Request**

   ```bash
   git push -u origin feature-branch-name

   gh pr create --title "Feature: [Description]" --body "$(cat <<'EOF'
   ## Summary
   - What was built
   - Why it was needed
   - Key decisions made

   ## Testing
   - Tests added/modified
   - Manual testing performed

   ## Before / After Screenshots
   | Before | After |
   |--------|-------|
   | ![before](URL) | ![after](URL) |

   ## Figma Design
   [Link if applicable]

   ---

   [![Compound Engineered](https://img.shields.io/badge/Compound-Engineered-6366f1)](https://github.com/EveryInc/compound-engineering-plugin) 🤖 Generated with [Claude Code](https://claude.com/claude-code)
   EOF
   )"
   ```

4. **Notify User**
   - Summarize what was completed
   - Link to PR
   - Note any follow-up work needed
   - Suggest next steps if applicable

---

## Swarm Mode (Optional)

For complex plans with multiple independent workstreams, enable swarm mode for parallel execution with coordinated agents.

### When to Use Swarm Mode

| Use Swarm Mode when... | Use Standard Mode when... |
|------------------------|---------------------------|
| Plan has 5+ independent tasks | Plan is linear/sequential |
| Multiple specialists needed (review + test + implement) | Single-focus work |
| Want maximum parallelism | Simpler mental model preferred |
| Large feature with clear phases | Small feature or bug fix |

### Enabling Swarm Mode

To trigger swarm execution, say:

> "Make a Task list and launch an army of agent swarm subagents to build the plan"

Or explicitly request: "Use swarm mode for this work"

### Swarm Workflow

When swarm mode is enabled, the workflow changes:

1. **Create Team**
   ```
   Teammate({ operation: "spawnTeam", team_name: "work-{timestamp}" })
   ```

2. **Create Task List with Dependencies**
   - Parse plan into TaskCreate items
   - Set up blockedBy relationships for sequential dependencies
   - Independent tasks have no blockers (can run in parallel)

3. **Spawn Specialized Teammates**
   ```
   Task({
     team_name: "work-{timestamp}",
     name: "implementer",
     subagent_type: "general-purpose",
     prompt: "Claim implementation tasks, execute, mark complete",
     run_in_background: true
   })

   Task({
     team_name: "work-{timestamp}",
     name: "tester",
     subagent_type: "general-purpose",
     prompt: "Claim testing tasks, run tests, mark complete",
     run_in_background: true
   })
   ```

4. **Coordinate and Monitor**
   - Team lead monitors task completion
   - Spawn additional workers as phases unblock
   - Handle plan approval if required

5. **Cleanup**
   ```
   Teammate({ operation: "requestShutdown", target_agent_id: "implementer" })
   Teammate({ operation: "requestShutdown", target_agent_id: "tester" })
   Teammate({ operation: "cleanup" })
   ```

See the `orchestrating-swarms` skill for detailed swarm patterns and best practices.

---

## Key Principles

### Start Fast, Execute Faster

- Get clarification once at the start, then execute
- Don't wait for perfect understanding - ask questions and move
- The goal is to **finish the feature**, not create perfect process

### The Plan is Your Guide

- Work documents should reference similar code and patterns
- Load those references and follow them
- Don't reinvent - match what exists

### Test As You Go

- Run tests after each change, not at the end
- Fix failures immediately
- Continuous testing prevents big surprises

### Quality is Built In

- Follow existing patterns
- Write tests for new code
- Run linting before pushing
- Use reviewer agents for complex/risky changes only

### Ship Complete Features

- Mark all tasks completed before moving on
- Don't leave features 80% done
- A finished feature that ships beats a perfect feature that doesn't

## Quality Checklist

Before creating PR, verify:

- [ ] All clarifying questions asked and answered
- [ ] All TodoWrite tasks marked completed
- [ ] Tests pass (run project's test command)
- [ ] Linting passes (use linting-agent)
- [ ] Code follows existing patterns
- [ ] Figma designs match implementation (if applicable)
- [ ] Before/after screenshots captured and uploaded (for UI changes)
- [ ] Commit messages follow conventional format
- [ ] PR description includes summary, testing notes, and screenshots
- [ ] PR description includes Compound Engineered badge

## When to Use Reviewer Agents

**Don't use by default.** Use reviewer agents only when:

- Large refactor affecting many files (10+)
- Security-sensitive changes (authentication, permissions, data access)
- Performance-critical code paths
- Complex algorithms or business logic
- User explicitly requests thorough review

For most features: tests + linting + following patterns is sufficient.

## Common Pitfalls to Avoid

- **Analysis paralysis** - Don't overthink, read the plan and execute
- **Skipping clarifying questions** - Ask now, not after building wrong thing
- **Ignoring plan references** - The plan has links for a reason
- **Testing at the end** - Test continuously or suffer later
- **Forgetting TodoWrite** - Track progress or lose track of what's done
- **80% done syndrome** - Finish the feature, don't move on early
- **Over-reviewing simple changes** - Save reviewer agents for complex work

---

## /compound

_Source: `marketplaces/every-marketplace/plugins/compound-engineering/commands/workflows/compound.md`_

---
name: workflows:compound
description: Document a recently solved problem to compound your team's knowledge
argument-hint: "[optional: brief context about the fix]"
---

# /compound

Coordinate multiple subagents working in parallel to document a recently solved problem.

## Purpose

Captures problem solutions while context is fresh, creating structured documentation in `docs/solutions/` with YAML frontmatter for searchability and future reference. Uses parallel subagents for maximum efficiency.

**Why "compound"?** Each documented solution compounds your team's knowledge. The first time you solve a problem takes research. Document it, and the next occurrence takes minutes. Knowledge compounds.

## Usage

```bash
/workflows:compound                    # Document the most recent fix
/workflows:compound [brief context]    # Provide additional context hint
```

## Execution Strategy: Two-Phase Orchestration

<critical_requirement>
**Only ONE file gets written - the final documentation.**

Phase 1 subagents return TEXT DATA to the orchestrator. They must NOT use Write, Edit, or create any files. Only the orchestrator (Phase 2) writes the final documentation file.
</critical_requirement>

### Phase 1: Parallel Research

<parallel_tasks>

Launch these subagents IN PARALLEL. Each returns text data to the orchestrator.

#### 1. **Context Analyzer**
   - Extracts conversation history
   - Identifies problem type, component, symptoms
   - Validates against schema
   - Returns: YAML frontmatter skeleton

#### 2. **Solution Extractor**
   - Analyzes all investigation steps
   - Identifies root cause
   - Extracts working solution with code examples
   - Returns: Solution content block

#### 3. **Related Docs Finder**
   - Searches `docs/solutions/` for related documentation
   - Identifies cross-references and links
   - Finds related GitHub issues
   - Returns: Links and relationships

#### 4. **Prevention Strategist**
   - Develops prevention strategies
   - Creates best practices guidance
   - Generates test cases if applicable
   - Returns: Prevention/testing content

#### 5. **Category Classifier**
   - Determines optimal `docs/solutions/` category
   - Validates category against schema
   - Suggests filename based on slug
   - Returns: Final path and filename

</parallel_tasks>

### Phase 2: Assembly & Write

<sequential_tasks>

**WAIT for all Phase 1 subagents to complete before proceeding.**

The orchestrating agent (main conversation) performs these steps:

1. Collect all text results from Phase 1 subagents
2. Assemble complete markdown file from the collected pieces
3. Validate YAML frontmatter against schema
4. Create directory if needed: `mkdir -p docs/solutions/[category]/`
5. Write the SINGLE final file: `docs/solutions/[category]/[filename].md`

</sequential_tasks>

### Phase 3: Optional Enhancement

**WAIT for Phase 2 to complete before proceeding.**

<parallel_tasks>

Based on problem type, optionally invoke specialized agents to review the documentation:

- **performance_issue** → `performance-oracle`
- **security_issue** → `security-sentinel`
- **database_issue** → `data-integrity-guardian`
- **test_failure** → `cora-test-reviewer`
- Any code-heavy issue → `kieran-rails-reviewer` + `code-simplicity-reviewer`

</parallel_tasks>

## What It Captures

- **Problem symptom**: Exact error messages, observable behavior
- **Investigation steps tried**: What didn't work and why
- **Root cause analysis**: Technical explanation
- **Working solution**: Step-by-step fix with code examples
- **Prevention strategies**: How to avoid in future
- **Cross-references**: Links to related issues and docs

## Preconditions

<preconditions enforcement="advisory">
  <check condition="problem_solved">
    Problem has been solved (not in-progress)
  </check>
  <check condition="solution_verified">
    Solution has been verified working
  </check>
  <check condition="non_trivial">
    Non-trivial problem (not simple typo or obvious error)
  </check>
</preconditions>

## What It Creates

**Organized documentation:**

- File: `docs/solutions/[category]/[filename].md`

**Categories auto-detected from problem:**

- build-errors/
- test-failures/
- runtime-errors/
- performance-issues/
- database-issues/
- security-issues/
- ui-bugs/
- integration-issues/
- logic-errors/

## Common Mistakes to Avoid

| ❌ Wrong | ✅ Correct |
|----------|-----------|
| Subagents write files like `context-analysis.md`, `solution-draft.md` | Subagents return text data; orchestrator writes one final file |
| Research and assembly run in parallel | Research completes → then assembly runs |
| Multiple files created during workflow | Single file: `docs/solutions/[category]/[filename].md` |

## Success Output

```
✓ Documentation complete

Subagent Results:
  ✓ Context Analyzer: Identified performance_issue in brief_system
  ✓ Solution Extractor: 3 code fixes
  ✓ Related Docs Finder: 2 related issues
  ✓ Prevention Strategist: Prevention strategies, test suggestions
  ✓ Category Classifier: `performance-issues`

Specialized Agent Reviews (Auto-Triggered):
  ✓ performance-oracle: Validated query optimization approach
  ✓ kieran-rails-reviewer: Code examples meet Rails standards
  ✓ code-simplicity-reviewer: Solution is appropriately minimal
  ✓ every-style-editor: Documentation style verified

File created:
- docs/solutions/performance-issues/n-plus-one-brief-generation.md

This documentation will be searchable for future reference when similar
issues occur in the Email Processing or Brief System modules.

What's next?
1. Continue workflow (recommended)
2. Link related documentation
3. Update other references
4. View documentation
5. Other
```

## The Compounding Philosophy

This creates a compounding knowledge system:

1. First time you solve "N+1 query in brief generation" → Research (30 min)
2. Document the solution → docs/solutions/performance-issues/n-plus-one-briefs.md (5 min)
3. Next time similar issue occurs → Quick lookup (2 min)
4. Knowledge compounds → Team gets smarter

The feedback loop:

```
Build → Test → Find Issue → Research → Improve → Document → Validate → Deploy
    ↑                                                                      ↓
    └──────────────────────────────────────────────────────────────────────┘
```

**Each unit of engineering work should make subsequent units of work easier—not harder.**

## Auto-Invoke

<auto_invoke> <trigger_phrases> - "that worked" - "it's fixed" - "working now" - "problem solved" </trigger_phrases>

<manual_override> Use /workflows:compound [context] to document immediately without waiting for auto-detection. </manual_override> </auto_invoke>

## Routes To

`compound-docs` skill

## Applicable Specialized Agents

Based on problem type, these agents can enhance documentation:

### Code Quality & Review
- **kieran-rails-reviewer**: Reviews code examples for Rails best practices
- **code-simplicity-reviewer**: Ensures solution code is minimal and clear
- **pattern-recognition-specialist**: Identifies anti-patterns or repeating issues

### Specific Domain Experts
- **performance-oracle**: Analyzes performance_issue category solutions
- **security-sentinel**: Reviews security_issue solutions for vulnerabilities
- **cora-test-reviewer**: Creates test cases for prevention strategies
- **data-integrity-guardian**: Reviews database_issue migrations and queries

### Enhancement & Documentation
- **best-practices-researcher**: Enriches solution with industry best practices
- **every-style-editor**: Reviews documentation style and clarity
- **framework-docs-researcher**: Links to Rails/gem documentation references

### When to Invoke
- **Auto-triggered** (optional): Agents can run post-documentation for enhancement
- **Manual trigger**: User can invoke agents after /workflows:compound completes for deeper review

## Related Commands

- `/research [topic]` - Deep investigation (searches docs/solutions/ for patterns)
- `/workflows:plan` - Planning workflow (references documented solutions)

---

## /plan

_Source: `marketplaces/every-marketplace/plugins/compound-engineering/commands/workflows/plan.md`_

---
name: workflows:plan
description: Transform feature descriptions into well-structured project plans following conventions
argument-hint: "[feature description, bug report, or improvement idea]"
---

# Create a plan for a new feature or bug fix

## Introduction

**Note: The current year is 2026.** Use this when dating plans and searching for recent documentation.

Transform feature descriptions, bug reports, or improvement ideas into well-structured markdown files issues that follow project conventions and best practices. This command provides flexible detail levels to match your needs.

## Feature Description

<feature_description> #$ARGUMENTS </feature_description>

**If the feature description above is empty, ask the user:** "What would you like to plan? Please describe the feature, bug fix, or improvement you have in mind."

Do not proceed until you have a clear feature description from the user.

### 0. Idea Refinement

**Check for brainstorm output first:**

Before asking questions, look for recent brainstorm documents in `docs/brainstorms/` that match this feature:

```bash
ls -la docs/brainstorms/*.md 2>/dev/null | head -10
```

**Relevance criteria:** A brainstorm is relevant if:
- The topic (from filename or YAML frontmatter) semantically matches the feature description
- Created within the last 14 days
- If multiple candidates match, use the most recent one

**If a relevant brainstorm exists:**
1. Read the brainstorm document
2. Announce: "Found brainstorm from [date]: [topic]. Using as context for planning."
3. Extract key decisions, chosen approach, and open questions
4. **Skip the idea refinement questions below** - the brainstorm already answered WHAT to build
5. Use brainstorm decisions as input to the research phase

**If multiple brainstorms could match:**
Use **AskUserQuestion tool** to ask which brainstorm to use, or whether to proceed without one.

**If no brainstorm found (or not relevant), run idea refinement:**

Refine the idea through collaborative dialogue using the **AskUserQuestion tool**:

- Ask questions one at a time to understand the idea fully
- Prefer multiple choice questions when natural options exist
- Focus on understanding: purpose, constraints and success criteria
- Continue until the idea is clear OR user says "proceed"

**Gather signals for research decision.** During refinement, note:

- **User's familiarity**: Do they know the codebase patterns? Are they pointing to examples?
- **User's intent**: Speed vs thoroughness? Exploration vs execution?
- **Topic risk**: Security, payments, external APIs warrant more caution
- **Uncertainty level**: Is the approach clear or open-ended?

**Skip option:** If the feature description is already detailed, offer:
"Your description is clear. Should I proceed with research, or would you like to refine it further?"

## Main Tasks

### 1. Local Research (Always Runs - Parallel)

<thinking>
First, I need to understand the project's conventions, existing patterns, and any documented learnings. This is fast and local - it informs whether external research is needed.
</thinking>

Run these agents **in parallel** to gather local context:

- Task repo-research-analyst(feature_description)
- Task learnings-researcher(feature_description)

**What to look for:**
- **Repo research:** existing patterns, CLAUDE.md guidance, technology familiarity, pattern consistency
- **Learnings:** documented solutions in `docs/solutions/` that might apply (gotchas, patterns, lessons learned)

These findings inform the next step.

### 1.5. Research Decision

Based on signals from Step 0 and findings from Step 1, decide on external research.

**High-risk topics → always research.** Security, payments, external APIs, data privacy. The cost of missing something is too high. This takes precedence over speed signals.

**Strong local context → skip external research.** Codebase has good patterns, CLAUDE.md has guidance, user knows what they want. External research adds little value.

**Uncertainty or unfamiliar territory → research.** User is exploring, codebase has no examples, new technology. External perspective is valuable.

**Announce the decision and proceed.** Brief explanation, then continue. User can redirect if needed.

Examples:
- "Your codebase has solid patterns for this. Proceeding without external research."
- "This involves payment processing, so I'll research current best practices first."

### 1.5b. External Research (Conditional)

**Only run if Step 1.5 indicates external research is valuable.**

Run these agents in parallel:

- Task best-practices-researcher(feature_description)
- Task framework-docs-researcher(feature_description)

### 1.6. Consolidate Research

After all research steps complete, consolidate findings:

- Document relevant file paths from repo research (e.g., `app/services/example_service.rb:42`)
- **Include relevant institutional learnings** from `docs/solutions/` (key insights, gotchas to avoid)
- Note external documentation URLs and best practices (if external research was done)
- List related issues or PRs discovered
- Capture CLAUDE.md conventions

**Optional validation:** Briefly summarize findings and ask if anything looks off or missing before proceeding to planning.

### 2. Issue Planning & Structure

<thinking>
Think like a product manager - what would make this issue clear and actionable? Consider multiple perspectives
</thinking>

**Title & Categorization:**

- [ ] Draft clear, searchable issue title using conventional format (e.g., `feat: Add user authentication`, `fix: Cart total calculation`)
- [ ] Determine issue type: enhancement, bug, refactor
- [ ] Convert title to filename: add today's date prefix, strip prefix colon, kebab-case, add `-plan` suffix
  - Example: `feat: Add User Authentication` → `2026-01-21-feat-add-user-authentication-plan.md`
  - Keep it descriptive (3-5 words after prefix) so plans are findable by context

**Stakeholder Analysis:**

- [ ] Identify who will be affected by this issue (end users, developers, operations)
- [ ] Consider implementation complexity and required expertise

**Content Planning:**

- [ ] Choose appropriate detail level based on issue complexity and audience
- [ ] List all necessary sections for the chosen template
- [ ] Gather supporting materials (error logs, screenshots, design mockups)
- [ ] Prepare code examples or reproduction steps if applicable, name the mock filenames in the lists

### 3. SpecFlow Analysis

After planning the issue structure, run SpecFlow Analyzer to validate and refine the feature specification:

- Task spec-flow-analyzer(feature_description, research_findings)

**SpecFlow Analyzer Output:**

- [ ] Review SpecFlow analysis results
- [ ] Incorporate any identified gaps or edge cases into the issue
- [ ] Update acceptance criteria based on SpecFlow findings

### 4. Choose Implementation Detail Level

Select how comprehensive you want the issue to be, simpler is mostly better.

#### 📄 MINIMAL (Quick Issue)

**Best for:** Simple bugs, small improvements, clear features

**Includes:**

- Problem statement or feature description
- Basic acceptance criteria
- Essential context only

**Structure:**

````markdown
---
title: [Issue Title]
type: [feat|fix|refactor]
date: YYYY-MM-DD
---

# [Issue Title]

[Brief problem/feature description]

## Acceptance Criteria

- [ ] Core requirement 1
- [ ] Core requirement 2

## Context

[Any critical information]

## MVP

### test.rb

```ruby
class Test
  def initialize
    @name = "test"
  end
end
```

## References

- Related issue: #[issue_number]
- Documentation: [relevant_docs_url]
````

#### 📋 MORE (Standard Issue)

**Best for:** Most features, complex bugs, team collaboration

**Includes everything from MINIMAL plus:**

- Detailed background and motivation
- Technical considerations
- Success metrics
- Dependencies and risks
- Basic implementation suggestions

**Structure:**

```markdown
---
title: [Issue Title]
type: [feat|fix|refactor]
date: YYYY-MM-DD
---

# [Issue Title]

## Overview

[Comprehensive description]

## Problem Statement / Motivation

[Why this matters]

## Proposed Solution

[High-level approach]

## Technical Considerations

- Architecture impacts
- Performance implications
- Security considerations

## Acceptance Criteria

- [ ] Detailed requirement 1
- [ ] Detailed requirement 2
- [ ] Testing requirements

## Success Metrics

[How we measure success]

## Dependencies & Risks

[What could block or complicate this]

## References & Research

- Similar implementations: [file_path:line_number]
- Best practices: [documentation_url]
- Related PRs: #[pr_number]
```

#### 📚 A LOT (Comprehensive Issue)

**Best for:** Major features, architectural changes, complex integrations

**Includes everything from MORE plus:**

- Detailed implementation plan with phases
- Alternative approaches considered
- Extensive technical specifications
- Resource requirements and timeline
- Future considerations and extensibility
- Risk mitigation strategies
- Documentation requirements

**Structure:**

```markdown
---
title: [Issue Title]
type: [feat|fix|refactor]
date: YYYY-MM-DD
---

# [Issue Title]

## Overview

[Executive summary]

## Problem Statement

[Detailed problem analysis]

## Proposed Solution

[Comprehensive solution design]

## Technical Approach

### Architecture

[Detailed technical design]

### Implementation Phases

#### Phase 1: [Foundation]

- Tasks and deliverables
- Success criteria
- Estimated effort

#### Phase 2: [Core Implementation]

- Tasks and deliverables
- Success criteria
- Estimated effort

#### Phase 3: [Polish & Optimization]

- Tasks and deliverables
- Success criteria
- Estimated effort

## Alternative Approaches Considered

[Other solutions evaluated and why rejected]

## Acceptance Criteria

### Functional Requirements

- [ ] Detailed functional criteria

### Non-Functional Requirements

- [ ] Performance targets
- [ ] Security requirements
- [ ] Accessibility standards

### Quality Gates

- [ ] Test coverage requirements
- [ ] Documentation completeness
- [ ] Code review approval

## Success Metrics

[Detailed KPIs and measurement methods]

## Dependencies & Prerequisites

[Detailed dependency analysis]

## Risk Analysis & Mitigation

[Comprehensive risk assessment]

## Resource Requirements

[Team, time, infrastructure needs]

## Future Considerations

[Extensibility and long-term vision]

## Documentation Plan

[What docs need updating]

## References & Research

### Internal References

- Architecture decisions: [file_path:line_number]
- Similar features: [file_path:line_number]
- Configuration: [file_path:line_number]

### External References

- Framework documentation: [url]
- Best practices guide: [url]
- Industry standards: [url]

### Related Work

- Previous PRs: #[pr_numbers]
- Related issues: #[issue_numbers]
- Design documents: [links]
```

### 5. Issue Creation & Formatting

<thinking>
Apply best practices for clarity and actionability, making the issue easy to scan and understand
</thinking>

**Content Formatting:**

- [ ] Use clear, descriptive headings with proper hierarchy (##, ###)
- [ ] Include code examples in triple backticks with language syntax highlighting
- [ ] Add screenshots/mockups if UI-related (drag & drop or use image hosting)
- [ ] Use task lists (- [ ]) for trackable items that can be checked off
- [ ] Add collapsible sections for lengthy logs or optional details using `<details>` tags
- [ ] Apply appropriate emoji for visual scanning (🐛 bug, ✨ feature, 📚 docs, ♻️ refactor)

**Cross-Referencing:**

- [ ] Link to related issues/PRs using #number format
- [ ] Reference specific commits with SHA hashes when relevant
- [ ] Link to code using GitHub's permalink feature (press 'y' for permanent link)
- [ ] Mention relevant team members with @username if needed
- [ ] Add links to external resources with descriptive text

**Code & Examples:**

````markdown
# Good example with syntax highlighting and line references


```ruby
# app/services/user_service.rb:42
def process_user(user)

# Implementation here

end
```

# Collapsible error logs

<details>
<summary>Full error stacktrace</summary>

`Error details here...`

</details>
````

**AI-Era Considerations:**

- [ ] Account for accelerated development with AI pair programming
- [ ] Include prompts or instructions that worked well during research
- [ ] Note which AI tools were used for initial exploration (Claude, Copilot, etc.)
- [ ] Emphasize comprehensive testing given rapid implementation
- [ ] Document any AI-generated code that needs human review

### 6. Final Review & Submission

**Pre-submission Checklist:**

- [ ] Title is searchable and descriptive
- [ ] Labels accurately categorize the issue
- [ ] All template sections are complete
- [ ] Links and references are working
- [ ] Acceptance criteria are measurable
- [ ] Add names of files in pseudo code examples and todo lists
- [ ] Add an ERD mermaid diagram if applicable for new model changes

## Output Format

**Filename:** Use the date and kebab-case filename from Step 2 Title & Categorization.

```
docs/plans/YYYY-MM-DD-<type>-<descriptive-name>-plan.md
```

Examples:
- ✅ `docs/plans/2026-01-15-feat-user-authentication-flow-plan.md`
- ✅ `docs/plans/2026-02-03-fix-checkout-race-condition-plan.md`
- ✅ `docs/plans/2026-03-10-refactor-api-client-extraction-plan.md`
- ❌ `docs/plans/2026-01-15-feat-thing-plan.md` (not descriptive - what "thing"?)
- ❌ `docs/plans/2026-01-15-feat-new-feature-plan.md` (too vague - what feature?)
- ❌ `docs/plans/2026-01-15-feat: user auth-plan.md` (invalid characters - colon and space)
- ❌ `docs/plans/feat-user-auth-plan.md` (missing date prefix)

## Post-Generation Options

After writing the plan file, use the **AskUserQuestion tool** to present these options:

**Question:** "Plan ready at `docs/plans/YYYY-MM-DD-<type>-<name>-plan.md`. What would you like to do next?"

**Options:**
1. **Open plan in editor** - Open the plan file for review
2. **Run `/deepen-plan`** - Enhance each section with parallel research agents (best practices, performance, UI)
3. **Run `/technical_review`** - Technical feedback from code-focused reviewers (DHH, Kieran, Simplicity)
4. **Review and refine** - Improve the document through structured self-review
5. **Start `/workflows:work`** - Begin implementing this plan locally
6. **Start `/workflows:work` on remote** - Begin implementing in Claude Code on the web (use `&` to run in background)
7. **Create Issue** - Create issue in project tracker (GitHub/Linear)

Based on selection:
- **Open plan in editor** → Run `open docs/plans/<plan_filename>.md` to open the file in the user's default editor
- **`/deepen-plan`** → Call the /deepen-plan command with the plan file path to enhance with research
- **`/technical_review`** → Call the /technical_review command with the plan file path
- **Review and refine** → Load `document-review` skill.
- **`/workflows:work`** → Call the /workflows:work command with the plan file path
- **`/workflows:work` on remote** → Run `/workflows:work docs/plans/<plan_filename>.md &` to start work in background for Claude Code web
- **Create Issue** → See "Issue Creation" section below
- **Other** (automatically provided) → Accept free text for rework or specific changes

**Note:** If running `/workflows:plan` with ultrathink enabled, automatically run `/deepen-plan` after plan creation for maximum depth and grounding.

Loop back to options after Simplify or Other changes until user selects `/workflows:work` or `/technical_review`.

## Issue Creation

When user selects "Create Issue", detect their project tracker from CLAUDE.md:

1. **Check for tracker preference** in user's CLAUDE.md (global or project):
   - Look for `project_tracker: github` or `project_tracker: linear`
   - Or look for mentions of "GitHub Issues" or "Linear" in their workflow section

2. **If GitHub:**

   Use the title and type from Step 2 (already in context - no need to re-read the file):

   ```bash
   gh issue create --title "<type>: <title>" --body-file <plan_path>
   ```

3. **If Linear:**

   ```bash
   linear issue create --title "<title>" --description "$(cat <plan_path>)"
   ```

4. **If no tracker configured:**
   Ask user: "Which project tracker do you use? (GitHub/Linear/Other)"
   - Suggest adding `project_tracker: github` or `project_tracker: linear` to their CLAUDE.md

5. **After creation:**
   - Display the issue URL
   - Ask if they want to proceed to `/workflows:work` or `/technical_review`

NEVER CODE! Just research and write the plan.

---

## /review

_Source: `marketplaces/every-marketplace/plugins/compound-engineering/commands/workflows/review.md`_

---
name: workflows:review
description: Perform exhaustive code reviews using multi-agent analysis, ultra-thinking, and worktrees
argument-hint: "[PR number, GitHub URL, branch name, or latest]"
---

# Review Command

<command_purpose> Perform exhaustive code reviews using multi-agent analysis, ultra-thinking, and Git worktrees for deep local inspection. </command_purpose>

## Introduction

<role>Senior Code Review Architect with expertise in security, performance, architecture, and quality assurance</role>

## Prerequisites

<requirements>
- Git repository with GitHub CLI (`gh`) installed and authenticated
- Clean main/master branch
- Proper permissions to create worktrees and access the repository
- For document reviews: Path to a markdown file or document
</requirements>

## Main Tasks

### 1. Determine Review Target & Setup (ALWAYS FIRST)

<review_target> #$ARGUMENTS </review_target>

<thinking>
First, I need to determine the review target type and set up the code for analysis.
</thinking>

#### Immediate Actions:

<task_list>

- [ ] Determine review type: PR number (numeric), GitHub URL, file path (.md), or empty (current branch)
- [ ] Check current git branch
- [ ] If ALREADY on the target branch (PR branch, requested branch name, or the branch already checked out for review) → proceed with analysis on current branch
- [ ] If DIFFERENT branch than the review target → offer to use worktree: "Use git-worktree skill for isolated Call `skill: git-worktree` with branch name
- [ ] Fetch PR metadata using `gh pr view --json` for title, body, files, linked issues
- [ ] Set up language-specific analysis tools
- [ ] Prepare security scanning environment
- [ ] Make sure we are on the branch we are reviewing. Use gh pr checkout to switch to the branch or manually checkout the branch.

Ensure that the code is ready for analysis (either in worktree or on current branch). ONLY then proceed to the next step.

</task_list>

#### Protected Artifacts

<protected_artifacts>
The following paths are compound-engineering pipeline artifacts and must never be flagged for deletion, removal, or gitignore by any review agent:

- `docs/plans/*.md` — Plan files created by `/workflows:plan`. These are living documents that track implementation progress (checkboxes are checked off by `/workflows:work`).
- `docs/solutions/*.md` — Solution documents created during the pipeline.

If a review agent flags any file in these directories for cleanup or removal, discard that finding during synthesis. Do not create a todo for it.
</protected_artifacts>

#### Parallel Agents to review the PR:

<parallel_tasks>

Run ALL or most of these agents at the same time:

1. Task kieran-rails-reviewer(PR content)
2. Task dhh-rails-reviewer(PR title)
3. If turbo is used: Task rails-turbo-expert(PR content)
4. Task git-history-analyzer(PR content)
5. Task dependency-detective(PR content)
6. Task pattern-recognition-specialist(PR content)
7. Task architecture-strategist(PR content)
8. Task code-philosopher(PR content)
9. Task security-sentinel(PR content)
10. Task performance-oracle(PR content)
11. Task devops-harmony-analyst(PR content)
12. Task data-integrity-guardian(PR content)
13. Task agent-native-reviewer(PR content) - Verify new features are agent-accessible

</parallel_tasks>

#### Conditional Agents (Run if applicable):

<conditional_agents>

These agents are run ONLY when the PR matches specific criteria. Check the PR files list to determine if they apply:

**If PR contains database migrations (db/migrate/*.rb files) or data backfills:**

14. Task data-migration-expert(PR content) - Validates ID mappings match production, checks for swapped values, verifies rollback safety
15. Task deployment-verification-agent(PR content) - Creates Go/No-Go deployment checklist with SQL verification queries

**When to run migration agents:**
- PR includes files matching `db/migrate/*.rb`
- PR modifies columns that store IDs, enums, or mappings
- PR includes data backfill scripts or rake tasks
- PR changes how data is read/written (e.g., changing from FK to string column)
- PR title/body mentions: migration, backfill, data transformation, ID mapping

**What these agents check:**
- `data-migration-expert`: Verifies hard-coded mappings match production reality (prevents swapped IDs), checks for orphaned associations, validates dual-write patterns
- `deployment-verification-agent`: Produces executable pre/post-deploy checklists with SQL queries, rollback procedures, and monitoring plans

</conditional_agents>

### 4. Ultra-Thinking Deep Dive Phases

<ultrathink_instruction> For each phase below, spend maximum cognitive effort. Think step by step. Consider all angles. Question assumptions. And bring all reviews in a synthesis to the user.</ultrathink_instruction>

<deliverable>
Complete system context map with component interactions
</deliverable>

#### Phase 3: Stakeholder Perspective Analysis

<thinking_prompt> ULTRA-THINK: Put yourself in each stakeholder's shoes. What matters to them? What are their pain points? </thinking_prompt>

<stakeholder_perspectives>

1. **Developer Perspective** <questions>

   - How easy is this to understand and modify?
   - Are the APIs intuitive?
   - Is debugging straightforward?
   - Can I test this easily? </questions>

2. **Operations Perspective** <questions>

   - How do I deploy this safely?
   - What metrics and logs are available?
   - How do I troubleshoot issues?
   - What are the resource requirements? </questions>

3. **End User Perspective** <questions>

   - Is the feature intuitive?
   - Are error messages helpful?
   - Is performance acceptable?
   - Does it solve my problem? </questions>

4. **Security Team Perspective** <questions>

   - What's the attack surface?
   - Are there compliance requirements?
   - How is data protected?
   - What are the audit capabilities? </questions>

5. **Business Perspective** <questions>
   - What's the ROI?
   - Are there legal/compliance risks?
   - How does this affect time-to-market?
   - What's the total cost of ownership? </questions> </stakeholder_perspectives>

#### Phase 4: Scenario Exploration

<thinking_prompt> ULTRA-THINK: Explore edge cases and failure scenarios. What could go wrong? How does the system behave under stress? </thinking_prompt>

<scenario_checklist>

- [ ] **Happy Path**: Normal operation with valid inputs
- [ ] **Invalid Inputs**: Null, empty, malformed data
- [ ] **Boundary Conditions**: Min/max values, empty collections
- [ ] **Concurrent Access**: Race conditions, deadlocks
- [ ] **Scale Testing**: 10x, 100x, 1000x normal load
- [ ] **Network Issues**: Timeouts, partial failures
- [ ] **Resource Exhaustion**: Memory, disk, connections
- [ ] **Security Attacks**: Injection, overflow, DoS
- [ ] **Data Corruption**: Partial writes, inconsistency
- [ ] **Cascading Failures**: Downstream service issues </scenario_checklist>

### 6. Multi-Angle Review Perspectives

#### Technical Excellence Angle

- Code craftsmanship evaluation
- Engineering best practices
- Technical documentation quality
- Tooling and automation assessment

#### Business Value Angle

- Feature completeness validation
- Performance impact on users
- Cost-benefit analysis
- Time-to-market considerations

#### Risk Management Angle

- Security risk assessment
- Operational risk evaluation
- Compliance risk verification
- Technical debt accumulation

#### Team Dynamics Angle

- Code review etiquette
- Knowledge sharing effectiveness
- Collaboration patterns
- Mentoring opportunities

### 4. Simplification and Minimalism Review

Run the Task code-simplicity-reviewer() to see if we can simplify the code.

### 5. Findings Synthesis and Todo Creation Using file-todos Skill

<critical_requirement> ALL findings MUST be stored in the todos/ directory using the file-todos skill. Create todo files immediately after synthesis - do NOT present findings for user approval first. Use the skill for structured todo management. </critical_requirement>

#### Step 1: Synthesize All Findings

<thinking>
Consolidate all agent reports into a categorized list of findings.
Remove duplicates, prioritize by severity and impact.
</thinking>

<synthesis_tasks>

- [ ] Collect findings from all parallel agents
- [ ] Discard any findings that recommend deleting or gitignoring files in `docs/plans/` or `docs/solutions/` (see Protected Artifacts above)
- [ ] Categorize by type: security, performance, architecture, quality, etc.
- [ ] Assign severity levels: 🔴 CRITICAL (P1), 🟡 IMPORTANT (P2), 🔵 NICE-TO-HAVE (P3)
- [ ] Remove duplicate or overlapping findings
- [ ] Estimate effort for each finding (Small/Medium/Large)

</synthesis_tasks>

#### Step 2: Create Todo Files Using file-todos Skill

<critical_instruction> Use the file-todos skill to create todo files for ALL findings immediately. Do NOT present findings one-by-one asking for user approval. Create all todo files in parallel using the skill, then summarize results to user. </critical_instruction>

**Implementation Options:**

**Option A: Direct File Creation (Fast)**

- Create todo files directly using Write tool
- All findings in parallel for speed
- Use standard template from `.claude/skills/file-todos/assets/todo-template.md`
- Follow naming convention: `{issue_id}-pending-{priority}-{description}.md`

**Option B: Sub-Agents in Parallel (Recommended for Scale)** For large PRs with 15+ findings, use sub-agents to create finding files in parallel:

```bash
# Launch multiple finding-creator agents in parallel
Task() - Create todos for first finding
Task() - Create todos for second finding
Task() - Create todos for third finding
etc. for each finding.
```

Sub-agents can:

- Process multiple findings simultaneously
- Write detailed todo files with all sections filled
- Organize findings by severity
- Create comprehensive Proposed Solutions
- Add acceptance criteria and work logs
- Complete much faster than sequential processing

**Execution Strategy:**

1. Synthesize all findings into categories (P1/P2/P3)
2. Group findings by severity
3. Launch 3 parallel sub-agents (one per severity level)
4. Each sub-agent creates its batch of todos using the file-todos skill
5. Consolidate results and present summary

**Process (Using file-todos Skill):**

1. For each finding:

   - Determine severity (P1/P2/P3)
   - Write detailed Problem Statement and Findings
   - Create 2-3 Proposed Solutions with pros/cons/effort/risk
   - Estimate effort (Small/Medium/Large)
   - Add acceptance criteria and work log

2. Use file-todos skill for structured todo management:

   ```bash
   skill: file-todos
   ```

   The skill provides:

   - Template location: `.claude/skills/file-todos/assets/todo-template.md`
   - Naming convention: `{issue_id}-{status}-{priority}-{description}.md`
   - YAML frontmatter structure: status, priority, issue_id, tags, dependencies
   - All required sections: Problem Statement, Findings, Solutions, etc.

3. Create todo files in parallel:

   ```bash
   {next_id}-pending-{priority}-{description}.md
   ```

4. Examples:

   ```
   001-pending-p1-path-traversal-vulnerability.md
   002-pending-p1-api-response-validation.md
   003-pending-p2-concurrency-limit.md
   004-pending-p3-unused-parameter.md
   ```

5. Follow template structure from file-todos skill: `.claude/skills/file-todos/assets/todo-template.md`

**Todo File Structure (from template):**

Each todo must include:

- **YAML frontmatter**: status, priority, issue_id, tags, dependencies
- **Problem Statement**: What's broken/missing, why it matters
- **Findings**: Discoveries from agents with evidence/location
- **Proposed Solutions**: 2-3 options, each with pros/cons/effort/risk
- **Recommended Action**: (Filled during triage, leave blank initially)
- **Technical Details**: Affected files, components, database changes
- **Acceptance Criteria**: Testable checklist items
- **Work Log**: Dated record with actions and learnings
- **Resources**: Links to PR, issues, documentation, similar patterns

**File naming convention:**

```
{issue_id}-{status}-{priority}-{description}.md

Examples:
- 001-pending-p1-security-vulnerability.md
- 002-pending-p2-performance-optimization.md
- 003-pending-p3-code-cleanup.md
```

**Status values:**

- `pending` - New findings, needs triage/decision
- `ready` - Approved by manager, ready to work
- `complete` - Work finished

**Priority values:**

- `p1` - Critical (blocks merge, security/data issues)
- `p2` - Important (should fix, architectural/performance)
- `p3` - Nice-to-have (enhancements, cleanup)

**Tagging:** Always add `code-review` tag, plus: `security`, `performance`, `architecture`, `rails`, `quality`, etc.

#### Step 3: Summary Report

After creating all todo files, present comprehensive summary:

````markdown
## ✅ Code Review Complete

**Review Target:** PR #XXXX - [PR Title] **Branch:** [branch-name]

### Findings Summary:

- **Total Findings:** [X]
- **🔴 CRITICAL (P1):** [count] - BLOCKS MERGE
- **🟡 IMPORTANT (P2):** [count] - Should Fix
- **🔵 NICE-TO-HAVE (P3):** [count] - Enhancements

### Created Todo Files:

**P1 - Critical (BLOCKS MERGE):**

- `001-pending-p1-{finding}.md` - {description}
- `002-pending-p1-{finding}.md` - {description}

**P2 - Important:**

- `003-pending-p2-{finding}.md` - {description}
- `004-pending-p2-{finding}.md` - {description}

**P3 - Nice-to-Have:**

- `005-pending-p3-{finding}.md` - {description}

### Review Agents Used:

- kieran-rails-reviewer
- security-sentinel
- performance-oracle
- architecture-strategist
- agent-native-reviewer
- [other agents]

### Next Steps:

1. **Address P1 Findings**: CRITICAL - must be fixed before merge

   - Review each P1 todo in detail
   - Implement fixes or request exemption
   - Verify fixes before merging PR

2. **Triage All Todos**:
   ```bash
   ls todos/*-pending-*.md  # View all pending todos
   /triage                  # Use slash command for interactive triage
   ```
````

3. **Work on Approved Todos**:

   ```bash
   /resolve_todo_parallel  # Fix all approved items efficiently
   ```

4. **Track Progress**:
   - Rename file when status changes: pending → ready → complete
   - Update Work Log as you work
   - Commit todos: `git add todos/ && git commit -m "refactor: add code review findings"`

### Severity Breakdown:

**🔴 P1 (Critical - Blocks Merge):**

- Security vulnerabilities
- Data corruption risks
- Breaking changes
- Critical architectural issues

**🟡 P2 (Important - Should Fix):**

- Performance issues
- Significant architectural concerns
- Major code quality problems
- Reliability issues

**🔵 P3 (Nice-to-Have):**

- Minor improvements
- Code cleanup
- Optimization opportunities
- Documentation updates

```

### 7. End-to-End Testing (Optional)

<detect_project_type>

**First, detect the project type from PR files:**

| Indicator | Project Type |
|-----------|--------------|
| `*.xcodeproj`, `*.xcworkspace`, `Package.swift` (iOS) | iOS/macOS |
| `Gemfile`, `package.json`, `app/views/*`, `*.html.*` | Web |
| Both iOS files AND web files | Hybrid (test both) |

</detect_project_type>

<offer_testing>

After presenting the Summary Report, offer appropriate testing based on project type:

**For Web Projects:**
```markdown
**"Want to run browser tests on the affected pages?"**
1. Yes - run `/test-browser`
2. No - skip
```

**For iOS Projects:**
```markdown
**"Want to run Xcode simulator tests on the app?"**
1. Yes - run `/xcode-test`
2. No - skip
```

**For Hybrid Projects (e.g., Rails + Hotwire Native):**
```markdown
**"Want to run end-to-end tests?"**
1. Web only - run `/test-browser`
2. iOS only - run `/xcode-test`
3. Both - run both commands
4. No - skip
```

</offer_testing>

#### If User Accepts Web Testing:

Spawn a subagent to run browser tests (preserves main context):

```
Task general-purpose("Run /test-browser for PR #[number]. Test all affected pages, check for console errors, handle failures by creating todos and fixing.")
```

The subagent will:
1. Identify pages affected by the PR
2. Navigate to each page and capture snapshots (using Playwright MCP or agent-browser CLI)
3. Check for console errors
4. Test critical interactions
5. Pause for human verification on OAuth/email/payment flows
6. Create P1 todos for any failures
7. Fix and retry until all tests pass

**Standalone:** `/test-browser [PR number]`

#### If User Accepts iOS Testing:

Spawn a subagent to run Xcode tests (preserves main context):

```
Task general-purpose("Run /xcode-test for scheme [name]. Build for simulator, install, launch, take screenshots, check for crashes.")
```

The subagent will:
1. Verify XcodeBuildMCP is installed
2. Discover project and schemes
3. Build for iOS Simulator
4. Install and launch app
5. Take screenshots of key screens
6. Capture console logs for errors
7. Pause for human verification (Sign in with Apple, push, IAP)
8. Create P1 todos for any failures
9. Fix and retry until all tests pass

**Standalone:** `/xcode-test [scheme]`

### Important: P1 Findings Block Merge

Any **🔴 P1 (CRITICAL)** findings must be addressed before merging the PR. Present these prominently and ensure they're resolved before accepting the PR.
```

---

## /brainstorm

_Source: `marketplaces/every-marketplace/plugins/compound-engineering/commands/workflows/brainstorm.md`_

---
name: workflows:brainstorm
description: Explore requirements and approaches through collaborative dialogue before planning implementation
argument-hint: "[feature idea or problem to explore]"
---

# Brainstorm a Feature or Improvement

**Note: The current year is 2026.** Use this when dating brainstorm documents.

Brainstorming helps answer **WHAT** to build through collaborative dialogue. It precedes `/workflows:plan`, which answers **HOW** to build it.

**Process knowledge:** Load the `brainstorming` skill for detailed question techniques, approach exploration patterns, and YAGNI principles.

## Feature Description

<feature_description> #$ARGUMENTS </feature_description>

**If the feature description above is empty, ask the user:** "What would you like to explore? Please describe the feature, problem, or improvement you're thinking about."

Do not proceed until you have a feature description from the user.

## Execution Flow

### Phase 0: Assess Requirements Clarity

Evaluate whether brainstorming is needed based on the feature description.

**Clear requirements indicators:**
- Specific acceptance criteria provided
- Referenced existing patterns to follow
- Described exact expected behavior
- Constrained, well-defined scope

**If requirements are already clear:**
Use **AskUserQuestion tool** to suggest: "Your requirements seem detailed enough to proceed directly to planning. Should I run `/workflows:plan` instead, or would you like to explore the idea further?"

### Phase 1: Understand the Idea

#### 1.1 Repository Research (Lightweight)

Run a quick repo scan to understand existing patterns:

- Task repo-research-analyst("Understand existing patterns related to: <feature_description>")

Focus on: similar features, established patterns, CLAUDE.md guidance.

#### 1.2 Collaborative Dialogue

Use the **AskUserQuestion tool** to ask questions **one at a time**.

**Guidelines (see `brainstorming` skill for detailed techniques):**
- Prefer multiple choice when natural options exist
- Start broad (purpose, users) then narrow (constraints, edge cases)
- Validate assumptions explicitly
- Ask about success criteria

**Exit condition:** Continue until the idea is clear OR user says "proceed"

### Phase 2: Explore Approaches

Propose **2-3 concrete approaches** based on research and conversation.

For each approach, provide:
- Brief description (2-3 sentences)
- Pros and cons
- When it's best suited

Lead with your recommendation and explain why. Apply YAGNI—prefer simpler solutions.

Use **AskUserQuestion tool** to ask which approach the user prefers.

### Phase 3: Capture the Design

Write a brainstorm document to `docs/brainstorms/YYYY-MM-DD-<topic>-brainstorm.md`.

**Document structure:** See the `brainstorming` skill for the template format. Key sections: What We're Building, Why This Approach, Key Decisions, Open Questions.

Ensure `docs/brainstorms/` directory exists before writing.

### Phase 4: Handoff

Use **AskUserQuestion tool** to present next steps:

**Question:** "Brainstorm captured. What would you like to do next?"

**Options:**
1. **Review and refine** - Improve the document through structured self-review
2. **Proceed to planning** - Run `/workflows:plan` (will auto-detect this brainstorm)
3. **Done for now** - Return later

**If user selects "Review and refine":**

Load the `document-review` skill and apply it to the brainstorm document.

When document-review returns "Review complete", present next steps:

1. **Move to planning** - Continue to `/workflows:plan` with this document
2. **Done for now** - Brainstorming complete. To start planning later: `/workflows:plan [document-path]`

## Output Summary

When complete, display:

```
Brainstorm complete!

Document: docs/brainstorms/YYYY-MM-DD-<topic>-brainstorm.md

Key decisions:
- [Decision 1]
- [Decision 2]

Next: Run `/workflows:plan` when ready to implement.
```

## Important Guidelines

- **Stay focused on WHAT, not HOW** - Implementation details belong in the plan
- **Ask one question at a time** - Don't overwhelm
- **Apply YAGNI** - Prefer simpler approaches
- **Keep outputs concise** - 200-300 words per section max

NEVER CODE! Just explore and document decisions.

---

## /create-agent-skill

_Source: `marketplaces/every-marketplace/plugins/compound-engineering/commands/create-agent-skill.md`_

---
name: create-agent-skill
description: Create or edit Claude Code skills with expert guidance on structure and best practices
allowed-tools: Skill(create-agent-skills)
argument-hint: [skill description or requirements]
disable-model-invocation: true
---

Invoke the create-agent-skills skill for: $ARGUMENTS

---

## /slfg

_Source: `marketplaces/every-marketplace/plugins/compound-engineering/commands/slfg.md`_

---
name: slfg
description: Full autonomous engineering workflow using swarm mode for parallel execution
argument-hint: "[feature description]"
disable-model-invocation: true
---

Swarm-enabled LFG. Run these steps in order, parallelizing where indicated.

## Sequential Phase

1. `/ralph-wiggum:ralph-loop "finish all slash commands" --completion-promise "DONE"`
2. `/workflows:plan $ARGUMENTS`
3. `/compound-engineering:deepen-plan`
4. `/workflows:work` — **Use swarm mode**: Make a Task list and launch an army of agent swarm subagents to build the plan

## Parallel Phase

After work completes, launch steps 5 and 6 as **parallel swarm agents** (both only need code to be written):

5. `/workflows:review` — spawn as background Task agent
6. `/compound-engineering:test-browser` — spawn as background Task agent

Wait for both to complete before continuing.

## Finalize Phase

7. `/compound-engineering:resolve_todo_parallel` — resolve any findings from the review
8. `/compound-engineering:feature-video` — record the final walkthrough and add to PR
9. Output `<promise>DONE</promise>` when video is in PR

Start with step 1 now.

---

## /test-xcode

_Source: `marketplaces/every-marketplace/plugins/compound-engineering/commands/test-xcode.md`_

---
name: test-xcode
description: Build and test iOS apps on simulator using XcodeBuildMCP
argument-hint: "[scheme name or 'current' to use default]"
disable-model-invocation: true
---

# Xcode Test Command

<command_purpose>Build, install, and test iOS apps on the simulator using XcodeBuildMCP. Captures screenshots, logs, and verifies app behavior.</command_purpose>

## Introduction

<role>iOS QA Engineer specializing in simulator-based testing</role>

This command tests iOS/macOS apps by:
- Building for simulator
- Installing and launching the app
- Taking screenshots of key screens
- Capturing console logs for errors
- Supporting human verification for external flows

## Prerequisites

<requirements>
- Xcode installed with command-line tools
- XcodeBuildMCP server connected
- Valid Xcode project or workspace
- At least one iOS Simulator available
</requirements>

## Main Tasks

### 0. Verify XcodeBuildMCP is Installed

<check_mcp_installed>

**First, check if XcodeBuildMCP tools are available.**

Try calling:
```
mcp__xcodebuildmcp__list_simulators({})
```

**If the tool is not found or errors:**

Tell the user:
```markdown
**XcodeBuildMCP not installed**

Please install the XcodeBuildMCP server first:

\`\`\`bash
claude mcp add XcodeBuildMCP -- npx xcodebuildmcp@latest
\`\`\`

Then restart Claude Code and run `/xcode-test` again.
```

**Do NOT proceed** until XcodeBuildMCP is confirmed working.

</check_mcp_installed>

### 1. Discover Project and Scheme

<discover_project>

**Find available projects:**
```
mcp__xcodebuildmcp__discover_projs({})
```

**List schemes for the project:**
```
mcp__xcodebuildmcp__list_schemes({ project_path: "/path/to/Project.xcodeproj" })
```

**If argument provided:**
- Use the specified scheme name
- Or "current" to use the default/last-used scheme

</discover_project>

### 2. Boot Simulator

<boot_simulator>

**List available simulators:**
```
mcp__xcodebuildmcp__list_simulators({})
```

**Boot preferred simulator (iPhone 15 Pro recommended):**
```
mcp__xcodebuildmcp__boot_simulator({ simulator_id: "[uuid]" })
```

**Wait for simulator to be ready:**
Check simulator state before proceeding with installation.

</boot_simulator>

### 3. Build the App

<build_app>

**Build for iOS Simulator:**
```
mcp__xcodebuildmcp__build_ios_sim_app({
  project_path: "/path/to/Project.xcodeproj",
  scheme: "[scheme_name]"
})
```

**Handle build failures:**
- Capture build errors
- Create P1 todo for each build error
- Report to user with specific error details

**On success:**
- Note the built app path for installation
- Proceed to installation step

</build_app>

### 4. Install and Launch

<install_launch>

**Install app on simulator:**
```
mcp__xcodebuildmcp__install_app_on_simulator({
  app_path: "/path/to/built/App.app",
  simulator_id: "[uuid]"
})
```

**Launch the app:**
```
mcp__xcodebuildmcp__launch_app_on_simulator({
  bundle_id: "[app.bundle.id]",
  simulator_id: "[uuid]"
})
```

**Start capturing logs:**
```
mcp__xcodebuildmcp__capture_sim_logs({
  simulator_id: "[uuid]",
  bundle_id: "[app.bundle.id]"
})
```

</install_launch>

### 5. Test Key Screens

<test_screens>

For each key screen in the app:

**Take screenshot:**
```
mcp__xcodebuildmcp__take_screenshot({
  simulator_id: "[uuid]",
  filename: "screen-[name].png"
})
```

**Review screenshot for:**
- UI elements rendered correctly
- No error messages visible
- Expected content displayed
- Layout looks correct

**Check logs for errors:**
```
mcp__xcodebuildmcp__get_sim_logs({ simulator_id: "[uuid]" })
```

Look for:
- Crashes
- Exceptions
- Error-level log messages
- Failed network requests

</test_screens>

### 6. Human Verification (When Required)

<human_verification>

Pause for human input when testing touches:

| Flow Type | What to Ask |
|-----------|-------------|
| Sign in with Apple | "Please complete Sign in with Apple on the simulator" |
| Push notifications | "Send a test push and confirm it appears" |
| In-app purchases | "Complete a sandbox purchase" |
| Camera/Photos | "Grant permissions and verify camera works" |
| Location | "Allow location access and verify map updates" |

Use AskUserQuestion:
```markdown
**Human Verification Needed**

This test requires [flow type]. Please:
1. [Action to take on simulator]
2. [What to verify]

Did it work correctly?
1. Yes - continue testing
2. No - describe the issue
```

</human_verification>

### 7. Handle Failures

<failure_handling>

When a test fails:

1. **Document the failure:**
   - Take screenshot of error state
   - Capture console logs
   - Note reproduction steps

2. **Ask user how to proceed:**
   ```markdown
   **Test Failed: [screen/feature]**

   Issue: [description]
   Logs: [relevant error messages]

   How to proceed?
   1. Fix now - I'll help debug and fix
   2. Create todo - Add to todos/ for later
   3. Skip - Continue testing other screens
   ```

3. **If "Fix now":**
   - Investigate the issue in code
   - Propose a fix
   - Rebuild and retest

4. **If "Create todo":**
   - Create `{id}-pending-p1-xcode-{description}.md`
   - Continue testing

</failure_handling>

### 8. Test Summary

<test_summary>

After all tests complete, present summary:

```markdown
## 📱 Xcode Test Results

**Project:** [project name]
**Scheme:** [scheme name]
**Simulator:** [simulator name]

### Build: ✅ Success / ❌ Failed

### Screens Tested: [count]

| Screen | Status | Notes |
|--------|--------|-------|
| Launch | ✅ Pass | |
| Home | ✅ Pass | |
| Settings | ❌ Fail | Crash on tap |
| Profile | ⏭️ Skip | Requires login |

### Console Errors: [count]
- [List any errors found]

### Human Verifications: [count]
- Sign in with Apple: ✅ Confirmed
- Push notifications: ✅ Confirmed

### Failures: [count]
- Settings screen - crash on navigation

### Created Todos: [count]
- `006-pending-p1-xcode-settings-crash.md`

### Result: [PASS / FAIL / PARTIAL]
```

</test_summary>

### 9. Cleanup

<cleanup>

After testing:

**Stop log capture:**
```
mcp__xcodebuildmcp__stop_log_capture({ simulator_id: "[uuid]" })
```

**Optionally shut down simulator:**
```
mcp__xcodebuildmcp__shutdown_simulator({ simulator_id: "[uuid]" })
```

</cleanup>

## Quick Usage Examples

```bash
# Test with default scheme
/xcode-test

# Test specific scheme
/xcode-test MyApp-Debug

# Test after making changes
/xcode-test current
```

## Integration with /workflows:review

When reviewing PRs that touch iOS code, the `/workflows:review` command can spawn this as a subagent:

```
Task general-purpose("Run /xcode-test for scheme [name]. Build, install on simulator, test key screens, check for crashes.")
```

---

## /agent-native-audit

_Source: `marketplaces/every-marketplace/plugins/compound-engineering/commands/agent-native-audit.md`_

---
name: agent-native-audit
description: Run comprehensive agent-native architecture review with scored principles
argument-hint: "[optional: specific principle to audit]"
disable-model-invocation: true
---

# Agent-Native Architecture Audit

Conduct a comprehensive review of the codebase against agent-native architecture principles, launching parallel sub-agents for each principle and producing a scored report.

## Core Principles to Audit

1. **Action Parity** - "Whatever the user can do, the agent can do"
2. **Tools as Primitives** - "Tools provide capability, not behavior"
3. **Context Injection** - "System prompt includes dynamic context about app state"
4. **Shared Workspace** - "Agent and user work in the same data space"
5. **CRUD Completeness** - "Every entity has full CRUD (Create, Read, Update, Delete)"
6. **UI Integration** - "Agent actions immediately reflected in UI"
7. **Capability Discovery** - "Users can discover what the agent can do"
8. **Prompt-Native Features** - "Features are prompts defining outcomes, not code"

## Workflow

### Step 1: Load the Agent-Native Skill

First, invoke the agent-native-architecture skill to understand all principles:

```
/compound-engineering:agent-native-architecture
```

Select option 7 (action parity) to load the full reference material.

### Step 2: Launch Parallel Sub-Agents

Launch 8 parallel sub-agents using the Task tool with `subagent_type: Explore`, one for each principle. Each agent should:

1. Enumerate ALL instances in the codebase (user actions, tools, contexts, data stores, etc.)
2. Check compliance against the principle
3. Provide a SPECIFIC SCORE like "X out of Y (percentage%)"
4. List specific gaps and recommendations

<sub-agents>

**Agent 1: Action Parity**
```
Audit for ACTION PARITY - "Whatever the user can do, the agent can do."

Tasks:
1. Enumerate ALL user actions in frontend (API calls, button clicks, form submissions)
   - Search for API service files, fetch calls, form handlers
   - Check routes and components for user interactions
2. Check which have corresponding agent tools
   - Search for agent tool definitions
   - Map user actions to agent capabilities
3. Score: "Agent can do X out of Y user actions"

Format:
## Action Parity Audit
### User Actions Found
| Action | Location | Agent Tool | Status |
### Score: X/Y (percentage%)
### Missing Agent Tools
### Recommendations
```

**Agent 2: Tools as Primitives**
```
Audit for TOOLS AS PRIMITIVES - "Tools provide capability, not behavior."

Tasks:
1. Find and read ALL agent tool files
2. Classify each as:
   - PRIMITIVE (good): read, write, store, list - enables capability without business logic
   - WORKFLOW (bad): encodes business logic, makes decisions, orchestrates steps
3. Score: "X out of Y tools are proper primitives"

Format:
## Tools as Primitives Audit
### Tool Analysis
| Tool | File | Type | Reasoning |
### Score: X/Y (percentage%)
### Problematic Tools (workflows that should be primitives)
### Recommendations
```

**Agent 3: Context Injection**
```
Audit for CONTEXT INJECTION - "System prompt includes dynamic context about app state"

Tasks:
1. Find context injection code (search for "context", "system prompt", "inject")
2. Read agent prompts and system messages
3. Enumerate what IS injected vs what SHOULD be:
   - Available resources (files, drafts, documents)
   - User preferences/settings
   - Recent activity
   - Available capabilities listed
   - Session history
   - Workspace state

Format:
## Context Injection Audit
### Context Types Analysis
| Context Type | Injected? | Location | Notes |
### Score: X/Y (percentage%)
### Missing Context
### Recommendations
```

**Agent 4: Shared Workspace**
```
Audit for SHARED WORKSPACE - "Agent and user work in the same data space"

Tasks:
1. Identify all data stores/tables/models
2. Check if agents read/write to SAME tables or separate ones
3. Look for sandbox isolation anti-pattern (agent has separate data space)

Format:
## Shared Workspace Audit
### Data Store Analysis
| Data Store | User Access | Agent Access | Shared? |
### Score: X/Y (percentage%)
### Isolated Data (anti-pattern)
### Recommendations
```

**Agent 5: CRUD Completeness**
```
Audit for CRUD COMPLETENESS - "Every entity has full CRUD"

Tasks:
1. Identify all entities/models in the codebase
2. For each entity, check if agent tools exist for:
   - Create
   - Read
   - Update
   - Delete
3. Score per entity and overall

Format:
## CRUD Completeness Audit
### Entity CRUD Analysis
| Entity | Create | Read | Update | Delete | Score |
### Overall Score: X/Y entities with full CRUD (percentage%)
### Incomplete Entities (list missing operations)
### Recommendations
```

**Agent 6: UI Integration**
```
Audit for UI INTEGRATION - "Agent actions immediately reflected in UI"

Tasks:
1. Check how agent writes/changes propagate to frontend
2. Look for:
   - Streaming updates (SSE, WebSocket)
   - Polling mechanisms
   - Shared state/services
   - Event buses
   - File watching
3. Identify "silent actions" anti-pattern (agent changes state but UI doesn't update)

Format:
## UI Integration Audit
### Agent Action → UI Update Analysis
| Agent Action | UI Mechanism | Immediate? | Notes |
### Score: X/Y (percentage%)
### Silent Actions (anti-pattern)
### Recommendations
```

**Agent 7: Capability Discovery**
```
Audit for CAPABILITY DISCOVERY - "Users can discover what the agent can do"

Tasks:
1. Check for these 7 discovery mechanisms:
   - Onboarding flow showing agent capabilities
   - Help documentation
   - Capability hints in UI
   - Agent self-describes in responses
   - Suggested prompts/actions
   - Empty state guidance
   - Slash commands (/help, /tools)
2. Score against 7 mechanisms

Format:
## Capability Discovery Audit
### Discovery Mechanism Analysis
| Mechanism | Exists? | Location | Quality |
### Score: X/7 (percentage%)
### Missing Discovery
### Recommendations
```

**Agent 8: Prompt-Native Features**
```
Audit for PROMPT-NATIVE FEATURES - "Features are prompts defining outcomes, not code"

Tasks:
1. Read all agent prompts
2. Classify each feature/behavior as defined in:
   - PROMPT (good): outcomes defined in natural language
   - CODE (bad): business logic hardcoded
3. Check if behavior changes require prompt edit vs code change

Format:
## Prompt-Native Features Audit
### Feature Definition Analysis
| Feature | Defined In | Type | Notes |
### Score: X/Y (percentage%)
### Code-Defined Features (anti-pattern)
### Recommendations
```

</sub-agents>

### Step 3: Compile Summary Report

After all agents complete, compile a summary with:

```markdown
## Agent-Native Architecture Review: [Project Name]

### Overall Score Summary

| Core Principle | Score | Percentage | Status |
|----------------|-------|------------|--------|
| Action Parity | X/Y | Z% | ✅/⚠️/❌ |
| Tools as Primitives | X/Y | Z% | ✅/⚠️/❌ |
| Context Injection | X/Y | Z% | ✅/⚠️/❌ |
| Shared Workspace | X/Y | Z% | ✅/⚠️/❌ |
| CRUD Completeness | X/Y | Z% | ✅/⚠️/❌ |
| UI Integration | X/Y | Z% | ✅/⚠️/❌ |
| Capability Discovery | X/Y | Z% | ✅/⚠️/❌ |
| Prompt-Native Features | X/Y | Z% | ✅/⚠️/❌ |

**Overall Agent-Native Score: X%**

### Status Legend
- ✅ Excellent (80%+)
- ⚠️ Partial (50-79%)
- ❌ Needs Work (<50%)

### Top 10 Recommendations by Impact

| Priority | Action | Principle | Effort |
|----------|--------|-----------|--------|

### What's Working Excellently

[List top 5 strengths]
```

## Success Criteria

- [ ] All 8 sub-agents complete their audits
- [ ] Each principle has a specific numeric score (X/Y format)
- [ ] Summary table shows all scores and status indicators
- [ ] Top 10 recommendations are prioritized by impact
- [ ] Report identifies both strengths and gaps

## Optional: Single Principle Audit

If $ARGUMENTS specifies a single principle (e.g., "action parity"), only run that sub-agent and provide detailed findings for that principle alone.

Valid arguments:
- `action parity` or `1`
- `tools` or `primitives` or `2`
- `context` or `injection` or `3`
- `shared` or `workspace` or `4`
- `crud` or `5`
- `ui` or `integration` or `6`
- `discovery` or `7`
- `prompt` or `features` or `8`

---

## /resolve_todo_parallel

_Source: `marketplaces/every-marketplace/plugins/compound-engineering/commands/resolve_todo_parallel.md`_

---
name: resolve_todo_parallel
description: Resolve all pending CLI todos using parallel processing
argument-hint: "[optional: specific todo ID or pattern]"
---

Resolve all TODO comments using parallel processing.

## Workflow

### 1. Analyze

Get all unresolved TODOs from the /todos/\*.md directory

If any todo recommends deleting, removing, or gitignoring files in `docs/plans/` or `docs/solutions/`, skip it and mark it as `wont_fix`. These are compound-engineering pipeline artifacts that are intentional and permanent.

### 2. Plan

Create a TodoWrite list of all unresolved items grouped by type.Make sure to look at dependencies that might occur and prioritize the ones needed by others. For example, if you need to change a name, you must wait to do the others. Output a mermaid flow diagram showing how we can do this. Can we do everything in parallel? Do we need to do one first that leads to others in parallel? I'll put the to-dos in the mermaid diagram flow‑wise so the agent knows how to proceed in order.

### 3. Implement (PARALLEL)

Spawn a pr-comment-resolver agent for each unresolved item in parallel.

So if there are 3 comments, it will spawn 3 pr-comment-resolver agents in parallel. liek this

1. Task pr-comment-resolver(comment1)
2. Task pr-comment-resolver(comment2)
3. Task pr-comment-resolver(comment3)

Always run all in parallel subagents/Tasks for each Todo item.

### 4. Commit & Resolve

- Commit changes
- Remove the TODO from the file, and mark it as resolved.
- Push to remote

---

## /release-docs

_Source: `marketplaces/every-marketplace/plugins/compound-engineering/commands/release-docs.md`_

---
name: release-docs
description: Build and update the documentation site with current plugin components
argument-hint: "[optional: --dry-run to preview changes without writing]"
disable-model-invocation: true
---

# Release Documentation Command

You are a documentation generator for the compound-engineering plugin. Your job is to ensure the documentation site at `plugins/compound-engineering/docs/` is always up-to-date with the actual plugin components.

## Overview

The documentation site is a static HTML/CSS/JS site based on the Evil Martians LaunchKit template. It needs to be regenerated whenever:

- Agents are added, removed, or modified
- Commands are added, removed, or modified
- Skills are added, removed, or modified
- MCP servers are added, removed, or modified

## Step 1: Inventory Current Components

First, count and list all current components:

```bash
# Count agents
ls plugins/compound-engineering/agents/*.md | wc -l

# Count commands
ls plugins/compound-engineering/commands/*.md | wc -l

# Count skills
ls -d plugins/compound-engineering/skills/*/ 2>/dev/null | wc -l

# Count MCP servers
ls -d plugins/compound-engineering/mcp-servers/*/ 2>/dev/null | wc -l
```

Read all component files to get their metadata:

### Agents
For each agent file in `plugins/compound-engineering/agents/*.md`:
- Extract the frontmatter (name, description)
- Note the category (Review, Research, Workflow, Design, Docs)
- Get key responsibilities from the content

### Commands
For each command file in `plugins/compound-engineering/commands/*.md`:
- Extract the frontmatter (name, description, argument-hint)
- Categorize as Workflow or Utility command

### Skills
For each skill directory in `plugins/compound-engineering/skills/*/`:
- Read the SKILL.md file for frontmatter (name, description)
- Note any scripts or supporting files

### MCP Servers
For each MCP server in `plugins/compound-engineering/mcp-servers/*/`:
- Read the configuration and README
- List the tools provided

## Step 2: Update Documentation Pages

### 2a. Update `docs/index.html`

Update the stats section with accurate counts:
```html
<div class="stats-grid">
  <div class="stat-card">
    <span class="stat-number">[AGENT_COUNT]</span>
    <span class="stat-label">Specialized Agents</span>
  </div>
  <!-- Update all stat cards -->
</div>
```

Ensure the component summary sections list key components accurately.

### 2b. Update `docs/pages/agents.html`

Regenerate the complete agents reference page:
- Group agents by category (Review, Research, Workflow, Design, Docs)
- Include for each agent:
  - Name and description
  - Key responsibilities (bullet list)
  - Usage example: `claude agent [agent-name] "your message"`
  - Use cases

### 2c. Update `docs/pages/commands.html`

Regenerate the complete commands reference page:
- Group commands by type (Workflow, Utility)
- Include for each command:
  - Name and description
  - Arguments (if any)
  - Process/workflow steps
  - Example usage

### 2d. Update `docs/pages/skills.html`

Regenerate the complete skills reference page:
- Group skills by category (Development Tools, Content & Workflow, Image Generation)
- Include for each skill:
  - Name and description
  - Usage: `claude skill [skill-name]`
  - Features and capabilities

### 2e. Update `docs/pages/mcp-servers.html`

Regenerate the MCP servers reference page:
- For each server:
  - Name and purpose
  - Tools provided
  - Configuration details
  - Supported frameworks/services

## Step 3: Update Metadata Files

Ensure counts are consistent across:

1. **`plugins/compound-engineering/.claude-plugin/plugin.json`**
   - Update `description` with correct counts
   - Update `components` object with counts
   - Update `agents`, `commands` arrays with current items

2. **`.claude-plugin/marketplace.json`**
   - Update plugin `description` with correct counts

3. **`plugins/compound-engineering/README.md`**
   - Update intro paragraph with counts
   - Update component lists

## Step 4: Validate

Run validation checks:

```bash
# Validate JSON files
cat .claude-plugin/marketplace.json | jq .
cat plugins/compound-engineering/.claude-plugin/plugin.json | jq .

# Verify counts match
echo "Agents in files: $(ls plugins/compound-engineering/agents/*.md | wc -l)"
grep -o "[0-9]* specialized agents" plugins/compound-engineering/docs/index.html

echo "Commands in files: $(ls plugins/compound-engineering/commands/*.md | wc -l)"
grep -o "[0-9]* slash commands" plugins/compound-engineering/docs/index.html
```

## Step 5: Report Changes

Provide a summary of what was updated:

```
## Documentation Release Summary

### Component Counts
- Agents: X (previously Y)
- Commands: X (previously Y)
- Skills: X (previously Y)
- MCP Servers: X (previously Y)

### Files Updated
- docs/index.html - Updated stats and component summaries
- docs/pages/agents.html - Regenerated with X agents
- docs/pages/commands.html - Regenerated with X commands
- docs/pages/skills.html - Regenerated with X skills
- docs/pages/mcp-servers.html - Regenerated with X servers
- plugin.json - Updated counts and component lists
- marketplace.json - Updated description
- README.md - Updated component lists

### New Components Added
- [List any new agents/commands/skills]

### Components Removed
- [List any removed agents/commands/skills]
```

## Dry Run Mode

If `--dry-run` is specified:
- Perform all inventory and validation steps
- Report what WOULD be updated
- Do NOT write any files
- Show diff previews of proposed changes

## Error Handling

- If component files have invalid frontmatter, report the error and skip
- If JSON validation fails, report and abort
- Always maintain a valid state - don't partially update

## Post-Release

After successful release:
1. Suggest updating CHANGELOG.md with documentation changes
2. Remind to commit with message: `docs: Update documentation site to match plugin components`
3. Remind to push changes

## Usage Examples

```bash
# Full documentation release
claude /release-docs

# Preview changes without writing
claude /release-docs --dry-run

# After adding new agents
claude /release-docs
```

---

## /technical_review

_Source: `marketplaces/every-marketplace/plugins/compound-engineering/commands/technical_review.md`_

---
name: technical_review
description: Have multiple specialized agents review the technical approach and architecture of a plan in parallel
argument-hint: "[plan file path or plan content]"
disable-model-invocation: true
---

Have @agent-dhh-rails-reviewer @agent-kieran-rails-reviewer @agent-code-simplicity-reviewer review the technical approach in this plan in parallel.

---

## /feature-video

_Source: `marketplaces/every-marketplace/plugins/compound-engineering/commands/feature-video.md`_

---
name: feature-video
description: Record a video walkthrough of a feature and add it to the PR description
argument-hint: "[PR number or 'current'] [optional: base URL, default localhost:3000]"
---

# Feature Video Walkthrough

<command_purpose>Record a video walkthrough demonstrating a feature, upload it, and add it to the PR description.</command_purpose>

## Introduction

<role>Developer Relations Engineer creating feature demo videos</role>

This command creates professional video walkthroughs of features for PR documentation:
- Records browser interactions using agent-browser CLI
- Demonstrates the complete user flow
- Uploads the video for easy sharing
- Updates the PR description with an embedded video

## Prerequisites

<requirements>
- Local development server running (e.g., `bin/dev`, `rails server`)
- agent-browser CLI installed
- Git repository with a PR to document
- `ffmpeg` installed (for video conversion)
- `rclone` configured (optional, for cloud upload - see rclone skill)
</requirements>

## Setup

**Check installation:**
```bash
command -v agent-browser >/dev/null 2>&1 && echo "Installed" || echo "NOT INSTALLED"
```

**Install if needed:**
```bash
npm install -g agent-browser && agent-browser install
```

See the `agent-browser` skill for detailed usage.

## Main Tasks

### 1. Parse Arguments

<parse_args>

**Arguments:** $ARGUMENTS

Parse the input:
- First argument: PR number or "current" (defaults to current branch's PR)
- Second argument: Base URL (defaults to `http://localhost:3000`)

```bash
# Get PR number for current branch if needed
gh pr view --json number -q '.number'
```

</parse_args>

### 2. Gather Feature Context

<gather_context>

**Get PR details:**
```bash
gh pr view [number] --json title,body,files,headRefName -q '.'
```

**Get changed files:**
```bash
gh pr view [number] --json files -q '.files[].path'
```

**Map files to testable routes** (same as playwright-test):

| File Pattern | Route(s) |
|-------------|----------|
| `app/views/users/*` | `/users`, `/users/:id`, `/users/new` |
| `app/controllers/settings_controller.rb` | `/settings` |
| `app/javascript/controllers/*_controller.js` | Pages using that Stimulus controller |
| `app/components/*_component.rb` | Pages rendering that component |

</gather_context>

### 3. Plan the Video Flow

<plan_flow>

Before recording, create a shot list:

1. **Opening shot**: Homepage or starting point (2-3 seconds)
2. **Navigation**: How user gets to the feature
3. **Feature demonstration**: Core functionality (main focus)
4. **Edge cases**: Error states, validation, etc. (if applicable)
5. **Success state**: Completed action/result

Ask user to confirm or adjust the flow:

```markdown
**Proposed Video Flow**

Based on PR #[number]: [title]

1. Start at: /[starting-route]
2. Navigate to: /[feature-route]
3. Demonstrate:
   - [Action 1]
   - [Action 2]
   - [Action 3]
4. Show result: [success state]

Estimated duration: ~[X] seconds

Does this look right?
1. Yes, start recording
2. Modify the flow (describe changes)
3. Add specific interactions to demonstrate
```

</plan_flow>

### 4. Setup Video Recording

<setup_recording>

**Create videos directory:**
```bash
mkdir -p tmp/videos
```

**Recording approach: Use browser screenshots as frames**

agent-browser captures screenshots at key moments, then combine into video using ffmpeg:

```bash
ffmpeg -framerate 2 -pattern_type glob -i 'tmp/screenshots/*.png' -vf "scale=1280:-1" tmp/videos/feature-demo.gif
```

</setup_recording>

### 5. Record the Walkthrough

<record_walkthrough>

Execute the planned flow, capturing each step:

**Step 1: Navigate to starting point**
```bash
agent-browser open "[base-url]/[start-route]"
agent-browser wait 2000
agent-browser screenshot tmp/screenshots/01-start.png
```

**Step 2: Perform navigation/interactions**
```bash
agent-browser snapshot -i  # Get refs
agent-browser click @e1    # Click navigation element
agent-browser wait 1000
agent-browser screenshot tmp/screenshots/02-navigate.png
```

**Step 3: Demonstrate feature**
```bash
agent-browser snapshot -i  # Get refs for feature elements
agent-browser click @e2    # Click feature element
agent-browser wait 1000
agent-browser screenshot tmp/screenshots/03-feature.png
```

**Step 4: Capture result**
```bash
agent-browser wait 2000
agent-browser screenshot tmp/screenshots/04-result.png
```

**Create video/GIF from screenshots:**

```bash
# Create directories
mkdir -p tmp/videos tmp/screenshots

# Create MP4 video (RECOMMENDED - better quality, smaller size)
# -framerate 0.5 = 2 seconds per frame (slower playback)
# -framerate 1 = 1 second per frame
ffmpeg -y -framerate 0.5 -pattern_type glob -i 'tmp/screenshots/*.png' \
  -c:v libx264 -pix_fmt yuv420p -vf "scale=1280:-2" \
  tmp/videos/feature-demo.mp4

# Create low-quality GIF for preview (small file, for GitHub embed)
ffmpeg -y -framerate 0.5 -pattern_type glob -i 'tmp/screenshots/*.png' \
  -vf "scale=640:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse" \
  -loop 0 tmp/videos/feature-demo-preview.gif
```

**Note:**
- The `-2` in MP4 scale ensures height is divisible by 2 (required for H.264)
- Preview GIF uses 640px width and 128 colors to keep file size small (~100-200KB)

</record_walkthrough>

### 6. Upload the Video

<upload_video>

**Upload with rclone:**

```bash
# Check rclone is configured
rclone listremotes

# Upload video, preview GIF, and screenshots to cloud storage
# Use --s3-no-check-bucket to avoid permission errors
rclone copy tmp/videos/ r2:kieran-claude/pr-videos/pr-[number]/ --s3-no-check-bucket --progress
rclone copy tmp/screenshots/ r2:kieran-claude/pr-videos/pr-[number]/screenshots/ --s3-no-check-bucket --progress

# List uploaded files
rclone ls r2:kieran-claude/pr-videos/pr-[number]/
```

Public URLs (R2 with public access):
```
Video: https://pub-4047722ebb1b4b09853f24d3b61467f1.r2.dev/pr-videos/pr-[number]/feature-demo.mp4
Preview: https://pub-4047722ebb1b4b09853f24d3b61467f1.r2.dev/pr-videos/pr-[number]/feature-demo-preview.gif
```

</upload_video>

### 7. Update PR Description

<update_pr>

**Get current PR body:**
```bash
gh pr view [number] --json body -q '.body'
```

**Add video section to PR description:**

If the PR already has a video section, replace it. Otherwise, append:

**IMPORTANT:** GitHub cannot embed external MP4s directly. Use a clickable GIF that links to the video:

```markdown
## Demo

[![Feature Demo]([preview-gif-url])]([video-mp4-url])

*Click to view full video*
```

Example:
```markdown
[![Feature Demo](https://pub-4047722ebb1b4b09853f24d3b61467f1.r2.dev/pr-videos/pr-137/feature-demo-preview.gif)](https://pub-4047722ebb1b4b09853f24d3b61467f1.r2.dev/pr-videos/pr-137/feature-demo.mp4)
```

**Update the PR:**
```bash
gh pr edit [number] --body "[updated body with video section]"
```

**Or add as a comment if preferred:**
```bash
gh pr comment [number] --body "## Feature Demo

![Demo]([video-url])

_Automated walkthrough of the changes in this PR_"
```

</update_pr>

### 8. Cleanup

<cleanup>

```bash
# Optional: Clean up screenshots
rm -rf tmp/screenshots

# Keep videos for reference
echo "Video retained at: tmp/videos/feature-demo.gif"
```

</cleanup>

### 9. Summary

<summary>

Present completion summary:

```markdown
## Feature Video Complete

**PR:** #[number] - [title]
**Video:** [url or local path]
**Duration:** ~[X] seconds
**Format:** [GIF/MP4]

### Shots Captured
1. [Starting point] - [description]
2. [Navigation] - [description]
3. [Feature demo] - [description]
4. [Result] - [description]

### PR Updated
- [x] Video section added to PR description
- [ ] Ready for review

**Next steps:**
- Review the video to ensure it accurately demonstrates the feature
- Share with reviewers for context
```

</summary>

## Quick Usage Examples

```bash
# Record video for current branch's PR
/feature-video

# Record video for specific PR
/feature-video 847

# Record with custom base URL
/feature-video 847 http://localhost:5000

# Record for staging environment
/feature-video current https://staging.example.com
```

## Tips

- **Keep it short**: 10-30 seconds is ideal for PR demos
- **Focus on the change**: Don't include unrelated UI
- **Show before/after**: If fixing a bug, show the broken state first (if possible)
- **Annotate if needed**: Add text overlays for complex features

---

## /heal-skill

_Source: `marketplaces/every-marketplace/plugins/compound-engineering/commands/heal-skill.md`_

---
name: heal-skill
description: Fix incorrect SKILL.md files when a skill has wrong instructions or outdated API references
argument-hint: [optional: specific issue to fix]
allowed-tools: [Read, Edit, Bash(ls:*), Bash(git:*)]
disable-model-invocation: true
---

<objective>
Update a skill's SKILL.md and related files based on corrections discovered during execution.

Analyze the conversation to detect which skill is running, reflect on what went wrong, propose specific fixes, get user approval, then apply changes with optional commit.
</objective>

<context>
Skill detection: !`ls -1 ./skills/*/SKILL.md | head -5`
</context>

<quick_start>
<workflow>
1. **Detect skill** from conversation context (invocation messages, recent SKILL.md references)
2. **Reflect** on what went wrong and how you discovered the fix
3. **Present** proposed changes with before/after diffs
4. **Get approval** before making any edits
5. **Apply** changes and optionally commit
</workflow>
</quick_start>

<process>
<step_1 name="detect_skill">
Identify the skill from conversation context:

- Look for skill invocation messages
- Check which SKILL.md was recently referenced
- Examine current task context

Set: `SKILL_NAME=[skill-name]` and `SKILL_DIR=./skills/$SKILL_NAME`

If unclear, ask the user.
</step_1>

<step_2 name="reflection_and_analysis">
Focus on $ARGUMENTS if provided, otherwise analyze broader context.

Determine:
- **What was wrong**: Quote specific sections from SKILL.md that are incorrect
- **Discovery method**: Context7, error messages, trial and error, documentation lookup
- **Root cause**: Outdated API, incorrect parameters, wrong endpoint, missing context
- **Scope of impact**: Single section or multiple? Related files affected?
- **Proposed fix**: Which files, which sections, before/after for each
</step_2>

<step_3 name="scan_affected_files">
```bash
ls -la $SKILL_DIR/
ls -la $SKILL_DIR/references/ 2>/dev/null
ls -la $SKILL_DIR/scripts/ 2>/dev/null
```
</step_3>

<step_4 name="present_proposed_changes">
Present changes in this format:

```
**Skill being healed:** [skill-name]
**Issue discovered:** [1-2 sentence summary]
**Root cause:** [brief explanation]

**Files to be modified:**
- [ ] SKILL.md
- [ ] references/[file].md
- [ ] scripts/[file].py

**Proposed changes:**

### Change 1: SKILL.md - [Section name]
**Location:** Line [X] in SKILL.md

**Current (incorrect):**
```
[exact text from current file]
```

**Corrected:**
```
[new text]
```

**Reason:** [why this fixes the issue]

[repeat for each change across all files]

**Impact assessment:**
- Affects: [authentication/API endpoints/parameters/examples/etc.]

**Verification:**
These changes will prevent: [specific error that prompted this]
```
</step_4>

<step_5 name="request_approval">
```
Should I apply these changes?

1. Yes, apply and commit all changes
2. Apply but don't commit (let me review first)
3. Revise the changes (I'll provide feedback)
4. Cancel (don't make changes)

Choose (1-4):
```

**Wait for user response. Do not proceed without approval.**
</step_5>

<step_6 name="apply_changes">
Only after approval (option 1 or 2):

1. Use Edit tool for each correction across all files
2. Read back modified sections to verify
3. If option 1, commit with structured message showing what was healed
4. Confirm completion with file list
</step_6>
</process>

<success_criteria>
- Skill correctly detected from conversation context
- All incorrect sections identified with before/after
- User approved changes before application
- All edits applied across SKILL.md and related files
- Changes verified by reading back
- Commit created if user chose option 1
- Completion confirmed with file list
</success_criteria>

<verification>
Before completing:

- Read back each modified section to confirm changes applied
- Ensure cross-file consistency (SKILL.md examples match references/)
- Verify git commit created if option 1 was selected
- Check no unintended files were modified
</verification>

---

## /quiz-me

_Source: `marketplaces/every-marketplace/plugins/coding-tutor/commands/quiz-me.md`_

Quiz me using the coding-tutor skill

---

## /sync-tutorials

_Source: `marketplaces/every-marketplace/plugins/coding-tutor/commands/sync-tutorials.md`_

# Sync Coding Tutor Tutorials

Commit and push your tutorials to the GitHub repository for backup and mobile reading.

## Instructions

1. **Go to the tutorials repo**: `cd ~/coding-tutor-tutorials`

2. **Check for changes**: Run `git status` to see what's new or modified

3. **If there are changes**:
   - Stage all changes: `git add -A`
   - Create a commit with a message summarizing what was added/updated (e.g., "Add tutorial on React hooks" or "Update quiz scores")
   - Push to origin: `git push`

4. **If no GitHub remote exists**:
   - Create the repo: `gh repo create coding-tutor-tutorials --private --source=. --push`

5. **Report results**: Tell the user what was synced or that everything is already up to date

## Notes

- The tutorials repo is at: `~/coding-tutor-tutorials/`
- Always use `--private` when creating the GitHub repo
- This is your personal learning journey - keep it backed up!

---

## /teach-me

_Source: `marketplaces/every-marketplace/plugins/coding-tutor/commands/teach-me.md`_

Teach me something using the coding-tutor skill

---

## /default-command

_Source: `marketplaces/every-marketplace/tests/fixtures/custom-paths/commands/default-command.md`_

---
name: default-command
---

Default command

---

## /disabled-command

_Source: `marketplaces/every-marketplace/tests/fixtures/sample-plugin/commands/disabled-command.md`_

---
name: deploy-docs
description: Deploy documentation site
disable-model-invocation: true
---

Deploy docs body.

---

## /model-command

_Source: `marketplaces/every-marketplace/tests/fixtures/sample-plugin/commands/model-command.md`_

---
name: workflows:work
description: Execute planned tasks step by step
model: gpt-4o
allowed-tools: WebFetch
---

Workflows work body.

---

## /todo-command

_Source: `marketplaces/every-marketplace/tests/fixtures/sample-plugin/commands/todo-command.md`_

---
name: workflows:plan
description: Create a structured plan from requirements
allowed-tools: Question, TodoWrite, TodoRead
---

Workflows plan body.

---

## /command-two

_Source: `marketplaces/every-marketplace/tests/fixtures/sample-plugin/commands/nested/command-two.md`_

---
name: plan_review
description: Review a plan with multiple agents
allowed-tools:
  - Read
  - Edit
---

Plan review body.

---

## /command-one

_Source: `marketplaces/every-marketplace/tests/fixtures/sample-plugin/commands/command-one.md`_

---
name: workflows:review
description: Run a multi-agent review workflow
allowed-tools: Read, Write, Edit, Bash(ls:*), Bash(git:*), Grep, Glob, List, Patch, Task
---

Workflows review body.

---

## /skill-command

_Source: `marketplaces/every-marketplace/tests/fixtures/sample-plugin/commands/skill-command.md`_

---
name: create-agent-skill
description: Create or edit a Claude Code skill
allowed-tools: Skill(create-agent-skills)
---

Create agent skill body.

---

## /pattern-command

_Source: `marketplaces/every-marketplace/tests/fixtures/sample-plugin/commands/pattern-command.md`_

---
name: report-bug
description: Report a bug with structured context
allowed-tools: Read(.env), Bash(git:*)
---

Report bug body.

---

## /triage-prs

_Source: `marketplaces/every-marketplace/.claude/commands/triage-prs.md`_

---
name: triage-prs
description: Triage all open PRs with parallel agents, label, group, and review one-by-one
argument-hint: "[optional: repo owner/name or GitHub PRs URL]"
disable-model-invocation: true
allowed-tools: Bash(gh *), Bash(git log *)
---

# Triage Open Pull Requests

Review, label, and act on all open PRs for a repository using parallel review agents. Produces a grouped triage report, applies labels, cross-references with issues, and walks through each PR for merge/comment decisions.

## Step 0: Detect Repository

Detect repo context:
- Current repo: !`gh repo view --json nameWithOwner -q .nameWithOwner 2>/dev/null || echo "no repo detected"`
- Current branch: !`git branch --show-current 2>/dev/null`

If `$ARGUMENTS` contains a GitHub URL or `owner/repo`, use that instead. Confirm the repo with the user if ambiguous.

## Step 1: Gather Context (Parallel)

Run these in parallel:

1. **List all open PRs:**
   ```bash
   gh pr list --repo OWNER/REPO --state open --limit 50
   ```

2. **List all open issues:**
   ```bash
   gh issue list --repo OWNER/REPO --state open --limit 50
   ```

3. **List existing labels:**
   ```bash
   gh label list --repo OWNER/REPO --limit 50
   ```

4. **Check recent merges** (to detect duplicate/superseded PRs):
   ```bash
   git log --oneline -20 main
   ```

## Step 2: Batch PRs by Theme

Group PRs into review batches of 4-6 based on apparent type:

- **Bug fixes** - titles with `fix`, `bug`, error descriptions
- **Features** - titles with `feat`, `add`, new functionality
- **Documentation** - titles with `docs`, `readme`, terminology
- **Configuration/Setup** - titles with `config`, `setup`, `install`
- **Stale/Old** - PRs older than 30 days

## Step 3: Parallel Review (Team of Agents)

Spawn one review agent per batch using the Task tool. Each agent should:

For each PR in their batch:
1. Run `gh pr view --repo OWNER/REPO <number> --json title,body,files,additions,deletions,author,createdAt`
2. Run `gh pr diff --repo OWNER/REPO <number>` (pipe to `head -200` for large diffs)
3. Determine:
   - **Description:** 1-2 sentence summary of the change
   - **Label:** Which existing repo label fits best
   - **Action:** merge / request changes / close / needs discussion
   - **Related PRs:** Any PRs in this or other batches that touch the same files or feature
   - **Quality notes:** Code quality, test coverage, staleness concerns

Instruct each agent to:
- Flag PRs that touch the same files (potential merge conflicts)
- Flag PRs that duplicate recently merged work
- Flag PRs that are part of a group solving the same problem differently
- Report findings as a markdown table
- Send findings back via message when done

## Step 4: Cross-Reference Issues

After all agents report, match issues to PRs:

- Check if any PR title/body mentions `Fixes #X` or `Closes #X`
- Check if any issue title matches a PR's topic
- Look for duplicate issues (same bug reported twice)

Build a mapping table:
```
| Issue | PR | Relationship |
|-------|-----|--------------|
| #158  | #159 | PR fixes issue |
```

## Step 5: Identify Themes

Group all issues into themes (3-6 themes):
- Count issues per theme
- Note which themes have PRs addressing them and which don't
- Flag themes with competing/overlapping PRs

## Step 6: Compile Triage Report

Present a single report with:

1. **Summary stats:** X open PRs, Y open issues, Z themes
2. **PR groups** with recommended actions:
   - Group name and related PRs
   - Per-PR: #, title, author, description, label, action
3. **Issue-to-PR mapping**
4. **Themes across issues**
5. **Suggested cleanup:** spam issues, duplicates, stale items

## Step 7: Apply Labels

After presenting the report, ask user:

> "Apply these labels to all PRs on GitHub?"

If yes, run `gh pr edit --repo OWNER/REPO <number> --add-label "<label>"` for each PR.

## Step 8: One-by-One Review

Use **AskUserQuestion** to ask:

> "Ready to walk through PRs one-by-one for merge/comment decisions?"

Then for each PR, ordered by priority (bug fixes first, then docs, then features, then stale):

### Show the PR:

```
### PR #<number> - <title>
Author: <author> | Files: <count> | +<additions>/-<deletions> | <age>
Label: <label>

<1-2 sentence description>

Fixes: <linked issues if any>
Related: <related PRs if any>
```

Show the diff (trimmed to key changes if large).

### Ask for decision:

Use **AskUserQuestion**:
- **Merge** - Merge this PR now
- **Comment & skip** - Leave a comment explaining why not merging, keep open
- **Close** - Close with a comment
- **Skip** - Move to next without action

### Execute decision:

- **Merge:** `gh pr merge --repo OWNER/REPO <number> --squash`
  - If PR fixes an issue, close the issue too
- **Comment & skip:** `gh pr comment --repo OWNER/REPO <number> --body "<comment>"`
  - Ask user what to say, or generate a grateful + specific comment
- **Close:** `gh pr close --repo OWNER/REPO <number> --comment "<reason>"`
- **Skip:** Move on

## Step 9: Post-Merge Cleanup

After all PRs are reviewed:

1. **Close resolved issues** that were fixed by merged PRs
2. **Close spam/off-topic issues** (confirm with user first)
3. **Summary of actions taken:**
   ```
   ## Triage Complete

   Merged: X PRs
   Commented: Y PRs
   Closed: Z PRs
   Skipped: W PRs

   Issues closed: A
   Labels applied: B
   ```

## Step 10: Post-Triage Options

Use **AskUserQuestion**:

1. **Run `/release-docs`** - Update documentation site if components changed
2. **Run `/changelog`** - Generate changelog for merged PRs
3. **Commit any local changes** - If version bumps needed
4. **Done** - Wrap up

## Important Notes

- **DO NOT merge without user approval** for each PR
- **DO NOT force push or destructive actions**
- Comments on declined PRs should be grateful and constructive
- When PRs conflict with each other, note this and suggest merge order
- When multiple PRs solve the same problem differently, flag for user to pick one
- Use Haiku model for review agents to save cost (they're doing read-only analysis)

---

## /feature-dev

_Source: `marketplaces/claude-plugins-official/plugins/feature-dev/commands/feature-dev.md`_

---
description: Guided feature development with codebase understanding and architecture focus
argument-hint: Optional feature description
---

# Feature Development

You are helping a developer implement a new feature. Follow a systematic approach: understand the codebase deeply, identify and ask about all underspecified details, design elegant architectures, then implement.

## Core Principles

- **Ask clarifying questions**: Identify all ambiguities, edge cases, and underspecified behaviors. Ask specific, concrete questions rather than making assumptions. Wait for user answers before proceeding with implementation. Ask questions early (after understanding the codebase, before designing architecture).
- **Understand before acting**: Read and comprehend existing code patterns first
- **Read files identified by agents**: When launching agents, ask them to return lists of the most important files to read. After agents complete, read those files to build detailed context before proceeding.
- **Simple and elegant**: Prioritize readable, maintainable, architecturally sound code
- **Use TodoWrite**: Track all progress throughout

---

## Phase 1: Discovery

**Goal**: Understand what needs to be built

Initial request: $ARGUMENTS

**Actions**:
1. Create todo list with all phases
2. If feature unclear, ask user for:
   - What problem are they solving?
   - What should the feature do?
   - Any constraints or requirements?
3. Summarize understanding and confirm with user

---

## Phase 2: Codebase Exploration

**Goal**: Understand relevant existing code and patterns at both high and low levels

**Actions**:
1. Launch 2-3 code-explorer agents in parallel. Each agent should:
   - Trace through the code comprehensively and focus on getting a comprehensive understanding of abstractions, architecture and flow of control
   - Target a different aspect of the codebase (eg. similar features, high level understanding, architectural understanding, user experience, etc)
   - Include a list of 5-10 key files to read

   **Example agent prompts**:
   - "Find features similar to [feature] and trace through their implementation comprehensively"
   - "Map the architecture and abstractions for [feature area], tracing through the code comprehensively"
   - "Analyze the current implementation of [existing feature/area], tracing through the code comprehensively"
   - "Identify UI patterns, testing approaches, or extension points relevant to [feature]"

2. Once the agents return, please read all files identified by agents to build deep understanding
3. Present comprehensive summary of findings and patterns discovered

---

## Phase 3: Clarifying Questions

**Goal**: Fill in gaps and resolve all ambiguities before designing

**CRITICAL**: This is one of the most important phases. DO NOT SKIP.

**Actions**:
1. Review the codebase findings and original feature request
2. Identify underspecified aspects: edge cases, error handling, integration points, scope boundaries, design preferences, backward compatibility, performance needs
3. **Present all questions to the user in a clear, organized list**
4. **Wait for answers before proceeding to architecture design**

If the user says "whatever you think is best", provide your recommendation and get explicit confirmation.

---

## Phase 4: Architecture Design

**Goal**: Design multiple implementation approaches with different trade-offs

**Actions**:
1. Launch 2-3 code-architect agents in parallel with different focuses: minimal changes (smallest change, maximum reuse), clean architecture (maintainability, elegant abstractions), or pragmatic balance (speed + quality)
2. Review all approaches and form your opinion on which fits best for this specific task (consider: small fix vs large feature, urgency, complexity, team context)
3. Present to user: brief summary of each approach, trade-offs comparison, **your recommendation with reasoning**, concrete implementation differences
4. **Ask user which approach they prefer**

---

## Phase 5: Implementation

**Goal**: Build the feature

**DO NOT START WITHOUT USER APPROVAL**

**Actions**:
1. Wait for explicit user approval
2. Read all relevant files identified in previous phases
3. Implement following chosen architecture
4. Follow codebase conventions strictly
5. Write clean, well-documented code
6. Update todos as you progress

---

## Phase 6: Quality Review

**Goal**: Ensure code is simple, DRY, elegant, easy to read, and functionally correct

**Actions**:
1. Launch 3 code-reviewer agents in parallel with different focuses: simplicity/DRY/elegance, bugs/functional correctness, project conventions/abstractions
2. Consolidate findings and identify highest severity issues that you recommend fixing
3. **Present findings to user and ask what they want to do** (fix now, fix later, or proceed as-is)
4. Address issues based on user decision

---

## Phase 7: Summary

**Goal**: Document what was accomplished

**Actions**:
1. Mark all todos complete
2. Summarize:
   - What was built
   - Key decisions made
   - Files modified
   - Suggested next steps

---

---

## /revise-claude-md

_Source: `marketplaces/claude-plugins-official/plugins/claude-md-management/commands/revise-claude-md.md`_

---
description: Update CLAUDE.md with learnings from this session
allowed-tools: Read, Edit, Glob
---

Review this session for learnings about working with Claude Code in this codebase. Update CLAUDE.md with context that would help future Claude sessions be more effective.

## Step 1: Reflect

What context was missing that would have helped Claude work more effectively?
- Bash commands that were used or discovered
- Code style patterns followed
- Testing approaches that worked
- Environment/configuration quirks
- Warnings or gotchas encountered

## Step 2: Find CLAUDE.md Files

```bash
find . -name "CLAUDE.md" -o -name ".claude.local.md" 2>/dev/null | head -20
```

Decide where each addition belongs:
- `CLAUDE.md` - Team-shared (checked into git)
- `.claude.local.md` - Personal/local only (gitignored)

## Step 3: Draft Additions

**Keep it concise** - one line per concept. CLAUDE.md is part of the prompt, so brevity matters.

Format: `<command or pattern>` - `<brief description>`

Avoid:
- Verbose explanations
- Obvious information
- One-off fixes unlikely to recur

## Step 4: Show Proposed Changes

For each addition:

```
### Update: ./CLAUDE.md

**Why:** [one-line reason]

\`\`\`diff
+ [the addition - keep it brief]
\`\`\`
```

## Step 5: Apply with Approval

Ask if the user wants to apply the changes. Only edit files they approve.

---

## /example-command

_Source: `marketplaces/claude-plugins-official/plugins/example-plugin/commands/example-command.md`_

---
description: An example slash command that demonstrates command frontmatter options (legacy format)
argument-hint: <required-arg> [optional-arg]
allowed-tools: [Read, Glob, Grep, Bash]
---

# Example Command (Legacy `commands/` Format)

> **Note:** This demonstrates the legacy `commands/*.md` layout. For new plugins, prefer the `skills/<name>/SKILL.md` directory format (see `skills/example-command/SKILL.md` in this plugin). Both are loaded identically — the only difference is file layout.

This command demonstrates slash command structure and frontmatter options.

## Arguments

The user invoked this command with: $ARGUMENTS

## Instructions

When this command is invoked:

1. Parse the arguments provided by the user
2. Perform the requested action using allowed tools
3. Report results back to the user

## Frontmatter Options Reference

Commands support these frontmatter fields:

- **description**: Short description shown in /help
- **argument-hint**: Hints for command arguments shown to user
- **allowed-tools**: Pre-approved tools for this command (reduces permission prompts)
- **model**: Override the model (e.g., "haiku", "sonnet", "opus")

## Example Usage

```
/example-command my-argument
/example-command arg1 arg2
```

---

## /code-review

_Source: `marketplaces/claude-plugins-official/plugins/code-review/commands/code-review.md`_

---
allowed-tools: Bash(gh issue view:*), Bash(gh search:*), Bash(gh issue list:*), Bash(gh pr comment:*), Bash(gh pr diff:*), Bash(gh pr view:*), Bash(gh pr list:*)
description: Code review a pull request
disable-model-invocation: false
---

Provide a code review for the given pull request.

To do this, follow these steps precisely:

1. Use a Haiku agent to check if the pull request (a) is closed, (b) is a draft, (c) does not need a code review (eg. because it is an automated pull request, or is very simple and obviously ok), or (d) already has a code review from you from earlier. If so, do not proceed.
2. Use another Haiku agent to give you a list of file paths to (but not the contents of) any relevant CLAUDE.md files from the codebase: the root CLAUDE.md file (if one exists), as well as any CLAUDE.md files in the directories whose files the pull request modified
3. Use a Haiku agent to view the pull request, and ask the agent to return a summary of the change
4. Then, launch 5 parallel Sonnet agents to independently code review the change. The agents should do the following, then return a list of issues and the reason each issue was flagged (eg. CLAUDE.md adherence, bug, historical git context, etc.):
   a. Agent #1: Audit the changes to make sure they compily with the CLAUDE.md. Note that CLAUDE.md is guidance for Claude as it writes code, so not all instructions will be applicable during code review.
   b. Agent #2: Read the file changes in the pull request, then do a shallow scan for obvious bugs. Avoid reading extra context beyond the changes, focusing just on the changes themselves. Focus on large bugs, and avoid small issues and nitpicks. Ignore likely false positives.
   c. Agent #3: Read the git blame and history of the code modified, to identify any bugs in light of that historical context
   d. Agent #4: Read previous pull requests that touched these files, and check for any comments on those pull requests that may also apply to the current pull request.
   e. Agent #5: Read code comments in the modified files, and make sure the changes in the pull request comply with any guidance in the comments.
5. For each issue found in #4, launch a parallel Haiku agent that takes the PR, issue description, and list of CLAUDE.md files (from step 2), and returns a score to indicate the agent's level of confidence for whether the issue is real or false positive. To do that, the agent should score each issue on a scale from 0-100, indicating its level of confidence. For issues that were flagged due to CLAUDE.md instructions, the agent should double check that the CLAUDE.md actually calls out that issue specifically. The scale is (give this rubric to the agent verbatim):
   a. 0: Not confident at all. This is a false positive that doesn't stand up to light scrutiny, or is a pre-existing issue.
   b. 25: Somewhat confident. This might be a real issue, but may also be a false positive. The agent wasn't able to verify that it's a real issue. If the issue is stylistic, it is one that was not explicitly called out in the relevant CLAUDE.md.
   c. 50: Moderately confident. The agent was able to verify this is a real issue, but it might be a nitpick or not happen very often in practice. Relative to the rest of the PR, it's not very important.
   d. 75: Highly confident. The agent double checked the issue, and verified that it is very likely it is a real issue that will be hit in practice. The existing approach in the PR is insufficient. The issue is very important and will directly impact the code's functionality, or it is an issue that is directly mentioned in the relevant CLAUDE.md.
   e. 100: Absolutely certain. The agent double checked the issue, and confirmed that it is definitely a real issue, that will happen frequently in practice. The evidence directly confirms this.
6. Filter out any issues with a score less than 80. If there are no issues that meet this criteria, do not proceed.
7. Use a Haiku agent to repeat the eligibility check from #1, to make sure that the pull request is still eligible for code review.
8. Finally, use the gh bash command to comment back on the pull request with the result. When writing your comment, keep in mind to:
   a. Keep your output brief
   b. Avoid emojis
   c. Link and cite relevant code, files, and URLs

Examples of false positives, for steps 4 and 5:

- Pre-existing issues
- Something that looks like a bug but is not actually a bug
- Pedantic nitpicks that a senior engineer wouldn't call out
- Issues that a linter, typechecker, or compiler would catch (eg. missing or incorrect imports, type errors, broken tests, formatting issues, pedantic style issues like newlines). No need to run these build steps yourself -- it is safe to assume that they will be run separately as part of CI.
- General code quality issues (eg. lack of test coverage, general security issues, poor documentation), unless explicitly required in CLAUDE.md
- Issues that are called out in CLAUDE.md, but explicitly silenced in the code (eg. due to a lint ignore comment)
- Changes in functionality that are likely intentional or are directly related to the broader change
- Real issues, but on lines that the user did not modify in their pull request

Notes:

- Do not check build signal or attempt to build or typecheck the app. These will run separately, and are not relevant to your code review.
- Use `gh` to interact with Github (eg. to fetch a pull request, or to create inline comments), rather than web fetch
- Make a todo list first
- You must cite and link each bug (eg. if referring to a CLAUDE.md, you must link it)
- For your final comment, follow the following format precisely (assuming for this example that you found 3 issues):

---

### Code review

Found 3 issues:

1. <brief description of bug> (CLAUDE.md says "<...>")

<link to file and line with full sha1 + line range for context, note that you MUST provide the full sha and not use bash here, eg. https://github.com/anthropics/claude-code/blob/1d54823877c4de72b2316a64032a54afc404e619/README.md#L13-L17>

2. <brief description of bug> (some/other/CLAUDE.md says "<...>")

<link to file and line with full sha1 + line range for context>

3. <brief description of bug> (bug due to <file and code snippet>)

<link to file and line with full sha1 + line range for context>

🤖 Generated with [Claude Code](https://claude.ai/code)

<sub>- If this code review was useful, please react with 👍. Otherwise, react with 👎.</sub>

---

- Or, if you found no issues:

---

### Code review

No issues found. Checked for bugs and CLAUDE.md compliance.

🤖 Generated with [Claude Code](https://claude.ai/code)

- When linking to code, follow the following format precisely, otherwise the Markdown preview won't render correctly: https://github.com/anthropics/claude-cli-internal/blob/c21d3c10bc8e898b7ac1a2d745bdc9bc4e423afe/package.json#L10-L15
  - Requires full git sha
  - You must provide the full sha. Commands like `https://github.com/owner/repo/blob/$(git rev-parse HEAD)/foo/bar` will not work, since your comment will be directly rendered in Markdown.
  - Repo name must match the repo you're code reviewing
  - # sign after the file name
  - Line range format is L[start]-L[end]
  - Provide at least 1 line of context before and after, centered on the line you are commenting about (eg. if you are commenting about lines 5-6, you should link to `L4-7`)

---

## /create-plugin

_Source: `marketplaces/claude-plugins-official/plugins/plugin-dev/commands/create-plugin.md`_

---
description: Guided end-to-end plugin creation workflow with component design, implementation, and validation
argument-hint: Optional plugin description
allowed-tools:
  [
    "Read",
    "Write",
    "Grep",
    "Glob",
    "Bash",
    "TodoWrite",
    "AskUserQuestion",
    "Skill",
    "Task",
  ]
---

# Plugin Creation Workflow

Guide the user through creating a complete, high-quality Claude Code plugin from initial concept to tested implementation. Follow a systematic approach: understand requirements, design components, clarify details, implement following best practices, validate, and test.

## Core Principles

- **Ask clarifying questions**: Identify all ambiguities about plugin purpose, triggering, scope, and components. Ask specific, concrete questions rather than making assumptions. Wait for user answers before proceeding with implementation.
- **Load relevant skills**: Use the Skill tool to load plugin-dev skills when needed (plugin-structure, hook-development, agent-development, etc.)
- **Use specialized agents**: Leverage agent-creator, plugin-validator, and skill-reviewer agents for AI-assisted development
- **Follow best practices**: Apply patterns from plugin-dev's own implementation
- **Progressive disclosure**: Create lean skills with references/examples
- **Use TodoWrite**: Track all progress throughout all phases

**Initial request:** $ARGUMENTS

---

## Phase 1: Discovery

**Goal**: Understand what plugin needs to be built and what problem it solves

**Actions**:

1. Create todo list with all 7 phases
2. If plugin purpose is clear from arguments:
   - Summarize understanding
   - Identify plugin type (integration, workflow, analysis, toolkit, etc.)
3. If plugin purpose is unclear, ask user:
   - What problem does this plugin solve?
   - Who will use it and when?
   - What should it do?
   - Any similar plugins to reference?
4. Summarize understanding and confirm with user before proceeding

**Output**: Clear statement of plugin purpose and target users

---

## Phase 2: Component Planning

**Goal**: Determine what plugin components are needed

**MUST load plugin-structure skill** using Skill tool before this phase.

**Actions**:

1. Load plugin-structure skill to understand component types
2. Analyze plugin requirements and determine needed components:
   - **Skills**: Specialized knowledge OR user-initiated actions (deploy, configure, analyze). Skills are the preferred format for both — see note below.
   - **Agents**: Autonomous tasks? (validation, generation, analysis)
   - **Hooks**: Event-driven automation? (validation, notifications)
   - **MCP**: External service integration? (databases, APIs)
   - **Settings**: User configuration? (.local.md files)

   > **Note:** The `commands/` directory is a legacy format. For new plugins, user-invoked slash commands should be created as skills in `skills/<name>/SKILL.md`. Both are loaded identically — the only difference is file layout. `commands/` remains an acceptable legacy alternative.

3. For each component type needed, identify:
   - How many of each type
   - What each one does
   - Rough triggering/usage patterns
4. Present component plan to user as table:
   ```
   | Component Type | Count | Purpose |
   |----------------|-------|---------|
   | Skills         | 5     | Hook patterns, MCP usage, deploy, configure, validate |
   | Agents         | 1     | Autonomous validation |
   | Hooks          | 0     | Not needed |
   | MCP            | 1     | Database integration |
   ```
5. Get user confirmation or adjustments

**Output**: Confirmed list of components to create

---

## Phase 3: Detailed Design & Clarifying Questions

**Goal**: Specify each component in detail and resolve all ambiguities

**CRITICAL**: This is one of the most important phases. DO NOT SKIP.

**Actions**:

1. For each component in the plan, identify underspecified aspects:
   - **Skills**: What triggers them? What knowledge do they provide? How detailed? For user-invoked skills: what arguments, what tools, interactive or automated?
   - **Agents**: When to trigger (proactive/reactive)? What tools? Output format?
   - **Hooks**: Which events? Prompt or command based? Validation criteria?
   - **MCP**: What server type? Authentication? Which tools?
   - **Settings**: What fields? Required vs optional? Defaults?

2. **Present all questions to user in organized sections** (one section per component type)

3. **Wait for answers before proceeding to implementation**

4. If user says "whatever you think is best", provide specific recommendations and get explicit confirmation

**Example questions for a skill**:

- What specific user queries should trigger this skill?
- Should it include utility scripts? What functionality?
- How detailed should the core SKILL.md be vs references/?
- Any real-world examples to include?

**Example questions for an agent**:

- Should this agent trigger proactively after certain actions, or only when explicitly requested?
- What tools does it need (Read, Write, Bash, etc.)?
- What should the output format be?
- Any specific quality standards to enforce?

**Output**: Detailed specification for each component

---

## Phase 4: Plugin Structure Creation

**Goal**: Create plugin directory structure and manifest

**Actions**:

1. Determine plugin name (kebab-case, descriptive)
2. Choose plugin location:
   - Ask user: "Where should I create the plugin?"
   - Offer options: current directory, ../new-plugin-name, custom path
3. Create directory structure using bash:
   ```bash
   mkdir -p plugin-name/.claude-plugin
   mkdir -p plugin-name/skills/<skill-name>   # one dir per skill, each with a SKILL.md
   mkdir -p plugin-name/agents                # if needed
   mkdir -p plugin-name/hooks                 # if needed
   # Note: plugin-name/commands/ is a legacy alternative to skills/ — prefer skills/
   ```
4. Create plugin.json manifest using Write tool:
   ```json
   {
     "name": "plugin-name",
     "version": "0.1.0",
     "description": "[brief description]",
     "author": {
       "name": "[author from user or default]",
       "email": "[email or default]"
     }
   }
   ```
5. Create README.md template
6. Create .gitignore if needed (for .claude/\*.local.md, etc.)
7. Initialize git repo if creating new directory

**Output**: Plugin directory structure created and ready for components

---

## Phase 5: Component Implementation

**Goal**: Create each component following best practices

**LOAD RELEVANT SKILLS** before implementing each component type:

- Skills: Load skill-development skill
- Legacy `commands/` format (only if user explicitly requests): Load command-development skill
- Agents: Load agent-development skill
- Hooks: Load hook-development skill
- MCP: Load mcp-integration skill
- Settings: Load plugin-settings skill

**Actions for each component**:

### For Skills:

1. Load skill-development skill using Skill tool
2. For each skill:
   - Ask user for concrete usage examples (or use from Phase 3)
   - Plan resources (scripts/, references/, examples/)
   - Create skill directory: `skills/<skill-name>/`
   - Write `SKILL.md` with:
     - Third-person description with specific trigger phrases
     - Lean body (1,500-2,000 words) in imperative form
     - References to supporting files
   - For user-invoked skills (slash commands): include `description`, `argument-hint`, and `allowed-tools` frontmatter; write instructions FOR Claude (not TO user)
   - Create reference files for detailed content
   - Create example files for working code
   - Create utility scripts if needed
3. Use skill-reviewer agent to validate each skill

### For legacy `commands/` format (only if user explicitly requests):

> Prefer `skills/<name>/SKILL.md` for new plugins. Use `commands/` only when maintaining an existing plugin that already uses this layout.

1. Load command-development skill using Skill tool
2. For each command:
   - Write command markdown with frontmatter
   - Include clear description and argument-hint
   - Specify allowed-tools (minimal necessary)
   - Write instructions FOR Claude (not TO user)
   - Provide usage examples and tips
   - Reference relevant skills if applicable

### For Agents:

1. Load agent-development skill using Skill tool
2. For each agent, use agent-creator agent:
   - Provide description of what agent should do
   - Agent-creator generates: identifier, whenToUse with examples, systemPrompt
   - Create agent markdown file with frontmatter and system prompt
   - Add appropriate model, color, and tools
   - Validate with validate-agent.sh script

### For Hooks:

1. Load hook-development skill using Skill tool
2. For each hook:
   - Create hooks/hooks.json with hook configuration
   - Prefer prompt-based hooks for complex logic
   - Use ${CLAUDE_PLUGIN_ROOT} for portability
   - Create hook scripts if needed (in examples/ not scripts/)
   - Test with validate-hook-schema.sh and test-hook.sh utilities

### For MCP:

1. Load mcp-integration skill using Skill tool
2. Create .mcp.json configuration with:
   - Server type (stdio for local, SSE for hosted)
   - Command and args (with ${CLAUDE_PLUGIN_ROOT})
   - extensionToLanguage mapping if LSP
   - Environment variables as needed
3. Document required env vars in README
4. Provide setup instructions

### For Settings:

1. Load plugin-settings skill using Skill tool
2. Create settings template in README
3. Create example .claude/plugin-name.local.md file (as documentation)
4. Implement settings reading in hooks/commands as needed
5. Add to .gitignore: `.claude/*.local.md`

**Progress tracking**: Update todos as each component is completed

**Output**: All plugin components implemented

---

## Phase 6: Validation & Quality Check

**Goal**: Ensure plugin meets quality standards and works correctly

**Actions**:

1. **Run plugin-validator agent**:
   - Use plugin-validator agent to comprehensively validate plugin
   - Check: manifest, structure, naming, components, security
   - Review validation report

2. **Fix critical issues**:
   - Address any critical errors from validation
   - Fix any warnings that indicate real problems

3. **Review with skill-reviewer** (if plugin has skills):
   - For each skill, use skill-reviewer agent
   - Check description quality, progressive disclosure, writing style
   - Apply recommendations

4. **Test agent triggering** (if plugin has agents):
   - For each agent, verify <example> blocks are clear
   - Check triggering conditions are specific
   - Run validate-agent.sh on agent files

5. **Test hook configuration** (if plugin has hooks):
   - Run validate-hook-schema.sh on hooks/hooks.json
   - Test hook scripts with test-hook.sh
   - Verify ${CLAUDE_PLUGIN_ROOT} usage

6. **Present findings**:
   - Summary of validation results
   - Any remaining issues
   - Overall quality assessment

7. **Ask user**: "Validation complete. Issues found: [count critical], [count warnings]. Would you like me to fix them now, or proceed to testing?"

**Output**: Plugin validated and ready for testing

---

## Phase 7: Testing & Verification

**Goal**: Test that plugin works correctly in Claude Code

**Actions**:

1. **Installation instructions**:
   - Show user how to test locally:
     ```bash
     cc --plugin-dir /path/to/plugin-name
     ```
   - Or copy to `.claude-plugin/` for project testing

2. **Verification checklist** for user to perform:
   - [ ] Skills load when triggered (ask questions with trigger phrases)
   - [ ] User-invoked skills appear in `/help` and execute correctly
   - [ ] Agents trigger on appropriate scenarios
   - [ ] Hooks activate on events (if applicable)
   - [ ] MCP servers connect (if applicable)
   - [ ] Settings files work (if applicable)

3. **Testing recommendations**:
   - For skills: Ask questions using trigger phrases from descriptions
   - For user-invoked skills: Run `/plugin-name:skill-name` with various arguments
   - For agents: Create scenarios matching agent examples
   - For hooks: Use `claude --debug` to see hook execution
   - For MCP: Use `/mcp` to verify servers and tools

4. **Ask user**: "I've prepared the plugin for testing. Would you like me to guide you through testing each component, or do you want to test it yourself?"

5. **If user wants guidance**, walk through testing each component with specific test cases

**Output**: Plugin tested and verified working

---

## Phase 8: Documentation & Next Steps

**Goal**: Ensure plugin is well-documented and ready for distribution

**Actions**:

1. **Verify README completeness**:
   - Check README has: overview, features, installation, prerequisites, usage
   - For MCP plugins: Document required environment variables
   - For hook plugins: Explain hook activation
   - For settings: Provide configuration templates

2. **Add marketplace entry** (if publishing):
   - Show user how to add to marketplace.json
   - Help draft marketplace description
   - Suggest category and tags

3. **Create summary**:
   - Mark all todos complete
   - List what was created:
     - Plugin name and purpose
     - Components created (X skills, Y agents, etc.)
     - Key files and their purposes
     - Total file count and structure
   - Next steps:
     - Testing recommendations
     - Publishing to marketplace (if desired)
     - Iteration based on usage

4. **Suggest improvements** (optional):
   - Additional components that could enhance plugin
   - Integration opportunities
   - Testing strategies

**Output**: Complete, documented plugin ready for use or publication

---

## Important Notes

### Throughout All Phases

- **Use TodoWrite** to track progress at every phase
- **Load skills with Skill tool** when working on specific component types
- **Use specialized agents** (agent-creator, plugin-validator, skill-reviewer)
- **Ask for user confirmation** at key decision points
- **Follow plugin-dev's own patterns** as reference examples
- **Apply best practices**:
  - Third-person descriptions for skills
  - Imperative form in skill bodies
  - Skill instructions written FOR Claude (not TO user)
  - Strong trigger phrases
  - ${CLAUDE_PLUGIN_ROOT} for portability
  - Progressive disclosure
  - Security-first (HTTPS, no hardcoded credentials)

### Key Decision Points (Wait for User)

1. After Phase 1: Confirm plugin purpose
2. After Phase 2: Approve component plan
3. After Phase 3: Proceed to implementation
4. After Phase 6: Fix issues or proceed
5. After Phase 7: Continue to documentation

### Skills to Load by Phase

- **Phase 2**: plugin-structure
- **Phase 5**: skill-development, agent-development, hook-development, mcp-integration, plugin-settings (as needed); command-development only for legacy `commands/` layout
- **Phase 6**: (agents will use skills automatically)

### Quality Standards

Every component must meet these standards:

- ✅ Follows plugin-dev's proven patterns
- ✅ Uses correct naming conventions
- ✅ Has strong trigger conditions (skills/agents)
- ✅ Includes working examples
- ✅ Properly documented
- ✅ Validated with utilities
- ✅ Tested in Claude Code

---

## Example Workflow

### User Request

"Create a plugin for managing database migrations"

### Phase 1: Discovery

- Understand: Migration management, database schema versioning
- Confirm: User wants to create, run, rollback migrations

### Phase 2: Component Planning

- Skills: 4 (migration best practices, create-migration, run-migrations, rollback)
- Agents: 1 (migration-validator)
- MCP: 1 (database connection)

### Phase 3: Clarifying Questions

- Which databases? (PostgreSQL, MySQL, etc.)
- Migration file format? (SQL, code-based?)
- Should agent validate before applying?
- What MCP tools needed? (query, execute, schema)

### Phase 4-8: Implementation, Validation, Testing, Documentation

---

**Begin with Phase 1: Discovery**

---

## /review-pr

_Source: `marketplaces/claude-plugins-official/plugins/pr-review-toolkit/commands/review-pr.md`_

---
description: "Comprehensive PR review using specialized agents"
argument-hint: "[review-aspects]"
allowed-tools: ["Bash", "Glob", "Grep", "Read", "Task"]
---

# Comprehensive PR Review

Run a comprehensive pull request review using multiple specialized agents, each focusing on a different aspect of code quality.

**Review Aspects (optional):** "$ARGUMENTS"

## Review Workflow:

1. **Determine Review Scope**
   - Check git status to identify changed files
   - Parse arguments to see if user requested specific review aspects
   - Default: Run all applicable reviews

2. **Available Review Aspects:**

   - **comments** - Analyze code comment accuracy and maintainability
   - **tests** - Review test coverage quality and completeness
   - **errors** - Check error handling for silent failures
   - **types** - Analyze type design and invariants (if new types added)
   - **code** - General code review for project guidelines
   - **simplify** - Simplify code for clarity and maintainability
   - **all** - Run all applicable reviews (default)

3. **Identify Changed Files**
   - Run `git diff --name-only` to see modified files
   - Check if PR already exists: `gh pr view`
   - Identify file types and what reviews apply

4. **Determine Applicable Reviews**

   Based on changes:
   - **Always applicable**: code-reviewer (general quality)
   - **If test files changed**: pr-test-analyzer
   - **If comments/docs added**: comment-analyzer
   - **If error handling changed**: silent-failure-hunter
   - **If types added/modified**: type-design-analyzer
   - **After passing review**: code-simplifier (polish and refine)

5. **Launch Review Agents**

   **Sequential approach** (one at a time):
   - Easier to understand and act on
   - Each report is complete before next
   - Good for interactive review

   **Parallel approach** (user can request):
   - Launch all agents simultaneously
   - Faster for comprehensive review
   - Results come back together

6. **Aggregate Results**

   After agents complete, summarize:
   - **Critical Issues** (must fix before merge)
   - **Important Issues** (should fix)
   - **Suggestions** (nice to have)
   - **Positive Observations** (what's good)

7. **Provide Action Plan**

   Organize findings:
   ```markdown
   # PR Review Summary

   ## Critical Issues (X found)
   - [agent-name]: Issue description [file:line]

   ## Important Issues (X found)
   - [agent-name]: Issue description [file:line]

   ## Suggestions (X found)
   - [agent-name]: Suggestion [file:line]

   ## Strengths
   - What's well-done in this PR

   ## Recommended Action
   1. Fix critical issues first
   2. Address important issues
   3. Consider suggestions
   4. Re-run review after fixes
   ```

## Usage Examples:

**Full review (default):**
```
/pr-review-toolkit:review-pr
```

**Specific aspects:**
```
/pr-review-toolkit:review-pr tests errors
# Reviews only test coverage and error handling

/pr-review-toolkit:review-pr comments
# Reviews only code comments

/pr-review-toolkit:review-pr simplify
# Simplifies code after passing review
```

**Parallel review:**
```
/pr-review-toolkit:review-pr all parallel
# Launches all agents in parallel
```

## Agent Descriptions:

**comment-analyzer**:
- Verifies comment accuracy vs code
- Identifies comment rot
- Checks documentation completeness

**pr-test-analyzer**:
- Reviews behavioral test coverage
- Identifies critical gaps
- Evaluates test quality

**silent-failure-hunter**:
- Finds silent failures
- Reviews catch blocks
- Checks error logging

**type-design-analyzer**:
- Analyzes type encapsulation
- Reviews invariant expression
- Rates type design quality

**code-reviewer**:
- Checks CLAUDE.md compliance
- Detects bugs and issues
- Reviews general code quality

**code-simplifier**:
- Simplifies complex code
- Improves clarity and readability
- Applies project standards
- Preserves functionality

## Tips:

- **Run early**: Before creating PR, not after
- **Focus on changes**: Agents analyze git diff by default
- **Address critical first**: Fix high-priority issues before lower priority
- **Re-run after fixes**: Verify issues are resolved
- **Use specific reviews**: Target specific aspects when you know the concern

## Workflow Integration:

**Before committing:**
```
1. Write code
2. Run: /pr-review-toolkit:review-pr code errors
3. Fix any critical issues
4. Commit
```

**Before creating PR:**
```
1. Stage all changes
2. Run: /pr-review-toolkit:review-pr all
3. Address all critical and important issues
4. Run specific reviews again to verify
5. Create PR
```

**After PR feedback:**
```
1. Make requested changes
2. Run targeted reviews based on feedback
3. Verify issues are resolved
4. Push updates
```

## Notes:

- Agents run autonomously and return detailed reports
- Each agent focuses on its specialty for deep analysis
- Results are actionable with specific file:line references
- Agents use appropriate models for their complexity
- All agents available in `/agents` list

---

## /new-sdk-app

_Source: `marketplaces/claude-plugins-official/plugins/agent-sdk-dev/commands/new-sdk-app.md`_

---
description: Create and setup a new Claude Agent SDK application
argument-hint: [project-name]
---

You are tasked with helping the user create a new Claude Agent SDK application. Follow these steps carefully:

## Reference Documentation

Before starting, review the official documentation to ensure you provide accurate and up-to-date guidance. Use WebFetch to read these pages:

1. **Start with the overview**: https://docs.claude.com/en/api/agent-sdk/overview
2. **Based on the user's language choice, read the appropriate SDK reference**:
   - TypeScript: https://docs.claude.com/en/api/agent-sdk/typescript
   - Python: https://docs.claude.com/en/api/agent-sdk/python
3. **Read relevant guides mentioned in the overview** such as:
   - Streaming vs Single Mode
   - Permissions
   - Custom Tools
   - MCP integration
   - Subagents
   - Sessions
   - Any other relevant guides based on the user's needs

**IMPORTANT**: Always check for and use the latest versions of packages. Use WebSearch or WebFetch to verify current versions before installation.

## Gather Requirements

IMPORTANT: Ask these questions one at a time. Wait for the user's response before asking the next question. This makes it easier for the user to respond.

Ask the questions in this order (skip any that the user has already provided via arguments):

1. **Language** (ask first): "Would you like to use TypeScript or Python?"

   - Wait for response before continuing

2. **Project name** (ask second): "What would you like to name your project?"

   - If $ARGUMENTS is provided, use that as the project name and skip this question
   - Wait for response before continuing

3. **Agent type** (ask third, but skip if #2 was sufficiently detailed): "What kind of agent are you building? Some examples:

   - Coding agent (SRE, security review, code review)
   - Business agent (customer support, content creation)
   - Custom agent (describe your use case)"
   - Wait for response before continuing

4. **Starting point** (ask fourth): "Would you like:

   - A minimal 'Hello World' example to start
   - A basic agent with common features
   - A specific example based on your use case"
   - Wait for response before continuing

5. **Tooling choice** (ask fifth): Let the user know what tools you'll use, and confirm with them that these are the tools they want to use (for example, they may prefer pnpm or bun over npm). Respect the user's preferences when executing on the requirements.

After all questions are answered, proceed to create the setup plan.

## Setup Plan

Based on the user's answers, create a plan that includes:

1. **Project initialization**:

   - Create project directory (if it doesn't exist)
   - Initialize package manager:
     - TypeScript: `npm init -y` and setup `package.json` with type: "module" and scripts (include a "typecheck" script)
     - Python: Create `requirements.txt` or use `poetry init`
   - Add necessary configuration files:
     - TypeScript: Create `tsconfig.json` with proper settings for the SDK
     - Python: Optionally create config files if needed

2. **Check for Latest Versions**:

   - BEFORE installing, use WebSearch or check npm/PyPI to find the latest version
   - For TypeScript: Check https://www.npmjs.com/package/@anthropic-ai/claude-agent-sdk
   - For Python: Check https://pypi.org/project/claude-agent-sdk/
   - Inform the user which version you're installing

3. **SDK Installation**:

   - TypeScript: `npm install @anthropic-ai/claude-agent-sdk@latest` (or specify latest version)
   - Python: `pip install claude-agent-sdk` (pip installs latest by default)
   - After installation, verify the installed version:
     - TypeScript: Check package.json or run `npm list @anthropic-ai/claude-agent-sdk`
     - Python: Run `pip show claude-agent-sdk`

4. **Create starter files**:

   - TypeScript: Create an `index.ts` or `src/index.ts` with a basic query example
   - Python: Create a `main.py` with a basic query example
   - Include proper imports and basic error handling
   - Use modern, up-to-date syntax and patterns from the latest SDK version

5. **Environment setup**:

   - Create a `.env.example` file with `ANTHROPIC_API_KEY=your_api_key_here`
   - Add `.env` to `.gitignore`
   - Explain how to get an API key from https://console.anthropic.com/

6. **Optional: Create .claude directory structure**:
   - Offer to create `.claude/` directory for agents, commands, and settings
   - Ask if they want any example subagents or slash commands

## Implementation

After gathering requirements and getting user confirmation on the plan:

1. Check for latest package versions using WebSearch or WebFetch
2. Execute the setup steps
3. Create all necessary files
4. Install dependencies (always use latest stable versions)
5. Verify installed versions and inform the user
6. Create a working example based on their agent type
7. Add helpful comments in the code explaining what each part does
8. **VERIFY THE CODE WORKS BEFORE FINISHING**:
   - For TypeScript:
     - Run `npx tsc --noEmit` to check for type errors
     - Fix ALL type errors until types pass completely
     - Ensure imports and types are correct
     - Only proceed when type checking passes with no errors
   - For Python:
     - Verify imports are correct
     - Check for basic syntax errors
   - **DO NOT consider the setup complete until the code verifies successfully**

## Verification

After all files are created and dependencies are installed, use the appropriate verifier agent to validate that the Agent SDK application is properly configured and ready for use:

1. **For TypeScript projects**: Launch the **agent-sdk-verifier-ts** agent to validate the setup
2. **For Python projects**: Launch the **agent-sdk-verifier-py** agent to validate the setup
3. The agent will check SDK usage, configuration, functionality, and adherence to official documentation
4. Review the verification report and address any issues

## Getting Started Guide

Once setup is complete and verified, provide the user with:

1. **Next steps**:

   - How to set their API key
   - How to run their agent:
     - TypeScript: `npm start` or `node --loader ts-node/esm index.ts`
     - Python: `python main.py`

2. **Useful resources**:

   - Link to TypeScript SDK reference: https://docs.claude.com/en/api/agent-sdk/typescript
   - Link to Python SDK reference: https://docs.claude.com/en/api/agent-sdk/python
   - Explain key concepts: system prompts, permissions, tools, MCP servers

3. **Common next steps**:
   - How to customize the system prompt
   - How to add custom tools via MCP
   - How to configure permissions
   - How to create subagents

## Important Notes

- **ALWAYS USE LATEST VERSIONS**: Before installing any packages, check for the latest versions using WebSearch or by checking npm/PyPI directly
- **VERIFY CODE RUNS CORRECTLY**:
  - For TypeScript: Run `npx tsc --noEmit` and fix ALL type errors before finishing
  - For Python: Verify syntax and imports are correct
  - Do NOT consider the task complete until the code passes verification
- Verify the installed version after installation and inform the user
- Check the official documentation for any version-specific requirements (Node.js version, Python version, etc.)
- Always check if directories/files already exist before creating them
- Use the user's preferred package manager (npm, yarn, pnpm for TypeScript; pip, poetry for Python)
- Ensure all code examples are functional and include proper error handling
- Use modern syntax and patterns that are compatible with the latest SDK version
- Make the experience interactive and educational
- **ASK QUESTIONS ONE AT A TIME** - Do not ask multiple questions in a single response

Begin by asking the FIRST requirement question only. Wait for the user's answer before proceeding to the next question.

---

## /commit-push-pr

_Source: `marketplaces/claude-plugins-official/plugins/commit-commands/commands/commit-push-pr.md`_

---
allowed-tools: Bash(git checkout --branch:*), Bash(git add:*), Bash(git status:*), Bash(git push:*), Bash(git commit:*), Bash(gh pr create:*)
description: Commit, push, and open a PR
---

## Context

- Current git status: !`git status`
- Current git diff (staged and unstaged changes): !`git diff HEAD`
- Current branch: !`git branch --show-current`

## Your task

Based on the above changes:

1. Create a new branch if on main
2. Create a single commit with an appropriate message
3. Push the branch to origin
4. Create a pull request using `gh pr create`
5. You have the capability to call multiple tools in a single response. You MUST do all of the above in a single message. Do not use any other tools or do anything else. Do not send any other text or messages besides these tool calls.

---

## /clean_gone

_Source: `marketplaces/claude-plugins-official/plugins/commit-commands/commands/clean_gone.md`_

---
description: Cleans up all git branches marked as [gone] (branches that have been deleted on the remote but still exist locally), including removing associated worktrees.
---

## Your Task

You need to execute the following bash commands to clean up stale local branches that have been deleted from the remote repository.

## Commands to Execute

1. **First, list branches to identify any with [gone] status**
   Execute this command:
   ```bash
   git branch -v
   ```
   
   Note: Branches with a '+' prefix have associated worktrees and must have their worktrees removed before deletion.

2. **Next, identify worktrees that need to be removed for [gone] branches**
   Execute this command:
   ```bash
   git worktree list
   ```

3. **Finally, remove worktrees and delete [gone] branches (handles both regular and worktree branches)**
   Execute this command:
   ```bash
   # Process all [gone] branches, removing '+' prefix if present
   git branch -v | grep '\[gone\]' | sed 's/^[+* ]//' | awk '{print $1}' | while read branch; do
     echo "Processing branch: $branch"
     # Find and remove worktree if it exists
     worktree=$(git worktree list | grep "\\[$branch\\]" | awk '{print $1}')
     if [ ! -z "$worktree" ] && [ "$worktree" != "$(git rev-parse --show-toplevel)" ]; then
       echo "  Removing worktree: $worktree"
       git worktree remove --force "$worktree"
     fi
     # Delete the branch
     echo "  Deleting branch: $branch"
     git branch -D "$branch"
   done
   ```

## Expected Behavior

After executing these commands, you will:

- See a list of all local branches with their status
- Identify and remove any worktrees associated with [gone] branches
- Delete all branches marked as [gone]
- Provide feedback on which worktrees and branches were removed

If no branches are marked as [gone], report that no cleanup was needed.


---

## /commit

_Source: `marketplaces/claude-plugins-official/plugins/commit-commands/commands/commit.md`_

---
allowed-tools: Bash(git add:*), Bash(git status:*), Bash(git commit:*)
description: Create a git commit
---

## Context

- Current git status: !`git status`
- Current git diff (staged and unstaged changes): !`git diff HEAD`
- Current branch: !`git branch --show-current`
- Recent commits: !`git log --oneline -10`

## Your task

Based on the above changes, create a single git commit.

You have the capability to call multiple tools in a single response. Stage and create the commit using a single message. Do not use any other tools or do anything else. Do not send any other text or messages besides these tool calls.

---

## /help

_Source: `marketplaces/claude-plugins-official/plugins/ralph-loop/commands/help.md`_

---
description: "Explain Ralph Loop plugin and available commands"
---

# Ralph Loop Plugin Help

Please explain the following to the user:

## What is Ralph Loop?

Ralph Loop implements the Ralph Wiggum technique - an iterative development methodology based on continuous AI loops, pioneered by Geoffrey Huntley.

**Core concept:**
```bash
while :; do
  cat PROMPT.md | claude-code --continue
done
```

The same prompt is fed to Claude repeatedly. The "self-referential" aspect comes from Claude seeing its own previous work in the files and git history, not from feeding output back as input.

**Each iteration:**
1. Claude receives the SAME prompt
2. Works on the task, modifying files
3. Tries to exit
4. Stop hook intercepts and feeds the same prompt again
5. Claude sees its previous work in the files
6. Iteratively improves until completion

The technique is described as "deterministically bad in an undeterministic world" - failures are predictable, enabling systematic improvement through prompt tuning.

## Available Commands

### /ralph-loop <PROMPT> [OPTIONS]

Start a Ralph loop in your current session.

**Usage:**
```
/ralph-loop "Refactor the cache layer" --max-iterations 20
/ralph-loop "Add tests" --completion-promise "TESTS COMPLETE"
```

**Options:**
- `--max-iterations <n>` - Max iterations before auto-stop
- `--completion-promise <text>` - Promise phrase to signal completion

**How it works:**
1. Creates `.claude/.ralph-loop.local.md` state file
2. You work on the task
3. When you try to exit, stop hook intercepts
4. Same prompt fed back
5. You see your previous work
6. Continues until promise detected or max iterations

---

### /cancel-ralph

Cancel an active Ralph loop (removes the loop state file).

**Usage:**
```
/cancel-ralph
```

**How it works:**
- Checks for active loop state file
- Removes `.claude/.ralph-loop.local.md`
- Reports cancellation with iteration count

---

## Key Concepts

### Completion Promises

To signal completion, Claude must output a `<promise>` tag:

```
<promise>TASK COMPLETE</promise>
```

The stop hook looks for this specific tag. Without it (or `--max-iterations`), Ralph runs infinitely.

### Self-Reference Mechanism

The "loop" doesn't mean Claude talks to itself. It means:
- Same prompt repeated
- Claude's work persists in files
- Each iteration sees previous attempts
- Builds incrementally toward goal

## Example

### Interactive Bug Fix

```
/ralph-loop "Fix the token refresh logic in auth.ts. Output <promise>FIXED</promise> when all tests pass." --completion-promise "FIXED" --max-iterations 10
```

You'll see Ralph:
- Attempt fixes
- Run tests
- See failures
- Iterate on solution
- In your current session

## When to Use Ralph

**Good for:**
- Well-defined tasks with clear success criteria
- Tasks requiring iteration and refinement
- Iterative development with self-correction
- Greenfield projects

**Not good for:**
- Tasks requiring human judgment or design decisions
- One-shot operations
- Tasks with unclear success criteria
- Debugging production issues (use targeted debugging instead)

## Learn More

- Original technique: https://ghuntley.com/ralph/
- Ralph Orchestrator: https://github.com/mikeyobrien/ralph-orchestrator

---

## /cancel-ralph

_Source: `marketplaces/claude-plugins-official/plugins/ralph-loop/commands/cancel-ralph.md`_

---
description: "Cancel active Ralph Loop"
allowed-tools: ["Bash(test -f .claude/ralph-loop.local.md:*)", "Bash(rm .claude/ralph-loop.local.md)", "Read(.claude/ralph-loop.local.md)"]
hide-from-slash-command-tool: "true"
---

# Cancel Ralph

To cancel the Ralph loop:

1. Check if `.claude/ralph-loop.local.md` exists using Bash: `test -f .claude/ralph-loop.local.md && echo "EXISTS" || echo "NOT_FOUND"`

2. **If NOT_FOUND**: Say "No active Ralph loop found."

3. **If EXISTS**:
   - Read `.claude/ralph-loop.local.md` to get the current iteration number from the `iteration:` field
   - Remove the file using Bash: `rm .claude/ralph-loop.local.md`
   - Report: "Cancelled Ralph loop (was at iteration N)" where N is the iteration value

---

## /ralph-loop

_Source: `marketplaces/claude-plugins-official/plugins/ralph-loop/commands/ralph-loop.md`_

---
description: "Start Ralph Loop in current session"
argument-hint: "PROMPT [--max-iterations N] [--completion-promise TEXT]"
allowed-tools: ["Bash(${CLAUDE_PLUGIN_ROOT}/scripts/setup-ralph-loop.sh:*)"]
hide-from-slash-command-tool: "true"
---

# Ralph Loop Command

Execute the setup script to initialize the Ralph loop:

```!
"${CLAUDE_PLUGIN_ROOT}/scripts/setup-ralph-loop.sh" $ARGUMENTS
```

Please work on the task. When you try to exit, the Ralph loop will feed the SAME PROMPT back to you for the next iteration. You'll see your previous work in files and git history, allowing you to iterate and improve.

CRITICAL RULE: If a completion promise is set, you may ONLY output it when the statement is completely and unequivocally TRUE. Do not output false promises to escape the loop, even if you think you're stuck or should exit for other reasons. The loop is designed to continue until genuine completion.

---

## /help

_Source: `marketplaces/claude-plugins-official/plugins/hookify/commands/help.md`_

---
description: Get help with the hookify plugin
allowed-tools: ["Read"]
---

# Hookify Plugin Help

Explain how the hookify plugin works and how to use it.

## Overview

The hookify plugin makes it easy to create custom hooks that prevent unwanted behaviors. Instead of editing `hooks.json` files, users create simple markdown configuration files that define patterns to watch for.

## How It Works

### 1. Hook System

Hookify installs generic hooks that run on these events:
- **PreToolUse**: Before any tool executes (Bash, Edit, Write, etc.)
- **PostToolUse**: After a tool executes
- **Stop**: When Claude wants to stop working
- **UserPromptSubmit**: When user submits a prompt

These hooks read configuration files from `.claude/hookify.*.local.md` and check if any rules match the current operation.

### 2. Configuration Files

Users create rules in `.claude/hookify.{rule-name}.local.md` files:

```markdown
---
name: warn-dangerous-rm
enabled: true
event: bash
pattern: rm\s+-rf
---

⚠️ **Dangerous rm command detected!**

This command could delete important files. Please verify the path.
```

**Key fields:**
- `name`: Unique identifier for the rule
- `enabled`: true/false to activate/deactivate
- `event`: bash, file, stop, prompt, or all
- `pattern`: Regex pattern to match

The message body is what Claude sees when the rule triggers.

### 3. Creating Rules

**Option A: Use /hookify command**
```
/hookify Don't use console.log in production files
```

This analyzes your request and creates the appropriate rule file.

**Option B: Create manually**
Create `.claude/hookify.my-rule.local.md` with the format above.

**Option C: Analyze conversation**
```
/hookify
```

Without arguments, hookify analyzes recent conversation to find behaviors you want to prevent.

## Available Commands

- **`/hookify`** - Create hooks from conversation analysis or explicit instructions
- **`/hookify:help`** - Show this help (what you're reading now)
- **`/hookify:list`** - List all configured hooks
- **`/hookify:configure`** - Enable/disable existing hooks interactively

## Example Use Cases

**Prevent dangerous commands:**
```markdown
---
name: block-chmod-777
enabled: true
event: bash
pattern: chmod\s+777
---

Don't use chmod 777 - it's a security risk. Use specific permissions instead.
```

**Warn about debugging code:**
```markdown
---
name: warn-console-log
enabled: true
event: file
pattern: console\.log\(
---

Console.log detected. Remember to remove debug logging before committing.
```

**Require tests before stopping:**
```markdown
---
name: require-tests
enabled: true
event: stop
pattern: .*
---

Did you run tests before finishing? Make sure `npm test` or equivalent was executed.
```

## Pattern Syntax

Use Python regex syntax:
- `\s` - whitespace
- `\.` - literal dot
- `|` - OR
- `+` - one or more
- `*` - zero or more
- `\d` - digit
- `[abc]` - character class

**Examples:**
- `rm\s+-rf` - matches "rm -rf"
- `console\.log\(` - matches "console.log("
- `(eval|exec)\(` - matches "eval(" or "exec("
- `\.env$` - matches files ending in .env

## Important Notes

**No Restart Needed**: Hookify rules (`.local.md` files) take effect immediately on the next tool use. The hookify hooks are already loaded and read your rules dynamically.

**Block or Warn**: Rules can either `block` operations (prevent execution) or `warn` (show message but allow). Set `action: block` or `action: warn` in the rule's frontmatter.

**Rule Files**: Keep rules in `.claude/hookify.*.local.md` - they should be git-ignored (add to .gitignore if needed).

**Disable Rules**: Set `enabled: false` in frontmatter or delete the file.

## Troubleshooting

**Hook not triggering:**
- Check rule file is in `.claude/` directory
- Verify `enabled: true` in frontmatter
- Confirm pattern is valid regex
- Test pattern: `python3 -c "import re; print(re.search('your_pattern', 'test_text'))"`
- Rules take effect immediately - no restart needed

**Import errors:**
- Check Python 3 is available: `python3 --version`
- Verify hookify plugin is installed correctly

**Pattern not matching:**
- Test regex separately
- Check for escaping issues (use unquoted patterns in YAML)
- Try simpler pattern first, then refine

## Getting Started

1. Create your first rule:
   ```
   /hookify Warn me when I try to use rm -rf
   ```

2. Try to trigger it:
   - Ask Claude to run `rm -rf /tmp/test`
   - You should see the warning

4. Refine the rule by editing `.claude/hookify.warn-rm.local.md`

5. Create more rules as you encounter unwanted behaviors

For more examples, check the `${CLAUDE_PLUGIN_ROOT}/examples/` directory.

---

## /list

_Source: `marketplaces/claude-plugins-official/plugins/hookify/commands/list.md`_

---
description: List all configured hookify rules
allowed-tools: ["Glob", "Read", "Skill"]
---

# List Hookify Rules

**Load hookify:writing-rules skill first** to understand rule format.

Show all configured hookify rules in the project.

## Steps

1. Use Glob tool to find all hookify rule files:
   ```
   pattern: ".claude/hookify.*.local.md"
   ```

2. For each file found:
   - Use Read tool to read the file
   - Extract frontmatter fields: name, enabled, event, pattern
   - Extract message preview (first 100 chars)

3. Present results in a table:

```
## Configured Hookify Rules

| Name | Enabled | Event | Pattern | File |
|------|---------|-------|---------|------|
| warn-dangerous-rm | ✅ Yes | bash | rm\s+-rf | hookify.dangerous-rm.local.md |
| warn-console-log | ✅ Yes | file | console\.log\( | hookify.console-log.local.md |
| check-tests | ❌ No | stop | .* | hookify.require-tests.local.md |

**Total**: 3 rules (2 enabled, 1 disabled)
```

4. For each rule, show a brief preview:
```
### warn-dangerous-rm
**Event**: bash
**Pattern**: `rm\s+-rf`
**Message**: "⚠️ **Dangerous rm command detected!** This command could delete..."

**Status**: ✅ Active
**File**: .claude/hookify.dangerous-rm.local.md
```

5. Add helpful footer:
```
---

To modify a rule: Edit the .local.md file directly
To disable a rule: Set `enabled: false` in frontmatter
To enable a rule: Set `enabled: true` in frontmatter
To delete a rule: Remove the .local.md file
To create a rule: Use `/hookify` command

**Remember**: Changes take effect immediately - no restart needed
```

## If No Rules Found

If no hookify rules exist:

```
## No Hookify Rules Configured

You haven't created any hookify rules yet.

To get started:
1. Use `/hookify` to analyze conversation and create rules
2. Or manually create `.claude/hookify.my-rule.local.md` files
3. See `/hookify:help` for documentation

Example:
```
/hookify Warn me when I use console.log
```

Check `${CLAUDE_PLUGIN_ROOT}/examples/` for example rule files.
```

---

## /configure

_Source: `marketplaces/claude-plugins-official/plugins/hookify/commands/configure.md`_

---
description: Enable or disable hookify rules interactively
allowed-tools: ["Glob", "Read", "Edit", "AskUserQuestion", "Skill"]
---

# Configure Hookify Rules

**Load hookify:writing-rules skill first** to understand rule format.

Enable or disable existing hookify rules using an interactive interface.

## Steps

### 1. Find Existing Rules

Use Glob tool to find all hookify rule files:
```
pattern: ".claude/hookify.*.local.md"
```

If no rules found, inform user:
```
No hookify rules configured yet. Use `/hookify` to create your first rule.
```

### 2. Read Current State

For each rule file:
- Read the file
- Extract `name` and `enabled` fields from frontmatter
- Build list of rules with current state

### 3. Ask User Which Rules to Toggle

Use AskUserQuestion to let user select rules:

```json
{
  "questions": [
    {
      "question": "Which rules would you like to enable or disable?",
      "header": "Configure",
      "multiSelect": true,
      "options": [
        {
          "label": "warn-dangerous-rm (currently enabled)",
          "description": "Warns about rm -rf commands"
        },
        {
          "label": "warn-console-log (currently disabled)",
          "description": "Warns about console.log in code"
        },
        {
          "label": "require-tests (currently enabled)",
          "description": "Requires tests before stopping"
        }
      ]
    }
  ]
}
```

**Option format:**
- Label: `{rule-name} (currently {enabled|disabled})`
- Description: Brief description from rule's message or pattern

### 4. Parse User Selection

For each selected rule:
- Determine current state from label (enabled/disabled)
- Toggle state: enabled → disabled, disabled → enabled

### 5. Update Rule Files

For each rule to toggle:
- Use Read tool to read current content
- Use Edit tool to change `enabled: true` to `enabled: false` (or vice versa)
- Handle both with and without quotes

**Edit pattern for enabling:**
```
old_string: "enabled: false"
new_string: "enabled: true"
```

**Edit pattern for disabling:**
```
old_string: "enabled: true"
new_string: "enabled: false"
```

### 6. Confirm Changes

Show user what was changed:

```
## Hookify Rules Updated

**Enabled:**
- warn-console-log

**Disabled:**
- warn-dangerous-rm

**Unchanged:**
- require-tests

Changes apply immediately - no restart needed
```

## Important Notes

- Changes take effect immediately on next tool use
- You can also manually edit .claude/hookify.*.local.md files
- To permanently remove a rule, delete its .local.md file
- Use `/hookify:list` to see all configured rules

## Edge Cases

**No rules to configure:**
- Show message about using `/hookify` to create rules first

**User selects no rules:**
- Inform that no changes were made

**File read/write errors:**
- Inform user of specific error
- Suggest manual editing as fallback

---

## /hookify

_Source: `marketplaces/claude-plugins-official/plugins/hookify/commands/hookify.md`_

---
description: Create hooks to prevent unwanted behaviors from conversation analysis or explicit instructions
argument-hint: Optional specific behavior to address
allowed-tools: ["Read", "Write", "AskUserQuestion", "Task", "Grep", "TodoWrite", "Skill"]
---

# Hookify - Create Hooks from Unwanted Behaviors

**FIRST: Load the hookify:writing-rules skill** using the Skill tool to understand rule file format and syntax.

Create hook rules to prevent problematic behaviors by analyzing the conversation or from explicit user instructions.

## Your Task

You will help the user create hookify rules to prevent unwanted behaviors. Follow these steps:

### Step 1: Gather Behavior Information

**If $ARGUMENTS is provided:**
- User has given specific instructions: `$ARGUMENTS`
- Still analyze recent conversation (last 10-15 user messages) for additional context
- Look for examples of the behavior happening

**If $ARGUMENTS is empty:**
- Launch the conversation-analyzer agent to find problematic behaviors
- Agent will scan user prompts for frustration signals
- Agent will return structured findings

**To analyze conversation:**
Use the Task tool to launch conversation-analyzer agent:
```
{
  "subagent_type": "general-purpose",
  "description": "Analyze conversation for unwanted behaviors",
  "prompt": "You are analyzing a Claude Code conversation to find behaviors the user wants to prevent.

Read user messages in the current conversation and identify:
1. Explicit requests to avoid something (\"don't do X\", \"stop doing Y\")
2. Corrections or reversions (user fixing Claude's actions)
3. Frustrated reactions (\"why did you do X?\", \"I didn't ask for that\")
4. Repeated issues (same problem multiple times)

For each issue found, extract:
- What tool was used (Bash, Edit, Write, etc.)
- Specific pattern or command
- Why it was problematic
- User's stated reason

Return findings as a structured list with:
- category: Type of issue
- tool: Which tool was involved
- pattern: Regex or literal pattern to match
- context: What happened
- severity: high/medium/low

Focus on the most recent issues (last 20-30 messages). Don't go back further unless explicitly asked."
}
```

### Step 2: Present Findings to User

After gathering behaviors (from arguments or agent), present to user using AskUserQuestion:

**Question 1: Which behaviors to hookify?**
- Header: "Create Rules"
- multiSelect: true
- Options: List each detected behavior (max 4)
  - Label: Short description (e.g., "Block rm -rf")
  - Description: Why it's problematic

**Question 2: For each selected behavior, ask about action:**
- "Should this block the operation or just warn?"
- Options:
  - "Just warn" (action: warn - shows message but allows)
  - "Block operation" (action: block - prevents execution)

**Question 3: Ask for example patterns:**
- "What patterns should trigger this rule?"
- Show detected patterns
- Allow user to refine or add more

### Step 3: Generate Rule Files

For each confirmed behavior, create a `.claude/hookify.{rule-name}.local.md` file:

**Rule naming convention:**
- Use kebab-case
- Be descriptive: `block-dangerous-rm`, `warn-console-log`, `require-tests-before-stop`
- Start with action verb: block, warn, prevent, require

**File format:**
```markdown
---
name: {rule-name}
enabled: true
event: {bash|file|stop|prompt|all}
pattern: {regex pattern}
action: {warn|block}
---

{Message to show Claude when rule triggers}
```

**Action values:**
- `warn`: Show message but allow operation (default)
- `block`: Prevent operation or stop session

**For more complex rules (multiple conditions):**
```markdown
---
name: {rule-name}
enabled: true
event: file
conditions:
  - field: file_path
    operator: regex_match
    pattern: \.env$
  - field: new_text
    operator: contains
    pattern: API_KEY
---

{Warning message}
```

### Step 4: Create Files and Confirm

**IMPORTANT**: Rule files must be created in the current working directory's `.claude/` folder, NOT the plugin directory.

Use the current working directory (where Claude Code was started) as the base path.

1. Check if `.claude/` directory exists in current working directory
   - If not, create it first with: `mkdir -p .claude`

2. Use Write tool to create each `.claude/hookify.{name}.local.md` file
   - Use relative path from current working directory: `.claude/hookify.{name}.local.md`
   - The path should resolve to the project's .claude directory, not the plugin's

3. Show user what was created:
   ```
   Created 3 hookify rules:
   - .claude/hookify.dangerous-rm.local.md
   - .claude/hookify.console-log.local.md
   - .claude/hookify.sensitive-files.local.md

   These rules will trigger on:
   - dangerous-rm: Bash commands matching "rm -rf"
   - console-log: Edits adding console.log statements
   - sensitive-files: Edits to .env or credentials files
   ```

4. Verify files were created in the correct location by listing them

5. Inform user: **"Rules are active immediately - no restart needed!"**

   The hookify hooks are already loaded and will read your new rules on the next tool use.

## Event Types Reference

- **bash**: Matches Bash tool commands
- **file**: Matches Edit, Write, MultiEdit tools
- **stop**: Matches when agent wants to stop (use for completion checks)
- **prompt**: Matches when user submits prompts
- **all**: Matches all events

## Pattern Writing Tips

**Bash patterns:**
- Match dangerous commands: `rm\s+-rf|chmod\s+777|dd\s+if=`
- Match specific tools: `npm\s+install\s+|pip\s+install`

**File patterns:**
- Match code patterns: `console\.log\(|eval\(|innerHTML\s*=`
- Match file paths: `\.env$|\.git/|node_modules/`

**Stop patterns:**
- Check for missing steps: (check transcript or completion criteria)

## Example Workflow

**User says**: "/hookify Don't use rm -rf without asking me first"

**Your response**:
1. Analyze: User wants to prevent rm -rf commands
2. Ask: "Should I block this command or just warn you?"
3. User selects: "Just warn"
4. Create `.claude/hookify.dangerous-rm.local.md`:
   ```markdown
   ---
   name: warn-dangerous-rm
   enabled: true
   event: bash
   pattern: rm\s+-rf
   ---

   ⚠️ **Dangerous rm command detected**

   You requested to be warned before using rm -rf.
   Please verify the path is correct.
   ```
5. Confirm: "Created hookify rule. It's active immediately - try triggering it!"

## Important Notes

- **No restart needed**: Rules take effect immediately on the next tool use
- **File location**: Create files in project's `.claude/` directory (current working directory), NOT the plugin's .claude/
- **Regex syntax**: Use Python regex syntax (raw strings, no need to escape in YAML)
- **Action types**: Rules can `warn` (default) or `block` operations
- **Testing**: Test rules immediately after creating them

## Troubleshooting

**If rule file creation fails:**
1. Check current working directory with pwd
2. Ensure `.claude/` directory exists (create with mkdir if needed)
3. Use absolute path if needed: `{cwd}/.claude/hookify.{name}.local.md`
4. Verify file was created with Glob or ls

**If rule doesn't trigger after creation:**
1. Verify file is in project `.claude/` not plugin `.claude/`
2. Check file with Read tool to ensure pattern is correct
3. Test pattern with: `python3 -c "import re; print(re.search(r'pattern', 'test text'))"`
4. Verify `enabled: true` in frontmatter
5. Remember: Rules work immediately, no restart needed

**If blocking seems too strict:**
1. Change `action: block` to `action: warn` in the rule file
2. Or adjust the pattern to be more specific
3. Changes take effect on next tool use

Use TodoWrite to track your progress through the steps.

---

