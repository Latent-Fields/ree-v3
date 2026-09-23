# ree-v3 code knowledge-graph snapshot

`knowledge-graph.json` + `meta.json` here are the **weekly committed snapshot** of the
Understand-Anything (`/understand`) graph of this repo. `meta.json` records when it was
analysed and at which commit.

- **Live copy:** `.ua/knowledge-graph.json` (gitignored). The Mac's
  `nightly-documentation-update` scheduled task regenerates it in place every night;
  `REE_assembly/serve.py`'s Code Atlas and `/understand-dashboard` / `/understand-explain`
  read that live copy.
- **This copy:** written only by that task's weekly step, which copies the live graph here
  and commits it in the same motion (`scripts/ree_commit.py`), so it is never left dirty.
- **Fresh clone / worktree:** seed the live copy from here before using the dashboard or an
  incremental `/understand` run:

  ```bash
  mkdir -p .ua && cp docs/understand/knowledge-graph.json docs/understand/meta.json .ua/
  ```

Why the split (2026-09-23): while `.ua/knowledge-graph.json` itself was tracked, every nightly
regeneration left a tracked file dirty in the shared checkout until the next weekly commit, and
`git pull --rebase --autostash` swept it into orphaned stashes.
