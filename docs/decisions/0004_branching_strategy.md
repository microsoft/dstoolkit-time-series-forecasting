---
status: proposed
---
# Branching Strategy

## Context and Problem Statement

Time Series Forecasting Framework (TSFF) needs a coherent branching strategy that meets disparate goals, needing to move fast while avoiding breaking production and being consistent with others. We should have a consistent branching, merging, and tagging strategy to meet these needs.
More concretely:

## Decision Drivers

- Ability to move fast, on multiple teams, all contributing to the framework.
- New and occasional contributors should be able to understand the branching/merging strategy easily, even if they don't come from software engineering backgrounds.
- Maintenance will be by a small and fluid team, so:
  - The burden of keeping new work from impacting production should be low.
  - Multiple concurrent production versions will not be maintained.

## Considered Options

- GitFlow
- GitHub Flow
- GitLab Flow
- Trunk-based Development

## Decision Outcome

I advocate using GitLab Flow for our branching strategy.

We have multiple options with regards to the name of our "main" branch and our "production" branch (if we decide to have one), we also need to decide whether we have branches for any additional environments, and whether we will use any tagging. My recommendations are as follows:

- Since `production` will always have the current code running in production, we do not need tags. However, this is only a mild recommendation, and if the consensus is that adding version tags when we "cut" a release is useful for clarity, I don't have reason to object.
- I do not advocate for any branches other than `main` and `production`, as additional branches just add complexity. Instead, I advocate that we should push for CI/CD of `main` into our Experimentation environments so that we will be constantly testing new features and any "bug bash" efforts will be starting from a more solid tested foundation. This also incentivises experimentation teams to check their features in, knowing that they will then appear in the environment for their colleagues to use.

**NOTE: Regardless of which strategy we use, developers _should_ frequently `fetch`/`sync` to avoid any "merge hell" when merging their new features back to `main`.**

## Pros and Cons of the Options

**NOTE: Please don't focus on the branch _names_ as much as their purposes. We may alter the names to suit our needs and our current setup, but keep their purposes intact.**

### GitFlow

GitFlow focuses on two long-lived branches: `main` and `develop`. Short-lived feature branches from `develop` are used for developing new features, merged back into `develop`. Release branches are created from `develop` to manage the release process, with bug-bashes and fixes being created there and merged back to `develop`. The release branch merges onto `main` once stable, and a new version tag is cut. Emergent bugs in production code are patched in hotfix branches from `main`, and then merged into `develop` and `main` once completed. Version tagging is used on `main` but tags are not used anywhere else.

![GitFlow branch diagram example](https://www.flagship.io/wp-content/uploads/gitflow-branching-strategy.png "GitFlow")

- Good because:
  - `main` is always deployable to production, and only during the "release window" does it differ from the code running in production.
  - New features are always done in their own branch, allowing developers to work simultaneously and easily sync to others' work.
- Bad because:
  - Maintaining two long-lived active branches can be a management burden
  - Merging bugfixes from release branches, and hotfixes to both `develop` and `main` can be a source of regressions unless strictly followed

### GitHub Flow

GitHub Flow forgoes a dedicated release branch for a single `main` branch. Feature branches are short-lived, forked from `main`, and merged back to it at the end of development, after merging `main`'s changes to the feature. `main` is always deployable to production. GitHub Flow is silent on both tags and on a hotfixing strategy.

![GitHub Flow branch diagram example](https://www.flagship.io/wp-content/uploads/github-flow-branching-model.jpeg "GitHub Flow")

- Good because:
  - It's very easy to understand
  - It lends itself well to CI/CD scenarios
- Bad because
  - It doesn't tell you how to know what is currently running in production, assuming that you will be pushing `main` to production as soon as it's checked in
  - As such, bugs can easily make it to production and require rollback and fixing. However GitHub Flow is silent on rollbacks, instead implying a roll-forward strategy (leaving prod broken until bug is fixed and rolled out)
  - Hotfixes are not expected: since `main` and production are expected to be the same, a hotfix is just a feature branch like any other

### GitLab Flow

GitLab Flow is somewhere between GitFlow and GitHub Flow. All development is done on short-lived feature branches from `main` and integrated back in as features are complete and bugs fixed. However, unlike GitHub Flow where `main` is expected to deploy immediately to production, GitLab Flow has one or more additional long-lived branches matching the target deployment environments (e.g. pre-prod, prod). `main` is, ideally, auto-deployed to a `staging` environment, and when a new release is cut `main` is merged onto the next branch in the release pipeline (e.g. `pre-prod`). The implementor can decide how many environments there are and how changes flow from one environment's branch to another, depending on their need. The implementor also decides whether they rely on that merge commit or create explicit version tags, to understand what's running in those environments and how it maps back to the code in `main`. GitLab Flow says only "serious" bug fixes should go into any branch but `main`, instead believing most hotfixes should be checked into `main` and then cherry-picked.

![GitLab Flow branch diagram example](https://www.flagship.io/wp-content/uploads/gitlab_flow_environment_branches.png "GitLab Flow")

- Good because:
  - The `production` branch is always the same code as that running in production.
  - New features are always done in their own branch, allowing developers to work simultaneously and easily sync to others' work.
  - Releasing `main` to production can be as simple as merging `main` into the production branch.
  - However, if we want to protect production from errors, we can introduce a `pre-prod` branch and environment for release testing.
  - We can still enable CI/CD from `main` into non-prod environments (e.g. a `staging` environment or possibly our Experimentation environments).
  - This technique is resilient if stakeholders makes later decisions to support multiple simultaneous versions of `tsff`, although with added complexity.
- Bad because:
  - Maintaining two long-lived active branches can be a management burden
  - Cherry-picking hotfixes from `main` to other environments' branches is difficult and prone to error

### Trunk-based Development

In reading "Trunk-based development" there is no substantive difference between it and GitHub Flow, other than the mention of using [Feature Flags](https://www.flagship.io/feature-flags/) to avoid enabling untested code in production. It has all of the same pros and cons as GitHub Flow.

![Trunk-based Development branch diagram example](https://www.flagship.io/wp-content/uploads/trunk-based-development-branching-strategy.png "Trunk-based Development")

## More Information

- [GitKraken's take on branching strategies](https://www.gitkraken.com/learn/git/best-practices/git-branch-strategy)
- [Flagship.io has a remarkably similar take](https://www.flagship.io/git-branching-strategies/)
- [GitFlow](https://nvie.com/posts/a-successful-git-branching-model/)
- [GitHub Flow](https://docs.github.com/en/get-started/quickstart/github-flow)
- [GitLab Flow](https://docs.gitlab.com/ee/topics/gitlab_flow.html)
- [Trunk-Based Development](https://www.flagship.io/glossary/trunk-based-development/)
