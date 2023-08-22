---
status: proposed
---
# Time Series Forecasting Accelerator (TSFA) Versioning Strategy

## Context and Problem Statement

TSFA is a collection of classes, "interfaces", and utilities used both in experimentation and for productionalizing models as part of the MLOps pipeline. As such, it needs to support experimentation workstreams and productionalized models while still being actively worked on. We need a way to version and release the code so that we can have stable production models, quick feature turnaround in experimentation environments while being stable enough to avoid impacting experimentation flow - all with a small team of people who may change regularly, and may have limited/delayed access to the ability to deploy. This necessitates both a strong versioning and a strong release packaging strategy - this ADR covers the versioning, which also encompasses version maintenance (i.e. how many versions we expect to support simultaneously), which I'm bundling together as they are highly related and unlikely to change independently. The actual release packaging, however, may change over time.

## Decision Drivers

- Data Scientists contribute to TSFA, so want a quick turnaround on new features being deployed in their environments
- A small contributor base will need to both build new features and maintain existing releases
- Production should remain stable or it can negatively impact the bottom line
- Different production environments can be required to update at the same cadence
  - There is no "production release freeze" that applies to one specific environment vs. another

## Considered Options

### Versioning Strategy

- Tool-assisted Semantic Versioning (e.g. `semver`)
- Semantic Versioning
- Date-based versioning
- Simple versioning (monotonically increasing)

### Version Support

- No explicit version support - all fixes go on latest, and the only mechanism for fixing production is roll-forward
- Explicit version support of a single version
- Explicit version support of N versions (N >= 2)

## Decision Outcome

**Chosen option:** [Semantic Versioning][semver], maintaining a single production release.

Semantic versioning because the version number makes it clear to everyone whether it's a breaking change and whether they will need to update their code to use it. For instance, a version bump from 1.0.0 to 2.0.0 is a breaking change, 2.0.0 to 2.1.0 is a non-breaking change, and 2.1.0 to 2.1.1 would be a bugfix.

A single production release because the EF team (even with champions involved) is a very small team and is devoted to rapid experimentation and features, and maintaining more than one production release is too heavy a burden. The team will support a single currently-deployed production release at any one time. All production environments should migrate to a new release when it is cut, avoiding the maintenance burden of maintaining multiple "live" releases.

## Pros and Cons of the Options - Versioning Strategy

### Tool-assisted Semantic Versioning

Tool-assisted [Semantic Versioning][semver] is just Semantic Versioning, but using a tool like [semver](https://github.com/fsaintjacques/semver-tool) (there are a few others) to manage version "bumping" as part of your CI/CD process.

- Good, because:
  - Semantic versioning maintains a clean, monotonically increasing version number that is widely understood in the industry
  - It matches the expectations from our code-with CSE partners
  - As a "standard", it deals with special cases and boundary conditions far beyond what we would proactively consider
  - Users can just compare version numbers to understand the impact of a new version (i.e. is it breaking, will they need to update their code, etc.)
  - A tool-based solution makes it easy to integrate into any current or future CI/CD pipeline
- Bad, because:
  - Semantic versions can be challenging to determine, especially where multiple committers are involved and the version maintainer may not have "line of sight" to whether a breaking change has been integrated
  - A tool-based solution may make some assumptions or produce wrong results that can be propagated without adequate oversight

### Semantic Versioning

[Semantic Versioning][semver] is a specification for versioning that specifies a four-part version string (three + optional modifier). These consist of `MAJOR` version bumps - those consisting of breaking changes that will require updates to existing dependent code in order to work - either because of changes to function signatures or changes to underlying behavior that differ from previous expectations. `MINOR` changes which consist of backward-compatible changes (note: even if you have the same API, if you've changed fundamental behaviors, you should still consider it a major change). Finally, `PATCH` changes which are backward compatible bug-fixes. Additional modifiers can be specified which modify the end of the version and can tag things like "release candidates" or metadata from build or CI/CD systems. These are all laid out clearly in [the spec][semver] which clearly gives not only the rules (in standard [MUST/SHALL/SHOULD/MAY terms](https://www.rfc-editor.org/rfc/rfc2119)) but also the [EBNF](https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form) for the versions themselves.

- Good, because:
  - Semantic versioning maintains a clean, monotonically increasing version number that is widely understood in the industry
  - It matches the expectations from our code-with CSE partners
  - As a "standard", it deals with special cases and boundary conditions far beyond what we would proactively consider
  - Users can just compare version numbers to understand the impact of a new version (i.e. is it breaking, will they need to update their code, etc.)
- Bad, because:
  - Semantic versions can be challenging to determine, especially where multiple committers are involved and the version maintainer may not have "line of sight" to whether a breaking change has been integrated

### Date-based Versioning

Date-based versioning is simple, with users just marking a new version with the date when it is created (e.g. `20220823`).

- Good, because:
  - It's immediately obvious how old a given release is
  - It's simple to use and understand
  - The version is monotonically increasing while also being semantically useful
- Bad, because:
  - Multiple versions on a single date need some way to disambiguate (e.g. `20220823.2`)
  - It's impossible to tell how substantial a change is based on version number alone

### Simple Versioning

Simple versioning is even more simple than date-based versioning, with folks just increasing the version number every time a new version is released (e.g. `Windows 11`).

- Good, because:
  - It's simple to use and understand
- Bad, because:
  - It's impossible to tell how substantial a change is based on version number alone
  - The version number has no semantic value

## Pros and Cons of the Options - Version Support

### No Explicit Version Support

No support is given for deployed versions - bug-fixes are made into main and there is no cherry-picking into any other versioned branches, they should roll forward to the new code. This is a typical CI/CD scenario and is often the right choice for fast-moving systems that can tolerate failure and fix quickly.

- Good, because:
  - Easy to maintain
  - Ensures production and development don't diverge significantly
  - Often doesn't require an active "support" team since production fixes are fixed in main and may even be deployed to production using CI/CD
- Bad, because:
  - Can cause more lengthy production outages
  - Rolling forward production can wind up shipping features that are not fully "baked" (can be mitigated with feature flags)

### Explicit Support of a Single Version

Every "release" branch contains the same release version. As main and release diverge, issues found in release are either fixed in the release branch and merged back to main, or cherry-picked from main into each release branch.

- Good, because:
  - Relatively easy to maintain
  - Multiple environments are guaranteed to be on the same release, and in fact can often use the same release branch
    - (typically only needing to diverge if they need to release on different schedules and there is a requirement that code in the branch match that in production up to some SLA for the release window)
  - Production tends to be more stable than in the "no explicit version support" solution since new features are rolled out only when they are considered "baked"
- Bad, because:
  - Harder than not supporting a release version
  - Multiple release environments are required to upgrade at roughly the same cadence
  - Typically requires at least a small support team to deal with production fixes and cherry-picking/reverse-merging, which can be error-prone

### Explicit Support of More Than One Version

Every production environment can be on a separate release. Bug-fixes for production issues take place within that branch, and are typically merged onto other branches (main, other releases) from there.

- Good, because:
  - Multiple environments can maintain their own release adoption cadence, typically useful when an environment needs to "lock" for certain periods (holidays, etc.) that don't apply to other environments
  - Production tends to be quite stable, as each production environment only adopts features when they wish, and can wait and see how those features perform in other environments before adoption
- Bad, because:
  - A significant support team is needed, as they need to maintain multiple versions, field tickets from many stakeholders, understand and fix often old code, and determine whether and how those fixes get applied onto other versions and the current mainline
  - Developers in mainline will dread support as production branches age, which can result in retention issues
  - Production users will avoid updating unless they see an obvious benefit to the new release, leading to a feedback loop as they get further behind and the upgrading process becomes even more of a burden

## More Information

- [Semantic Versioning][semver]
- [Code-With (CSE) Engineering Playbook - Component Versioning](https://microsoft.github.io/code-with-engineering-playbook/source-control/component-versioning/)

[semver]: https://semver.org/ "Semantic Versioning Specification"
