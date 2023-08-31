# Contributing to Time Series Forecasting Accelerator (TSFA)

This project welcomes contributions and suggestions.

- [Issues and Bugs](#issue)
- [Feature Requests](#feature)
- [Submission Guidelines](#submit)

## <a name="issue"></a> Found an Issue?

If you find a bug in the source code or a mistake in the documentation, you can help us by [submitting an issue](#submit-issue) to the Repository. Even better, you can [submit a Pull Request](#submit-pr) with a fix.

## <a name="feature"></a> Want a Feature?

You can *request* a new feature by [submitting an issue](#submit-issue) to the Repository.

If you would like to *implement* a new feature, please submit an issue with a proposal for your work first, to be sure that it is aligned with existing work on the accelerator and prioritized and tracked accordingly. The proposal doesn't need to be anything too long or detailed - the goal is to make sure that the team is aware of the change, that they can help with any design direction, and advise of any similar work that might have been or is being done. Also, if it's a significant piece of work, the Design Authority will need to be involved in the approval and prioritization.

See [submitting a PR](#submit-pr) for details on writing code for your new feature.

**Small Features** can be crafted and directly [submitted as a Pull Request](#submit-pr).

## <a name="submit"></a> Submission Guidelines

### <a name="submit-issue"></a> Submitting an Issue

Before you submit an issue, search the bugs/tasks/stories to make sure it's not already part of the backlog.

#### Submitting Bugs

You can create an issue on github and tag it as a bug or a feature enhancement. Describing the issue in detail will increase the chances of your issue being dealt with quickly:

- **Overview of the Issue** - if an error is being thrown, a non-minified stack trace helps.
- **Version** - what version is affected (e.g. 0.1.2)
- **Motivation for or Use Case** - explain what are you trying to do and why the current behavior is a bug for you
- **Operating System** - is there anything different about your experimentation environment from the baseline (i.e. other libraries installed)?
- **Reproduce the Error** - provide a live example or a unambiguous set of steps and also provide a configuration file with parameters used and sample of the dataset used to reproduce the error.
- **Related Issues** - has a similar issue been reported before?
- **Suggest a Fix** - if you have any ideas on what the problem might be or how it might be fixed, please add that as well

#### Submitting Feature Requests

If your issue is a request for a new feature, please open an issue and tag it as a feature enhancement and provide as much information as possible about what you'd like changed/added and any thoughts you have on the implementation

### <a name="submit-pr"></a> Submitting a Pull Request (PR)

If you'd like to fix your own bug or write your own feature. Before you start writing code:

- Review the repository for an open or completed PRs that relates to your submission. Also search through both open and closed issues on github to minimize duplication.
- Make your changes in a new git branch. See our [branching strategy](./docs/decisions/0003-Branch-Strategy.md).
- Adhere to our [coding standard](./docs/coding_standards.md).

Before you submit your Pull Request (PR) consider the following guidelines:

- Commit your changes using a descriptive commit message
- Any substantial changes should have accompanying unit tests and a `module_sample` notebooks illustrating how to use the new module or functionality.
- If your change involves new features or models, please ensure that these are added to the `docs`
- If your change involves any significant decisions where you've considered trade-offs and chosen the option you think is best, consider authoring an ADR in [./docs/decisions](./docs/decisions)
- Create a PR, including evidence that (like screenshots) that end-to-end testing has been conducted to ensure that the modifications integrate with the existing codebase and help the reviewers understand how they're expected to work
- The TSFA team will review it and get back to you, possibly with suggested changes:
  - If you have questions, please reach out to the reviewers and we'll go over our feedback and come to consensus.
  - Make the required updates and push to remote, then ensure the PR is updated.

Thank you for your contribution!
