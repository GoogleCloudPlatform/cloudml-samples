# Contributing to Cloud ML Engine Samples

You want to contribute to Cloud ML Engine samples? That's awesome! Please refer to the short guide below.

# Contributing Guide

## Notebook Guide
| Criteria        | Samples        |
| ------------- |:-------------:|
| Introduction | Yes |
| Prerequisites | Yes |
| Setup | Yes |
| Steps for completion with code and instructions | Yes |
| Next Steps | Optional |

## Code Guide

We are happy to see your enthusiasm. This table lists the criteria and how important is it to follow when you make your contributions. 

<br/>

| Criteria        | Core Samples        | Contrib Samples |
| ------------- |:-------------:| -----:|
| Include requirements.txt for dependencies     | Yes | Yes |
| Include unit tests     | Yes | Optional |
| Include integration tests     | Yes | Optional |
| Include core TF API aka no contrib   | Yes | Optional |
| Include README     | Yes | Yes |
| Maintain version dependency updates     | Yes | Optional |


## Expectations

### Contributors

- You should have read the Python Dev Guide and in particular the [Google Python Style Guide](https://github.com/google/styleguide/blob/gh-pages/pyguide.md).

- Having already read the documentation makes the review smoother, as fewer obvious mistakes would have to be pointed out, and gives you a head start to the progress.

- Your Pull Request should be small and include a single logical change.

- Smaller changes are easier to review and evaluate, and in general smaller self-contained PRs make it easier to debug or, if needed, rollback.

- If your code appears to include generic functionality, make sure you have not reinvented the wheel.

- Check other repos and, for instance, prefer adapting code that already exists over a new implementation.

- Your change should pass existing unit tests.

- Also pay attention to what is covered by those tests.

- You should provide a sufficiently detailed description of your Pull Request.

- Good descriptions make it easy to understand what the goal of the change is. Writing a good description may also make it apparent if multiple changes are being bundled in the same Pull Request that could be split.

- The LGTM from the code owner ("code review") should ensure both the change intention and implementation are suitable for the codebase being changed. The review from Google team can then focus on the specificities of the Python language.

- Your Pull Request should not have any pending commits (modified in Local Branch but not uploaded).

- If you made changes to your Pull Request before sending it to for approval, make sure you uploaded it, too.

- You should be able to make (possibly significant) changes to your Pull Request.

- Because of how Pull Request reviews work, it is possible that your reviewer will ask you to refactor the code you're modifying. If you have a long series of changes that would have to be updated, this can involve a lot of work. Consider this before building a long chain of dependent Pull Requests, preferring instead to make changes independent and self-contained.


### Reviewers

- Your reviewer will respond to your Pull Request within one business day (according to their location schedule, be mindful of timezone and local holidays).

- Please note that in some cases you may be assigned an "overflow reviewer", in which case you may experience further delay.

- Reviewers may not have full knowledge of the frameworks or local conventions of your codebase.

- This is the primary reason why due dilligence and testing is needed before sending the code for review. It also means that while the reviewer may be able to point out logical problems with your code, this is not certain.

- The review can be streamlined by pointing out upfront known limitations or team-specific limitations of the code, for instance by providing links to supporting documentation after the code reviewer is assigned.

- Approvers may recuse themselves if a higher urgency interrupt make the review likely to slip through time, or if they don't feel comfortable to review the changelist (e.g. for lack of expertise on a framework).

- If this is to happen, the reviewer is expected to either find an alternative new reviewer.
