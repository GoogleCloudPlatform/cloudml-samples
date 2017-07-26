# Cloud ML Samples testing

This repository uses a Google service Kokoro for CI testing. The following guide covers how to modify existing tests, and write new tests.

## Running Tests

All presubmits will automatically be run on all pull requests which are made by a repository administrator, or have the label `kokoro:run`. The script [`testing/ubuntu/presubmit/run_if_changed.sh`](testing/ubuntu/presubmit/run_if_changed.sh) can be used to only run tests when the pull request changes a relevant repository.

Tests past if their corresponding script exits with status 0.

## Modifying Existing Tests

As specified above, tests will have a corresponding scripts which they run. Simply modify these scripts in a pull request, and they will be run. We welcome external contributions which add or fix tests.

## Adding a new test environment

NOTE: this must be done by a Googler.

### Adding a new Presubmit

In this repository, in `testing/[OS]/presubmit/`, add a Kokoro build config file. At the very minimum you config file must provide 3 environment variables:

* `CMLE_REQUIREMENTS_FILE` a pip formatted `.txt` file with the list of python requirements for your test.
* `CMLE_TEST_BASE_DIR` the desired working directory for your test. Note the test will only be run if there are changes relevant to this working directory.
* `CMLE_TEST_SCRIPT` the testing script to run if changes relevant to your directory have been made.

You can see [`testing/ubuntu/presubmit/flowers.cfg`](testing/ubuntu/presubmit/flowers.cfg) for an example of such a file.

In Google3 you must add a corresponding job config in `//depot/google3/devtools/kokoro/config/prod/cloudml_engine/samples/` with the subdirectory *and* name matching your build config.

You can use `//depot/google3/devtools/config/prod/cloudml_engine/samples/ubuntu/presubmit/flowers.cfg` as a template, and should only need to modify `build_badge_path` (if you want a build badge for your test), and `commit_status_context` which will determine how your test is named in the Github UI.



