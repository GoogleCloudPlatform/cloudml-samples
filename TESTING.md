# Cloud ML Samples testing

This repository uses a Google service Kokoro for CI testing. The following guide covers how to modify existing tests, and write new tests.

## Running Tests

All presubmits will automatically be run on all pull requests which are made by a repository administrator, or have the label `kokoro:run`.

For Ubuntu for example a presubmit will then chain together the following scripts:
1. [`testing/ubuntu/presubmit/run_if_changed.sh`](testing/ubuntu/presubmit/run_if_changed.sh) will exit if there are no changes in `${CMLE_TEST_BASE_DIR}`, otherwise it will run
2. [`testing/ubuntu/setup_and_run_test.sh`](testing/ubuntu/setup_and_run_test.sh) which performs setup common to our examples, and then runs the test specified by
3. `${CMLE_TEST_SCRIPT}` which is specified in the build config file defining the test (for example [`testing/ubuntu/presubmit/flowers.cfg`](testing/ubuntu/presubmit/flowers.cfg)

Tests past if the script chain exits with status 0.

### Getting Logs Access

Logs are provided through Stackdriver logging and the Cloud Console. Frequent contributors or contributors of complex examples may want to get Log access for the project `cloudml-samples-testbed` which will allow them to follow through on the links.

## Modifying Existing Tests

As specified above, tests will have a corresponding scripts which they run. Simply modify these scripts in a pull request, and they will be run. We welcome external contributions which add or fix tests.

## Adding a new test environment

NOTE: this must be done by a Googler.

### Adding a new Presubmit

In this repository, in `testing/[OS]/presubmit/`, add a Kokoro build config file. At the very minimum your config file must provide 3 environment variables:

* `CMLE_REQUIREMENTS_FILE` a pip formatted `.txt` file with the list of python requirements for your test.
* `CMLE_TEST_BASE_DIR` the desired working directory for your test. Note the test will only be run if there are changes relevant to this working directory.
* `CMLE_TEST_SCRIPT` the testing script to run if changes relevant to your directory have been made.

You can see [`testing/ubuntu/presubmit/flowers.cfg`](testing/ubuntu/presubmit/flowers.cfg) for an example of such a file.

In Google3 you must add a corresponding job config in `//depot/google3/devtools/kokoro/config/prod/cloudml_engine/samples/` with the subdirectory *and* name matching your build config.

You can use `//depot/google3/devtools/config/prod/cloudml_engine/samples/ubuntu/presubmit/flowers.cfg` as a template, and should only need to modify `commit_status_context` which will determine how your test is named in the Github UI.

### Adding a new Release (periodic tests)

Follow the directions as above except add your build config in `testing/[OS]/release/` and your job config in `//depot/google3/devtools/config/prod/cloudml_engine/samples/[OS]/release`.

Additionally, instead of `commit_status_context` you'll want to set a `build_badge_path` which can be embedded using the corresponding public address in a `README` doc. See [`flowers/README.md`](flowers/README.md) for an example of the public address format.
