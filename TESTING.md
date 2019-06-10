# AI Platform testing

This repository uses a Google service Kokoro for CI testing. The following guide covers how to modify existing tests, and write new tests.

## Running Tests

All presubmits will automatically be run on all pull requests which are made by a repository administrator, or have the label `kokoro:run`.

For Ubuntu for example a presubmit will then chain together the following scripts:

1. [`.kokoro/tests/run_tests.sh`](.kokoro/tests/run_tests.sh) will exit if there are no changes in `${CMLE_TEST_BASE_DIR}`, otherwise it will run
2. [`.kokoro/tests/`](.kokoro/tests/) which performs setup common to our examples, and then runs the test specified by
`${CMLE_TEST_SCRIPT}` which is specified in the build config file defining the test.

Tests past if the script chain exits with status 0.

### Getting Logs Access

Logs are provided through Stackdriver logging and the Cloud Console. Frequent contributors or contributors of complex examples may want to get Log access for the project `cloudml-samples-testbed` which will allow them to follow through on the links.

## Modifying Existing Tests

As specified above, tests will have a corresponding scripts which they run. Simply modify these scripts in a pull request, and they will be run. We welcome external contributions which add or fix tests.

## Adding a new test environment

NOTE: this must be done by a Googler.

### Adding a new Presubmit

In this repository, in `.kokoro/`, add a Kokoro folder and build config file. At the very minimum your config file must provide 3 environment variables:

* `CMLE_TEST_DIR` the desired working directory for your test. Note the test will only be run if there are changes relevant to this working directory.
* `CMLE_TEST_SCRIPT` the testing script to run if changes relevant to your directory have been made.

You can see [`.kokoro/census/common.cfg`](.kokoro/census/common.cfg) for an example of such a file.

In Google3 you must add a corresponding job config in `//google3/devtools/kokoro/config/prod/cloud-devrel/cloudml-samples/` with the subdirectory *and* name matching your build config.

You can use `//google3/devtools/kokoro/config/prod/cloud-devrel/cloudml-samples/samples/census/common.cfg` as a template, and should only need to modify `build_config_dir` which will determine how your test is named in the Github UI.
