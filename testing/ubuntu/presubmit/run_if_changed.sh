#!/bin/bash
set -e

### Ignore this test if there are no relevant changes
cd ${CMLE_REPO_DIR}/${CMLE_TEST_BASE_DIR}

DIFF=`git diff master $KOKORO_GITHUB_PULL_REQUEST_COMMIT $PWD`

echo "DIFF:\n $DIFF"

if [ -z  $DIFF ]
then
    echo "TEST IGNORED; directory not modified in pull request $KOKORO_GITHUB_PULL_REQUEST_NUMBER"
    exit 0
fi

${CMLE_REPO_DIR}/testing/ubuntu/setup_and_run_test.sh
