---
name: Bug report
about: For problems running the sample code please provide the following information.

---

**Describe the bug**
A clear and concise description of what the bug is. Be sure to convey here whether it occurred locally or on the server (AI Platform, Google Dataflow)

**What sample is this bug related to?**

**Source code / logs**
Include any logs or source code that would be helpful to diagnose the problem. If including tracebacks, please include the full traceback. Large logs and files should be attached. Try to provide a reproducible test case that is the bare minimum necessary to generate the problem.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

**Expected behavior**
A clear and concise description of what you expected to happen.

**System Information**
- **OS Platform and Distribution (e.g., Linux Ubuntu 16.04)**:
- **Framework and version (Tensorflow, scikit-learn, XGBoost)**:
- **Python version**:
- **Exact command to reproduce**:
- **Tensorflow Transform environment (if applicable, see below)**:

To obtain the Tensorflow and Tensorflow Transform environment do

```
pip freeze |grep tensorflow
pip freeze |grep apache-beam
```
**Additional context**
Add any other context about the problem here.
