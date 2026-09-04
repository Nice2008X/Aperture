# Security Policy

## Supported Versions

Aperture doesn't cut versioned releases — it's developed on `main`, and the
latest commit there is the only supported one. Security fixes land as
regular commits, not backports to older tags.

## Reporting a Vulnerability

Please **do not** report security vulnerabilities through public GitHub
issues, discussions, or pull requests.

Instead, report them privately using one of the following channels:

1. **GitHub Private Vulnerability Reporting (preferred)**
   Go to the [Security tab](https://github.com/Nice2008X/Aperture/security) of this repository and select **"Report a vulnerability"** to open a private advisory. This keeps the report and any discussion confidential until a fix is available.

2. **Email**
   If you're unable to use GitHub's private reporting, email **aixu2008nice@gmail.com** with details of the issue. Please include "SECURITY" in the subject line.

Please include as much of the following as you can:

- A description of the vulnerability and its potential impact
- Steps to reproduce, including a minimal example if possible
- The affected version/commit
- Any suggested fix or mitigation, if known

### What to expect

- Acknowledgement of your report within a few days.
- An initial assessment of the issue and its severity.
- Coordination on a fix and disclosure timeline before any public details are shared. We ask that you give us reasonable time to address the issue before any public disclosure.

### Scope

Aperture is a local-first tool: a Python/FastAPI backend that downloads
Hugging Face checkpoints onto `data/models/`, loads them onto a real GPU
with PyTorch + `transformers`, and a browser frontend that talks to that
backend over HTTP/SSE. It's designed to run on `localhost` for a single
user, not as an internet-facing multi-tenant service — the backend's CORS
policy is deliberately permissive for that local-dev setup. Reports of
particular interest include (but aren't limited to):

- Path traversal or arbitrary file access/deletion via the backend's model
  download, load, or delete endpoints
- Repo-id or file-path validation bypasses that reach the filesystem or an
  outbound network request in an unintended way
- Any way a downloaded model repo's contents (`config.json`, tokenizer
  files, weights) could trigger code execution rather than just being
  parsed as data
- Cross-site scripting (XSS) or other injection issues in the frontend
- Dependency vulnerabilities with a credible exploit path in this project
- Supply-chain issues in the build or release process
- Anything that would make the CORS/local-dev assumptions above unsafe for
  a more exposed deployment (e.g. binding beyond localhost by default)

Thank you for helping keep Aperture and its users safe.
