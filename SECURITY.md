# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in BayesianGPU, please report it responsibly:

1. **Do NOT open a public GitHub issue**
2. Email security concerns to **security@bayesiangpu.io**
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

We will acknowledge receipt within 48 hours and provide a detailed response within 7 days.

## Security Measures

### Docker Images

- **Non-root users**: All images run as non-privileged users
- **Minimal base images**: Reduced attack surface
- **Signed images**: Container signatures via Cosign/Sigstore
- **Security scanning**: Trivy scans before push

## Security Best Practices for Users

- Keep dependencies up to date
- Review WASM binary integrity before deployment
- Use Subresource Integrity (SRI) hashes when loading from CDN
- Report any suspected vulnerabilities through the process above

## Updates

This security policy is reviewed quarterly. Last update: 2026-01-14.
