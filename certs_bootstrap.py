"""
certs_bootstrap.py
==================

Some Windows machines run antivirus / corporate proxies that intercept HTTPS and
present their OWN root certificate. That root lives in the Windows certificate
store, but Python libraries (requests, and especially yfinance's curl_cffi
backend) ship their own CA bundle and don't look at the Windows store — so every
outbound HTTPS call fails with:

    SSL: CERTIFICATE_VERIFY_FAILED / unable to get local issuer certificate

This module fixes that WITHOUT disabling certificate verification. It exports
every trusted root from the Windows "ROOT" and "CA" stores into a single .pem
bundle and points the relevant environment variables at it, so both `requests`
and `curl_cffi` (yfinance) verify against the exact certs Windows trusts.

Import this module FIRST, before importing requests / yfinance:

    import certs_bootstrap  # noqa: F401  (must be first)

It is a no-op on non-Windows platforms and fails safe (never raises).
"""

from __future__ import annotations

import base64
import os
import sys

_BUNDLE_NAME = "win_ca_bundle.pem"


def _build_windows_ca_bundle() -> None:
    # Only relevant on Windows; other OSes use their normal system trust.
    if not sys.platform.startswith("win"):
        return

    try:
        import ssl
    except Exception:
        return

    if not hasattr(ssl, "enum_certificates"):
        return

    try:
        pem_blocks = []
        seen = set()
        for store in ("ROOT", "CA"):
            try:
                for cert_bytes, enc_type, _trust in ssl.enum_certificates(store):
                    if enc_type != "x509_asn":
                        continue
                    if cert_bytes in seen:
                        continue
                    seen.add(cert_bytes)
                    b64 = base64.encodebytes(cert_bytes).decode("ascii").strip()
                    pem_blocks.append(
                        "-----BEGIN CERTIFICATE-----\n"
                        + b64
                        + "\n-----END CERTIFICATE-----\n"
                    )
            except Exception:
                # A single store failing shouldn't abort the whole bundle.
                continue

        if not pem_blocks:
            return

        bundle_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), _BUNDLE_NAME)
        with open(bundle_path, "w", encoding="ascii") as fh:
            fh.write("\n".join(pem_blocks))

        # requests (and urllib3) honour REQUESTS_CA_BUNDLE.
        # curl_cffi (used by yfinance) honours CURL_CA_BUNDLE / SSL_CERT_FILE.
        os.environ.setdefault("REQUESTS_CA_BUNDLE", bundle_path)
        os.environ.setdefault("CURL_CA_BUNDLE", bundle_path)
        os.environ.setdefault("SSL_CERT_FILE", bundle_path)
    except Exception:
        # Never let cert bootstrapping crash the app; worst case the user sees
        # the original SSL errors and can fall back to another network.
        pass


_build_windows_ca_bundle()
