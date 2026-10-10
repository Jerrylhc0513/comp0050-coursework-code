#!/usr/bin/env python3
"""
check_email.py - email settings diagnosis (never prints the password)

Usage: python check_email.py
Checks: whether .env is read, the password length, whether ports 465 / 587 of the mail server are
reachable, and whether login works. The output contains no password, so it is safe to share.
"""
import os
import smtplib
import socket
import ssl
from pathlib import Path

ENV = Path(os.environ.get("ENV_FILE", Path(__file__).resolve().parent / ".env"))
try:
    from dotenv import load_dotenv
    if ENV.exists():
        load_dotenv(ENV, override=True)
        print(f"[1] Read {ENV}")
    else:
        print(f"[1] {ENV} not found! In the quant folder run: copy .env.example .env")
except ImportError:
    print("[1] python-dotenv is not installed: pip install python-dotenv")

sender = (os.environ.get("EMAIL_SENDER") or "").strip()
pw = (os.environ.get("EMAIL_APP_PASSWORD") or "").replace(" ", "")
host_env = os.environ.get("EMAIL_SMTP_HOST")
port_env = os.environ.get("EMAIL_SMTP_PORT")
domain = sender.split("@")[-1].lower() if "@" in sender else ""
DEFAULT_HOSTS = {"gmail.com": "smtp.gmail.com", "163.com": "smtp.163.com", "126.com": "smtp.126.com",
                 "qq.com": "smtp.qq.com", "foxmail.com": "smtp.qq.com",
                 "outlook.com": "smtp-mail.outlook.com", "hotmail.com": "smtp-mail.outlook.com"}
host = host_env or DEFAULT_HOSTS.get(domain, "smtp.gmail.com")

print(f"[2] Sender: {sender or '(empty)'}")
print(f"    Password length: {len(pw)}" + (" (a Gmail app password has 16)" if domain == "gmail.com" and len(pw) != 16 else ""))
print(f"    Server: {host}" + (" (from EMAIL_SMTP_HOST in .env)" if host_env else " (chosen from the sender address)"))
if host_env and domain and DEFAULT_HOSTS.get(domain) and host_env != DEFAULT_HOSTS[domain]:
    print(f"    ! The sender is {domain} but EMAIL_SMTP_HOST is {host_env}; they do not match. "
          f"Delete the EMAIL_SMTP_HOST and EMAIL_SMTP_PORT lines from .env")
if port_env:
    print(f"    Port: {port_env} (from .env)")

print("[3] Network:")
for port in (465, 587):
    try:
        socket.create_connection((host, port), timeout=10).close()
        print(f"    {host}:{port} reachable")
    except OSError as e:
        print(f"    {host}:{port} not reachable: {e}")

print("[4] Login:")
ctx = ssl.create_default_context()
for port in (465, 587):
    try:
        if port == 465:
            s = smtplib.SMTP_SSL(host, port, timeout=20, context=ctx)
        else:
            s = smtplib.SMTP(host, port, timeout=20)
            s.ehlo()
            s.starttls(context=ctx)
            s.ehlo()
        s.login(sender, pw)
        s.quit()
        print(f"    port {port}: login OK")
    except smtplib.SMTPAuthenticationError as e:
        print(f"    port {port}: connected but login rejected (wrong password): {e.smtp_code} {e.smtp_error[:120]!r}")
    except Exception as e:
        print(f"    port {port}: {type(e).__name__}: {e}")

print("\nHow to read this:")
print("  - both ports 'not reachable' or 'Connection unexpectedly closed': network problem. Turn off VPN / proxy, or try another network")
print("  - 'login rejected': password problem. Use an app password (Gmail) / authorisation code (163, QQ) with 2-step verification on")
print("  - 'login OK' on either port: settings are fine; python daily_signal.py test-email picks the working port")
