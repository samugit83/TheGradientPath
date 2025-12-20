# Clickjacking (UI Redressing)

## Table of Contents

1. [Overview Diagram](#overview-diagram)
2. [Introduction and Core Concepts](#introduction-and-core-concepts)
   - [Definition](#definition)
   - [How the Attack Works](#how-the-attack-works)
   - [Impact](#impact)
   - [Attack Vectors](#attack-vectors)
3. [Defense Principles](#defense-principles)
   - [Core Protection Principles](#core-protection-principles)
   - [When and Where to Apply Defenses](#when-and-where-to-apply-defenses)
4. [Mitigation Strategies](#mitigation-strategies)
   - [Primary Mitigation: Content Security Policy (CSP) frame-ancestors](#primary-mitigation-content-security-policy-csp-frame-ancestors)
   - [Secondary Mitigation: X-Frame-Options Header](#secondary-mitigation-x-frame-options-header)
   - [Client-Side Defenses: Frame Busting Scripts](#client-side-defenses-frame-busting-scripts)
   - [Additional Security Measures](#additional-security-measures)
   - [Implementation Considerations](#implementation-considerations)
5. [Real-World Attack Scenarios](#real-world-attack-scenarios)
   - [Scenario 1: Social Media Like/Follow Hijacking](#scenario-1-social-media-likefollow-hijacking)
   - [Scenario 2: Banking Payment Authorization](#scenario-2-banking-payment-authorization)
   - [Scenario 3: Account Settings Manipulation](#scenario-3-account-settings-manipulation)
   - [Scenario 4: Multi-Step Form Submission Attack](#scenario-4-multi-step-form-submission-attack)
6. [Operating Techniques](#operating-techniques)
   - [Standard Approach](#standard-approach)
   - [AI Approach](#ai-approach)
7. [References](#references)

---

## Overview Diagram

```mermaid
flowchart TB
    subgraph Legend["Legend"]
        direction LR
        L1[Attack Types]
        L2[Attack Vectors]
        L3[Attack Context]
        L4[Defense Mechanisms]
        L5[Impact Categories]
        
        style L1 fill:#ffffff,stroke:#ff0000,stroke-width:3px,color:#000000
        style L2 fill:#ffffff,stroke:#ff8c00,stroke-width:3px,color:#000000
        style L3 fill:#ffffff,stroke:#0066cc,stroke-width:3px,color:#000000
        style L4 fill:#ffffff,stroke:#228b22,stroke-width:3px,color:#000000
        style L5 fill:#ffffff,stroke:#800080,stroke-width:3px,color:#000000
    end

    subgraph AttackTypes["Attack Types"]
        direction TB
        CJ[Clickjacking<br/>UI Redressing]
        LJ[Likejacking]
        CU[Cursorjacking]
        FG[Filejacking]
        DJ[Double Clickjacking]
        
        style CJ fill:#ffffff,stroke:#ff0000,stroke-width:3px,color:#000000
        style LJ fill:#ffffff,stroke:#ff0000,stroke-width:3px,color:#000000
        style CU fill:#ffffff,stroke:#ff0000,stroke-width:3px,color:#000000
        style FG fill:#ffffff,stroke:#ff0000,stroke-width:3px,color:#000000
        style DJ fill:#ffffff,stroke:#ff0000,stroke-width:3px,color:#000000
    end

    subgraph Vectors["Attack Vectors"]
        direction TB
        TI[Transparent iframes]
        OL[Overlay Elements]
        OP[Opacity Manipulation]
        ZI[Z-Index Stacking]
        PO[Pointer Events]
        
        style TI fill:#ffffff,stroke:#ff8c00,stroke-width:3px,color:#000000
        style OL fill:#ffffff,stroke:#ff8c00,stroke-width:3px,color:#000000
        style OP fill:#ffffff,stroke:#ff8c00,stroke-width:3px,color:#000000
        style ZI fill:#ffffff,stroke:#ff8c00,stroke-width:3px,color:#000000
        style PO fill:#ffffff,stroke:#ff8c00,stroke-width:3px,color:#000000
    end

    subgraph Context["Attack Context"]
        direction TB
        MA[Malicious Website]
        PE[Phishing Email]
        AD[Malvertising]
        SM[Social Media]
        
        style MA fill:#ffffff,stroke:#0066cc,stroke-width:3px,color:#000000
        style PE fill:#ffffff,stroke:#0066cc,stroke-width:3px,color:#000000
        style AD fill:#ffffff,stroke:#0066cc,stroke-width:3px,color:#000000
        style SM fill:#ffffff,stroke:#0066cc,stroke-width:3px,color:#000000
    end

    subgraph Defenses["Defense Mechanisms"]
        direction TB
        CSP[CSP frame-ancestors]
        XFO[X-Frame-Options]
        FB[Frame Busting Scripts]
        SCS[SameSite Cookies]
        UAC[User Action Confirmation]
        
        style CSP fill:#ffffff,stroke:#228b22,stroke-width:3px,color:#000000
        style XFO fill:#ffffff,stroke:#228b22,stroke-width:3px,color:#000000
        style FB fill:#ffffff,stroke:#228b22,stroke-width:3px,color:#000000
        style SCS fill:#ffffff,stroke:#228b22,stroke-width:3px,color:#000000
        style UAC fill:#ffffff,stroke:#228b22,stroke-width:3px,color:#000000
    end

    subgraph Impacts["Attack Impacts"]
        direction TB
        UA[Unauthorized Actions]
        DT[Data Theft]
        AC[Account Compromise]
        FT[Financial Theft]
        PM[Privacy Violations]
        
        style UA fill:#ffffff,stroke:#800080,stroke-width:3px,color:#000000
        style DT fill:#ffffff,stroke:#800080,stroke-width:3px,color:#000000
        style AC fill:#ffffff,stroke:#800080,stroke-width:3px,color:#000000
        style FT fill:#ffffff,stroke:#800080,stroke-width:3px,color:#000000
        style PM fill:#ffffff,stroke:#800080,stroke-width:3px,color:#000000
    end

    %% Attack Type relationships
    CJ --> LJ
    CJ --> CU
    CJ --> FG
    CJ --> DJ

    %% Attack vectors enable attacks
    TI --> CJ
    OL --> CJ
    OP --> CJ
    ZI --> CJ
    PO --> CU

    %% Context to Vectors
    MA --> TI
    PE --> MA
    AD --> MA
    SM --> MA

    %% Attacks lead to Impacts
    CJ --> UA
    CJ --> DT
    LJ --> PM
    CU --> AC
    FG --> DT
    DJ --> FT

    %% Defenses block attacks
    CSP -.->|blocks| TI
    XFO -.->|blocks| TI
    FB -.->|detects| TI
    SCS -.->|limits| UA
    UAC -.->|prevents| FT

    %% Subgraph styling - transparent background with border only
    style Legend fill:#ffffff10,stroke:#666666,stroke-width:2px
    style AttackTypes fill:#ffffff10,stroke:#ff0000,stroke-width:2px
    style Vectors fill:#ffffff10,stroke:#ff8c00,stroke-width:2px
    style Context fill:#ffffff10,stroke:#0066cc,stroke-width:2px
    style Defenses fill:#ffffff10,stroke:#228b22,stroke-width:2px
    style Impacts fill:#ffffff10,stroke:#800080,stroke-width:2px
```

### Key Relationships

| Connection | Description |
|------------|-------------|
| **Attack Types → Variants** | Clickjacking is the parent attack with specialized variants (Likejacking, Cursorjacking, Filejacking, Double Clickjacking) targeting specific actions |
| **Vectors → Attack Types** | Transparent iframes and overlay elements are the technical enablers that make clickjacking possible |
| **Context → Vectors** | Malicious websites host the attack infrastructure; phishing emails and social media drive traffic to these sites |
| **Attack Types → Impacts** | Successful attacks result in unauthorized actions, data theft, account compromise, and financial losses |
| **Defenses → Vectors** | CSP frame-ancestors and X-Frame-Options prevent iframe embedding; frame busting detects embedding attempts |

---

## Introduction and Core Concepts

### Definition

**Clickjacking** (also known as **UI Redressing**) is a malicious technique in which an attacker tricks a user into clicking on something different from what the user perceives, potentially revealing confidential information or taking control of their computer while clicking on seemingly innocuous objects, including web pages.

The attack exploits the browser's ability to embed external content (typically through HTML iframes) and layer it invisibly over a deceptive interface. The victim believes they are interacting with a visible page but are actually clicking on hidden elements from a legitimate, targeted website where they may already be authenticated.

> [!IMPORTANT]
> Clickjacking is classified under **CWE-1021: Improper Restriction of Rendered UI Layers or Frames** and is listed in the OWASP Top 10 under **A05:2021 – Security Misconfiguration**.

### How the Attack Works

The clickjacking attack operates through a multi-layer deception technique:

1. **Target Identification**: The attacker identifies a sensitive action on a target website (e.g., "Transfer Funds", "Delete Account", "Change Email")

2. **Malicious Page Construction**: The attacker creates a web page that:
   - Embeds the target website in a transparent or hidden iframe
   - Positions the target's sensitive button directly under a visible, attractive element
   - Uses CSS properties like `opacity: 0`, `z-index`, and precise positioning

3. **Victim Luring**: The attacker lures the victim to the malicious page through:
   - Phishing emails
   - Social engineering
   - Malicious advertisements
   - Compromised websites

4. **Invisible Interaction**: When the victim clicks on the visible element (e.g., "Claim Your Prize"), they actually click on the hidden target button, executing the attacker's intended action

5. **Session Exploitation**: Since the victim is already authenticated on the target site (cookies are sent automatically), the action is executed with the victim's privileges

### Impact

Clickjacking attacks can result in severe consequences:

| Impact Category | Description | Severity |
|----------------|-------------|----------|
| **Financial Loss** | Unauthorized fund transfers, purchases, or payment authorizations | Critical |
| **Account Takeover** | Password or email changes, security setting modifications | Critical |
| **Data Breach** | Exposure of personal information, privacy violations | High |
| **Reputation Damage** | Unauthorized social media posts, likes, or follows | Medium |
| **Malware Installation** | Triggering drive-by downloads or plugin installations | High |
| **Privilege Escalation** | Granting permissions to malicious applications | High |

### Attack Vectors

Clickjacking attacks utilize various technical vectors:

| Vector | Technique | Description |
|--------|-----------|-------------|
| **Classic Clickjacking** | Transparent iframe overlay | Target site rendered in invisible iframe over malicious visible content |
| **Likejacking** | Social media targeting | Hijacking social media "Like" or "Follow" buttons for spam propagation |
| **Cursorjacking** | Cursor displacement | Displaying a fake cursor offset from the real cursor position |
| **Filejacking** | File dialog manipulation | Tricking users into selecting files for unauthorized upload |
| **Double Clickjacking** | Double-click exploitation | Using the first click to reposition elements before the second click |
| **Drag-and-Drop Hijacking** | HTML5 drag-drop abuse | Exploiting drag-and-drop operations to move data to attacker-controlled areas |
| **Stroke Jacking** | Keyboard input capture | Capturing keystrokes by focusing hidden input fields under visible buttons |

---

## Defense Principles

### Core Protection Principles

Effective clickjacking defense is built on these fundamental principles:

1. **Frame Control**: Explicitly declare whether and how your content can be embedded in frames
   - Adopt a "deny by default" approach
   - Whitelist only trusted origins when framing is legitimately required

2. **Defense in Depth**: Implement multiple layers of protection
   - Server-side headers as primary defense
   - Client-side frame detection as secondary measure
   - UI design considerations as tertiary protection

3. **Least Privilege**: Sensitive actions should require additional verification
   - Re-authentication for critical operations
   - Out-of-band confirmation for high-risk actions
   - Time-delayed or multi-step processes

4. **Session Integrity**: Ensure session context matches expected interaction patterns
   - Implement anti-CSRF tokens alongside anti-clickjacking measures
   - Use SameSite cookie attributes to limit cross-origin abuse

### When and Where to Apply Defenses

| Scenario | Recommended Defense Level | Justification |
|----------|--------------------------|---------------|
| **All pages** | CSP frame-ancestors | Baseline protection should be universal |
| **Authentication pages** | Strict deny + additional verification | Login/logout pages are high-value targets |
| **Payment/Financial actions** | Strict deny + re-authentication | Financial operations require maximum protection |
| **Account settings** | Strict deny | Settings changes can lead to account takeover |
| **Public embeddable widgets** | Selective allow with CSP | Must whitelist specific trusted origins |
| **APIs and non-UI endpoints** | Headers still recommended | Defense in depth even for non-HTML responses |

> [!WARNING]
> Clickjacking protections should be applied at the server level. Client-side-only defenses can be bypassed and should only serve as supplementary measures.

---

## Mitigation Strategies

### Primary Mitigation: Content Security Policy (CSP) frame-ancestors

The `frame-ancestors` directive in CSP is the modern, recommended approach to prevent clickjacking. It specifies valid parents that may embed a page using frame, iframe, object, or embed elements.

**How It Works**:
- Server sends CSP header with `frame-ancestors` directive
- Browser checks if the current page's parent is in the allowed list
- If not allowed, the browser refuses to render the page in the frame

**Directive Values**:

| Value | Effect |
|-------|--------|
| `'none'` | Prevents all framing (most restrictive) |
| `'self'` | Only allows framing by the same origin |
| `https://trusted.com` | Allows framing only by the specified origin |
| `https://*.trusted.com` | Allows framing by any subdomain of the specified domain |

**Advantages**:
- More flexible than X-Frame-Options
- Supports multiple origins
- Supports wildcards for subdomains
- Part of the broader CSP security framework
- Cannot be bypassed by double-framing attacks

**Limitations**:
- Not supported by very old browsers (IE 11 and below)
- Requires careful configuration to avoid breaking legitimate embedding

### Secondary Mitigation: X-Frame-Options Header

X-Frame-Options is the legacy method for clickjacking protection. While superseded by CSP frame-ancestors, it provides important backward compatibility.

**Directive Values**:

| Value | Effect |
|-------|--------|
| `DENY` | Prevents all framing |
| `SAMEORIGIN` | Only allows framing by the same origin |
| `ALLOW-FROM uri` | Allows framing only by the specified URI (deprecated, limited support) |

**Best Practice**: Deploy both headers for maximum compatibility:
- Use CSP `frame-ancestors` as primary defense
- Include X-Frame-Options as fallback for older browsers

> [!NOTE]
> When both headers are present, CSP `frame-ancestors` takes precedence in browsers that support it.

### Client-Side Defenses: Frame Busting Scripts

Frame busting scripts attempt to detect when a page is being framed and break out of the frame. While not reliable as a sole defense, they provide defense in depth.

**Common Techniques**:
- Checking `window.self !== window.top`
- Attempting to redirect the top window to the framed page
- Hiding page content when framing is detected

**Limitations**:
- Can be bypassed using iframe sandbox attributes
- May be disabled by browser extensions
- JavaScript-dependent (fails if JS is blocked)
- Double-framing attacks can circumvent simple checks

> [!CAUTION]
> Never rely on frame busting scripts as your only protection. Attackers can neutralize them using `sandbox="allow-scripts"` in the iframe, which prevents the script from accessing `window.top`.

### Additional Security Measures

| Measure | Description | Effectiveness |
|---------|-------------|---------------|
| **SameSite Cookies** | Set `SameSite=Strict` or `SameSite=Lax` to prevent cookies from being sent in cross-site framing contexts | High |
| **Re-authentication** | Require password or MFA for sensitive actions | High |
| **User Interaction Verification** | CAPTCHA or deliberate user gestures before critical actions | Medium |
| **Referrer Validation** | Check that requests originate from expected sources | Medium |
| **UI Design** | Use distinctive, hard-to-replicate interfaces for sensitive actions | Low-Medium |

### Implementation Considerations

**Priority Order for Implementation**:

1. **CSP frame-ancestors** – Primary, modern defense
2. **X-Frame-Options** – Secondary, legacy compatibility
3. **SameSite Cookies** – Session protection layer
4. **Frame Busting** – Client-side supplementary defense
5. **Re-authentication** – High-value action protection

**Configuration Recommendations**:

| Application Type | Recommended Configuration |
|------------------|---------------------------|
| Internal applications | `frame-ancestors 'none'` |
| Public websites | `frame-ancestors 'none'` or `'self'` |
| Embeddable widgets | `frame-ancestors 'self' https://allowed-partner.com` |
| API endpoints | Include headers even on non-HTML responses |

> [!TIP]
> Test your configuration using browser developer tools. Check the Network tab to verify headers are correctly sent, and use the Console to catch any CSP violation reports.

---

## Real-World Attack Scenarios

### Scenario 1: Social Media Like/Follow Hijacking

**Context**: An attacker wants to artificially inflate their social media presence by hijacking the "Follow" button on a popular social platform.

#### Attack Flow

1. Attacker creates a malicious webpage disguised as a game or contest
2. The page contains a transparent iframe loading the target social media profile
3. The "Follow" button is precisely positioned under a visible "Play Now" button
4. Victims, already logged into the social platform, click "Play Now"
5. Their click actually triggers the hidden "Follow" button
6. The victim unknowingly follows the attacker's account

```mermaid
sequenceDiagram
    autonumber
    participant A as Attacker
    participant V as Victim
    participant M as Malicious Site
    participant S as Social Media Platform

    A->>M: Create malicious page with transparent iframe
    A->>A: Position "Follow" button under "Play Now" button
    A->>V: Send phishing link via email/social media
    V->>M: Visit malicious site
    M->>S: Load social media profile in hidden iframe
    S->>M: Return profile page (victim already authenticated)
    V->>M: Click visible "Play Now" button
    M->>S: Click registers on hidden "Follow" button
    S->>S: Process follow action with victim's session
    S->>V: Victim now follows attacker's account
    V->>V: Unaware of the unauthorized action
```

#### Mitigation Application

The social media platform implements CSP frame-ancestors to prevent their pages from being embedded:

```mermaid
sequenceDiagram
    autonumber
    participant A as Attacker
    participant V as Victim
    participant M as Malicious Site
    participant S as Social Media Platform
    participant B as Victim's Browser

    A->>M: Create malicious page with transparent iframe
    A->>V: Send phishing link
    V->>M: Visit malicious site
    M->>B: Page attempts to load social media in iframe
    B->>S: Request profile page for iframe
    S->>B: Response with CSP: frame-ancestors 'self'
    B->>B: Check if parent origin matches frame-ancestors
    B->>B: Malicious site origin NOT in allowed list
    B->>M: Refuse to render social media page in iframe
    M->>V: Display empty iframe or error
    V->>V: Attack fails - no hidden button to click
    Note over B,S: CSP frame-ancestors blocks the embedding attempt
```

---

### Scenario 2: Banking Payment Authorization

**Context**: An attacker attempts to trick a victim into authorizing a fraudulent money transfer on their online banking portal.

#### Attack Flow

1. Attacker identifies the transfer confirmation button's position on the bank's website
2. Creates a fake "prize claim" page with the bank's transfer page in an invisible iframe
3. The "Confirm Transfer" button aligns with the "Claim $1000 Prize" button
4. Attacker pre-fills the transfer form using URL parameters or POST manipulation
5. Victim clicks to claim their "prize" but authorizes a real transfer

```mermaid
sequenceDiagram
    autonumber
    participant A as Attacker
    participant V as Victim
    participant M as Malicious Site
    participant B as Banking Portal

    A->>M: Create prize claim page with invisible iframe
    A->>M: Pre-configure transfer (recipient: attacker's account)
    A->>V: Email "You've won $1000! Click to claim"
    V->>M: Click link, visit malicious page
    M->>B: Load banking transfer page in hidden iframe
    B->>M: Return transfer page (victim is logged in)
    Note over M: Transfer form pre-filled with attacker's details
    V->>M: Click "Claim $1000 Prize" button
    M->>B: Click hits "Confirm Transfer" button
    B->>B: Validate session cookie (valid)
    B->>B: Process transfer request
    B->>A: Funds transferred to attacker's account
    V->>V: Realizes money is missing (too late)
```

#### Mitigation Application

The bank implements multiple layers of defense:

```mermaid
sequenceDiagram
    autonumber
    participant A as Attacker
    participant V as Victim
    participant M as Malicious Site
    participant B as Banking Portal
    participant BR as Victim's Browser

    A->>M: Create malicious page with iframe
    V->>M: Visit malicious page
    M->>BR: Attempt to load banking portal in iframe
    BR->>B: Request transfer page
    B->>BR: Response includes multiple headers
    Note over B,BR: X-Frame-Options: DENY<br/>CSP: frame-ancestors 'none'<br/>Set-Cookie: SameSite=Strict
    BR->>BR: X-Frame-Options DENY check
    BR->>BR: CSP frame-ancestors 'none' check
    BR->>BR: Parent origin is malicious site ≠ bank
    BR->>M: Block iframe content rendering
    Note over BR: Additionally, SameSite cookies<br/>wouldn't be sent cross-origin
    M->>V: Empty iframe displayed
    V->>M: Click "Claim Prize" hits nothing
    Note over V,B: Attack completely neutralized
```

---

### Scenario 3: Account Settings Manipulation

**Context**: An attacker wants to change a victim's email address on a SaaS platform to enable subsequent account takeover via password reset.

#### Attack Flow

1. Attacker identifies the account settings page with email change functionality
2. Creates a page with a fake survey requiring multiple clicks
3. Hidden iframe contains the email settings form
4. Sequential clicks fill in the attacker's email and submit the form
5. Victim's account email is changed; attacker requests password reset

```mermaid
sequenceDiagram
    autonumber
    participant A as Attacker
    participant V as Victim
    participant M as Malicious Site
    participant S as SaaS Platform
    participant E as Email Provider

    A->>M: Create fake survey with multiple click targets
    A->>M: Each click maps to email change form actions
    A->>V: Share "Quick survey - win a gift card!"
    V->>M: Visit malicious survey page
    M->>S: Load account settings in transparent iframe
    S->>M: Return settings page (victim logged in)
    V->>M: Click "Start Survey" (clicks hidden email field)
    V->>M: Click answer buttons (types attacker's email)
    V->>M: Click "Submit Survey" (clicks "Save Changes")
    M->>S: Form submission with new email
    S->>S: Update user email to attacker@evil.com
    S->>E: Send confirmation to new email
    A->>E: Receive confirmation, complete change
    A->>S: Request password reset
    S->>A: Reset link sent to attacker@evil.com
    A->>S: Reset password, take over account
```

#### Mitigation Application

The SaaS platform implements comprehensive protections:

```mermaid
sequenceDiagram
    autonumber
    participant A as Attacker
    participant V as Victim
    participant M as Malicious Site
    participant S as SaaS Platform
    participant BR as Victim's Browser

    A->>M: Create malicious survey page
    V->>M: Visit page
    M->>BR: Attempt to iframe SaaS settings page
    BR->>S: Request settings page
    S->>BR: CSP: frame-ancestors 'self'
    BR->>BR: Origin check fails (malicious ≠ SaaS)
    BR->>M: Iframe blocked
    
    Note over S: Additional defense layer
    alt If frame-ancestors bypassed somehow
        V->>M: Click actions
        M->>S: Attempt to submit email change
        S->>S: Check anti-CSRF token
        S->>S: Token missing/invalid (cross-origin)
        S->>M: Reject request (403 Forbidden)
        
        Note over S: Third defense layer
        S->>V: Email change requires password re-entry
        M->>S: Cannot provide password
        S->>M: Change rejected
    end
    
    Note over V,S: Multiple layers ensure protection
```

---

### Scenario 4: Multi-Step Form Submission Attack

**Context**: An attacker wants to trick a victim into submitting a multi-step form that grants OAuth permissions to a malicious application.

#### Attack Flow

1. Attacker identifies an OAuth consent flow with multiple confirmation steps
2. Creates an engaging "memory game" that requires clicking in sequence
3. Each game click corresponds to a step in the OAuth authorization flow
4. The victim completes the "game" while unknowingly granting full account access
5. Attacker's application gains access to victim's data and actions

```mermaid
sequenceDiagram
    autonumber
    participant A as Attacker
    participant V as Victim
    participant M as Malicious Game Site
    participant O as OAuth Provider
    participant AP as Attacker's App

    A->>AP: Register malicious OAuth application
    A->>M: Create memory game matching OAuth flow clicks
    A->>M: Position game elements over OAuth buttons
    A->>V: Advertise "Fun memory challenge!"
    V->>M: Visit game site
    M->>O: Load OAuth consent page in hidden iframe
    O->>M: Return consent form (victim logged in)
    V->>M: Game Click 1: "Select scope"
    M->>O: Click selects "read/write" permissions
    V->>M: Game Click 2: "Flip next card"
    M->>O: Click confirms application access
    V->>M: Game Click 3: "Win!"
    M->>O: Click submits final authorization
    O->>AP: Issue access token to attacker's app
    AP->>O: Use token to access victim's data
    A->>A: Full account access obtained
```

#### Mitigation Application

The OAuth provider implements robust protections:

```mermaid
sequenceDiagram
    autonumber
    participant A as Attacker
    participant V as Victim
    participant M as Malicious Game Site
    participant O as OAuth Provider
    participant BR as Victim's Browser

    A->>M: Create malicious memory game
    V->>M: Visit game site
    M->>BR: Attempt to load OAuth consent in iframe
    BR->>O: Request OAuth consent page
    O->>BR: Response headers set
    Note over O,BR: CSP: frame-ancestors 'none'<br/>X-Frame-Options: DENY
    BR->>BR: Framing check fails
    BR->>M: Refuse to render OAuth page
    
    Note over O: Defense in depth measures
    O->>O: OAuth consent requires<br/>user interaction verification
    O->>O: State parameter validates<br/>request origin
    O->>O: Consent page uses<br/>unique visual design
    
    alt Direct access attempt
        V->>O: If victim visits OAuth directly
        O->>V: Show clear application name/permissions
        O->>V: Require explicit typed confirmation
        V->>V: Recognize suspicious request
        V->>O: Deny authorization
    end
    
    Note over V,O: Multi-layer protection prevents attack
```

---

## Operating Techniques

This section details the operational methods used to execute clickjacking attacks, covering both standard tooling and emerging AI-driven approaches.

### Standard Approach

The standard approach relies on specialized security testing tools to automate the creation of clickjacking Proof-of-Concepts (PoCs). The most prominent industry tool for this purpose is **Burp Suite Clickbandit**.

#### Tools Used
*   **Burp Suite Professional/Community**: A comprehensive platform for web security testing.
*   **Clickbandit**: A JavaScript-based tool integrated into Burp Suite that simplifies the creation of clickjacking attacks.
*   **Web Browser**: Any modern browser (Chrome, Firefox, etc.) to render the target page.

#### Operational Instructions

The following workflow describes how to use Burp Clickbandit to generate a PoC:

1.  **Initialize Clickbandit**: In Burp Suite, navigating to the **Burp** menu and selecting **Burp Clickbandit**.
2.  **Copy Script**: Click the "Copy Clickbandit to clipboard" button in the dialog that appears.
3.  **Prepare Target**: Open your web browser and navigate to the target application page you wish to test (ensure you are authenticated if testing sensitive actions).
4.  **Inject Script**: Open the browser's Developer Tools (F12), go to the **Console** tab, paste the script, and hit Enter.
5.  **Record Actions**:
    *   A Clickbandit banner will appear at the top of the page.
    *   Click "Start" to begin recording.
    *   Perform the sequence of clicks you want the victim to execute (e.g., clicking "Delete Account", then "Confirm").
6.  **Finalize Attack**: Click "Finish" on the banner. The tool will now display an overlay of the attack structure.
7.  **Review and Adjust**:
    *   Use the transparency slider to see how the hidden iframe aligns with the decoy visible buttons.
    *   Adjust the zoom or position if necessary.
8.  **Export PoC**: Click "Save" to download the generated HTML file. This file contains the complete clickjacking attack code ready for deployment.

#### Workflow Diagram

```mermaid
flowchart TD
    subgraph Setup["1. Setup Phase"]
        direction TB
        B[Open Burp Suite] --> C[Select 'Burp Clickbandit']
        C --> D[Copy JS Snippet]
        D --> E[Open Target URL]
    end

    subgraph Injection["2. Injection Phase"]
        direction TB
        E --> F[Open DevTools <br/> Console]
        F --> G[Paste & Run Snippet]
        G --> H[Clickbandit Interface <br/> Loads in Browser]
    end

    subgraph Execution["3. Execution Phase"]
        direction TB
        H --> I[Click 'Start Recording']
        I --> J["Perform Target Clicks <br/> (e.g., Transfer Funds)"]
        J --> K[Click 'Finish']
    end

    subgraph Output["4. Output Phase"]
        direction TB
        K --> L[Review Overlay <br/> Alignment]
        L --> M[Save HTML PoC]
    end

    Setup --> Injection
    Injection --> Execution
    Execution --> Output

    style Setup fill:#ffffff,stroke:#333,stroke-width:2px
    style Injection fill:#ffffff,stroke:#333,stroke-width:2px
    style Execution fill:#ffffff,stroke:#333,stroke-width:2px
    style Output fill:#ffffff,stroke:#333,stroke-width:2px
```

### AI Approach

The AI approach leverages Large Language Models (LLMs) and computer vision capabilities to streamline the complex manual work of positioning and CSS generation. While specific "AI Clickjacking Tools" are less standardized than defensive ones, attackers use AI agents to dynamically analyze and construct attacks.

#### Tools Used
*   **Multimodal LLMs (e.g., GPT-4o, Claude 3.5 Sonnet)**: For analyzing screenshots and DOM structures.
*   **Python with Selenium/Playwright**: For automated browser interaction and coordinate extraction.
*   **Computer Vision Libraries (OpenCV)**: For precise visual element detection.

#### Process: AI-Assisted Dynamic Analysis

In this workflow, an AI agent replaces the manual "recording" phase by analytically determining the perfect overlay coordinates.

1.  **Target Acquisition**: The attacker provides the AI agent with the target URL and the specific objective (e.g., "Click the 'Follow' button").
2.  **Visual Analysis**:
    *   The AI uses a headless browser to capture a screenshot and the DOM tree of the target page.
    *   It identifies the target element's exact pixel coordinates `(x, y)` and dimensions `(w, h)` relative to the viewport.
3.  **Context Construction**:
    *   The AI generates a "Lure Scenario" based on the target audience (e.g., a "Claim Prize" button matching the dimensions of the "Transfer" button).
4.  **Code Generation**:
    *   The AI writes the complete HTML/CSS structure, automatically calculating the negative margins and `z-index` required to center the target element under the lure.
    *   It can generate adaptive JavaScript that recalculates coordinates if the browser window is resized, a common failure point for static PoCs.
5.  **Validation**: The AI simulates a user visit to the generated page to verify the click intercepts correctly before final deployment.

#### AI-Driven Workflow Diagram

```mermaid
flowchart TD
    subgraph Input["1. Target Input"]
        U[Attacker] -->|URL + Goal| AI[AI Agent]
    end

    subgraph Analysis["2. AI Analysis"]
        direction TB
        AI -->|Headless Browser| DOM[Fetch DOM Tree]
        AI -->|Computer Vision| VIS[Analyze Screenshot]
        DOM & VIS --> COORD[Calculate Exact <br/> Target Coordinates]
    end

    subgraph Generation["3. Asset Generation"]
        direction TB
        COORD --> LURE[Design Matching <br/> Lure UI]
        COORD --> CSS[Generate Adaptive <br/> CSS/JS Code]
    end

    subgraph Validation["4. Verification"]
        direction TB
        CSS & LURE --> POC[Assemble Attack Page]
        POC --> SIM[Simulate Victim Click]
        SIM -->|Success?| FINAL[Export Optimized PoC]
        SIM -->|Fail?| RE[Adjust Coordinates]
        RE -.-> COORD
    end

    Input --> Analysis
    Analysis --> Generation
    Generation --> Validation

    style Input fill:#ffffff,stroke:#800080,stroke-width:2px
    style Analysis fill:#ffffff,stroke:#800080,stroke-width:2px
    style Generation fill:#ffffff,stroke:#800080,stroke-width:2px
    style Validation fill:#ffffff,stroke:#800080,stroke-width:2px
```

---

