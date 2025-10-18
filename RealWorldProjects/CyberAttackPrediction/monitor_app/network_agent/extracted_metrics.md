# Network Monitor Agent - Extracted Metrics

## Features Sent to ML Service

The following metrics are extracted from network flows and sent to the ML service for training or prediction. Think of these metrics as the "vital signs" of network traffic that help identify normal behavior versus potential attacks.

### Basic Flow Metrics

- **duration** 
  - *What it measures*: The total time span of a network conversation in seconds, from when the first packet is sent until the last packet is received
  - *Why it matters*: Short-duration flows with high packet counts might indicate scanning attacks, while unusually long flows could suggest data exfiltration

- **total_packets** 
  - *What it measures*: The complete count of all data packets exchanged in both directions during the conversation
  - *Why it matters*: Abnormally high packet counts in a short time can indicate DDoS attacks or port scanning activities

- **forward_packets** 
  - *What it measures*: Number of packets traveling from the client/attacker to your server
  - *Why it matters*: A high forward packet count with low reverse packets might indicate a one-way attack like SYN flooding

- **reverse_packets** 
  - *What it measures*: Number of packets sent back from your server to the client
  - *Why it matters*: The ratio between forward and reverse packets helps identify the type of communication pattern

- **total_bytes** 
  - *What it measures*: The total amount of data transferred in the flow, counting every byte in every packet
  - *Why it matters*: Unusually large byte transfers might indicate data theft, while tiny byte counts with many packets could suggest reconnaissance

- **forward_bytes** 
  - *What it measures*: Total amount of data sent from the client to your server
  - *Why it matters*: Large forward bytes might indicate someone uploading malware or attempting buffer overflow attacks

- **reverse_bytes** 
  - *What it measures*: Total amount of data sent from your server back to the client
  - *Why it matters*: Unexpectedly large reverse bytes could indicate successful data exfiltration or information leakage

### Packet Size Statistics

- **min_packet_size** 
  - *What it measures*: The size of the smallest packet in the entire flow (in bytes)
  - *Why it matters*: Many attack tools use fixed minimum packet sizes; unusual minimums can be attack signatures

- **max_packet_size** 
  - *What it measures*: The size of the largest packet in the flow (in bytes)
  - *Why it matters*: Maximum packet sizes can reveal attempts to exploit buffer overflows or fragmentation attacks

- **avg_packet_size** 
  - *What it measures*: The average size of all packets in the flow (total bytes divided by total packets)
  - *Why it matters*: Normal web traffic has predictable average sizes; deviations might indicate malicious activity

- **forward_avg_packet_size** 
  - *What it measures*: Average size of packets sent from client to server
  - *Why it matters*: Attack traffic often has different packet size patterns than legitimate requests

- **reverse_avg_packet_size** 
  - *What it measures*: Average size of packets sent from server to client
  - *Why it matters*: Helps identify if the server is responding normally or sending unusual amounts of data

### Timing Features

- **packets_per_second** 
  - *What it measures*: The rate at which packets are transmitted (total packets divided by duration)
  - *Why it matters*: Extremely high rates can indicate automated attacks or scanning tools

- **bytes_per_second** 
  - *What it measures*: The data transmission rate throughout the flow
  - *Why it matters*: Bandwidth consumption patterns help distinguish between normal usage and flooding attacks

- **forward_packets_per_second** 
  - *What it measures*: Rate of packets coming from the client/attacker
  - *Why it matters*: High forward rates with low reverse rates often indicate denial-of-service attempts

- **reverse_packets_per_second** 
  - *What it measures*: Rate of packets being sent by your server
  - *Why it matters*: Unusual server response rates might indicate it's being overwhelmed or exploited

### TCP-Specific Features

- **tcp_flags_count** 
  - *What it measures*: The variety of different TCP control flags seen in the flow
  - *Why it matters*: Normal connections use predictable flag combinations; unusual patterns indicate attacks

- **syn_count** 
  - *What it measures*: Count of SYN (synchronize) flags, which initiate new connections
  - *Why it matters*: Multiple SYN flags without proper responses indicate SYN flood attacks

- **fin_count** 
  - *What it measures*: Count of FIN (finish) flags, which gracefully close connections
  - *Why it matters*: Proper connection termination patterns differ between normal traffic and attacks

- **rst_count** 
  - *What it measures*: Count of RST (reset) flags, which abruptly terminate connections
  - *Why it matters*: High RST counts often indicate port scanning or failed attack attempts

- **ack_count** 
  - *What it measures*: Count of ACK (acknowledgment) flags, confirming data receipt
  - *Why it matters*: ACK patterns help identify the health and legitimacy of TCP connections

- **forward_tcp_flags** 
  - *What it measures*: Variety of TCP flags used by the client in their packets
  - *Why it matters*: Attack tools often use unusual flag combinations not seen in normal traffic

- **reverse_tcp_flags** 
  - *What it measures*: Variety of TCP flags used by the server in response packets
  - *Why it matters*: Server flag patterns can reveal if it's responding normally or under stress

### IP-Level Features

- **src_port** 
  - *What it measures*: The port number used by the client/source (like apartment numbers in a building)
  - *Why it matters*: Certain attacks originate from specific port ranges; unusual source ports can be suspicious

- **dst_port** 
  - *What it measures*: The port number on your server being accessed (80 for HTTP, 443 for HTTPS, etc.)
  - *Why it matters*: Attacks often target specific services; unusual destination ports might indicate scanning

- **protocol** 
  - *What it measures*: The IP protocol number (6 for TCP, 17 for UDP, 1 for ICMP, etc.)
  - *Why it matters*: Different protocols have different security implications; some attacks use specific protocols

- **forward_ttl** 
  - *What it measures*: Time-To-Live value from client packets (how many network hops the packet can make)
  - *Why it matters*: TTL values can reveal the attacker's operating system and detect spoofed IP addresses

- **reverse_ttl** 
  - *What it measures*: Time-To-Live value from server response packets
  - *Why it matters*: Consistent TTL values help verify legitimate server responses

- **tcp_window_size_forward** 
  - *What it measures*: The amount of data the client can receive before acknowledgment is needed
  - *Why it matters*: Window size patterns can identify the client's OS and detect certain attack tools

- **tcp_window_size_reverse** 
  - *What it measures*: The amount of data the server can send before needing acknowledgment
  - *Why it matters*: Server window sizes can indicate congestion or attempts to slow down connections

### Flow State Features

- **is_bidirectional** 
  - *What it measures*: A binary flag (1 or 0) indicating whether packets flowed in both directions
  - *Why it matters*: One-way flows are often attacks (like UDP floods), while bidirectional flows are usually legitimate

- **connection_state** 
  - *What it measures*: The final state of the connection ('CON' for established, 'FIN' for properly closed, 'INT' for interrupted)
  - *Why it matters*: Attack traffic often has abnormal connection states compared to legitimate traffic patterns

## Understanding the Big Picture

These 29 metrics work together like pieces of a puzzle. While any single metric might seem normal, the ML model looks at all of them together to detect patterns that humans might miss. For example:

- A flow with high **forward_packets** but low **reverse_packets** and many **syn_count** flags likely indicates a SYN flood attack
- Normal web browsing has predictable **avg_packet_size** and **tcp_window_size** patterns that differ from automated attack tools
- The combination of **duration**, **packets_per_second**, and **connection_state** can reveal scanning attempts versus legitimate connections

The ML service uses these metrics to learn what "normal" looks like for your server, then alerts when it sees patterns that deviate from normal behavior.