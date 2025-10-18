#!/usr/bin/env python3
"""
NETWORK MONITORING AGENT - Simplified Scapy Version
==================================================

This agent uses Scapy to capture network flows and extract only
features that can be reliably obtained from packet analysis.

Features:
- Real-time flow capture using Scapy
- Extracts only scapy-available features
- Support for train/predict modes
- Sequential training with batching
- No CSV output

Requirements:
- Run with sudo (for packet capture)
- Install: pip install requests psutil scapy

Flow Analysis:
- Extracts only reliable packet-level features
- Groups packets by flow (5-tuple)
- Supports both training and prediction modes
"""

import time
import requests
import json
from datetime import datetime
from typing import Dict, Any, List
import logging
import signal
import sys
import os
import socket
import psutil
import threading
import tempfile
import queue

# Scapy imports
from scapy.all import sniff, IP, TCP, UDP

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/network_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Scapy-extractable features only
SCAPY_FEATURES = [
    # Basic flow metrics
    "duration", "total_packets", "forward_packets", "reverse_packets",
    "total_bytes", "forward_bytes", "reverse_bytes",
    
    # Packet size statistics
    "min_packet_size", "max_packet_size", "avg_packet_size",
    "forward_avg_packet_size", "reverse_avg_packet_size",
    
    # Timing features
    "packets_per_second", "bytes_per_second",
    "forward_packets_per_second", "reverse_packets_per_second",
    
    # TCP-specific features (when available)
    "tcp_flags_count", "syn_count", "fin_count", "rst_count", "ack_count",
    "forward_tcp_flags", "reverse_tcp_flags",
    
    # IP-level features
    "src_port", "dst_port", "protocol",
    "forward_ttl", "reverse_ttl",
    "tcp_window_size_forward", "tcp_window_size_reverse",
    
    # Flow state
    "is_bidirectional", "connection_state"
]

# Ports to exclude from monitoring (SSH, DNS, etc.)
EXCLUDED_PORTS = []

class NetworkMonitor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.interface = config.get('interface', 'eth0')
        self.mode = config.get('mode', 'predict')  # 'train' or 'predict'
        self.batch_size = config.get('batch_size', 30)  # For training mode
        self.label = config.get('label', 0)  # Label for training mode (0=benign, 1=attack)
        
        # Endpoints
        base_url = config.get('base_url', 'http://15.160.68.117:8080')
        self.predict_endpoint = f"{base_url}/predict"
        self.train_endpoint = f"{base_url}/train"
        
        # Flow processing settings
        self.flow_timeout = config.get('flow_timeout', 5.0) 
        self.capture_window = config.get('capture_window', 5)  
        self.max_packets_per_flow = config.get('max_packets_per_flow', 50)  # Standard packet count
        
        self.packet_queue = queue.Queue(maxsize=10000)
        
        # Extract ML endpoint IP for filtering
        self.ml_endpoint_ip = self._extract_ml_endpoint_ip(base_url)
        
        # Determine server's local IP address
        self.server_ip = self._get_server_ip()
        
        # Running state
        self.running = True
        self.flow_count = 0
        self.packet_count = 0
        
        # Training batch management
        self.training_batch = []
        self.training_queue = queue.Queue()  # Sequential training queue
        self.training_in_progress = False
        
        # Threading for packet capture and processing
        self.temp_pcap_dir = tempfile.mkdtemp(prefix='netflow_')
        
        # Flow tracking
        self.active_flows = {}
        
        logger.info(f"Initialized Network Monitor in {self.mode.upper()} mode")
        logger.info(f"  - Interface: {self.interface}")
        logger.info(f"  - Server IP: {self.server_ip}")
        logger.info(f"  - Predict Endpoint: {self.predict_endpoint}")
        logger.info(f"  - Train Endpoint: {self.train_endpoint}")
        logger.info(f"  - ML Endpoint IP: {self.ml_endpoint_ip} (excluded from monitoring)")
        logger.info(f"  - Flow Timeout: {self.flow_timeout}s")
        logger.info(f"  - Batch Size (train mode): {self.batch_size}")
        if self.mode == 'train':
            label_type = "BENIGN" if self.label == 0 else "ATTACK"
            logger.info(f"  - Training Label: {self.label} ({label_type})")
        logger.info(f"  - Available Features: {len(SCAPY_FEATURES)}")

    def _extract_ml_endpoint_ip(self, base_url: str) -> str:
        """Extract IP address from ML endpoint URL."""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(base_url)
            return parsed.hostname or '15.160.68.117'
        except:
            return '15.160.68.117'

    def _get_server_ip(self) -> str:
        """Detect the server's primary IP address."""
        try:
            # Method 1: Connect to a remote address to determine local IP
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                local_ip = s.getsockname()[0]
                return local_ip
        except:
            pass
            
        try:
            # Method 2: Use psutil to get interface addresses
            for interface_name in psutil.net_if_addrs():
                if interface_name == self.interface:
                    addresses = psutil.net_if_addrs()[interface_name]
                    for addr in addresses:
                        if addr.family == socket.AF_INET and not addr.address.startswith('127.'):
                            return addr.address
        except:
            pass
            
        # Fallback
        logger.warning("Could not detect server IP, using fallback 127.0.0.1")
        return "127.0.0.1"

    def start_monitoring(self):
        """Start the network monitoring process using Scapy."""
        logger.info("🚀 Starting network monitoring...")
        
        try:
            # Start packet capture thread
            capture_thread = threading.Thread(target=self._capture_packets, daemon=True)
            capture_thread.start()
            
            # Start flow processing thread
            processing_thread = threading.Thread(target=self._process_flows, daemon=True)
            processing_thread.start()
            
            # Start training queue processor (for train mode)
            if self.mode == 'train':
                training_thread = threading.Thread(target=self._process_training_queue, daemon=True)
                training_thread.start()
            
            # Keep main thread alive
            while self.running:
                time.sleep(1)
                
        except Exception as e:
            logger.error(f"Network monitoring error: {str(e)}")
            raise
        finally:
            self._cleanup()

    def _capture_packets(self):
        """Capture packets using scapy."""
        logger.info("🎯 Starting packet capture...")
        
        def packet_handler(packet):
            if not self.running:
                return False
            
            try:
                # Extract basic packet info
                src_ip = packet[IP].src if packet.haslayer(IP) else None
                dst_ip = packet[IP].dst if packet.haslayer(IP) else None
                
                if not src_ip or not dst_ip:
                    return self.running
                
                # SKIP ML ENDPOINT TRAFFIC
                if (src_ip == self.ml_endpoint_ip or dst_ip == self.ml_endpoint_ip):
                    return self.running
                
                # SKIP EXCLUDED PORTS
                src_port = None
                dst_port = None
                
                if packet.haslayer(TCP):
                    src_port = packet[TCP].sport
                    dst_port = packet[TCP].dport
                elif packet.haslayer(UDP):
                    src_port = packet[UDP].sport
                    dst_port = packet[UDP].dport
                
                # Filter out excluded ports
                if src_port in EXCLUDED_PORTS or dst_port in EXCLUDED_PORTS:
                    return self.running
                
                # Only process packets involving our server
                if src_ip == self.server_ip or dst_ip == self.server_ip:
                    if not self.packet_queue.full():
                        self.packet_queue.put(packet)
                        
                        # Increment packet count and log periodically
                        self.packet_count += 1
                        if self.packet_count % 500 == 0:
                            logger.info(f"📦 Captured {self.packet_count} packets")
                    else:
                        logger.warning("⚠️ Packet queue full, dropping packet")
                
                return self.running
                
            except Exception as e:
                logger.debug(f"Error processing packet: {str(e)}")
                return self.running
        
        try:
            sniff(iface=self.interface, prn=packet_handler, stop_filter=lambda x: not self.running)
        except Exception as e:
            logger.error(f"Error in packet capture: {str(e)}")
            self.running = False

    def _process_flows(self):
        """Process captured packets into flows."""
        logger.info("Starting flow processing thread...")
        
        window_packets = []
        last_process_time = time.time()
        
        while self.running:
            try:
                # Collect packets for the capture window
                try:
                    packet = self.packet_queue.get(timeout=1)
                    window_packets.append(packet)
                        
                except queue.Empty:
                    # No packets, check if we should process what we have
                    if window_packets and (time.time() - last_process_time) > self.capture_window:
                        logger.info(f"⏰ Processing window: {len(window_packets)} packets (timeout)")
                        self._process_packet_window(window_packets)
                        window_packets = []
                        last_process_time = time.time()
                    continue
                
                # Process window when it's full or timeout reached
                should_process = (
                    len(window_packets) >= self.max_packets_per_flow or
                    (time.time() - last_process_time) > self.capture_window
                )
                
                if should_process and window_packets:
                    trigger = 'size' if len(window_packets) >= self.max_packets_per_flow else 'timeout'
                    logger.info(f"🔄 Processing window: {len(window_packets)} packets ({trigger})")
                    self._process_packet_window(window_packets)
                    window_packets = []
                    last_process_time = time.time()
                    
            except Exception as e:
                logger.error(f"Flow processing error: {str(e)}")
                time.sleep(1)

    def _process_packet_window(self, packets):
        """Process a window of packets into flows."""
        if not packets:
            return
            
        logger.info(f"Processing window of {len(packets)} packets...")
        
        try:
            # Generate flows directly from packets
            self._generate_flows_from_packets(packets)
                
        except Exception as e:
            logger.error(f"Error processing packet window: {str(e)}")

    def _generate_flows_from_packets(self, packets):
        """Generate simplified bidirectional flows using 5-tuple classification."""
        try:
            if not packets:
                return
                
            current_time = time.time()
            
            # Process packets into bidirectional flows using 5-tuple classification
            for packet in packets:
                try:
                    if packet.haslayer('IP'):
                        ip_layer = packet['IP']
                        src_ip = ip_layer.src
                        dst_ip = ip_layer.dst
                        protocol = ip_layer.proto
                        
                        # SKIP ML ENDPOINT TRAFFIC
                        if (src_ip == self.ml_endpoint_ip or dst_ip == self.ml_endpoint_ip):
                            continue
                        
                        src_port = 0
                        dst_port = 0
                        
                        if packet.haslayer('TCP'):
                            tcp_layer = packet['TCP']
                            src_port = tcp_layer.sport
                            dst_port = tcp_layer.dport
                        elif packet.haslayer('UDP'):
                            udp_layer = packet['UDP']
                            src_port = udp_layer.sport
                            dst_port = udp_layer.dport
                        
                        # Filter out excluded ports
                        if src_port in EXCLUDED_PORTS or dst_port in EXCLUDED_PORTS:
                            continue

                        # Create bidirectional flow key using 5-tuple
                        if (src_ip < dst_ip) or (src_ip == dst_ip and src_port < dst_port):
                            # Forward direction
                            flow_key = (src_ip, dst_ip, src_port, dst_port, protocol)
                            is_forward = True
                        else:
                            # Reverse direction - swap src/dst for consistent key
                            flow_key = (dst_ip, src_ip, dst_port, src_port, protocol)
                            is_forward = False
                        
                        # Initialize flow if not exists
                        if flow_key not in self.active_flows:
                            self.active_flows[flow_key] = {
                                'flow_key': flow_key,
                                'src_ip': flow_key[0],
                                'dst_ip': flow_key[1],
                                'src_port': flow_key[2],
                                'dst_port': flow_key[3],
                                'protocol': protocol,
                                'forward_packets': [],
                                'reverse_packets': [],
                                'start_time': packet.time,
                                'end_time': packet.time,
                                'last_activity': current_time,
                                'tcp_flags_forward': set(),
                                'tcp_flags_reverse': set(),
                            }
                        
                        flow_data = self.active_flows[flow_key]
                        
                        # Determine actual packet direction relative to normalized flow
                        actual_forward = ((src_ip == flow_key[0] and dst_ip == flow_key[1] and 
                                         src_port == flow_key[2] and dst_port == flow_key[3]) or
                                        (is_forward and src_ip == flow_key[0] and dst_ip == flow_key[1]))
                        
                        # Add packet to appropriate direction
                        if actual_forward:
                            flow_data['forward_packets'].append(packet)
                            if packet.haslayer('TCP'):
                                flow_data['tcp_flags_forward'].add(packet['TCP'].flags)
                        else:
                            flow_data['reverse_packets'].append(packet)
                            if packet.haslayer('TCP'):
                                flow_data['tcp_flags_reverse'].add(packet['TCP'].flags)
                        
                        # Update flow metadata
                        flow_data['end_time'] = packet.time
                        flow_data['last_activity'] = current_time
                        
                except Exception as e:
                    # Silently skip problematic packets
                    continue
            
            # Complete flows based on timeout or standard duration
            flows_to_complete = []
            
            for flow_key, flow_data in list(self.active_flows.items()):
                should_complete = False
                flow_duration = current_time - flow_data['start_time']
                idle_time = current_time - flow_data['last_activity']
                
                # Complete flows based on standard criteria
                if (flow_duration >= self.flow_timeout or  # Standard duration reached
                    idle_time > 2.0 or  # 2 second idle timeout
                    self._has_tcp_termination(flow_data)):  # TCP termination
                    should_complete = True
                
                if should_complete:
                    flows_to_complete.append((flow_key, flow_data))
                    
            # Process completed flows
            for flow_key, flow_data in flows_to_complete:
                total_packets = len(flow_data['forward_packets']) + len(flow_data['reverse_packets'])
                if total_packets > 0:  # Only process flows with packets
                    logger.info(f"📊 Completing flow: {total_packets} packets (fwd: {len(flow_data['forward_packets'])}, rev: {len(flow_data['reverse_packets'])})")
                    self._process_completed_flow(flow_data)
                del self.active_flows[flow_key]
            
            # Log active flows
            if flows_to_complete:
                logger.info(f"🔧 Active flows: {len(self.active_flows)}, Completing: {len(flows_to_complete)}")
            
        except Exception as e:
            logger.error(f"Error generating flows: {str(e)}")

    def _has_tcp_termination(self, flow_data: Dict[str, Any]) -> bool:
        """Check if flow has TCP termination flags (FIN, RST) in either direction."""
        try:
            # Check for FIN (0x01) or RST (0x04) flags in either direction
            all_flags = flow_data.get('tcp_flags_forward', set()) | flow_data.get('tcp_flags_reverse', set())
            for flags in all_flags:
                if flags & 0x01 or flags & 0x04:  # FIN or RST
                    return True
            return False
        except:
            return False

    def _process_completed_flow(self, flow_data: Dict[str, Any]):
        """Process a completed flow and extract scapy-available features."""
        try:
            # Get flow information
            src_ip = flow_data['src_ip']
            dst_ip = flow_data['dst_ip']
            
            # Skip flows that don't involve our server
            if not (src_ip == self.server_ip or dst_ip == self.server_ip):
                return
                
            # Skip very short flows (less than 2 packets)
            total_packets = len(flow_data['forward_packets']) + len(flow_data['reverse_packets'])
            if total_packets < 2:
                return
            
            self.flow_count += 1
            
            # Determine remote IP and direction
            remote_ip = dst_ip if src_ip == self.server_ip else src_ip
            direction = "outbound" if src_ip == self.server_ip else "inbound"
            
            logger.info(f"Flow {self.flow_count}: {src_ip}:{flow_data['src_port']} <-> {dst_ip}:{flow_data['dst_port']} [{direction}]")
            
            # Extract scapy-available features
            flow_features = self._extract_scapy_features(flow_data)
            
            # Send to appropriate endpoint based on mode
            if self.mode == 'predict':
                self._send_for_prediction(flow_features, remote_ip)
            elif self.mode == 'train':
                self._add_to_training_batch(flow_features, remote_ip)
            
        except Exception as e:
            logger.error(f"Error processing completed flow: {str(e)}")

    def _extract_scapy_features(self, flow_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract scapy-available features from bidirectional flow data."""
        try:
            # Get packets from both directions  
            forward_packets = flow_data.get('forward_packets', [])
            reverse_packets = flow_data.get('reverse_packets', [])
            
            if not forward_packets and not reverse_packets:
                return self._get_default_features()
            
            # Calculate duration
            duration = flow_data['end_time'] - flow_data['start_time']
            if duration <= 0:
                duration = 0.001  # Minimum duration to avoid division by zero
            
            # Packet and byte counts
            forward_packet_count = len(forward_packets)
            reverse_packet_count = len(reverse_packets)
            forward_bytes = sum(len(p) for p in forward_packets)
            reverse_bytes = sum(len(p) for p in reverse_packets)
            total_packets = forward_packet_count + reverse_packet_count
            
            features = {}
            
            # Basic flow metrics
            features['duration'] = duration
            features['total_packets'] = total_packets
            features['forward_packets'] = forward_packet_count
            features['reverse_packets'] = reverse_packet_count
            features['total_bytes'] = forward_bytes + reverse_bytes
            features['forward_bytes'] = forward_bytes
            features['reverse_bytes'] = reverse_bytes
            
            # Packet size statistics
            all_packets = forward_packets + reverse_packets
            if all_packets:
                packet_sizes = [len(p) for p in all_packets]
                features['min_packet_size'] = min(packet_sizes)
                features['max_packet_size'] = max(packet_sizes)
                features['avg_packet_size'] = sum(packet_sizes) / len(packet_sizes)
            else:
                features['min_packet_size'] = 0
                features['max_packet_size'] = 0
                features['avg_packet_size'] = 0

            # Direction-specific packet size statistics
            if forward_packets:
                forward_sizes = [len(p) for p in forward_packets]
                features['forward_avg_packet_size'] = sum(forward_sizes) / len(forward_sizes)
            else:
                features['forward_avg_packet_size'] = 0
                
            if reverse_packets:
                reverse_sizes = [len(p) for p in reverse_packets]
                features['reverse_avg_packet_size'] = sum(reverse_sizes) / len(reverse_sizes)
            else:
                features['reverse_avg_packet_size'] = 0

            # Timing features
            features['packets_per_second'] = total_packets / duration if duration > 0 else 0
            features['bytes_per_second'] = (forward_bytes + reverse_bytes) / duration if duration > 0 else 0
            features['forward_packets_per_second'] = forward_packet_count / duration if duration > 0 else 0
            features['reverse_packets_per_second'] = reverse_packet_count / duration if duration > 0 else 0

            # TCP-specific features (when available)
            forward_flags = flow_data.get('tcp_flags_forward', set())
            reverse_flags = flow_data.get('tcp_flags_reverse', set())
            all_flags = forward_flags | reverse_flags
            
            features['tcp_flags_count'] = len(all_flags)
            features['syn_count'] = sum(1 for flags in all_flags if flags & 0x02)  # SYN flag
            features['fin_count'] = sum(1 for flags in all_flags if flags & 0x01)  # FIN flag
            features['rst_count'] = sum(1 for flags in all_flags if flags & 0x04)  # RST flag
            features['ack_count'] = sum(1 for flags in all_flags if flags & 0x10)  # ACK flag
            features['forward_tcp_flags'] = len(forward_flags)
            features['reverse_tcp_flags'] = len(reverse_flags)

            # IP-level features
            features['src_port'] = flow_data['src_port']
            features['dst_port'] = flow_data['dst_port']
            features['protocol'] = flow_data['protocol']
            
            # TTL features (from first packet in each direction)
            features['forward_ttl'] = self._get_ttl_from_packets(forward_packets, default=64)
            features['reverse_ttl'] = self._get_ttl_from_packets(reverse_packets, default=64)
            
            # TCP window size features (from first packet in each direction)
            features['tcp_window_size_forward'] = self._get_window_size_from_packets(forward_packets, default=0)
            features['tcp_window_size_reverse'] = self._get_window_size_from_packets(reverse_packets, default=0)

            # Flow state
            features['is_bidirectional'] = 1 if forward_packet_count > 0 and reverse_packet_count > 0 else 0
            features['connection_state'] = self._determine_connection_state(flow_data)

            return features
            
        except Exception as e:
            logger.error(f"Error extracting scapy features: {str(e)}")
            return self._get_default_features()

    def _get_ttl_from_packets(self, packets, default=64) -> int:
        """Extract TTL from first packet."""
        try:
            if packets and packets[0].haslayer('IP'):
                return packets[0]['IP'].ttl
        except:
            pass
        return default

    def _get_window_size_from_packets(self, packets, default=0) -> int:
        """Extract TCP window size from first packet."""
        try:
            if packets and packets[0].haslayer('TCP'):
                return packets[0]['TCP'].window
        except:
            pass
        return default

    def _get_default_features(self) -> Dict[str, Any]:
        """Return default values for all scapy features."""
        return {feature: 0 for feature in SCAPY_FEATURES}

    def _send_for_prediction(self, flow_features: Dict[str, Any], remote_ip: str):
        """Send flow features to ML model for prediction."""
        try:
            logger.info(f"🚀 Sending flow to ML model for prediction: {self.server_ip} <-> {remote_ip}")
            
            # Create a clean feature set with only scapy features
            clean_features = {}
            for feature in SCAPY_FEATURES:
                if feature in flow_features:
                    clean_features[feature] = flow_features[feature]
                else:
                    # Set default values for missing features
                    clean_features[feature] = 0

            # Log the payload in structured JSON format
            payload_log = {
                "timestamp": datetime.now().isoformat(),
                "flow_id": f"flow_{self.flow_count}",
                "server_ip": self.server_ip,
                "remote_ip": remote_ip,
                "ml_endpoint": self.predict_endpoint,
                "payload": clean_features
            }
            
            logger.info(f"📤 ML Payload: {json.dumps(payload_log, indent=2, sort_keys=True)}")

            # Send to ML endpoint
            response = requests.post(
                self.predict_endpoint,
                json=clean_features,
                timeout=10.0
            )
            
            if response.status_code == 200:
                prediction = response.json()
                attack_prob = prediction.get('attack_probability', 0)
                prediction_result = prediction.get('prediction', 0)
                
                logger.info(f"✅ ML Prediction: {prediction_result} (prob: {attack_prob:.3f})")
                
            else:
                logger.warning(f"❌ ML model returned status {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error calling ML model for prediction: {str(e)}")

    def _add_to_training_batch(self, flow_features: Dict[str, Any], remote_ip: str):
        """Add flow features to the training batch for sequential training."""
        try:
            # Create a clean feature set with only scapy features
            clean_features = {}
            for feature in SCAPY_FEATURES:
                if feature in flow_features:
                    clean_features[feature] = flow_features[feature]
                else:
                    clean_features[feature] = 0

            self.training_batch.append({
                "flow_id": f"flow_{self.flow_count}",
                "server_ip": self.server_ip,
                "remote_ip": remote_ip,
                "features": clean_features,
                "label": self.label  # Use configurable label instead of hardcoded 0
            })
            
            label_type = "BENIGN" if self.label == 0 else "ATTACK"
            logger.info(f"📥 Added flow to training batch (label: {self.label} - {label_type}). Current batch size: {len(self.training_batch)}")

            if len(self.training_batch) >= self.batch_size:
                logger.info(f"📊 Batch size reached ({self.batch_size}). Queuing for training...")
                # Add batch to training queue for sequential processing
                self.training_queue.put(list(self.training_batch))  # Copy the batch
                self.training_batch = []  # Clear current batch

        except Exception as e:
            logger.error(f"Error adding flow to training batch: {str(e)}")

    def _process_training_queue(self):
        """Process training batches sequentially from the queue."""
        logger.info("🔄 Starting training queue processor...")
        
        while self.running:
            try:
                # Get batch from queue (blocking with timeout)
                try:
                    batch = self.training_queue.get(timeout=5)
                    self._send_training_batch_to_ml(batch)
                    self.training_queue.task_done()
                except queue.Empty:
                    # No batches to process, continue
                    continue
                    
            except Exception as e:
                logger.error(f"Error in training queue processor: {str(e)}")
                time.sleep(1)

    def _send_training_batch_to_ml(self, batch: List[Dict[str, Any]]):
        """Send a training batch to the ML model for training."""
        if not batch:
            return

        logger.info(f"🚀 Sending training batch of {len(batch)} flows to {self.train_endpoint}")
        logger.info(f"First flow in batch: {json.dumps(batch[0], indent=2, sort_keys=True)}")

        try:
            payload = {
                "flows": batch,
                "batch_size": len(batch),
                "timestamp": datetime.now().isoformat()
            }

            response = requests.post(
                self.train_endpoint,
                json=payload,
                timeout=120.0  # Increased timeout for training
            )

            if response.status_code == 200:
                training_result = response.json()
                logger.info(f"✅ Training batch completed successfully. Response: {training_result}")
            else:
                logger.warning(f"❌ Training batch failed. Status: {response.status_code}. Response: {response.text}")
                
        except Exception as e:
            logger.error(f"Error sending training batch to ML: {str(e)}")

    def _determine_connection_state(self, flow_data: Dict[str, Any]) -> str:
        """Determine connection state based on TCP flags."""
        try:
            if self._has_tcp_termination(flow_data):
                return 'FIN'
            
            # Check if we have bidirectional traffic
            if len(flow_data.get('forward_packets', [])) > 0 and len(flow_data.get('reverse_packets', [])) > 0:
                return 'CON'  # Connected
            else:
                return 'INT'  # Interrupted
        except:
            return 'CON'

    def _cleanup(self):
        """Clean up temporary files and resources."""
        try:
            # Remove temporary PCAP directory
            import shutil
            if os.path.exists(self.temp_pcap_dir):
                shutil.rmtree(self.temp_pcap_dir)
                logger.info(f"Cleaned up temporary directory: {self.temp_pcap_dir}")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

    def stop(self):
        """Stop the monitoring agent."""
        logger.info("Stopping network monitoring agent...")
        self.running = False


def main():
    # Get the default network interface automatically
    def get_default_interface():
        """Get the default network interface with internet connectivity."""
        try:
            # Get default gateway interface
            gateways = psutil.net_if_addrs()
            stats = psutil.net_if_stats()
            
            for interface_name, interface_addresses in gateways.items():
                # Skip loopback and inactive interfaces
                if interface_name.startswith('lo') or not stats[interface_name].isup:
                    continue
                    
                # Look for interfaces with IP addresses
                for address in interface_addresses:
                    if address.family == 2:  # IPv4
                        return interface_name
            
            # Fallback to eth0 if nothing found
            return 'eth0'
        except:
            return 'eth0'
    
    # Configuration
    config = {
        'interface': get_default_interface(),
        'base_url': 'http://15.160.68.117:8080',
        'mode': 'train',  # 'train' or 'predict'
        'flow_timeout': 5.0,  # Standard 5 second flows
        'capture_window': 5,  # 5 second capture windows
        'max_packets_per_flow': 50,  # Standard packet count
        'batch_size': 10,  # For training mode
        'label': 0
    }
    
    logger.info(f"🚀 Starting Network Monitor")
    logger.info(f"   Interface: {config['interface']}")
    logger.info(f"   ML Base URL: {config['base_url']}")
    logger.info(f"   Mode: {config['mode'].upper()}")
    logger.info(f"   Flow Timeout: {config['flow_timeout']}s")
    logger.info(f"   Capture Window: {config['capture_window']}s")
    logger.info(f"   Max Packets Per Flow: {config['max_packets_per_flow']}")
    if config['mode'] == 'train':
        logger.info(f"   Training Batch Size: {config['batch_size']}")
    
    # Create and start monitor
    monitor = NetworkMonitor(config)
    
    # Handle graceful shutdown
    def signal_handler(sig, frame):
        monitor.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        monitor.start_monitoring()
    except KeyboardInterrupt:
        monitor.stop()


if __name__ == "__main__":
    main() 