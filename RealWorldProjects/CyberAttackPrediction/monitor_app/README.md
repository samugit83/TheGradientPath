# 🛡️ Monitor App - Network Traffic Analysis & ML Integration

## 📺 Video Tutorial
**[Watch the Complete Video Tutorial on YouTube](https://youtu.be/3-mH1ynRf7U)** - Complete guide for deploying the ML Cyber Attack Prediction System.

## 📋 Project Overview

The **Monitor App** is a comprehensive network monitoring solution that combines a Next.js web application with a Python-based network monitoring agent. This project is part of a larger **Cyber Attack Prediction** system designed to detect and predict network security threats using machine learning.

### Architecture Overview

This project (`monitor-app`) works in conjunction with the `ml-service` component:
- **monitor-app**: Deployed on the monitoring server, captures network traffic and extracts features
- **ml-service**: Deployed on a separate ML server, provides prediction and incremental training capabilities

The network monitor agent extracts real-time network flow data and communicates with the ML service to:
- Get real-time attack predictions (predict mode)
- Perform incremental model training with labeled data (train mode)

## 🚀 Local Development Setup

### Prerequisites

- Node.js 22.x or higher
- Python 3.8+
- npm or yarn package manager
- sudo privileges (for packet capture)

### 1. Next.js Application Setup

```bash
# Navigate to the monitor-app directory
cd monitor-app

# Install dependencies
npm install

# Run the development server
npm run dev
```

The Next.js application will be available at `http://localhost:3000`

### 2. Network Monitor Agent Setup

The network monitor agent requires Python and several dependencies. Here's how to set it up locally with a virtual environment:

```bash
# Navigate to the network agent directory
cd monitor-app/network_agent

# Create a Python virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt

# Run the network monitor agent (requires sudo for packet capture)
sudo venv/bin/python network_monitor_agent.py
```

#### Configuration

Before running the agent, you may want to modify the configuration in `network_monitor_agent.py`:

```python
config = {
    'interface': 'eth0',                    # Your network interface
    'base_url': 'http://15.160.68.117:8080', # ML service endpoint
    'mode': 'predict',                      # 'predict' or 'train'
    'flow_timeout': 5.0,                    # Flow completion timeout
    'capture_window': 5,                    # Packet capture window
    'max_packets_per_flow': 50,             # Standard flow packet count
    'batch_size': 30,                       # Training batch size (train mode)
    'label': 0                              # 0=benign, 1=attack (train mode)
}
```

For local development, update the `base_url` to point to your local ML service instance.

## 🔧 Network Monitor Agent Features

The network monitor agent uses **Scapy** for real-time packet capture and flow analysis:

### Operating Modes

1. **Predict Mode** (Default):
   - Sends individual network flows to the ML service for real-time threat detection
   - Receives immediate predictions about potential attacks

2. **Train Mode**:
   - Collects flows into batches for incremental model training
   - Supports labeled data (benign or attack traffic)
   - Uses sequential processing to avoid overwhelming the training endpoint

### Flow Analysis

The agent creates **bidirectional network flows** using 5-tuple classification:
- Source IP + Destination IP + Source Port + Destination Port + Protocol

Features extracted include:
- Packet counts and byte statistics
- Timing features (packets/bytes per second)
- TCP flags and window sizes
- Flow state indicators

## 📦 CI/CD Integration

This project is integrated with AWS CodeDeploy for automated deployment. The following components manage the deployment pipeline:

### 1. Build Configuration (`buildspec-dual-asg.yml`)

The buildspec file defines the build process for AWS CodeBuild:
- **Install phase**: Sets up Node.js 22 runtime and installs all dependencies
- **Build phase**: Builds the Next.js application for production
- **Post-build phase**: Prepares the artifact with only production dependencies
- **Artifacts**: Packages the entire application for deployment

### 2. Deployment Configuration (`appspec.yml`)

The appspec file orchestrates the deployment process on EC2 instances:
- **Files section**: Defines what gets deployed and where
- **Permissions**: Sets appropriate file permissions for the application
- **Hooks**: Executes deployment scripts at different lifecycle stages

### 3. CodeDeploy Scripts (`code-deploy-scripts/`)

These scripts handle the deployment lifecycle:
- **before_install.sh**: Prepares the environment before deployment
- **install.sh**: Installs dependencies and sets up the application
- **after_install.sh**: Post-installation configuration
- **stop_app.sh**: Gracefully stops the running application
- **start_app.sh**: Starts the application and services
- **validate_service.sh**: Verifies the deployment was successful

## 🔄 Git Setup & Deployment Trigger

To set up version control and trigger the automated deployment pipeline:

### 1. Install Git (if not already installed)

```bash
# On Ubuntu/Debian
sudo apt update
sudo apt install git

# On CentOS/RHEL
sudo yum install git

# On macOS
brew install git

# Verify installation
git --version
```

### 2. Configure Git

```bash
# Set your identity
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Verify configuration
git config --list
```

### 3. Initialize Local Repository

```bash
# Navigate to your project root (parent of monitor-app and ml-service)
cd /path/to/CyberAttackPrediction

# Initialize git repository
git init

# Add all files to staging
git add .

# Create initial commit
git commit -m "Initial commit: Network monitoring and ML service setup"
```

### 4. Create Remote Repository

#### Using GitHub:

1. Go to [GitHub.com](https://github.com) and sign in
2. Click the "+" icon in the top right corner
3. Select "New repository"
4. Fill in the repository details:
   - Repository name: `CyberAttackPrediction`
   - Description: "Network traffic analysis and attack prediction system"
   - Choose public or private
5. **Do NOT** initialize with README, .gitignore, or license (since you already have local files)
6. Click "Create repository"

#### Using GitLab or Bitbucket:

The process is similar - create a new repository without initializing it with any files.

### 5. Connect Local to Remote Repository

```bash
# Add the remote repository (replace with your repository URL)
git remote add origin https://github.com/yourusername/CyberAttackPrediction.git

# Verify remote was added
git remote -v

# Push to master branch
git push -u origin master
```

### 6. Subsequent Updates

After the initial setup, pushing changes to trigger the deployment pipeline:

```bash
# Check status of changes
git status

# Add changed files
git add .

# Commit with descriptive message
git commit -m "Update: Description of changes"

# Push to master to trigger deployment
git push origin master
```

**Note**: Pushing to the master branch will automatically trigger the CodeDeploy pipeline, which will:
1. Run CodeBuild to build the application
2. Create deployment artifacts
3. Deploy to the configured Auto Scaling Groups
4. Execute the deployment scripts defined in appspec.yml

## 📁 Project Structure

```
monitor-app/
├── app/                      # Next.js application source
│   ├── api/                  # API routes
│   ├── layout.tsx            # Root layout
│   └── page.tsx              # Main page component
├── network_agent/            # Python network monitoring agent
│   ├── network_monitor_agent.py  # Main agent script
│   ├── requirements.txt      # Python dependencies
│   └── README.md             # Agent documentation
├── code-deploy-scripts/      # AWS CodeDeploy lifecycle scripts
├── nginx/                    # Nginx configuration
├── public/                   # Static assets
├── appspec.yml              # CodeDeploy configuration
├── buildspec-dual-asg.yml   # CodeBuild configuration
├── package.json             # Node.js dependencies
└── README.md                # This file
```

## 🔍 Monitoring and Logs

### Development Logs

```bash
# Next.js development server logs
npm run dev

# Network agent logs (in development)
sudo venv/bin/python network_monitor_agent.py
# Logs are written to /var/log/network_monitor.log
```

### Production Logs

Once deployed, you can monitor the services:

```bash
# Check network monitor service status
sudo systemctl status network-monitor

# View network monitor logs
sudo journalctl -u network-monitor -f

# View application logs
tail -f /var/log/network_monitor.log
```

## 🛠️ Troubleshooting

### Common Issues

1. **Permission denied for packet capture**:
   - Ensure you're running the network agent with sudo
   - Check that your user has appropriate permissions

2. **ML service connection refused**:
   - Verify the ML service is running
   - Check the base_url configuration in the agent
   - Ensure network connectivity between servers

3. **Build failures**:
   - Check Node.js version (requires 22.x)
   - Clear npm cache: `npm cache clean --force`
   - Delete node_modules and reinstall: `rm -rf node_modules && npm install`

## 📚 Additional Resources

- [Network Monitor Agent Documentation](network_agent/README.md) - Detailed agent setup and operation guide
- [ML Service Documentation](../ml-service/README.md) - ML service setup and API documentation
- [AWS CodeDeploy Documentation](https://docs.aws.amazon.com/codedeploy/) - Official AWS CodeDeploy guide

## 📄 License

This project is part of the CyberAttackPrediction system developed for network security analysis and threat detection.
