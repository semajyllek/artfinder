#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

# Define color outputs for clean terminal reporting
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0;m' # No Color

echo -e "${BLUE}============ ARTFINDER LOCAL AUTOMATION ENGINE ============${NC}"

# 1. Verify Google Cloud SDK installation footprint
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}❌ Error: 'gcloud' CLI is not installed on this system.${NC}"
    echo "Please download and install the Google Cloud SDK before running this pipeline."
    exit 1
fi

# 2. Check for existing local Google Application Default Credentials
ADC_PATH="$HOME/.config/gcloud/application_default_credentials.json"
if [ ! -f "$ADC_PATH" ]; then
    echo -e "${YELLOW}⚠️ Application Default Credentials (ADC) missing or unrecognized.${NC}"
    echo "Launching Google browser authentication window now..."
    gcloud auth application-default login
else
    echo -e "${GREEN}✅ Valid Cloud Authentication Signature Detected at: $ADC_PATH${NC}"
fi

# 3. Handle Python Virtual Environment Setup
if [ ! -d "venv" ]; then
    echo -e "${BLUE}📦 Instantiating pristine localized virtual environment ('venv')...${NC}"
    python3 -m venv venv
else
    echo -e "${GREEN}🔄 Leveraging existing local virtual environment context.${NC}"
fi

# 4. Activate environment and install dependencies
echo -e "${BLUE}🔌 Activating environment and syncing requirements pins...${NC}"
source venv/bin/activate

# Upgrade pip to prevent package compilation errors
pip install --upgrade -q pip
pip install -r requirements.txt

# 5. Fire off the Map-Reduce multi-core extraction engine
echo -e "${GREEN}🚀 Pipeline configuration synchronized. Initializing local multi-core array...${NC}"
echo -e "${BLUE}===========================================================${NC}\n"

python3 run_local.py --rebuild

echo -e "\n${GREEN}✨ Processing loop successfully finalized! All assets safely vaulted.${NC}"
