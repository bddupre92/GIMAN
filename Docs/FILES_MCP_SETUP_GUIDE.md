# Files.com MCP Setup Guide for GIMAN Project

## 🎯 Goal: Use Files.com MCP for Professional GIMAN Organization

### Step 1: Complete MCP Server Setup

1. **Get Files.com API Key:**
   - Sign up at https://files.com
   - Go to API & SDKs → API Keys
   - Create a new API key with read/write permissions

2. **Configure VS Code MCP Server:**
   Add this to your VS Code `settings.json`:
   ```json
   {
     "mcp.servers": {
       "files-com": {
         "command": "uv",
         "args": [
           "--directory",
           "/Users/blair.dupre/files-mcp",
           "run",
           "files-com-mcp"
         ],
         "env": {
           "FILES_API_KEY": "your_actual_api_key_here"
         }
       }
     }
   }
   ```

### Step 2: GIMAN Organization Strategy with Files.com

#### Phase 1: Archive Large Development Files to Cloud
```bash
# These will be uploaded to Files.com for cloud storage:
- archive/temp_files/temp_checkpoints (2.9GB)
- archive/development/phase* files (experimental versions)  
- archive/logs/ (development logs)
```

#### Phase 2: Maintain Clean Local Structure
```
GIMAN_Local/
├── src/giman_pipeline/           # Production system
├── config/                       # Configuration files
├── data/                         # Active datasets
├── notebooks/                    # Current analysis
├── results/                      # Current outputs
├── phase3_1_real_data_integration.py  # Core dependency
├── phase4_unified_giman_system.py     # Core system
├── explainability_Gemini.py           # Current explainability
├── giman_research_analytics.py        # Research tools
└── README.md                          # Documentation
```

#### Phase 3: Cloud Archive Structure on Files.com
```
GIMAN_Archive/
├── development_history/
│   ├── phase1/                   # Phase 1 development
│   ├── phase2/                   # Phase 2 components  
│   ├── phase3/                   # Phase 3 iterations
│   └── phase4/                   # Phase 4 alternatives
├── experimental_checkpoints/
│   └── temp_checkpoints/         # 2.9GB training data
├── logs_and_debug/
│   └── *.log files              # Development logs
└── documentation/
    └── analysis_reports/         # Historical reports
```

### Step 3: Files.com MCP Commands for Organization

Once configured, you can use these MCP commands:

#### Upload Archive to Cloud:
```python
# Archive large development files to Files.com
await files_com.upload_folder("archive/temp_files", "GIMAN_Archive/experimental_checkpoints/")
await files_com.upload_folder("archive/development", "GIMAN_Archive/development_history/")
```

#### Create Cloud Organization:
```python
# Create organized folder structure on Files.com
await files_com.create_folder("GIMAN_Archive")
await files_com.create_folder("GIMAN_Archive/development_history")
await files_com.create_folder("GIMAN_Archive/experimental_checkpoints")
```

#### Share Organized Project:
```python
# Share clean project structure with team
await files_com.create_share_link("GIMAN_Production", permissions="read")
```

### Benefits of This Approach:

✅ **Cloud Storage**: 2.9GB+ archives stored in cloud, not local disk
✅ **Team Collaboration**: Share organized structure with collaborators  
✅ **Version Control**: Keep development history without cluttering local
✅ **Professional Structure**: Clean local environment for production work
✅ **Automated Workflows**: Script file organization and archival processes

### Next Steps:

1. **Get Files.com API key** (free tier available)
2. **Configure MCP server** in VS Code settings
3. **Test connection** with simple upload
4. **Execute organized archival** of development files
5. **Maintain clean local** production environment

### Immediate Action (Without Files.com):
While setting up Files.com, we can continue local organization:

```bash
# Continue Phase 2: Archive development files locally
mkdir -p archive/development/{phase1,phase2,phase3,phase4}
mv phase1_prognostic_development.py archive/development/phase1/
mv phase2_*.py archive/development/phase2/
# ... etc
```

This approach gives you both immediate organization and a path to professional cloud-based file management!