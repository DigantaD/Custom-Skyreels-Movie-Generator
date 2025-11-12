# SkyReels Movie Generator

A production-ready pipeline for generating promotional videos using SkyReels-V2 T2V (Text-to-Video) and I2V (Image-to-Video) models. This project converts scripts into segmented video content with automatic stitching and post-processing.

## Features

- **Text-to-Video (T2V)**: Generate video segments from text prompts
- **Image-to-Video (I2V)**: Generate video from starting images with continuity
- **Script Segmentation**: Automatically break scripts into logical video segments
- **Video Stitching**: Seamlessly concatenate multiple segments into final video
- **Post-Processing**: Normalization, transitions, fade effects, and brightness adjustment
- **Docker Support**: Ready for containerized deployment
- **Configurable**: YAML-based configuration for all parameters
- **GPU Optimized**: Automatic GPU detection and memory management

## Project Structure

```
Skyreels-Movie-Generator/
├── configs/
│   ├── model_config.yaml      # Model and generation parameters
│   └── prompts.json           # Prompt templates (optional)
├── scripts/
│   ├── segment_generator.py   # T2V/I2V segment generation
│   ├── video_stitcher.py      # Video concatenation
│   ├── post_processor.py      # Video post-processing
│   └── sample_script.txt      # P-TAL store example script
├── utils/
│   ├── model_loader.py        # Model loading and management
│   ├── prompt_parser.py       # Script to prompt conversion
│   └── video_utils.py         # Video manipulation utilities
├── outputs/
│   ├── segments/              # Individual video segments
│   └── final/                 # Final stitched videos
├── main.py                    # Main pipeline orchestrator
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Docker container definition
├── docker-compose.yml         # Docker Compose configuration
├── setup.py                   # Package installation
├── check_setup.py             # Setup verification script
└── README.md                   # This file
```

## Prerequisites

- **Python**: 3.8 or higher
- **GPU**: CUDA-capable GPU recommended
  - **1.3B Model**: 16GB+ VRAM (e.g., RTX 3090, RTX 4090)
  - **14B Model**: 24GB+ VRAM (e.g., A100, H100)
  - CPU-only mode is possible but extremely slow (not recommended)
- **Docker**: Optional, for containerized deployment
- **SkyReels-V2**: Repository needs to be cloned for integration (see Integration section)

## Installation

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd Skyreels-Movie-Generator
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Setup

```bash
python check_setup.py
```

This will verify:
- Python version
- Required packages
- GPU availability
- Project structure

### 5. Integrate SkyReels-V2

**⚠️ IMPORTANT**: This project requires integration with the actual SkyReels-V2 inference code.

```bash
# Clone SkyReels-V2 repository
cd ..
git clone https://github.com/SkyworkAI/SkyReels-V2.git
cd SkyReels-V2

# Install SkyReels dependencies
pip install -r requirements.txt

# Return to our project
cd ../Skyreels-Movie-Generator
```

See the [SkyReels-V2 Integration](#skyreels-v2-integration) section below for detailed integration steps.

## Usage

### Basic Text-to-Video (T2V)

```bash
python main.py \
    --script scripts/sample_script.txt \
    --mode t2v \
    --output outputs/final/video.mp4
```

### Image-to-Video (I2V)

```bash
python main.py \
    --script scripts/sample_script.txt \
    --mode i2v \
    --image path/to/starting_image.jpg \
    --output outputs/final/video.mp4
```

### Advanced Options

```bash
python main.py \
    --script scripts/sample_script.txt \
    --mode t2v \
    --output outputs/final/video.mp4 \
    --config configs/model_config.yaml
```

### Example: P-TAL Store Video

The included sample script (`scripts/sample_script.txt`) demonstrates generating a promotional video for a P-TAL store:

```bash
python main.py \
    --script scripts/sample_script.txt \
    --mode t2v \
    --output outputs/final/ptal_store_video.mp4
```

## Configuration

Edit `configs/model_config.yaml` to customize:

### Model Settings
- **Size**: `1.3b` (lower VRAM) or `14b` (better quality)
- **Model Paths**: Local paths or Hugging Face IDs
- **Auto-download**: Models download automatically from Hugging Face

### Generation Parameters
- **Resolution**: `360p`, `480p`, `540p`, or `720p`
- **Segment Duration**: Default duration per segment (seconds)
- **Inference Steps**: Number of diffusion steps (higher = better quality, slower)
- **Guidance Scale**: Prompt adherence (higher = more faithful to prompt)
- **FPS**: Frames per second (default: 8)
- **Seed**: Random seed for reproducibility (null for random)

### Performance Settings
- **Device**: `cuda` or `cpu` (CPU not recommended)
- **Model Offloading**: Enable to save VRAM (slower but uses less memory)
- **XFormers**: Enable attention optimizations
- **Batch Size**: For batch processing

### Video Settings
- **Output Format**: `mp4`, `avi`, etc.
- **Codec**: Video codec (e.g., `libx264`)
- **Transitions**: Enable/disable transitions between segments
- **Transition Duration**: Duration of transitions (seconds)

## SkyReels-V2 Integration

⚠️ **IMPORTANT**: This project provides a complete pipeline structure, but you need to integrate the actual SkyReels-V2 inference code from the original repository.

### Integration Steps

#### 1. Clone SkyReels-V2 Repository

```bash
git clone https://github.com/SkyworkAI/SkyReels-V2.git
cd SkyReels-V2
```

#### 2. Study the Inference Code

Key files to examine:
- `skyreels_v2_infer/` - Contains the actual inference code
- `generate_video.py` - Example T2V generation
- `generate_video_df.py` - Diffusion Forcing generation

#### 3. Adapt Model Loading

In `utils/model_loader.py`, replace the placeholder model loading with actual SkyReels-V2 loading:

```python
# Example adaptation (check actual SkyReels code):
from skyreels_v2_infer import SkyReelsPipeline  # Adjust import path

pipeline = SkyReelsPipeline.from_pretrained(
    model_dir,
    torch_dtype=torch.float16,
    device_map="auto"
)
```

**Location**: Lines 65-109 (T2V), Lines 145-178 (I2V)

#### 4. Adapt Generation Calls

In `scripts/segment_generator.py`, replace the placeholder generation calls:

```python
# Example adaptation (check actual SkyReels API):
output = model.generate(
    prompt=prompt,
    num_frames=num_frames,
    height=height,
    width=width,
    num_inference_steps=self.num_inference_steps,
    guidance_scale=self.guidance_scale
)
```

**Location**: Lines 95-166 (T2V), Lines 212-281 (I2V)

#### 5. Handle Output Format

SkyReels may output videos in different formats. Adapt `_save_frames_as_video()` in `segment_generator.py` to handle the actual output format.

### Model Download

Models are available on Hugging Face:
- **T2V**: `Skywork/SkyReels-V2-DF-14B-720P-Diffusers`
- **I2V**: `Skywork/SkyReels-V2-I2V-14B-720P`

Or use the 1.3B models for lower VRAM requirements.

Models will be automatically downloaded on first run, or manually:

```bash
huggingface-cli download Skywork/SkyReels-V2-DF-14B-720P-Diffusers
huggingface-cli download Skywork/SkyReels-V2-I2V-14B-720P
```

### Testing Integration

1. Test with a single segment first
2. Verify output format matches expectations
3. Check memory usage and performance
4. Adjust parameters in `configs/model_config.yaml`
5. Test full pipeline with sample script

### Common Integration Issues

1. **Import Errors**: Make sure SkyReels-V2 is in Python path or install as package
2. **Model Format**: Check if models need specific loading methods
3. **Output Format**: Verify video/frame output format matches our processing
4. **Memory**: Use model offloading if VRAM is limited

## Deployment

### Azure VM Deployment

#### Prerequisites
- Azure VM with GPU (NC-series or ND-series recommended)
- NVIDIA drivers installed
- Docker with NVIDIA runtime (optional)

#### Steps

1. **Clone Repository**
   ```bash
   git clone <your-repo-url>
   cd Skyreels-Movie-Generator
   ```

2. **Install Dependencies**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Integrate SkyReels-V2**
   - Clone SkyReels-V2 repository
   - Follow integration steps above

4. **Test Generation**
   ```bash
   python main.py \
       --script scripts/sample_script.txt \
       --mode t2v \
       --output outputs/final/test_video.mp4
   ```

### Docker Deployment

#### Build Image

```bash
docker build -t skyreels-generator:latest .
```

#### Run Container

```bash
docker run --gpus all \
    -v $(pwd)/outputs:/app/outputs \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/configs:/app/configs \
    skyreels-generator:latest \
    python main.py --script /app/scripts/sample_script.txt --output /app/outputs/final/video.mp4
```

#### Using Docker Compose

```bash
docker-compose up
```

### On-Premise Deployment

#### Option 1: Direct Python Installation

1. Follow Azure VM steps (Steps 1-4)
2. Set up as systemd service (optional)
3. Configure firewall if exposing API

#### Option 2: Docker Deployment

1. Install Docker and NVIDIA Container Toolkit
2. Build and run using Docker Compose
3. Configure volumes for persistent storage

#### Systemd Service Example

Create `/etc/systemd/system/skyreels.service`:

```ini
[Unit]
Description=SkyReels Movie Generator
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/Skyreels-Movie-Generator
ExecStart=/path/to/venv/bin/python main.py --script /path/to/script.txt
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable skyreels
sudo systemctl start skyreels
```

### Environment Variables

Create `.env` file:

```bash
CUDA_VISIBLE_DEVICES=0
HF_TOKEN=your_huggingface_token  # If needed for private models
MODEL_CACHE_DIR=/path/to/models
OUTPUT_DIR=/path/to/outputs
```

### Monitoring

#### GPU Usage
```bash
watch -n 1 nvidia-smi
```

#### Disk Space
```bash
df -h
du -sh outputs/ models/
```

#### Logs
Check `outputs/` directory for generation logs and segment files.

## Troubleshooting

### Out of Memory

- Reduce resolution in `configs/model_config.yaml` (e.g., `540p` → `480p`)
- Enable model offloading: `enable_offload: true`
- Use smaller model: `size: "1.3b"`
- Reduce number of inference steps
- Reduce segment duration

### Slow Generation

- Check GPU utilization: `nvidia-smi`
- Verify GPU is being used (not CPU)
- Reduce number of inference steps
- Use lower resolution
- Disable model offloading if you have enough VRAM

### Model Download Issues

- Check internet connection
- Verify Hugging Face token if needed
- Manually download models using `huggingface-cli`
- Check disk space

### Integration Issues

- Ensure SkyReels-V2 repository is cloned and accessible
- Verify Python path includes SkyReels code
- Check model loading code matches SkyReels API
- Review error messages for specific issues

### Video Stitching Issues

- Verify all segments have same resolution and FPS
- Check FFmpeg is installed (for FFmpeg-based stitching)
- Ensure sufficient disk space
- Check video file permissions

## Implementation Status

### ✅ Completed Components

- ✅ Complete project structure
- ✅ Script parsing and segmentation
- ✅ Prompt engineering
- ✅ Video stitching (FFmpeg and OpenCV)
- ✅ Post-processing (normalization, effects)
- ✅ Model loading framework (ready for SkyReels integration)
- ✅ T2V/I2V generation interfaces (ready for SkyReels integration)
- ✅ Docker support
- ✅ Configuration management
- ✅ Error handling
- ✅ GPU detection
- ✅ Comprehensive documentation

### ⚠️ Integration Required

- ⚠️ Model loading: Needs SkyReels-V2 custom pipeline
- ⚠️ Video generation: Needs SkyReels-V2 API calls

**Status**: ~90% Complete

The project has a complete, production-ready structure with all pipeline components implemented. Only the actual SkyReels-V2 model integration remains, which requires studying their inference code and adapting the model loading and generation calls.

## Workflow

1. **Input**: Script text file (e.g., `scripts/sample_script.txt`)
2. **Parsing**: Script is segmented into logical video segments
3. **Generation**: Each segment is generated using T2V or I2V
4. **Stitching**: Segments are concatenated into final video
5. **Output**: Final video saved to `outputs/final/`

## Use Case: P-TAL Store Video

The included sample script (`scripts/sample_script.txt`) demonstrates generating a promotional video for a P-TAL store in Vasant Kunj, including:

- Store entrance shots
- Product showcases (brass, copper, kansa cookware)
- Influencer interactions
- Wide shots and closing scenes

## File Descriptions

### Core Modules

- **main.py**: Entry point, orchestrates entire pipeline
- **utils/prompt_parser.py**: Converts script text to video prompts
- **utils/model_loader.py**: Loads and manages SkyReels models
- **scripts/segment_generator.py**: Generates video segments
- **scripts/video_stitcher.py**: Concatenates video segments
- **scripts/post_processor.py**: Video post-processing operations
- **utils/video_utils.py**: Video manipulation utilities

### Configuration

- **configs/model_config.yaml**: All model and generation parameters
- **scripts/sample_script.txt**: Example P-TAL store script

### Development Tools

- **check_setup.py**: Setup verification script
- **setup.py**: Package installation

## Testing Checklist

Once SkyReels is integrated:

- [ ] Single segment T2V generation
- [ ] Single segment I2V generation
- [ ] Full pipeline with sample script
- [ ] Video stitching quality
- [ ] GPU memory usage
- [ ] Error handling
- [ ] Docker container functionality

## Future Enhancements

- [ ] API endpoint for remote generation
- [ ] Batch processing support
- [ ] Advanced transitions and effects
- [ ] Audio overlay support
- [ ] Real-time preview
- [ ] Web UI

## License

This project uses SkyReels-V2 models. Please refer to the original SkyReels-V2 repository for licensing information.

## Acknowledgments

- **SkyReels-V2** by SkyworkAI
- Based on: https://github.com/SkyworkAI/SkyReels-V2

## Support

For issues or questions:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review the [SkyReels-V2 Integration](#skyreels-v2-integration) section
3. Verify setup with `python check_setup.py`
4. Check SkyReels-V2 repository for model-specific issues
