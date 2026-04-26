# Pixels2GenAI

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9-3.12](https://img.shields.io/badge/Python-3.9--3.12-blue.svg)](https://www.python.org/)
[![Docs](https://img.shields.io/badge/Docs-Live-green.svg)](https://burakkagann.github.io/Pixels2GenAI/)

An open-source educational platform that teaches generative art and AI through 15 progressive modules. The curriculum starts with pixel manipulation and NumPy operations, then moves through fractals, simulations, neural networks, and generative models.

**Live site:** https://burakkagann.github.io/Pixels2GenAI/

## Curriculum

The modules are grouped into four skill levels:

**Creative Coding Foundations (Modules 0-6)**

- 0 - Foundations & Definitions
- 1 - Pixel Fundamentals
- 2 - Geometry & Mathematics
- 3 - Transformations & Effects
- 4 - Fractals & Recursion
- 5 - Simulation & Emergent Behavior
- 6 - Noise & Procedural Generation

**Machine Learning & Animation (Modules 7-9)**

- 7 - Classical Machine Learning
- 8 - Animation & Time
- 9 - Intro to Neural Networks

**Real-Time & AI Integration (Modules 10-13)**

- 10 - TouchDesigner Fundamentals
- 11 - Interactive Systems
- 12 - Generative AI Models
- 13 - AI + TouchDesigner Integration

**Data & Capstone (Modules 14-15)**

- 14 - Data as Material
- 15 - Capstone Project

## Getting Started

### Prerequisites

- Python 3.11 recommended (supports 3.9-3.12)
- GPU recommended for neural network modules (7+), not required

### Installation

```bash
git clone https://github.com/burakkagann/Pixels2GenAI.git
cd Pixels2GenAI
python -m venv .venv
```

Activate the virtual environment:

```bash
# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

Install dependencies based on which modules you need:

```bash
# Core only (Modules 0-6)
pip install .

# With machine learning packages (Modules 7-13)
pip install .[ml]

# Everything
pip install .[all]
```

Or use the requirements file directly:

```bash
pip install -r requirements.txt
```

## Building the Documentation

```bash
pip install .[docs]
sphinx-build -b html . build/html   # macOS / Linux
make.bat html                       # Windows
```

The built site will be in `build/html/`. Open `build/html/index.html` in a browser to view it.

## Project Structure

```
Pixels2GenAI/
├── content/          # 16 learning modules (Module_00 through Module_15)
├── _static/          # Custom CSS, JavaScript, images
├── _templates/       # Sphinx templates
├── conf.py           # Sphinx configuration
├── index.rst         # Landing page
├── pyproject.toml    # Package configuration and dependencies
├── requirements.txt  # Pinned runtime dependencies
└── make.bat          # Windows build script
```

Each module follows this structure:

```
content/Module_XX_topic/
└── X.Y_subtopic/
    └── X.Y.Z_exercise/
        ├── README.rst    # Tutorial documentation
        ├── *.py          # Python scripts
        └── *.png         # Generated output images
```

## Contributing

Contributions are welcome. Open an issue or submit a pull request at:

https://github.com/burakkagann/Pixels2GenAI/issues

## License

MIT License. See [LICENSE](LICENSE) for details.
