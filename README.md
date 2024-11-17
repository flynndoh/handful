# Handful

## Intended Use

Real-time image processing to control servo motor actuation in 1DOF robotic fingers.

## Get Started 

### Clone the repository
```bash
git clone git@github.com:flynndoh/handful.git
cd handful
```

### Create and activate virtual environment
```bash
python -m venv .venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows
```

### Install dependencies
```bash
pip install -e ".[dev]"
```

### Run!
```bash
handful --stream_url http://192.168.0.117:8080/stream
```
By default, navigate to http://localhost:5000

### Run the CLI (Work in progress, likely broken)
```bash
handful-cli --help
```

