# Neuro_Connect

A Python script for identifying nuclei and connections from stained images.

## Description

This project includes a Python script that processes two TIFF images: one for nuclei and one for cell walls. The script performs image segmentation, identifies nuclei, computes connections between nuclei, and generates various output files including metrics and connection graphs.

## Files

- `process_images.py`: The main script that processes the images and generates the outputs.
- `requirements.txt`: A file listing all the dependencies required to run the script.

## Setup

### Prerequisites

- Python 3.x
- pip (Python package installer)

### Installation

1. **Clone the repository:**

   ```
   git clone https://github.com/yourusername/Neuro_Connect.git
   cd Neuro_Connect
   ```

2. **Create and activate a virtual environment (optional but recommended):**

   ```
   python -m venv env
   source env/bin/activate    # On Windows, use `env\\Scripts\\activate`
   ```

3. **Install the dependencies:**

   ```
   pip install -r requirements.txt
   ```

## Usage

Run the script with the paths to your two TIFF images as arguments:

```
python process_images.py path/to/nuclei_image.tif path/to/cell_walls_image.tif
```

### Example

```
python process_images.py data/nuclei_image.tif data/cell_walls_image.tif
```

## Output

The script generates the following output files named based on the first image provided:

1. **Final Combined Image**: `nuclei_image_final_combined.png`
2. **Centroids CSV**: `nuclei_image_centroids.csv`
3. **Metrics TXT**: `nuclei_image_metrics.txt`
4. **Nuclei Connection Graph**: `cell_walls_image_nuclei_connection_graph_100px.png`
5. **Connection Metrics CSV**: `cell_walls_image_connection_metrics.csv`
6. **Edges CSV**: `cell_walls_image_edges.csv`

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
