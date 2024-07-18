import argparse
import numpy as np
import cv2
from skimage import measure, morphology, segmentation
from scipy import ndimage as ndi
from PIL import Image
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os

def process_nuclei(tif_path_nuclei):
    # Load and convert TIFF image
    tif_image = Image.open(tif_path_nuclei).convert("L")
    image_np = np.array(tif_image)
    png_path = tif_path_nuclei.replace(".tif", ".png")
    tif_image.save(png_path, format="PNG")

    # Initial segmentation
    _, binary_image = cv2.threshold(image_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cleaned_image = morphology.remove_small_objects(binary_image > 0, min_size=30)
    cleaned_image = ndi.binary_fill_holes(cleaned_image)
    label_image = measure.label(cleaned_image)
    borders = segmentation.find_boundaries(label_image)

    # Nuclei analysis (Initial segmentation)
    props = measure.regionprops(label_image)
    nuclei_areas = [prop.area for prop in props]
    nuclei_centroids = [prop.centroid for prop in props]
    mean_area = np.mean(nuclei_areas)
    median_area = np.median(nuclei_areas)
    std_area = np.std(nuclei_areas)
    threshold_upper_stricter = 2 * median_area

    # Mark outliers
    output_image = np.dstack([image_np, image_np, image_np])
    output_image[borders] = [255, 0, 0]
    for prop in props:
        y, x = prop.centroid
        color = (0, 0, 255) if prop.area > threshold_upper_stricter else (0, 255, 0)
        cv2.drawMarker(output_image, (int(x), int(y)), color, markerType=cv2.MARKER_CROSS, thickness=1)
    outliers_large_stricter = [(prop.centroid, prop.area) for prop in props if prop.area > threshold_upper_stricter]
    Image.fromarray(output_image).save(png_path.replace(".png", "_segmented_with_stricter_outliers.png"))

    # Masks for step 2
    large_outliers_mask = np.zeros_like(image_np, dtype=bool)
    for prop in props:
        if prop.area > threshold_upper_stricter:
            large_outliers_mask[prop.coords[:, 0], prop.coords[:, 1]] = True

    # Step 2 processing
    large_outliers_image_np = np.zeros_like(image_np)
    large_outliers_image_np[large_outliers_mask] = image_np[large_outliers_mask]
    _, binary_large_outliers = cv2.threshold(large_outliers_image_np, 73, 255, cv2.THRESH_BINARY)
    cleaned_large_outliers = morphology.remove_small_objects(binary_large_outliers > 0, min_size=10)
    cleaned_large_outliers = ndi.binary_fill_holes(cleaned_large_outliers)
    label_large_outliers = measure.label(cleaned_large_outliers)
    borders_large_outliers = segmentation.find_boundaries(label_large_outliers)
    output_image_large_outliers_processed = np.dstack([large_outliers_image_np, large_outliers_image_np, large_outliers_image_np])
    output_image_large_outliers_processed[borders_large_outliers] = [255, 0, 0]
    for prop in measure.regionprops(label_large_outliers):
        y, x = prop.centroid
        cv2.drawMarker(output_image_large_outliers_processed, (int(x), int(y)), (0, 255, 0), markerType=cv2.MARKER_CROSS, thickness=1)
    processed_large_outliers_path = png_path.replace(".png", "_large_outliers_processed.png")
    Image.fromarray(output_image_large_outliers_processed).save(processed_large_outliers_path)

    # Collecting centroids from step 2 processing
    step2_centroids = [prop.centroid for prop in measure.regionprops(label_large_outliers)]
    nuclei_centroids.extend(step2_centroids)  # Combining centroids from both steps

    # Combine step 1 and step 2
    final_image = np.zeros_like(output_image)
    final_image[large_outliers_mask] = output_image_large_outliers_processed[large_outliers_mask]
    final_image[~large_outliers_mask] = output_image[~large_outliers_mask]
    final_combined_path = png_path.replace(".png", "_final_combined.png")
    Image.fromarray(final_image).save(final_combined_path)

    # Save centroids to CSV (Combining initial and step 2 centroids)
    centroids_df = pd.DataFrame(nuclei_centroids, columns=["centroid_y", "centroid_x"])
    centroids_csv_path = png_path.replace(".png", "_centroids.csv")
    centroids_df.to_csv(centroids_csv_path, index=False)

    # Save metrics
    metrics = {
        "mean_area": mean_area,
        "median_area": median_area,
        "std_area": std_area,
        "threshold_upper_strict": threshold_upper_stricter,
        "outliers_large_strict": len(outliers_large_stricter),
        "total_nuclei": len(nuclei_areas)
    }
    metrics_df = pd.DataFrame([metrics])
    metrics_txt_path = png_path.replace(".png", "_metrics.txt")
    metrics_df.to_csv(metrics_txt_path, index=False)

    output_files = {
        "segmented_with_stricter_outliers": png_path.replace(".png", "_segmented_with_stricter_outliers.png"),
        "large_outliers_processed": processed_large_outliers_path,
        "final_combined": final_combined_path,
        "centroids": centroids_csv_path,
        "metrics": metrics_txt_path
    }

    return output_files


def process_connections(tif_path_cell_walls, centroids_csv_path):
    # Load and convert the cell wall TIFF image
    tif_image_cell_walls = Image.open(tif_path_cell_walls).convert("L")
    image_np_cell_walls = np.array(tif_image_cell_walls)
    png_path_cell_walls = tif_path_cell_walls.replace(".tif", ".png")
    tif_image_cell_walls.save(png_path_cell_walls, format="PNG")

    # Create a binary mask for the cell wall image
    _, binary_image_cell_walls = cv2.threshold(image_np_cell_walls, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_image_path = png_path_cell_walls.replace(".png", "_binary_mask.png")
    Image.fromarray(binary_image_cell_walls).save(binary_image_path)

    # Extract centroids of nuclei from the previously generated CSV file
    centroids_df = pd.read_csv(centroids_csv_path)
    nuclei_centroids = centroids_df[['centroid_y', 'centroid_x']].to_numpy()

    # Create a graph for nuclei connections
    G = nx.Graph()

    # Add nuclei centroids as nodes
    for i, centroid in enumerate(nuclei_centroids):
        G.add_node(i, pos=(centroid[1], centroid[0]))

    # Calculate the distance matrix between all nuclei centroids
    centroid_array = np.array(nuclei_centroids)
    dist_matrix = np.linalg.norm(centroid_array[:, None] - centroid_array, axis=2)

    # Define a distance threshold for connections
    distance_threshold = 100  # Threshold selected from previous analysis

    # Determine connections based on the distance threshold
    connection_matrix = dist_matrix < distance_threshold
    np.fill_diagonal(connection_matrix, False)  # Remove self-connections

    # Add edges to the graph based on connections
    for i, j in zip(*np.where(connection_matrix)):
        G.add_edge(i, j)

    # Draw the graph overlaid on the cell wall image
    pos = nx.get_node_attributes(G, 'pos')

    plt.figure(figsize=(10, 10))
    plt.imshow(image_np_cell_walls, cmap='gray')
    nx.draw(G, pos, node_color='red', edge_color='blue', node_size=20, with_labels=False)
    plt.title("Nuclei Connection Graph (Threshold: 100 pixels)")
    output_graph_path = tif_path_cell_walls.replace(".tif", "_nuclei_connection_graph_100px.png")
    plt.savefig(output_graph_path)
    plt.close()

    # Calculate and save metrics
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    average_degree = np.mean([deg for node, deg in G.degree()])
    density = nx.density(G)

    metrics = {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "average_degree": average_degree,
        "density": density
    }

    metrics_df = pd.DataFrame([metrics])
    metrics_csv_path = png_path_cell_walls.replace(".png", "_connection_metrics.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)

    # Save edges to CSV
    edges = list(G.edges)
    edges_df = pd.DataFrame(edges, columns=["node1", "node2"])
    edges_csv_path = png_path_cell_walls.replace(".png", "_edges.csv")
    edges_df.to_csv(edges_csv_path, index=False)
    
       # Save edges to CSV
    edges = list(G.edges)
    edges_df = pd.DataFrame(edges, columns=["node1", "node2"])
    edges_csv_path = png_path_cell_walls.replace(".png", "_edges.csv")
    edges_df.to_csv(edges_csv_path, index=False)

    # Output paths
    output_files = {
        "connection_graph": output_graph_path,
        "connection_metrics": metrics_csv_path,
        "edges": edges_csv_path
    }

    return output_files

def main():
    parser = argparse.ArgumentParser(description="Process two TIFF images: one for nuclei and one for cell walls. Generates analysis results and metrics.")
    parser.add_argument("nuclei_image", type=str, help="Path to the nuclei TIFF image.")
    parser.add_argument("cell_walls_image", type=str, help="Path to the cell walls TIFF image.")
    args = parser.parse_args()

    nuclei_image_path = args.nuclei_image
    cell_walls_image_path = args.cell_walls_image

    print(f"Processing nuclei image: {nuclei_image_path}")
    nuclei_outputs = process_nuclei(nuclei_image_path)
    print("Nuclei processing complete. Generated files:")
    for key, value in nuclei_outputs.items():
        print(f"{key}: {value}")

    print(f"Processing cell walls image: {cell_walls_image_path}")
    connections_outputs = process_connections(cell_walls_image_path, nuclei_outputs["centroids"])
    print("Cell walls processing complete. Generated files:")
    for key, value in connections_outputs.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()