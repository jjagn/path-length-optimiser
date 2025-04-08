import rasterio
import matplotlib.pyplot as plt
import numpy as np

def visualize_dem(geotiff_path, cmap='terrain', title='Digital Elevation Model'):
    """
    Visualize a Digital Elevation Model (DEM) from a GeoTIFF file.
    
    Parameters:
    -----------
    geotiff_path : str
        Path to the input GeoTIFF file
    cmap : str, optional
        Colormap to use for visualization (default is 'terrain')
    title : str, optional
        Title for the visualization plot
    
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object containing the DEM visualization
    """
    try:
        # Open the GeoTIFF file
        with rasterio.open(geotiff_path) as src:
            # Read the first band (elevation data)
            elevation_data = src.read(1)
            
            # Get geospatial metadata
            transform = src.transform
            crs = src.crs
    
        # Create a figure with subplots for different visualizations
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(title, fontsize=16)
        
        # 1. Grayscale Elevation Map
        im1 = axs[0, 0].imshow(elevation_data, cmap='gray', origin='upper')
        axs[0, 0].set_title('Grayscale Elevation')
        plt.colorbar(im1, ax=axs[0, 0], shrink=0.8)
        
        # 2. Colored Terrain Map
        im2 = axs[0, 1].imshow(elevation_data, cmap=cmap, origin='upper')
        axs[0, 1].set_title(f'Terrain Map ({cmap} colormap)')
        plt.colorbar(im2, ax=axs[0, 1], shrink=0.8)
        
        # 3. 3D Surface Plot
        from mpl_toolkits.mplot3d import Axes3D
        X, Y = np.meshgrid(np.arange(elevation_data.shape[1]), 
                           np.arange(elevation_data.shape[0]))
        axs[1, 0] = fig.add_subplot(2, 2, 3, projection='3d')
        surf = axs[1, 0].plot_surface(X, Y, elevation_data, 
                                      cmap=cmap, edgecolor='none')
        axs[1, 0].set_title('3D Surface Elevation')
        fig.colorbar(surf, ax=axs[1, 0], shrink=0.8)
        
        # 4. Contour Plot
        contour = axs[1, 1].contourf(elevation_data, cmap=cmap, levels=20)
        axs[1, 1].set_title('Elevation Contours')
        fig.colorbar(contour, ax=axs[1, 1], shrink=0.8)
        
        # Adjust layout and display metadata
        plt.tight_layout()
        fig.text(0.5, 0.02, 
                f'CRS: {crs} | Geotransform: {transform}', 
                ha='center', fontsize=10)
        
        return fig
    
    except Exception as e:
        print(f"Error processing DEM data: {e}")
        return None

def main():
    # Example usage
    geotiff_path = 'dem/DEM_BQ27_2020_1000_5016.tif'  # Replace with your GeoTIFF file path
    
    # Visualize with default terrain colormap
    fig = visualize_dem(geotiff_path)
    
    # Optional: try different colormaps
    # fig = visualize_dem(geotiff_path, cmap='viridis', title='Custom DEM Visualization')
    
    plt.show()

if __name__ == '__main__':
    main()
