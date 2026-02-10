import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from osgeo import gdal
from matplotlib.image import AxesImage

class DEMVisualizer:
    def __init__(self):
        self.dem_files = []
        self.dem_mosaic = None
        self.dem_extent = None
        self.overlay_image = None
        self.overlay_extent = None
        self.overlay_alpha = 0.7
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.ax.set_title('DEM Visualizer - Pan/Zoom with Mouse: Left=Pan, Right=Zoom, Middle=Reset')
        
        # Create buttons for controlling overlay transparency
        ax_alpha_up = plt.axes([0.7, 0.05, 0.1, 0.05])
        ax_alpha_down = plt.axes([0.8, 0.05, 0.1, 0.05])
        ax_reset_view = plt.axes([0.6, 0.05, 0.1, 0.05])
        self.btn_alpha_up = Button(ax_alpha_up, 'Alpha +')
        self.btn_alpha_down = Button(ax_alpha_down, 'Alpha -')
        self.btn_reset_view = Button(ax_reset_view, 'Reset View')
        self.btn_alpha_up.on_clicked(self.increase_alpha)
        self.btn_alpha_down.on_clicked(self.decrease_alpha)
        self.btn_reset_view.on_clicked(self.reset_view)
        
        # Variables for image dragging and pan/zoom
        self.dragging_overlay = False
        self.dragging_pan = False
        self.last_pos = None
        self.overlay_artist = None
        self.dem_artist = None
        self.xlim = None
        self.ylim = None
        
        # Connect events
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
    
    def load_dem_files(self, folder_path, center_coord=None, view_height_km=None, downsample_factor=4):
        """Load DEM files with proper extent handling during downsampling"""
        self.dem_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                          if f.lower().endswith(('.tif', '.tiff', '.dem'))]
        
        if not self.dem_files:
            raise ValueError("No DEM files found in the specified folder")

        # Create virtual mosaic
        vrt = gdal.BuildVRT("mosaic.vrt", self.dem_files)
        
        # Get ORIGINAL geotransform and extent (before any downsampling)
        gt = vrt.GetGeoTransform()
        original_extent = (
            gt[0],  # x_min
            gt[0] + vrt.RasterXSize * gt[1],  # x_max
            gt[3] + vrt.RasterYSize * gt[5],  # y_min
            gt[3]   # y_max
        )
        
        # Apply downsampling while maintaining correct geographic extent
        if downsample_factor > 1:
            xsize = int(vrt.RasterXSize / downsample_factor)
            ysize = int(vrt.RasterYSize / downsample_factor)
            
            # Warp with cubic resampling for smoother results
            warped_vrt = gdal.Warp("", vrt, format='VRT', 
                                  width=xsize, height=ysize,
                                  resampleAlg='cubic')
            self.dem_mosaic = warped_vrt.ReadAsArray()
            warped_vrt = None
        else:
            self.dem_mosaic = vrt.ReadAsArray()
        
        # Clip negative values to 0 (e.g., ocean depths)
        self.dem_mosaic = self.dem_mosaic.clip(min=0)
        
        # The ACTUAL geographic extent never changes
        self.dem_extent = original_extent  # Note: Using original extent!
        self.xlim = (original_extent[0], original_extent[1])
        self.ylim = (original_extent[2], original_extent[3])
        
        # Display with correct extent regardless of downsampling
        self.dem_artist = self.ax.imshow(
            self.dem_mosaic,
            extent=self.dem_extent,  # Using original geographic extent
            cmap='terrain',
            origin='upper'
        )
        
        # Set initial view if requested
        if center_coord and view_height_km:
            self.set_initial_view(center_coord, view_height_km)
        else:
            self.ax.set_xlim(self.xlim)
            self.ax.set_ylim(self.ylim)
        
        vrt = None  # Cleanup

    def set_initial_view(self, center_coord, view_height_km):
        """Set view centered on coordinates with height in km"""
        center_lon, center_lat = center_coord
        
        # Convert view height from km to degrees
        # 1 degree ≈ 111.32 km (latitude)
        lat_span_deg = view_height_km / 111.32
        
        # Longitude span depends on latitude
        lon_span_deg = view_height_km / (111.32 * math.cos(math.radians(center_lat)))
        
        # Calculate bounds
        x_min = center_lon - lon_span_deg/2
        x_max = center_lon + lon_span_deg/2
        y_min = center_lat - lat_span_deg/2
        y_max = center_lat + lat_span_deg/2
        
        # Apply to plot
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)    

    def load_overlay_image(self, image_path):
        """Load an image to overlay on the DEM"""
        img = plt.imread(image_path)
        
        # Set initial extent to match DEM bounds but centered
        dem_width = self.dem_extent[1] - self.dem_extent[0]
        dem_height = self.dem_extent[3] - self.dem_extent[2]
        
        # Calculate aspect ratio of the image
        img_aspect = img.shape[1] / img.shape[0]
        
        # Set initial width to 1/4 of DEM width, height proportional
        init_width = dem_width / 4
        init_height = init_width / img_aspect
        
        # Center the image on the DEM
        center_x = (self.dem_extent[0] + self.dem_extent[1]) / 2
        center_y = (self.dem_extent[2] + self.dem_extent[3]) / 2
        
        x_min = center_x - init_width/2
        x_max = center_x + init_width/2
        y_min = center_y - init_height/2
        y_max = center_y + init_height/2
        
        self.overlay_extent = (x_min, x_max, y_min, y_max)
        
        # Display the overlay image
        self.overlay_artist = self.ax.imshow(
            img, 
            extent=self.overlay_extent, 
            alpha=self.overlay_alpha,
            zorder=10  # Ensure the overlay is on top
        )
        
        plt.draw()
    
    def increase_alpha(self, event):
        """Increase overlay transparency"""
        self.overlay_alpha = min(1.0, self.overlay_alpha + 0.1)
        if self.overlay_artist:
            self.overlay_artist.set_alpha(self.overlay_alpha)
            plt.draw()
    
    def decrease_alpha(self, event):
        """Decrease overlay transparency"""
        self.overlay_alpha = max(0.1, self.overlay_alpha - 0.1)
        if self.overlay_artist:
            self.overlay_artist.set_alpha(self.overlay_alpha)
            plt.draw()
    
    def reset_view(self, event):
        """Reset the view to show entire DEM"""
        self.ax.set_xlim(self.xlim)
        self.ax.set_ylim(self.ylim)
        plt.draw()
    
    def on_click(self, event):
        """Handle mouse click events"""
        if event.inaxes != self.ax:
            return
        
        # Left click - check for overlay drag or start pan
        if event.button == 1:
            if self.overlay_artist and event.xdata and event.ydata:
                x, y = event.xdata, event.ydata
                x_min, x_max, y_min, y_max = self.overlay_extent
                
                if x_min <= x <= x_max and y_min <= y <= y_max:
                    self.dragging_overlay = True
                    self.last_pos = (x, y)
                    return
            
            # If not dragging overlay, start pan
            self.dragging_pan = True
            self.last_pos = (event.xdata, event.ydata)
        
        # Middle click - reset view
        elif event.button == 2:
            self.reset_view(event)
            return
        
        # Right click - normal click to get coordinates
        elif event.button == 3 and event.xdata and event.ydata:
            x, y = event.xdata, event.ydata
            
            # Get elevation from DEM mosaic
            if self.dem_mosaic is not None:
                # Convert coordinates to pixel indices
                gt = gdal.Open(self.dem_files[0]).GetGeoTransform()
                
                # Calculate pixel coordinates
                px = int((x - gt[0]) / gt[1])
                py = int((y - gt[3]) / gt[5])
                
                # Ensure we're within bounds
                if 0 <= px < self.dem_mosaic.shape[1] and 0 <= py < self.dem_mosaic.shape[0]:
                    elevation = self.dem_mosaic[py, px]
                    print(f"Clicked at: Longitude={x:.6f}, Latitude={y:.6f}, Elevation={elevation:.2f} m")
                else:
                    print("Clicked outside DEM bounds")
    
    def on_release(self, event):
        """Handle mouse release events"""
        self.dragging_overlay = False
        self.dragging_pan = False
        self.last_pos = None
    
    def on_motion(self, event):
        """Handle mouse motion events"""
        if event.inaxes != self.ax or not event.xdata or not event.ydata:
            return
        
        x, y = event.xdata, event.ydata
        
        # Handle overlay dragging
        if self.dragging_overlay and self.last_pos and self.overlay_artist:
            last_x, last_y = self.last_pos
            
            # Calculate movement delta
            dx = x - last_x
            dy = y - last_y
            
            # Update overlay extent
            x_min, x_max, y_min, y_max = self.overlay_extent
            width = x_max - x_min
            height = y_max - y_min
            
            new_x_min = x_min + dx
            new_x_max = x_max + dx
            new_y_min = y_min + dy
            new_y_max = y_max + dy
            
            self.overlay_extent = (new_x_min, new_x_max, new_y_min, new_y_max)
            
            # Update the artist
            self.overlay_artist.set_extent(self.overlay_extent)
            
            self.last_pos = (x, y)
            plt.draw()
        
        # Handle panning
        elif self.dragging_pan and self.last_pos:
            last_x, last_y = self.last_pos
            dx = last_x - x
            dy = last_y - y
            
            # Get current limits
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            
            # Apply pan
            self.ax.set_xlim(xlim[0] + dx, xlim[1] + dx)
            self.ax.set_ylim(ylim[0] + dy, ylim[1] + dy)
            
            self.last_pos = (x, y)
            plt.draw()
    
    def on_scroll(self, event):
        """Handle scroll events for zooming"""
        if event.inaxes != self.ax:
            return
        
        # Get current limits
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        
        # Calculate zoom factor (1.2 for zoom in, 0.8 for zoom out)
        zoom_factor = 1.2 if event.button == 'down' else 0.8
        
        # Get cursor position in data coordinates
        x, y = event.xdata, event.ydata
        
        # Calculate new limits centered on cursor position
        new_width = (xlim[1] - xlim[0]) * zoom_factor
        new_height = (ylim[1] - ylim[0]) * zoom_factor
        
        new_xlim = (x - (x - xlim[0]) * zoom_factor, 
                    x + (xlim[1] - x) * zoom_factor)
        new_ylim = (y - (y - ylim[0]) * zoom_factor, 
                    y + (ylim[1] - y) * zoom_factor)
        
        # Apply new limits
        self.ax.set_xlim(new_xlim)
        self.ax.set_ylim(new_ylim)
        plt.draw()
    
    def show(self):
        """Show the visualization"""
        plt.tight_layout()
        plt.show()

import math

def degrees_to_meters(lon_deg, lat_deg, ref_lat=None):
    """
    Convert decimal degrees to meters using Haversine formula
    ref_lat: Reference latitude for longitudinal distance calculation
             (if None, uses input lat_deg)
    """
    if ref_lat is None:
        ref_lat = lat_deg
        
    # Earth's radius in meters
    R = 6371000 
    
    # Latitude distance (constant)
    lat_m = lat_deg * (math.pi/180) * R
    
    # Longitude distance (varies with latitude)
    lon_m = lon_deg * (math.pi/180) * R * math.cos(math.radians(ref_lat))
    
    return lon_m, lat_m


# Example usage
if __name__ == "__main__":
    visualizer = DEMVisualizer()

    center_coord = (-43.58340, 172.75340)  
    view_height_km = 2.6  # Show 2.6km tall view
    
    # Load DEM files from a folder (change this to your DEM folder path)
    dem_folder = "./chc-dem"
    visualizer.load_dem_files(
            dem_folder,
            # center_coord=center_coord,
            # view_height_km=view_height_km,
            downsample_factor=10
        )
    
    # Load an overlay image (change this to your image path)
    # overlay_image = "download.png"
    # visualizer.load_overlay_image(overlay_image)
    
    visualizer.show()
