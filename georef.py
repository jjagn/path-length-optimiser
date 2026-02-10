import numpy as np


class AffineGeoref:
    """Least-squares affine transform from pixel coordinates to map coordinates.
    
    Fits: [easting]   = [a b c] @ [x]
          [northing]   = [d e f]   [y]
                                    [1]
    """

    def __init__(self):
        self.easting_coeffs = None  # [a, b, c]
        self.northing_coeffs = None  # [d, e, f]
        self.residuals_easting = None
        self.residuals_northing = None

    def fit(self, datums):
        """Fit affine transform from a list of DatumPoint objects with known x, y, easting, northing.
        
        Requires at least 3 datum points for an affine fit (2 for exact, 3+ for least-squares).
        """
        if len(datums) < 2:
            raise ValueError("Need at least 2 datum points to fit affine transform")

        A = np.array([[d.x, d.y, 1.0] for d in datums])
        E = np.array([d.easting for d in datums])
        N = np.array([d.northing for d in datums])

        result_e = np.linalg.lstsq(A, E, rcond=None)
        result_n = np.linalg.lstsq(A, N, rcond=None)

        self.easting_coeffs = result_e[0]
        self.northing_coeffs = result_n[0]

        if len(result_e[1]) > 0:
            self.residuals_easting = float(np.sqrt(result_e[1][0] / len(datums)))
        if len(result_n[1]) > 0:
            self.residuals_northing = float(np.sqrt(result_n[1][0] / len(datums)))

    def transform(self, x, y):
        """Transform pixel coordinates to easting, northing."""
        if self.easting_coeffs is None or self.northing_coeffs is None:
            raise RuntimeError("Affine transform not fitted. Call fit() first.")
        v = np.array([x, y, 1.0])
        easting = float(self.easting_coeffs @ v)
        northing = float(self.northing_coeffs @ v)
        return easting, northing

    def report(self):
        """Print fit quality."""
        print("Affine Georeferencing Transform:")
        print(f"  Easting  coeffs: {self.easting_coeffs}")
        print(f"  Northing coeffs: {self.northing_coeffs}")
        if self.residuals_easting is not None:
            print(f"  Easting  RMS residual: {self.residuals_easting:.2f} m")
        if self.residuals_northing is not None:
            print(f"  Northing RMS residual: {self.residuals_northing:.2f} m")


def georeference_controls(controls, georef):
    """Apply affine transform to set easting/northing on all controls."""
    for ctrl in controls:
        ctrl.easting, ctrl.northing = georef.transform(ctrl.x, ctrl.y)
