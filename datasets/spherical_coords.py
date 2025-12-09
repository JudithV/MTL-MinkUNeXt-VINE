# Judith Vilella Cantos. Miguel Hernández University of Elche.
# Code adapted from the MinkLoc3D-SI repository (https://github.com/KamilZywanowski/MinkLoc3D-SI) 
# to apply to more datasets used in our work.
import numpy as np

class SphericalCoords:
    def to_spherical(points, dataset_name):
        spherical_points = []
        if dataset_name == "vmd":
            # Cartesian to spherical conversion
            #APPROACH A: Conventional projection to polar space
            """r = np.linalg.norm(points, axis=1)
            mask_distance = r <= 5.0

            points_sel = points[mask_distance]
            x, y, z = points_sel.T
            r_sel = r[mask_distance]
            theta = np.degrees(np.arctan2(y, x))  # horizontal
            phi = np.degrees(np.arcsin(z / r_sel)) - 38  # vertical"""
            
            #APPROACH B: Spherical projection with a custom FoV that imitates the robotic platform's POV
            if points.shape[-1] == 4:
                x, y, z, intensity = points.T
            else:
                x, y, z = points.T
                intensity = None
            mask_region = (x >= 0) & (x <= 10) & (y >= -2.5) & (y <= 2.5)

            x_sel, y_sel, z_sel = x[mask_region], y[mask_region], z[mask_region]
            if intensity is not None:
                intensity = intensity[mask_region]
            r = np.sqrt(x_sel**2 + y_sel**2 + z_sel**2)

            # Spherical conversion
            theta = np.degrees(np.arctan2(y_sel, x_sel))       # Horizontal
            phi = np.degrees(np.arccos(z_sel / r)) - 38        # Vertical


            # New FoV(90° × 120°)
            theta_min, theta_max = -45, 45  # Horizontal (90°)
            phi_min, phi_max = -60, 60     # Vertical (120°)
            # Filters
            mask_theta = (theta >= theta_min) & (theta <= theta_max)
            mask_phi = (phi >= phi_min) & (phi <= phi_max)
            mask_fov = mask_theta & mask_phi
            # Apply masks
            # Original: r[mask_fov], theta[mask_fov], phi[mask_fov]
            if points.shape[-1] == 4:
                spherical_points = np.stack((r[mask_fov], theta[mask_fov], phi[mask_fov], intensity[mask_fov]), axis=1)
            else:
                spherical_points = np.stack((r[mask_fov], theta[mask_fov], phi[mask_fov]), axis=1)
        else:
            for point in points:
                if (np.abs(point[:3]) < 1e-4).all():
                    continue

                r = np.linalg.norm(point[:3])

                # Theta is calculated as an angle measured from the y-axis towards the x-axis
                # Shifted to range (0, 360)
                theta = np.rad2deg(np.arctan2(point[1], point[0]))
                if theta < 0:
                    theta += 360

                if dataset_name == "usyd":
                    # VLP-16 has 2 deg VRes and (+15, -15 VFoV).
                    # Phi calculated from the vertical axis, so (75, 105)
                    # Shifted to (0, 30)
                    phi = np.rad2deg(np.arccos(point[2] / r)) - 75
                elif dataset_name in ['intensityOxford', 'Oxford']:
                    # Oxford scans are built from a 2D scanner.
                    # Phi calculated from the vertical axis, so (0, 180)
                    phi = np.rad2deg(np.arccos(point[2] / r))

                elif dataset_name == 'KITTI':
                    # HDL-64 has 0.4 deg VRes and (+2, -24.8 VFoV).
                    # Phi calculated from the vertical axis, so (88, 114.8)
                    # Shifted to (0, 26.8)
                    phi = np.rad2deg(np.arccos(point[2] / r)) - 88
                elif dataset_name == 'nclt':
                    # HDL-32 has 0.4 deg VRes and (+10.67°, -30.67° VFoV).
                    # Phi calculated from the vertical axis, so (79.33, 120.67)
                    # Shifted to (0, 41.33)
                    phi = np.rad2deg(np.arcsin(point[2] / r)) - 79.33
                
                elif dataset_name == 'blt':
                    # OS1-16 has 0.4 deg VRes and (+16.6, -16.6 VFoV).
                    # Phi calculated from the vertical axis, so (73.4, 106.6)
                    # Shifted to (0, 33.2)
                    phi = np.rad2deg(np.arccos(point[2] / r)) - 73.4
                
                elif dataset_name == 'arvc':
                    # OS1-128 has 0.4 deg VRes and (+22.5, -22.5 VFoV).
                    # Phi calculated from the vertical axis, so (67.5, 112.5)
                    # Shifted to (0, 45)
                    phi = np.rad2deg(np.arccos(point[2] / r)) - 67.5
                
                if point.shape[-1] == 4:
                    spherical_points.append([r, theta, phi, point[3]])
                else:
                    spherical_points.append([r, theta, phi])
        return np.array(spherical_points)
