import numpy as np
import matplotlib.pyplot as plt

DEG2RAD = np.pi / 180
RAD2DEG = 180 / np.pi

def calc_line_breaks(points, dmax=0.5):
    # where the line rolls over the break in x coord or y coord
    points2 = np.roll(points, 1, 0)
    points2[0, :] = points[0,:]
    d = np.abs(points2 - points)
    
    return np.argwhere(np.logical_or(d[:,0] > dmax, d[:,1] > dmax)).reshape(-1)

def plot_line_map(points, breaks=None, ax=None, figsize=(6,3), tickdata=None, color="k", linewidth=1):
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        ax.set_aspect("equal")
        
    if breaks is None:
        ax.plot(points[:,0], points[:,1], color=color, linewidth=linewidth)
    else:
        last_br = 0
        for br in breaks:
            ax.plot(points[last_br:br, 0], points[last_br:br, 1], color=color, linewidth=linewidth)
            last_br = br
            
        ax.plot(points[last_br:, 0], points[last_br:, 1], color=color, linewidth=linewidth)
        
    if tickdata is None:
        ax.set_xticks([])
        ax.set_yticks([])
        
    plt.show()
    
    return ax

def make_grid(parallels, meridians, numsteps=100):
    # inputs in radians, output is cartesian
    x_parallel = np.linspace(-np.pi, np.pi, numsteps)
    y_meridian = np.linspace(-np.pi/2, np.pi/2, numsteps)
    
    grid = np.zeros(shape=((numsteps + 1) * (len(parallels) + len(meridians)),2))
    
    i = 0
    
    for p in parallels:
        grid[i*(numsteps+1):(i+1)*(numsteps+1)-1,0] = x_parallel
        grid[i*(numsteps+1):(i+1)*(numsteps+1)-1,1] = p
        grid[(i+1)*(numsteps+1)-1,:] = np.nan
        i += 1
        
    for m in meridians:
        grid[i*(numsteps+1):(i+1)*(numsteps+1)-1,0] = m
        grid[i*(numsteps+1):(i+1)*(numsteps+1)-1,1] = y_meridian
        grid[(i+1)*(numsteps+1)-1,:] = np.nan
        i += 1
        
    return spherical_to_cartesian(grid)
        
    
def correct_ticks(ax, x_y, center, half_width=180, step=60):
    num_ticks = int(2 * half_width / step) + 1
    tick_locs = [-half_width + i*step for i in range(num_ticks)]
    tick_labels = [(val + half_width + center) % (2 * half_width) - half_width for val in tick_locs]
    
    if x_y == "x":
        ax.set_xticks(tick_locs)
        ax.set_xticklabels(tick_labels)
    else:
        ax.set_yticks(tick_locs)
        ax.set_yticklabels(tick_labels)
        
    return ax
    
def rotmat_euler321(anga, angb, angc):
    # dcm that rotates from base to rotated frame, all in radians
    rota = np.array([[np.cos(anga),np.sin(anga),0],[-np.sin(anga),np.cos(anga),0],[0,0,1]])
    rotb = np.array([[np.cos(angb), 0, np.sin(angb)],[0,1,0],[-np.sin(angb), 0, np.cos(angb)]])
    rotc = np.array([[1, 0, 0], [0, np.cos(angc), -np.sin(angc)], [0, np.sin(angc), np.cos(angc)]])
    
    return np.dot(rotc, np.dot(rotb, rota))

def rotmat_euler323(anga, angb, angc):
    # dcm that rotates from base to rotated frame, all in radians
    rota = np.array([[np.cos(anga),np.sin(anga),0],[-np.sin(anga),np.cos(anga),0],[0,0,1]])
    rotb = np.array([[np.cos(angb), 0, np.sin(angb)],[0,1,0],[-np.sin(angb), 0, np.cos(angb)]])
    rotc = np.array([[np.cos(angc),np.sin(angc),0],[-np.sin(angc),np.cos(angc),0],[0,0,1]])
    
    return np.dot(rotc, np.dot(rotb, rota))

def spherical_to_cartesian(coords):
    coords2 = np.zeros(shape=(coords.shape[0],3))
    coords2[:,0] = np.cos(coords[:,1]) * np.cos(coords[:,0])
    coords2[:,1] = np.cos(coords[:,1]) * np.sin(coords[:,0])
    coords2[:,2] = np.sin(coords[:,1])
    return coords2

def cartesian_to_spherical(coords):
    coords2 = np.zeros(shape=(coords.shape[0],2))
    coords2[:,0] = np.arctan2(coords[:,1],coords[:,0])
    coords2[:,1] = np.arcsin(coords[:,2])
    return coords2

def transform_spherical(coords, center_lon=0, center_lat=0, heading=0):
    rotmat = rotmat_euler321(center_lon, center_lat, heading)
    coords2_cartesian = np.dot(coords, rotmat.transpose())
    return cartesian_to_spherical(coords2_cartesian)

def transform_mercator(coords, truncate_north, truncate_south=None, ref_lat=0,
                       axis_lon=0, axis_lat=np.pi/2, axis_roll=0,
                       return_ref_parallels=False):
    # all in radians
    # ref_lat is latitude where scale is equal to 1 (after axis rotation)
    # axis rotation happens 3-2-3
    coords_rotated = np.dot(coords, rotmat_euler323(axis_lon, axis_lat - np.pi/2, axis_roll).transpose())
    coords_spherical = cartesian_to_spherical(coords_rotated)
    
    # scale factor: 1 = K * sec(lat) --> K = cos(lat)
    K = np.cos(ref_lat)
    
    if truncate_south is None:
        truncate_south = -truncate_north
        
    coords_spherical[coords_spherical[:,1]>truncate_north, :] = np.nan
    coords_spherical[coords_spherical[:,1]<truncate_south, :] = np.nan
    
    coords_mercator = np.zeros(shape=coords_spherical.shape)
    coords_mercator[:,0] = K * coords_spherical[:,0]
    coords_mercator[:,1] = K * np.log(np.tan(np.pi/4 + coords_spherical[:,1]/2))
    
    if return_ref_parallels:
        ref_parallels = np.array([[-np.pi, ref_lat], [0, ref_lat], [np.pi, ref_lat],
                                  [np.nan, np.nan],
                                  [-np.pi, -ref_lat], [0, -ref_lat], [np.pi, -ref_lat]])
        
        ref_parallels[:,0] = K * ref_parallels[:,0]
        ref_parallels[:,1] = K * np.log(np.tan(np.pi/4 + ref_parallels[:,1]/2))
        
        return coords_mercator, ref_parallels
    
    else:
        return coords_mercator


def transform_lambert_cylindrical(coords, truncate_north, truncate_south=None, ref_lat=0,
                                  return_ref_parallels=False):
    # equal area cylindrical projection - lambert has ref lat at 0, gall-peters at 45
    # scale factor horz: k = K * sec(lat) 
    # scale factor vert: h = y'
    # equal area constraint: h = 1 / k --> y' = 1/K * cos(lat) --> y = 1/K * sin(lat)
    # ref_lat is where k = h = 1 (scaling is 1, no distortion)
    
    coords_spherical = cartesian_to_spherical(coords)
    
    # 1 = K * sec(ref_lat)
    K = np.cos(ref_lat)
    
    if truncate_south is None:
        truncate_south = -truncate_north
        
    coords_spherical[coords_spherical[:,1]>truncate_north, :] = np.nan
    coords_spherical[coords_spherical[:,1]<truncate_south, :] = np.nan
    
    coords_lambert = np.zeros(shape=coords_spherical.shape)
    coords_lambert[:,0] = K * coords_spherical[:,0]
    coords_lambert[:,1] = 1 / K * np.sin(coords_spherical[:,1])
    
    if return_ref_parallels:
        ref_parallels = np.array([[-np.pi, ref_lat], [0, ref_lat], [np.pi, ref_lat],
                                  [np.nan, np.nan],
                                  [-np.pi, -ref_lat], [0, -ref_lat], [np.pi, -ref_lat]])
        
        ref_parallels[:,0] = K * ref_parallels[:,0]
        ref_parallels[:,1] = 1 / K * np.sin(ref_parallels[:,1])
        
        return coords_lambert, ref_parallels
    
    else:
        return coords_lambert 
    
    
def transform_sinusoidal(coords, center_lon=0, center_lat=0, heading=0):
    # axis rotation happens 3-2-1
    coords_spherical = transform_spherical(coords, center_lon, center_lat, heading)
    
    coords_sin = np.zeros(shape=coords_spherical.shape)
    coords_sin[:,0] = coords_spherical[:,0] * np.cos(coords_spherical[:,1])
    coords_sin[:,1] = coords_spherical[:,1]
    
    return coords_sin


def transform_sinusoidal_interrupted(coords, interrupts,
                                     center_lon=0, center_lat=0, heading=0,
                                     interrupt_style="center", include_borders=True):
    # style is center or break
    # interrupt longitudes are defined after all rotations have taken place
    if interrupt_style == "center":
        interrupt_centers = interrupts
        interrupt_extents = [[-180 * DEG2RAD, 180 * DEG2RAD] for _ in range(len(interrupt_centers))]
        
        for i in range(1,len(interrupt_centers)):
            d = interrupt_centers[i] - interrupt_centers[i-1]
            interrupt_extents[i-1][1] = interrupt_centers[i-1] + d/2
            interrupt_extents[i][0] = interrupt_centers[i] - d/2
        
    elif interrupt_style == "break":
        interrupt_extents = [[-180 * DEG2RAD, 180 * DEG2RAD] for _ in range(len(interrupts)+1)]
        
        for i in range(len(interrupts)):
            interrupt_extents[i][1] = interrupts[i]
            interrupt_extents[i+1][0] = interrupts[i]
            
        interrupt_centers = [(e[0]+e[1])/2 for e in interrupt_extents]
        
    # borders of the interrupt regions
    interrupt_borders = []

    for i in range(len(interrupt_centers)):
        ey = np.linspace(-np.pi/2, np.pi/2,100)
        ex_left = (interrupt_extents[i][0] - interrupt_centers[i]) * np.cos(ey) + interrupt_centers[i]
        ex_right = (interrupt_extents[i][1] - interrupt_centers[i]) * np.cos(ey) + interrupt_centers[i]

        interrupt_borders.append(np.concatenate([np.stack([ex_left, ey], axis=1),
                                                 np.array([[np.nan, np.nan]]),
                                                 np.stack([ex_right, ey], axis=1),
                                                 np.array([[np.nan, np.nan]])], axis=0))

    interrupt_borders = np.concatenate(interrupt_borders, axis=0)
    
    # transform actual coordinates - one transform for each interrupt
    coords_spherical = transform_spherical(coords, center_lon, center_lat, heading)
    coords_sin_interrupted = []
    
    for center, extent in zip(interrupt_centers, interrupt_extents):
        csi = np.array(coords_spherical)
        csi[csi[:,0]<extent[0],:] = np.nan
        csi[csi[:,0]>extent[1],:] = np.nan

        csi[:,0] = (csi[:,0] - center) * np.cos(csi[:,1]) + center

        coords_sin_interrupted.append(np.concatenate([csi, np.array([[np.nan, np.nan]])], axis=0))
        
    coords_sin_interrupted = np.concatenate(coords_sin_interrupted, axis=0)
    
    if include_borders:
        return np.concatenate([coords_sin_interrupted, interrupt_borders], axis=0)
    else:
        return coords_sin_interrupted

