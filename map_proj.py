import numpy as np
from scipy.optimize import newton
import matplotlib.pyplot as plt

DEG2RAD = np.pi / 180
RAD2DEG = 180 / np.pi
M2KM = 1000
RE = 6378 * M2KM

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


def transform_mollweide(coords, center_lon, center_lat, heading, include_borders=True):
    coords_spherical = transform_spherical(coords, center_lon, center_lat, heading)
    
    if include_borders:
        n = 100
        border_lat = np.linspace(-np.pi/2, np.pi/2, n)
        coords_border = np.zeros(shape=(2*n + 2, 2))
        coords_border[0,:] = np.nan
        coords_border[1:n+1,0] = -np.pi
        coords_border[1:n+1,1] = border_lat
        coords_border[n+1,:] = np.nan
        coords_border[n+2:, 0] = np.pi
        coords_border[n+2:, 1] = border_lat
        
        coords_spherical = np.concatenate([coords_spherical, coords_border], axis=0)
        
    def mollweide_fun(theta, lat):
        return 2 * theta + np.sin(2 * theta) - np.pi * np.sin(lat)
    def mollweide_dot(theta, lat):
        return 2 + 2 * np.cos(2 * theta)
        
    # newton will not like working on nans
    isnan = np.isnan(coords_spherical[:,0])
    
    fun = lambda t: mollweide_fun(t, coords_spherical[~isnan, 1])
    prime = lambda t: mollweide_dot(t, coords_spherical[~isnan, 1])
    theta = newton(fun, coords_spherical[~isnan,1], prime)

    coords_mollweide = np.zeros(shape=coords_spherical.shape)
    coords_mollweide[isnan,:] = np.nan
    coords_mollweide[~isnan,0] = 2 * np.sqrt(2) / np.pi * coords_spherical[~isnan,0] * np.cos(theta)
    coords_mollweide[~isnan,1] = np.sqrt(2) * np.sin(theta)
    
    return coords_mollweide


def transform_gnomonic(coords, truncate_lat=np.pi/4, tangent_lon=0, tangent_lat=np.pi/2, heading=0, include_borders=True):
    # truncate_lat: how many degrees of latitude away from tangent point to show
    # transformation points the +z axis (lat=90) at tangent point
    ez = np.array([np.cos(tangent_lon)*np.cos(tangent_lat), np.sin(tangent_lon)*np.cos(tangent_lat), np.sin(tangent_lat)])

    if tangent_lat == np.pi/2:
        ey = np.array([0,1,0])
    else:
        ey = np.array([0,0,1])

    ey = ey - np.dot(ey, ez) * ez
    ey = ey / np.linalg.norm(ey)

    ex = np.cross(ey, ez)

    rot_heading = np.array([[np.cos(heading), -np.sin(heading), 0],
                            [np.sin(heading), np.cos(heading), 0],
                            [0, 0, 1]])
    rotmat = np.dot(rot_heading, np.stack([ex, ey, ez], axis=0))
    coords_spherical = cartesian_to_spherical(np.dot(coords, rotmat.transpose()))

    # polar coords are theta, r
    coords_gn_polar = np.zeros(shape=coords_spherical.shape)
    coords_gn_polar[:,0] = coords_spherical[:,0]
    coords_gn_polar[:,1] = np.tan(np.pi/2 - coords_spherical[:,1])

    ind_truncate = coords_spherical[:,1] < np.pi/2 - truncate_lat
    coords_gn_polar[ind_truncate,:] = np.nan

    x = coords_gn_polar[:,1] * np.cos(coords_gn_polar[:,0])
    y = coords_gn_polar[:,1] * np.sin(coords_gn_polar[:,0])
    
    return np.stack([x, y], axis=1)


def transform_perspective_near_side(coords, center_lat, center_lon, alt, heading=0, include_limb=True):
    # altitude in kilometers, angles in radians
    # set altitude to np.inf for orthographic projection
    
    r_norm = np.array([np.cos(center_lon)*np.cos(center_lat),
                       np.sin(center_lon)*np.cos(center_lat),
                       np.sin(center_lat)])

    if alt == np.inf:
        r = 0 * r_norm
    else:
        r = (alt + RE) * r_norm

    # build frame with r as aligned, north as constrained, then rotate by heading
    p1 = r_norm

    if p1[0] == 0 and p1[1] == 0:
        p2 = np.array([0, -1, 0])
    elif alt == np.inf:
        p2 = np.array([1, 0, 0])
    else:
        p2 = np.array([-1, 0, 0])

    p2 = p2 - np.dot(p1, p2) * p1
    p2 = p2 / np.linalg.norm(p2)

    p3 = np.cross(p1, p2)

    heading_rot = np.array([[1, 0, 0],
                            [0, np.cos(heading), np.sin(heading)],
                            [0, -np.sin(heading), np.cos(heading)]])
    
    rotmat_ecef_to_p = np.dot(heading_rot, np.stack([p1, p2, p3]))

    # put all points in p frame, relative to r
    coords_p = np.dot(rotmat_ecef_to_p, (coords*RE - r.reshape(1,3)).transpose()).transpose()

    # don't plot anything that is further away than earth limb
    if alt == np.inf:
        coords_p[coords_p[:,0] < 0,:] = np.nan
    else:
        limbdist = np.sqrt((alt+RE)**2 - RE**2)
        coords_p_dist = np.linalg.norm(coords_p, axis=1)
        coords_p[coords_p_dist > limbdist,:] = np.nan

    # add the limb
    if include_limb:
        if alt == np.inf:
            limbdist = RE
            limbang = np.pi/2
        else:
            limbang = np.arcsin(RE/(RE+alt))
            
        t = np.linspace(0,2*np.pi,100)
        coords_limb = np.stack([limbdist * np.cos(limbang) * np.ones(shape=t.shape),
                                limbdist * np.sin(limbang) * np.cos(t),
                                limbdist * np.sin(limbang) * np.sin(t)], axis=1)
        coords_p = np.concatenate([coords_p, np.array([[np.nan, np.nan, np.nan]]), coords_limb], axis=0)
        
    # scale p1 coordinate to nadir tangent plane with earth (forgot this earlier)
    if alt != np.inf:
        coords_p = coords_p / coords_p[:,0].reshape(-1,1) * alt
    
    # project by taking last two coordinates
    return coords_p[:,1:]


def transform_perspective_far_side(coords, center_lat, center_lon, alt, heading=0, truncate_ang=0.37*np.pi, include_limb=True):
    # altitude is km above ground on far side of earth (opposite center_lat/lon point)
    # truncate angle is half angle
    # alt=0 is stereographic
    r_center_norm = np.array([np.cos(center_lon)*np.cos(center_lat),
                              np.sin(center_lon)*np.cos(center_lat),
                              np.sin(center_lat)])
    
    # transform from ecef to p-frame, which points towards center point with heading ccw from north/180lon
    p1 = r_center_norm

    if p1[0] == 0 and p1[1] == 0:
        p2 = np.array([0, -1, 0])
    else:
        p2 = np.array([0, 1, 0])

    p2 = p2 - np.dot(p1, p2) * p1
    p2 = p2 / np.linalg.norm(p2)

    p3 = np.cross(p1, p2)

    heading_rot = np.array([[1, 0, 0],
                            [0, np.cos(heading), -np.sin(heading)],
                            [0, np.sin(heading), np.cos(heading)]])
    
    rotmat_ecef_to_p = np.dot(heading_rot, np.stack([p1, p2, p3]))
    
    # r_opp is the projection origin point, then rotate to p
    r_opp_norm = -r_center_norm
    r_opp = (RE + alt) * r_opp_norm
    
    coords_p = np.dot(rotmat_ecef_to_p, (coords*RE - r_opp.reshape(1,3)).transpose()).transpose()

    # remove points that are below the horizon on the same side as r_opp
    # d is the distance along the p1 axis from r_opp to the horizon point
    d = alt + RE * (1 - RE / (RE + alt))
    coords_p[coords_p[:,0] < d, :] = np.nan
    
    # normalize all points to allow truncation
    coords_p = coords_p / np.linalg.norm(coords_p, axis=1, keepdims=True)
    
    coords_p[coords_p[:,0] < np.cos(truncate_ang), :] = np.nan
    
    # scale each point to 1 on p1 axis
    coords_p = coords_p / coords_p[:,0].reshape(-1,1)
    
    # project by taking only last two coords
    coords_proj = coords_p[:,1:]
    
    if include_limb:
        view_ang = np.arcsin(RE/(RE+alt))
        limb_ang = min(view_ang, truncate_ang)
        limb_dist = np.tan(limb_ang)
        t = np.linspace(0,2*np.pi,100)
        coords_limb = limb_dist * np.stack([np.cos(t), np.sin(t)], axis=1)
        coords_proj = np.concatenate([coords_proj, np.array([[np.nan, np.nan]]), coords_limb], axis=0)
    
    return coords_proj