"""
Functions / Objects for occupancy grid maps.

!!IMPORTANT!! xy-rc coordinate relation:
2D map. (xmin, ymin) is located at the bottom left corner.
rows are counted from the bottom to top.
columns are counted from left to right.

Matplotlib.pyplot displays a matrix with (r,c)=(0,0) located at the top left.
Therefore, in the display_wrapper, the map is flipped.
"""
import numpy as np
import yaml
from metadata import METADATA

def round(x):
    if x >= 0:
        return int(x+0.5)
    else:
        return int(x-0.5)

class GridMap(object):
    def __init__(self, map_path, margin2wall=0.5):
        map_config = yaml.load(open(map_path+".yaml", "r"), Loader=yaml.CLoader)
        self.map = np.loadtxt(map_path+".cfg")
        if 'empty' in map_path:
            self.map = None
        else:
            self.map_linear = np.squeeze(self.map.astype(np.int8).reshape(-1, 1))
        self.mapdim = map_config['mapdim']
        self.mapres = np.array(map_config['mapres'])
        self.mapmin = np.array(map_config['mapmin'])
        self.mapmax = np.array(map_config['mapmax'])
        self.margin2wall = margin2wall
        self.origin = map_config['origin']
        self.visit_freq_map = None
        self.visit_map = None

    def generate_map(self, **kwargs):
        pass

    def reset_visit_map(self):
        self.visit_map = np.zeros(self.mapdim)

    def reset_visit_freq_map(self):
        self.visit_freq_map = np.zeros(self.mapdim)

    def decay_visit_freq_map(self, decay_factor):
        if self.visit_freq_map is not None:
            self.visit_freq_map *= decay_factor

    def se2_to_cell(self, pos):
        pos = pos[:2]
        cell_idx = (pos - self.mapmin)/self.mapres - 0.5
        return round(cell_idx[0]), round(cell_idx[1])

    def cell_to_se2(self, cell_idx):
        return ( np.array(cell_idx) + 0.5 ) * self.mapres + self.mapmin

    def is_collision_ray_cell(self, cell):
        """
        cell : cell r, c index from left bottom.
        """
        idx = cell[0] + self.mapdim[0] * cell[1]
        if (cell[0] < 0) or (cell[1] < 0) or (cell[0] >= self.mapdim[0]) or (cell[1] >= self.mapdim[1]):
            return True
        elif (self.map is not None) and self.map_linear[idx] == 1:
            return True
        else:
            return False

    def is_collision(self, pos, margin=None):
        if not(self.in_bound(pos)):
            return True
        else:
            if self.map is not None:
                cell = np.minimum([self.mapdim[0]-1,self.mapdim[1]-1] , self.se2_to_cell(pos))
                if margin is None:
                    margin = self.margin2wall
                if margin == 0.0:
                    return self.is_collision_ray_cell(cell)
                n = np.ceil(margin/self.mapres).astype(np.int16)
                for r_add in np.arange(-n[1],n[1],1):
                    for c_add in np.arange(-n[0],n[0],1):
                        x_c = np.clip(cell[0]+r_add, 0, self.mapdim[0]-1).astype(np.int16)
                        y_c = np.clip(cell[1]+c_add,0,self.mapdim[1]-1).astype(np.int16)
                        idx = x_c + self.mapdim[0] * y_c
                        if self.map_linear[idx] == 1:
                            return True
        return False

    def in_bound(self, pos):
        return not((pos[0] < self.mapmin[0] + self.margin2wall)
            or (pos[0] > self.mapmax[0] - self.margin2wall)
            or (pos[1] < self.mapmin[1] + self.margin2wall)
            or (pos[1] > self.mapmax[1] - self.margin2wall))

    def in_bound_cell(self, cell):
        """
        cell : cell r, c index from left bottom.
        """
        if (cell[0] < 0) or (cell[1] < 0) or (cell[0] >= self.mapdim[0]) or (cell[1] >= self.mapdim[1]):
            return False
        return True

    def is_blocked(self, start_pos, end_pos):
        if self.map is None:
            return False
        start_rc = self.se2_to_cell(start_pos)
        end_rc = self.se2_to_cell(end_pos)
        ray_cells = bresenham2D(start_rc[0], start_rc[1], end_rc[0], end_rc[1])
        i = 0
        while(i < ray_cells.shape[-1]):
            if self.is_collision_ray_cell(ray_cells[:,i]):
                return True
            i += 1
        return False

    def get_front_obstacle(self, odom, r_max, **kwargs):
        """
        Return radial and angular distances of the front obstacle/boundary cell
        """
        ro_min_t = None
        start_rc = self.se2_to_cell(odom[:2])
        end_pt_global_frame = coord_change2g(np.array([r_max*np.cos(0.0), r_max*np.sin(0.0)]), odom[-1]) + odom[:2]
        if self.map is None:
            if not(self.in_bound(end_pt_global_frame)):
                end_rc = self.se2_to_cell(end_pt_global_frame)
                ray_cells = bresenham2D(start_rc[0], start_rc[1], end_rc[0], end_rc[1])
                i = 0
                while(i < ray_cells.shape[-1]):
                    pt = self.cell_to_se2(ray_cells[:,i])
                    if not(self.in_bound(pt)):
                        break
                    i += 1
                if i < ray_cells.shape[-1]: # break!
                    ro_min_t = np.sqrt(np.sum(np.square(pt - odom[:2])))
        else:
            end_rc = self.se2_to_cell(end_pt_global_frame)
            ray_cells = bresenham2D(start_rc[0], start_rc[1], end_rc[0], end_rc[1])
            i = 0
            while(i < ray_cells.shape[-1]): # break!
                if self.is_collision_ray_cell(ray_cells[:,i]):
                    break
                i += 1
            if i < ray_cells.shape[-1]:
                ro_min_t = np.sqrt(np.sum(np.square(self.cell_to_se2(ray_cells[:,i]) - odom[:2])))
        if ro_min_t is None:
            return None
        else:
            return ro_min_t, 0.0

    def get_closest_obstacle(self, odom, fov, r_max, ang_res=0.05):
        """
        Return radial and angular distances of the closest obstacle/boundary cell
        """
        odom = np.array(odom)
        ang_grid = np.arange(-.5*fov, .5*fov, ang_res)
        closest_obstacle = (r_max, 0.0)
        start_rc = self.se2_to_cell(odom[:2])
        cs_ang_grid = r_max * np.array([np.cos(ang_grid), np.sin(ang_grid)])
        end_pt_global_frame = coord_change2g(cs_ang_grid, odom[-1]) + odom[:2, np.newaxis]
        end_rc = se2_to_cell_batch(end_pt_global_frame.T, self.mapmin, self.mapres)
        for j in range(len(ang_grid)):
            ray_cells = bresenham2D(start_rc[0], start_rc[1], end_rc[0][j], end_rc[1][j])
            i = 0
            if self.map is None:
                while(i < ray_cells.shape[-1]):
                    pt = self.cell_to_se2(ray_cells[:,i])
                    if not(self.in_bound(pt)):
                        break
                    i += 1
                if i < ray_cells.shape[-1]: # break!
                    ro_min_t = np.sqrt(np.sum(np.square(pt - odom[:2])))
                    if ro_min_t < closest_obstacle[0]:
                        closest_obstacle = (ro_min_t, ang_grid[j])
            else:
                while(i < ray_cells.shape[-1]): # break!
                    if self.is_collision_ray_cell(ray_cells[:,i]):
                        break
                    i += 1
                if i < ray_cells.shape[-1]:
                    ro_min_t = np.sqrt(np.sum(np.square(self.cell_to_se2(ray_cells[:,i]) - odom[:2])))
                    if ro_min_t < closest_obstacle[0]:
                        closest_obstacle = (ro_min_t, ang_grid[j])
        if closest_obstacle[0] == r_max:
            return None
        else:
            return closest_obstacle

    def update_visit_freq_map(self, fov, r_max, odom, decay_factor=1.0, ang_res=0.05, observed=True):
        """
        Update the visit frequency map from the given odometry.
        """
        odom = np.array(odom)
        ang_grid = np.arange(-.5*fov, .5*fov, ang_res)
        closest_obstacle = (r_max, 0.0)
        start_rc = self.se2_to_cell(odom[:2])
        cs_ang_grid = r_max * np.array([np.cos(ang_grid), np.sin(ang_grid)])
        end_pt_global_frame = coord_change2g(cs_ang_grid, odom[-1]) + odom[:2, np.newaxis]
        end_rc = se2_to_cell_batch(end_pt_global_frame.T, self.mapmin, self.mapres)
        self.decay_visit_freq_map(decay_factor)
        visit_map_tmp = np.zeros(self.mapdim)
        for j in range(len(ang_grid)):
            ray_cells = bresenham2D(start_rc[0], start_rc[1], end_rc[0][j], end_rc[1][j])
            i = 0
            while(i < ray_cells.shape[-1]): # break!
                if self.is_collision_ray_cell(ray_cells[:,i]):
                    break
                if self.visit_freq_map is not None:
                    self.visit_freq_map[ray_cells[0,i], ray_cells[1,i]] = 1.0
                if self.visit_map is not None and not(observed):
                    # self.visit_map[ray_cells[0,i], ray_cells[1,i]] += 1.0
                    visit_map_tmp[ray_cells[0,i], ray_cells[1,i]] = 1.0
                i += 1
        if self.visit_map is not None and not(observed):
            self.visit_map += visit_map_tmp

    def local_map_helper(self, im_size, odom, local_mapmin, R, get_visit_freq=False):
        """
        Helper function to generate a local map from the global map.
        For the visit frequency local map, it assigns 2.0 for an obstacle cell,
        and a value in (0,1) for an empty cell with its visit frequency value.
        """
        local_map = np.zeros((im_size, im_size))
        local_visit_freq_map = np.zeros((im_size, im_size)) if get_visit_freq else None
        for r in range(im_size):
            for c in range(im_size):
                xy_local = cell_to_se2([r,c], local_mapmin, self.mapres)
                xy_global = np.matmul(R, xy_local) + odom[:2]
                cell_global = self.se2_to_cell(xy_global)
                local_map[c,r] = int(self.is_collision_ray_cell(cell_global))
                if get_visit_freq:
                    # Cells with an obstacle have 2.0. Others have visit frequency value from the global map.
                    if local_map[c,r] == 1 :
                        local_visit_freq_map[c,r] = 2.0
                    elif self.in_bound(xy_global):
                        local_visit_freq_map[c,r] += self.visit_freq_map[cell_global[0], cell_global[1]]

        local_mapmin_g = np.matmul(R, local_mapmin) + odom[:2]
        return local_map, local_mapmin_g, local_visit_freq_map

    def local_map(self, im_size, odom, get_visit_freq=False):
        """
        Parameters:
        ---------
        im_size : the number of rows/columns
        """
        R=np.array([[np.cos(odom[2] - np.pi/2), -np.sin(odom[2] - np.pi/2)],
                  [np.sin(odom[2] - np.pi/2), np.cos(odom[2] - np.pi/2)]])
        local_mapmin = np.array([-im_size/2*self.mapres[0], 0.0])
        return self.local_map_helper(im_size, odom, local_mapmin, R, get_visit_freq)

    def local_visit_map(self, im_size, odom, get_visit_freq=True):
        """
        Return a local visit frequency map around the odom. odom position should
        be the center.
        Parameters:
        ---------
        im_size : the number of rows/columns
        """
        R=np.array([[np.cos(odom[2] - np.pi/2), -np.sin(odom[2] - np.pi/2)],
                  [np.sin(odom[2] - np.pi/2), np.cos(odom[2] - np.pi/2)]])
        local_mapmin = np.array([-im_size/2*self.mapres[0], -im_size/2*self.mapres[0]])
        return self.local_map_helper(im_size, odom, local_mapmin, R, get_visit_freq)

    def local_visit_map_surroundings(self, im_size, odom):
        """
        Return local visit frequency map of the surrounding areas.
        """
        R=np.array([[np.cos(odom[2] - np.pi/2), -np.sin(odom[2] - np.pi/2)],
                  [np.sin(odom[2] - np.pi/2), np.cos(odom[2] - np.pi/2)]])
        local_visit_maps = []
        local_mapmin_g = []
        local_mapmin = [  # From left, right, front, back
            np.array([-im_size/2*3*self.mapres[0], 0.0]),
            np.array([im_size/2*self.mapres[0], 0.0]),
            np.array([-im_size/2*self.mapres[0], im_size*self.mapres[1]]),
            np.array([-im_size/2*self.mapres[0], -im_size*self.mapres[1]]),
            ]
        for i in range(4):
            _, local_mapmin_g_i, local_map_visit_i = self.local_map_helper(im_size,
                                                        odom, local_mapmin[i], R,
                                                        get_visit_freq=True)
            local_visit_maps.append(local_map_visit_i)
            local_mapmin_g.append(local_mapmin_g_i)
        return None, local_mapmin_g, np.array(local_visit_maps)

def bresenham2D(sx, sy, ex, ey):
    """
    Bresenham's ray tracing algorithms in 2D from ESE650 2017 TA resources
    Inputs:
        (sx, sy)  start point of ray
        (ex, ey)  end point of ray
    Outputs:
        Indicies for x-axis and y-axis
    """
    sx = int(round(sx))
    sy = int(round(sy))
    ex = int(round(ex))
    ey = int(round(ey))
    dx = abs(ex-sx)
    dy = abs(ey-sy)
    steep = abs(dy)>abs(dx)
    if steep:
        dx,dy = dy,dx # swap

    if dy == 0:
        q = np.zeros((dx+1,1))
    else:
        q = np.append(0,np.greater_equal(np.diff(np.mod(np.arange( np.floor(dx/2), -dy*dx+np.floor(dx/2)-1,-dy),dx)),0))

    if steep:
        if sy <= ey:
            y = np.arange(sy,ey+1)
        else:
            y = np.arange(sy,ey-1,-1)
        if sx <= ex:
            x = sx + np.cumsum(q)
        else:
            x = sx - np.cumsum(q)
    else:
        if sx <= ex:
            x = np.arange(sx,ex+1)
        else:
            x = np.arange(sx,ex-1,-1)
        if sy <= ey:
            y = sy + np.cumsum(q)
        else:
            y = sy - np.cumsum(q)
    return np.vstack((x,y)).astype(np.int16)

def coord_change2g(vec, ang):
    assert(len(vec) == 2)
    # R * v
    return np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])@vec

def se2_to_cell(pos, mapmin, mapres):
    pos = pos[:2]
    cell_idx = (pos - mapmin)/mapres - 0.5
    return round(cell_idx[0]), round(cell_idx[1])

def cell_to_se2(cell_idx, mapmin, mapres):
    return ( np.array(cell_idx) + 0.5 ) * mapres + mapmin

def se2_to_cell_batch(pos, mapmin, mapres):
    """
    Coversion for Batch input : pos = [batch_size, 2 or 3]

    OUTPUT: [batch_size,], [batch_size,]
    """
    return np.round((pos[:,0] - mapmin[0])/mapres[0] - 0.5), np.round((pos[:,1] - mapmin[1])/mapres[1] - 0.5)

def cell_to_se2_batch(cell_idx, mapmin, mapres):
    """
    Coversion for Batch input : cell_idx = [batch_size, 2]

    OUTPUT: [batch_size, 2]
    """
    return (cell_idx[:,0] + 0.5) * mapres[0] + mapmin[0], (cell_idx[:,1] + 0.5) * mapres[1] + mapmin[1]

def generate_trajectory(map_obj):
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib import pyplot as plt
    from scipy.interpolate import interp1d

    ax = plt.gca()
    fig = plt.gcf()
    implot = ax.imshow(map_obj.map, cmap='gray_r', origin='lower', extent=[map_obj.mapmin[0], map_obj.mapmax[0], map_obj.mapmin[1], map_obj.mapmax[1]])

    path_points = []
    def onclick(event):
        if event.xdata != None and event.ydata != None:
            ax.plot(event.xdata, event.ydata, 'ro')
            plt.draw()
            path_points.append([event.xdata, event.ydata])
    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    plt.show()
    print("%d points selected"%len(path_points))
    path_points = np.array(path_points)
    interpol_num = 5
    passed = False
    vel = 0.2
    T_step = 100
    while(not passed):
        ax = plt.gca()
        fig = plt.gcf()
        implot = ax.imshow(map_obj.map, cmap='gray_r', origin='lower', extent=[map_obj.mapmin[0], map_obj.mapmax[0], map_obj.mapmin[1], map_obj.mapmax[1]])

        sorted_paths = []
        newx = []
        newy = []

        t_done = False
        t = 1
        while(not t_done):
            tmp = [path_points[t-1]]
            tmp_x = []
            while((path_points[t-1,0] < path_points[t,0])):
                tmp.append(path_points[t])
                distance = np.sqrt(np.sum(np.square( path_points[t] - path_points[t-1] )))
                num = distance / (0.5 * vel)
                tmp_x.extend(np.linspace(path_points[t-1,0], path_points[t,0], num=num, endpoint=False))
                t += 1
                if t >= len(path_points):
                    t_done = True
                    break
            if len(tmp) > 1:
                tmp = np.array(tmp)
                f0 = interp1d(tmp[:,0], tmp[:,1], kind='cubic')
                newx.extend(tmp_x)
                newy.extend(f0(np.array(tmp_x)))
                sorted_paths.append(tmp)
                if t_done:
                    break

            tmp = [path_points[t-1]]
            tmp_x = []
            while((path_points[t-1,0] >= path_points[t,0])):
                tmp.append(path_points[t])
                distance = np.sqrt(np.sum(np.square( path_points[t] - path_points[t-1] )))
                num = distance / (0.5 * vel)
                tmp_x.extend(np.linspace(path_points[t-1,0], path_points[t,0], num=num, endpoint=False))
                t += 1
                if t >= len(path_points):
                    t_done = True
                    break
            tmp = np.array(tmp)
            f0 = interp1d(np.flip(tmp[:,0], -1), np.flip(tmp[:,1], -1), kind='cubic')
            newx.extend(tmp_x)
            newy.extend(f0(np.array(tmp_x)))
            sorted_paths.append(tmp)

        ax.plot(path_points[:,0], path_points[:,1],'ro')
        ax.plot(newx, newy,'b.')
        print("Total point number: %d"%len(newx))
        plt.show()
        if len(newx) > T_step:
            passed = True
        else:
            interpol_num += 1
    newx = np.array(newx)
    newy = np.array(newy)
    x_vel = (newx[1:]-newx[:-1])/0.5
    y_vel = (newy[1:]-newy[:-1])/0.5
    theta = np.arctan2(newy[1:]-newy[:-1], newx[1:]-newx[:-1])

    final = np.concatenate((newx[:-1, np.newaxis], newy[:-1, np.newaxis], theta[:,np.newaxis], x_vel[:,np.newaxis], y_vel[:,np.newaxis]), axis=1)
    np.save(open("path_sh_1.npy", "wb"), final)

def generate_map(mapname, mapdim=(8,4), mapres=0.1):
    new_map = np.zeros((int(mapdim[0]/mapres), int(mapdim[1]/mapres)), dtype=np.int8)
    """
    0 0 0 0
    0 1 0 0
    0 0 0 0
    0 0 1 0
    0 0 0 0
    0 1 0 0
    0 1 0 0
    0 0 0 0
    """
    # Boundary
    new_map[0,:] = 1.0
    new_map[:,0] = 1.0
    new_map[-1,:] = 1.0
    new_map[:,-1] = 1.0

    # Obstacles
    new_map[int(1.0/mapres):int(2.0/mapres), int(1.0/mapres):int(2.0/mapres)] = 1.0
    new_map[int(3.0/mapres):int(4.0/mapres), int(2.0/mapres):int(3.0/mapres)] = 1.0
    new_map[int(5.0/mapres):int(7.0/mapres), int(1.0/mapres):int(2.0/mapres)] = 1.0

    new_map = new_map.astype(np.int8)
    np.savetxt(mapname, new_map, fmt='%d')
