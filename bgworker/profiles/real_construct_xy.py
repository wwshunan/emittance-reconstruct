import os
from .wrdis import *
import math
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import sys
import datetime
from pathlib import Path

mass = 938.272
w = 1.5210049
gamma = 1 + w / mass
btgm = (gamma ** 2 - 1) ** 0.5

lattice_name = 'MEBT_emittace.dat'
mebt_txt = '''FREQ 162.5  
DRIFT 161.65 100 0
superpose_map 0 0 0 0 0 0
MATCH_FAM_GRAD 6 0
FIELD_MAP 90 300 0 100 {} 0 0 0 quad1
superpose_map 192 0 0 0 0 0
MATCH_FAM_GRAD 6 0
FIELD_MAP 90 300 0 100 -{} 0 0 0 quad2
superpose_map 379 0 0 0 0 0
MATCH_FAM_GRAD 6 0
FIELD_MAP 90 300 0 100 {} 0 0 0 quad1
superpose_map 629 0 0 0 0 0
;MATCH_FAM_FIELD 6 0
;SET_SYNC_PHASE 
;FIELD_MAP 7700 240 -90 100 0 0 0 0 buncher
DRIFT 190 100 0
DRIFT 77 100 0

;MATCH_FAM_GRAD 6 0
QUAD 80 0 100 0 0 0 0 0 0

DRIFT 122.5 100 0

;slit
DRIFT 375.5 100 0
end
'''

hebt_txt = '''DRIFT 742.56 30 0
superpose_map 0 0 0 0 0 0
MATCH_FAM_GRAD 5 0
FIELD_MAP 0090 600 0 31.5 -{} 0 0 0 Q_H2 
superpose_map 420 0 0 0 0 0
MATCH_FAM_GRAD 5 0
FIELD_MAP 0090 600 0 31.5 {} 0 0 0 Q_H1
DRIFT 1195.02 70 0  
end
'''

def get_mask(nparts, test_wires, refer_wires, x_index):
    w_x = np.ones(nparts, dtype=bool)

    for i, m in enumerate(refer_wires):
        step_num = refer_wires[i].shape[0]
        x_min = min(m[:, 1])
        x_max = max(m[:, 1])
        dx_min = m[:, 1][1] - m[:, 1][0]
        x_min -= dx_min / 2
        dx_max = m[:, 1][-1] - m[:, 1][-2]
        x_max += dx_max / 2
        for j, xtest in enumerate(test_wires[i][:, x_index]):
            if xtest < x_min or xtest > x_max:
                w_x[j] = False
    return w_x

def weighted_avg_and_var(axis1, axis2, weights=None):
    average1 = np.average(axis1, weights=weights)
    average2 = np.average(axis2, weights=weights)
    variance = np.average((axis1 - average1) * (axis2 -average2), weights=weights)
    return (average1, average2, variance)

def load_refer(fnames, bg_noise):
    refer_wires = []
    for k, r in enumerate(fnames):
        refer_data = np.loadtxt(r)
        #refer_data = refer_data[refer_data[:, 1] > 10]

        rm_lines = []
        for i in range(len(refer_data)):
            if refer_data[i][2] < bg_noise:
                rm_lines.append(i)
            else:
                break
        for i in reversed(range(len(refer_data))):
            if refer_data[i][2] < bg_noise:
                rm_lines.append(i)
            else:
                break
        mask = np.ones(len(refer_data), dtype=bool)
        mask[rm_lines] = False
        refer_data = refer_data[mask]

        for i in range(len(refer_data)):
            if refer_data[i][2] < bg_noise:
                refer_data[i][2] = 0

        refer_data[:, 1] = refer_data[:, 1] - np.average(refer_data[:, 1], weights=refer_data[:, 2]) 
        
        refer_data[:, 1] *= 0.1

        refer_wires.append(refer_data)
    return refer_wires

def load_test(i_test, test_wires_fname, bg_noise):
    test_init = readDis(i_test)
    test_wires = []

    for t in test_wires_fname:
        test_wires.append(exitDis(t))

    return test_init, test_wires 

def run(current, dx, dxp, dy, dyp, mass, charge, project):
    if project.startswith('HEBT'):
        txt = hebt_txt
        lattice_name = 'HEBT.dat'
    else:
        txt = mebt_txt
        lattice_name = 'MEBT.dat'
    txt_r = txt.format(*current)

    with open(lattice_name, 'w') as f:
        f.write(txt_r)

    cmd_str = './TraceWin {} dst_file1=RFQ.dst mass1={} charge1={}'.format(project, mass, charge)
    cmd_str = '{} x1={} xp1={}'.format(cmd_str, dx * 10, dxp * 1000)
    cmd_str = '{} y1={} yp1={}'.format(cmd_str, dy * 10, dyp * 1000)
    os.system(cmd_str)
    file_renamed = Path('testdata') / ('_'.join(current) + '.plt')
    if os.path.exists(file_renamed):
        os.remove(file_renamed)
    os.rename(Path('results') / 'dtl1.plt', file_renamed)

def uniformRec(xWidth, yWidth, nparts):
    xs = []
    ys = []
    for i in range(nparts):
        xs.append(np.random.random() * xWidth - xWidth / 2.)
        ys.append(np.random.random() * yWidth - yWidth / 2.)
    return np.array(xs), np.array(ys)

def halo(xs, xps, bin_num, nparts, thick_prop):
    x_hist = np.histogram(xs, bin_num)
    xs_halo = []
    xps_halo = []
    for i in range(nparts):
        while True:
            j = np.random.choice(bin_num)
            xps_bin = []
            for x, xp in zip(xs, xps):
                if x > x_hist[1][j] and x < x_hist[1][j+1]:
                    xps_bin.append(xp)
            if xps_bin:
                break
        xp_bin_min = min(xps_bin)
        xp_bin_max = max(xps_bin)
        xp_bin_width = xp_bin_max - xp_bin_min
        x = x_hist[1][j] + (x_hist[1][j+1] - x_hist[1][j]) * np.random.random() 
        if i % 2 == 0:
            xp = xp_bin_min - xp_bin_width * thick_prop * np.random.random()
        else:
            xp = xp_bin_max + xp_bin_width * thick_prop * np.random.random()
        xs_halo.append(x)
        xps_halo.append(xp)
    return xs_halo, xps_halo

def gen_halo(xs, xps, thick_prop, nparts, weights):
    bin_num = 100
    
    x_min = min(xs)
    x_max = max(xs)
    xp_min = min(xps)
    xp_max = max(xps)
    x_width = x_max - x_min
    xp_width = xp_max - xp_min
    #dx = x_width * thick_prop
    #dxp = xp_width * thick_prop
    r = xp_width / (x_width + xp_width)
    nparts_x = nparts * (1 - r)
    nparts_xp = nparts - nparts_x
    xs_halo_h, xps_halo_h = halo(xs, xps, bin_num, nparts_x, thick_prop)
    xps_halo_v, xs_halo_v = halo(xps, xs, bin_num, nparts_xp, thick_prop)
    xs_halo = xs_halo_h + xs_halo_v
    xps_halo = xps_halo_h + xps_halo_v
        
    return xs_halo, xps_halo

def hist_and_weight(nparts, test_init, test_wires, refer_wires, mask, x_index, xp_index, save_figs):
    test_init = test_init[mask]
    weights = np.zeros((test_init.shape[0], len(test_wires)))
    test_wires_mask = []
    for i, t in enumerate(test_wires):
        test_wires_mask.append(t[mask])
        #fig = plt.figure(i+30)
        #plt.scatter(t[:, x_index], t[:, xp_index], s=1, edgecolor='b')
        #plt.scatter(test_wires_mask[i][:, x_index], test_wires_mask[i][:, xp_index], s=1, edgecolor='r')
        #plt.show()
        refer_xs = refer_wires[i][:, 1]
        refer_min = min(refer_xs)
        refer_max = max(refer_xs)
        dx_min = refer_xs[1] - refer_xs[0]
        dx_max = refer_xs[-1] - refer_xs[-2]
        x_bin_boundary = [(refer_xs[j] + refer_xs[j+1]) / 2 for j in range(len(refer_xs)-1)]
        x_bin_boundary.insert(0, refer_min - dx_min / 2)
        x_bin_boundary.append(refer_max + dx_max / 2)
        x_min = min(test_wires_mask[i][:, x_index])
        x_max = max(test_wires_mask[i][:, x_index])
        mask_start = 0
        mask_stop = len(refer_xs) - 1
        for j in range(len(x_bin_boundary)-1):
            if x_min >= x_bin_boundary[j] and x_min < x_bin_boundary[j+1]:
                mask_start = j
                break
        for j in range(len(x_bin_boundary)-1):
            if x_max >= x_bin_boundary[j] and x_max < x_bin_boundary[j+1]:
                mask_stop = j
                break

        x_bin_boundary = x_bin_boundary[mask_start:mask_stop+2]
        row_mask = np.ones(len(refer_xs), dtype=bool)
        row_mask[:mask_start] = False
        row_mask[mask_stop+1:] = False
        refer_cutoff = refer_wires[i][row_mask]
        x_hist_refer = refer_cutoff[:, 2]
        x_hist_refer = x_hist_refer / sum(x_hist_refer)
        hist_test = np.histogram(test_wires_mask[i][:, x_index], bins=x_bin_boundary, density=True)
        #h = np.histogram(test_wires_mask[i][:, 0], bins=100)
        #h_global = h[0] / float(sum(h[0]))
        h_global = refer_wires[i][:, 2] / sum(refer_wires[i][:, 2])
        x_hist_test = hist_test[0] / float(sum(hist_test[0]))
        with open('diff.txt', 'a') as f:
            f.write('%s\t' % np.average((x_hist_refer - x_hist_test) ** 2) ** 0.5)

        #plt.figure(i)
        #plt.scatter(test_wires_mask[i][:, 0], test_wires_mask[i][:, 1], s=1, edgecolor='b')
        #plt.scatter(refer_wires[i][:, 0], refer_wires[i][:, 1], s=1, edgecolor='r')
        #k = 0
        #while True:
        #    k += 1
        #    if os.path.exists('figs/slit_%s_%s.png' % (i, k)):
        #        continue
        #    plt.savefig('figs/slit_%s_%s.png' % (i, k))
        #    break
        #plt.close()

        if save_figs:
            plt.figure(i)
            plt.plot(refer_cutoff[:, 1], x_hist_refer, 'g')
            plt.plot(refer_cutoff[:, 1], hist_test[0] / float(sum(hist_test[0])), 'r')
            plt.plot(refer_wires[i][:, 1],h_global, 'b')

            k = 0
            while True:
                k += 1
                if os.path.exists(Path('figs') / 'slit{}_{}_{}.png'.format(x_index, i, k)):
                    continue
                plt.savefig(Path('figs') / 'slit{}_{}_{}.png'.format(x_index, i, k))
                break
            plt.close()

        for j, x in enumerate(test_wires_mask[i][:, x_index]):
            for k in range(len(x_bin_boundary)):
                if x >= x_bin_boundary[k] and x < x_bin_boundary[k+1]:
                    grid = k
                    break
            w = x_hist_refer[grid] / hist_test[0][grid]
            weights[j, i] = w 

    return weights, test_init

def part_regen(xs, xps, weights, xgrid, nparts_t, halo):
    _, _, var_x = weighted_avg_and_var(xs, xs, weights)
    _, _, var_xp = weighted_avg_and_var(xps, xps, weights)
    avg_x, avg_xp, covar = weighted_avg_and_var(xs, xps, weights)
    rot_angle = np.arctan(2 * covar / (var_x - var_xp)) / 2

    x_old = xs - avg_x
    xp_old = xps - avg_xp
    x_new = np.cos(rot_angle) * x_old + np.sin(rot_angle) * xp_old
    xp_new = -np.sin(rot_angle) * x_old + np.cos(rot_angle) * xp_old

    x_min = min(x_new)
    x_max = max(x_new)
    xp_min = min(xp_new)
    xp_max = max(xp_new)
    #r = (x_max - x_min) / (xp_max - xp_min)
    #xpgrid = int(xgrid / r)
    xpgrid = 80
    dx = (x_max - x_min) / xgrid
    dxp = (xp_max - xp_min) / xpgrid
    x_min = x_min - dx / 2
    x_max = x_max + dx / 2
    xp_min = xp_min - dxp / 2
    xp_max = xp_max + dxp / 2
    xgrid += 1
    xpgrid += 1

    particle_density = np.zeros((xgrid, xpgrid))
    x_cell = np.floor((x_new - x_min) / dx).astype(np.int)
    xp_cell = np.floor((xp_new - xp_min) / dxp).astype(np.int)
    coords = np.c_[x_cell, xp_cell]
    cell_indices, indxs = np.unique(coords, axis=0, return_inverse=True)
    weights = np.bincount(indxs, weights)
    particle_density[cell_indices[:, 0], cell_indices[:, 1]] = weights

    rst_phase_space = np.zeros((nparts_t, 2))
    dens_sum = np.sum(particle_density)
    particle_density = particle_density / dens_sum 

    halo_nparts = 0
    thick_prop = 0.1
    if halo:
        nparts = 5000
        xs_halo, xps_halo = gen_halo(x_new, xp_new, thick_prop, nparts, weights)
        halo_nparts += len(xs_halo)
    else:
        nparts = 0
        xs_halo = []
        xps_halo = []
            
    particle_density = particle_density.reshape(xgrid * xpgrid)
    num_part = nparts_t - halo_nparts
    x_rand = np.zeros(num_part)
    y_rand = np.zeros(num_part)
    cell_indeces = np.random.choice(xgrid * xpgrid,
                                    num_part, p=particle_density)
    x_rand = (np.random.random(num_part) + 
            cell_indeces // xpgrid) * dx + x_min
    xp_rand = (np.random.random(num_part) +
            cell_indeces % xpgrid) * dxp + xp_min
    
    x_rand = np.concatenate((x_rand, xs_halo))
    xp_rand = np.concatenate((xp_rand, xps_halo))
    rst_phase_space = np.c_[x_rand, xp_rand]
        
    x_new = np.cos(rot_angle) * rst_phase_space[:, 0] - np.sin(rot_angle) * rst_phase_space[:, 1]
    xp_new = np.sin(rot_angle) * rst_phase_space[:, 0] + np.cos(rot_angle) * rst_phase_space[:, 1]
    x_new = x_new + avg_x
    xp_new = xp_new + avg_xp
    return x_new, xp_new

def max_and_min(data):
    dxs = []
    x_mins = []
    x_maxs = []
    for i, m in enumerate(data):
        step_num = len(m)
        x_min = min(m[:, 1])
        x_max = max(m[:, 1])
        dx = (x_max - x_min) / (step_num - 1)
        x_min = x_min - dx / 2
        x_max = x_max + dx / 2
        dxs.append(dx)
        x_mins.append(x_min)
        x_maxs.append(x_max)
    return x_mins, x_maxs, dxs

def find_centers(iter_depth, profile_num, nparts, I, mass, charge, distribution, project, bg_noise, basedir):
    #os.chdir('bgworker')
    basedir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(basedir)
    xWidth = 16
    xpWidth = 14e-2
    #nparts = 98180
    #I = 0.9818
    xgrid = 80
    current = np.loadtxt(Path('..') / '..' / 'final-results' / 'lattice.txt', dtype=int)
    xs, xps = uniformRec(xWidth, xpWidth, nparts)
    partran_dist = readDis(distribution) 
    partran_dist[:, 0] = xs
    partran_dist[:, 1] = xps
    partran_dist[:, 2] = xs
    partran_dist[:, 3] = xps
    generate_new_dis(partran_dist, I, mass * 1e-6)
    i_test = 'RFQ.dst'

    process_profile(current, iter_depth, bg_noise, partran_dist, i_test, nparts, I, mass, charge, project, xgrid, basedir)

def new_dis_one_plane(xgrid, nparts, test_init, it, test_wires, refer_wires, which='x'):
    if which == 'x':
        x_index = 0
        xp_index = 1
    else:
        x_index = 2
        xp_index = 3

    mask_x = get_mask(nparts, test_wires, refer_wires, x_index)
    save_figs = (it % 5 == 0)
        
    if it == 0:
        t_mask = test_init[mask_x]
        if which == 'x':
            x_min = min(t_mask[:, 0])
            x_max = max(t_mask[:, 0])
            xp_min = min(t_mask[:, 1])
            xp_max = max(t_mask[:, 1])
        else:
            x_min = min(t_mask[:, 2])
            x_max = max(t_mask[:, 2])
            xp_min = min(t_mask[:, 3])
            xp_max = max(t_mask[:, 3])
        x_width = x_max - x_min
        xp_width = xp_max - xp_min
        x, xp = uniformRec(x_width, xp_width, nparts)
    else:
        weights_x, test_init_x = hist_and_weight(nparts, test_init, test_wires, refer_wires, mask_x, x_index, xp_index, save_figs)
        xs, xps = test_init_x[:, x_index], test_init_x[:, xp_index]
        weights_x =  np.mean(weights_x, axis=1) 
        #weights_x = (weights_x_avg +  weights_x_avg * (np.random.rand() * 2 - 1)) if (it > 5 and it < 25) else weights_x_avg
        if it < 1:
            x, xp = part_regen(xs, xps, weights_x, xgrid, nparts, halo=True)
        else:
            x, xp = part_regen(xs, xps, weights_x, xgrid, nparts, halo=False)
    return x, xp

def process_profile(current, iter_depth, bg_noise, partran_dist, i_test, nparts, I, mass, charge, project, xgrid, basedir):
    current_str = current.astype('str')
    raw_names = ['_'.join(c.tolist()) for c in current_str]
    x_refer_wires_fname = [Path('..') / '..' / 'final-results' / ('x_' + '_'.join(c.tolist())) for c in current_str]
    y_refer_wires_fname = [Path('..') / '..' / 'final-results' / ('y_' + '_'.join(c.tolist())) for c in current_str]
    x_refer_wires = load_refer(x_refer_wires_fname, bg_noise)
    y_refer_wires = load_refer(y_refer_wires_fname, bg_noise)

    dx = 0
    dxp = 0
    dy = 0
    dyp = 0
    for i in range(iter_depth):
        test_wires_fname = []
        for c in current_str:
            run(c, dx, dxp, dy, dyp, mass, charge, project)
            test_wires_fname.append(Path('testdata') / ('_'.join(c) + '.plt'))
            
        test_init, test_wires = load_test(i_test, test_wires_fname, bg_noise) 
        if i == iter_depth - 1:
            offsets = []
            for j in range(len(test_wires)):
                offsets.append(np.average(test_wires[j][:, 0]))
            return offsets

        x, xp = new_dis_one_plane(xgrid, nparts, test_init, i, test_wires, x_refer_wires, which='x')
        y, yp = new_dis_one_plane(xgrid, nparts, test_init, i, test_wires, y_refer_wires, which='y')

        partran_dist[:, 0] = x
        partran_dist[:, 1] = xp
        partran_dist[:, 2] = y
        partran_dist[:, 3] = yp
        dx = np.average(x)
        dxp = np.average(xp)
        dy = np.average(y)
        dyp = np.average(yp)

        with open('twissparameters.txt', 'a') as f:
            sigma_x = weighted_avg_and_var(x * 10, x * 10)[2]
            sigma_xp = weighted_avg_and_var(xp * 1000, xp * 1000)[2]
            sigma_x_xp = weighted_avg_and_var(x * 10, xp * 1000)[2]
            emit_x = (sigma_x * sigma_xp -sigma_x_xp ** 2) ** 0.5
            beta_x = sigma_x / emit_x
            gamma_x = sigma_xp / emit_x
            alpha_x = -sigma_x_xp / emit_x
            emit_x = btgm * emit_x
            

            sigma_y = weighted_avg_and_var(y * 10, y * 10)[2]
            sigma_yp = weighted_avg_and_var(yp * 1000, yp * 1000)[2]
            sigma_y_yp = weighted_avg_and_var(y * 10, yp * 1000)[2]
            emit_y = (sigma_y * sigma_yp -sigma_y_yp ** 2) ** 0.5
            beta_y = sigma_y / emit_y
            gamma_y = sigma_yp / emit_y
            alpha_y= -sigma_y_yp / emit_y
            emit_y = btgm * emit_y
            f.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(
                emit_x, beta_x, gamma_x, alpha_x, emit_y, beta_y, gamma_y, alpha_y))

        generate_new_dis(partran_dist, I, mass * 1e-6)
        heatmap, xedges, yedges = np.histogram2d(x, xp, bins=300)
        heatmap_mask = np.ma.masked_where(heatmap==0, heatmap)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        fig = plt.figure(20)
        ax = fig.gca()
        ax.set_facecolor('white')
        ax.imshow(heatmap_mask.T, extent=extent, origin='lower', aspect='auto', interpolation='bicubic')

        k = 0
        while True:
            k += 1
            if os.path.exists(Path('figs')/ 'initx_{}.png'.format(k)):
                continue
            fig.savefig(Path('figs') / 'initx_{}.png'.format(k))
            break
        plt.close()
        heatmap, xedges, yedges = np.histogram2d(y, yp, bins=300)
        heatmap_mask = np.ma.masked_where(heatmap==0, heatmap)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        fig = plt.figure(21)
        ax = fig.gca()
        ax.set_facecolor('white')
        ax.imshow(heatmap_mask.T, extent=extent, origin='lower', aspect='auto', interpolation='bicubic')

        k = 0
        while True:
            k += 1
            if os.path.exists(Path('figs') / 'inity_{}.png'.format(k)):
                continue
            fig.savefig(Path('figs') / 'inity_{}.png'.format(k))
            break
        plt.close()
    

if __name__ == '__main__':
    prefix = 'final-results'
    find_centers(50, 10, 98180, 0.9818, 0, 'final-results')
