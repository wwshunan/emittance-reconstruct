from flask import Flask, request, render_template, redirect, url_for, session, jsonify
from bgworker.profiles.real_construct_xy import find_centers
from bgworker.direct.reconstruct import DirectReconstruction
from scipy.signal import savgol_filter
from scipy.interpolate import griddata
from flask_caching import Cache
from werkzeug import secure_filename
from flask_bootstrap import Bootstrap
from flask_moment import Moment
from celery import Celery
import random
import os, re, base64, io, shutil, time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from sklearn.neighbors import KernelDensity
import zipfile, io
from flask import send_file
from pathlib import Path

UPLOAD_FOLDER = 'raw-data'
app = Flask(__name__)
cache = Cache(app, config={'CACHE_TYPE': 'simple'})
app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
celery = Celery(app.name, backend="redis", broker=app.config['CELERY_BROKER_URL'])
bootstrap = Bootstrap(app)
moment = Moment(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['BOOTSTRAP_SERVE_LOCAL'] = True
basedir = os.path.dirname(os.path.abspath(__file__))
app.config['SECRET_KEY'] = "I don't want to say!"

sections = {
    'mebt': {
        'PROTON': {
            'mass': 938.27203,
            'charge': 1,
            'distribution': 'RFQ_P_base.dst',
            'project': 'MEBT.ini'
        },
        'HE3PLUS': {
            'mass': 2814.816,
            'charge': 2,
            'distribution': 'RFQ_3He_base.dst',
            'project': '',
        },
        'HE4PLUS': {
            'mass': 3753.07,
            'charge': 2,
            'distribution': 'RFQ_4He_base.dst',
            'project': 'MEBT_4He.ini'
        },
    },
    'hebt': {
        'PROTON': {
            'mass': 938.27203,
            'charge': 1,
            'distribution': 'HEBT_P_base.dst',
            'project': 'HEBT.ini'
        },
        'HE3PLUS': {
            'mass': 2814.816,
            'charge': 2,
            'distribution': 'HEBT_3He_base.dst',
            'project': '',
        },
        'HE4PLUS': {
            'mass': 3753.07,
            'charge': 2,
            'distribution': 'HEBT_4He_base.dst',
            'project': 'HEBT_4He.ini'
        },
    },
}

in_hebt = True

def monotune(data, mono_points):
    filter_data = savgol_filter(data[:, 2], 3, 2)
    grad = np.gradient(filter_data)
    for i in range(len(grad)):
        if (grad[i:i + mono_points] > 0).all():
            low_limit = i
            break
    for i in reversed(range(len(grad))):
        if (grad[i - mono_points:i] < 0).all():
            high_limit = i
            break
    data = data[low_limit:high_limit + 1]
    data[:, 2] = data[:, 2] - min(data[:, 2])
    return data
	
def gauss_fit(data_in):
    top = (data_in.max(axis=0))[2]*0.1
    data = data_in[data_in[:, 2]<top]
    noise = data[:, 2].reshape(-1, 1)

    #核密度估计
    band=np.max(noise)-np.min(noise)
    kde=KernelDensity(kernel='gaussian',bandwidth=0.001).fit(noise)
    X_plot = np.linspace(np.min(noise)-0.1*band, np.max(noise)+0.1*band, 1000)[:, np.newaxis]
    log_dens = np.reshape(kde.score_samples(X_plot),(-1,1))

    #本底噪声平均值
    a=np.argmax(log_dens)
    mean=X_plot[a]
    minnoise=np.min(noise)
    maxnoise=mean*2-minnoise

    #开始处理
    if session['kind'] == 'profile':
        #data=data0[data0['beam']>maxnoise[0]]
        max_value_index = np.argmax(data_in[:, 2])
        #valid_signal_idx = data_in[:, 2] > maxnoise[0]
        #left_boundary = np.nonzero(valid_signal_idx)[0].min()
        #right_boundary = np.nonzero(valid_signal_idx)[0].max() + 1
        left_noises = np.nonzero(data_in[:max_value_index, 2] < maxnoise[0])[0]
        left_boundary = left_noises.max() if left_noises.size != 0 else 0
        right_noises = np.nonzero(data_in[max_value_index:, 2] < maxnoise[0])[0]
        right_boundary = right_noises.min() + max_value_index if right_noises.size != 0 else -1
        data_in -= mean
        data = data_in[left_boundary:right_boundary]
        #data_in[~valid_signal_idx] = 0
    else:
        normal_signal_idx = data_in[:, 2] > maxnoise
        data = data_in[normal_signal_idx] - mean
    return data
	
def three_rms(data):
    weights_positive = np.abs(data[:, 2]) 
    mean = np.average(data[:, 1], weights=weights_positive)
    sigma = np.average((data[:, 1]-mean)**2, weights=weights_positive) ** 0.5
    left_boundary = max(data[0, 1], mean - 3*sigma)
    right_boundary = min(data[-1, 1], mean + 3*sigma)
    data_3rms = (data[:, 1] > left_boundary) & (data[:, 1] < right_boundary)
    data = data[data_3rms]	
    data = data - data.min()
    return data
	
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/twiss-parameters', methods=['POST'])
def twiss_params():
    distance = float(request.form['distance'])
    gamma = float(request.form['gamma'])
    btgm = np.sqrt(gamma**2 - 1)
    fnames = cache.cache.get('files')
    for f in fnames:
        data = np.loadtxt('final-results/{}'.format(f))
        x, xp, density = data[:, 0], data[:, 1], data[:, 2]
        x_avg  = np.average(x, weights=density)
        x = x - x_avg
        xp_avg = np.average(xp, weights=density)
        xp = xp - xp_avg
        xp_mrad = (xp - x) / distance
        x_2 = np.average(x**2, weights=density)
        xp_2 = np.average(xp_mrad**2, weights=density)
        x_xp = np.average(x*xp_mrad, weights=density)
        rms_emit = np.sqrt(x_2 * xp_2 - x_xp**2)
        alpha = -x_xp / rms_emit
        beta = x_2 / rms_emit
        norm_rms_emit = btgm * rms_emit
    return r'<p>$\alpha: {0:.3f}$</p> <p>$\beta: {1:.3f}$</p><p>$\tilde\epsilon: {2:.3f}$</p>'.format(
        alpha, beta, norm_rms_emit)

@app.route('/direct-emittance/')
def direct_emittance():
    disable = session.get('disable_rm_btn', True)
    session['disable_rm_btn'] = True
    session['kind'] = 'direct'
    if_exist = False
    graph = cache.cache.get('graph')
    cache.cache.set('graph', None)
    if graph is not None:
        if_exist = True
    target_template = 'direct-emittance.html'

    minMaxInputShow = False
    minInput = 0
    maxInput = 100
    step = (maxInput - minInput) / 100
    decimals = 2

    fnames = cache.cache.get('files')
    if fnames:
        for f in fnames:
            data = np.loadtxt('raw-data/{}'.format(f), skiprows=2)

            minMaxInputShow = False
            minInput = data[:, 2].min()
            maxInput = data[:, 2].max()
            step = (maxInput - minInput) / 100
            decimals = 4

        cache.cache.set('initData', data, 0)

    return render_template(target_template, graph=graph, if_exist=if_exist,
                           disable_rm_btn=disable, minMaxInputShow=minMaxInputShow,
                           minInput=minInput, maxInput=maxInput, step=step, decimals=decimals)

@app.route('/profiles/')
def profiles():
    disable = session.get('disable_rm_btn', True)
    session['disable_rm_btn'] = True
    session['kind'] = 'profile'
    if_exist = False
    graph = cache.cache.get('graph')
    cache.cache.set('graph', None)
    if graph is not None:
        if_exist = True
    target_template = 'profile.html'
    return render_template(target_template, graph=graph, if_exist=if_exist, disable_rm_btn=disable)

@app.route('/noise-trunc', methods=['POST'])
def noise_trunc():
    noiseLimit = float(request.form['noiseLimit'])
    print(noiseLimit)
    fnames = cache.cache.get('files')
    for f in fnames:
        data = cache.cache.get('initData')
        reserve_idx = data[:, 2] > noiseLimit
        data = data[reserve_idx]
        np.savetxt('final-results/%s' % f, data)

    direct_emit_final = {}
    data_type = session['kind']
    skip_rows = 0
    graph_url = graph_display('final-results', direct_emit_final, data_type, skip_rows)

    return 'data:image/png;base64,{}'.format(graph_url)

@app.route('/rm-isolate-noise', methods=['POST'])
def rm_isolate_noise():
    direction = request.form['direction']
    x1 = float(request.form['x1'])
    y1 = float(request.form['y1'])
    x2 = float(request.form['x2'])
    y2 = float(request.form['y2'])

    k = (y2 - y1) / (x2 - x1)
    intercept = k * x1 - y1
    fnames = cache.cache.get('files')
    for f in fnames:
        data = np.loadtxt('final-results/{}'.format(f))
        if direction == 'above':
            reserve_idx = (data[:, 1] - data[:, 0] * k + intercept) < 0
            print(reserve_idx)
        else:
            reserve_idx = (data[:, 1] - data[:, 0] * k + intercept) >= 0
            print(reserve_idx)
        data = data[reserve_idx]
        np.savetxt('final-results/%s' % f, data)

    cache.cache.set('initData', data, 0)
    direct_emit_final = {}
    data_type = session['kind']
    skip_rows = 0
    graph_url = graph_display('final-results', direct_emit_final, data_type, skip_rows)

    return 'data:image/png;base64,{}'.format(graph_url)

@app.route('/rm-direct-emit-noise', methods=['POST'])
def rm_direct_emit_noise():
    if os.path.exists('final-results'):
        shutil.rmtree('final-results')
    os.mkdir('final-results')

    fnames = cache.cache.get('files')
    for f in fnames:
        data = np.loadtxt('raw-data/{}'.format(f))
        data = gauss_fit(data)
        np.savetxt('final-results/%s' % f, data)

    cache.cache.set('initData', data, 0)
    direct_emit_final = {}
    data_type = session['kind']
    skip_rows = 0
    graph_url = graph_display('final-results', direct_emit_final, data_type, skip_rows)

    return 'data:image/png;base64,{}'.format(graph_url)

@app.route('/rm-profile-noise', methods=['POST'])
def rm_profile_noise():
    mono_points = int(request.form['mono_points'])
    rm_noise_method = request.form['rm_noise_method']
	
    fnames = cache.cache.get('files')
	
    if os.path.exists('final-results'):
        shutil.rmtree('final-results')
    os.mkdir('final-results')
    lattice = open('final-results/{}'.format('lattice.txt'), 'w')

    for f in fnames:
        data = np.loadtxt('average-profiles/{}'.format(f))
        if rm_noise_method == '0':
            data = gauss_fit(data)
        elif rm_noise_method == '1':
            data = three_rms(data)
        else:
            data = monotune(data, mono_points)
        np.savetxt('final-results/%s' % f, data)
        quad_current = f.split('_')[1:]
        
        if f.startswith('x'):
            lattice.write(' '.join(quad_current))
            lattice.write('\n')
    lattice.close()

    profile_final = {}
    data_type = session['kind']
    skip_rows = 0
    graph_url = graph_display('final-results', profile_final, data_type, skip_rows)

    return 'data:image/png;base64,{}'.format(graph_url)

@app.route('/medianize')
def medianzie():
    fnames = cache.cache.get('files')
    if os.path.exists('average-profiles'):
        shutil.rmtree('average-profiles')
    os.mkdir('average-profiles')

    for f in fnames:
        profiles = ['raw-data/{}'.format(fname) for fname in fnames[f]]
        datas = []
        for name in profiles:
            data = np.loadtxt(name, skiprows=2)
            datas.append(data)
        datas = np.array(datas)
        data = np.median(datas, axis=0)
        data = data[np.argsort(data[:, 0])]
        np.savetxt('average-profiles/{}'.format(f), np.c_[np.zeros(data.shape[0]), data])

    profile_averaged = {}
    data_type = session['kind']
    skip_rows = 0
    graph_url = graph_display('average-profiles', profile_averaged, data_type, skip_rows)
    return 'data:image/png;base64,{}'.format(graph_url)

@app.route('/download-distribute')
def downloadDistrubute():
    filename = os.path.join('bgworker', 'profiles', 'RFQ.dst')
    return send_file(filename, attachment_filename='RFQ.dst', as_attachment=True)

@app.route('/download/')
def download_file():
    zipf = zipfile.ZipFile('profiles.zip', 'w', zipfile.ZIP_DEFLATED)
    for root, dirs, files in os.walk('final-results'):
        for file in files:
            zipf.write('final-results/' + file)
    zipf.close()
    return send_file('profiles.zip',
                     mimetype='zip',
                     attachment_filename='profiles.zip',
                     as_attachment=True)

@app.route('/upload/', methods = ['GET','POST'])
def upload_file():
    if request.method =='POST':
        data_type = session['kind']
        files = request.files.getlist('file[]',None)
        if files:
            in_hebt = True
            shutil.rmtree('raw-data')
            os.mkdir('raw-data')
            for file in files:
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            graph_url = process_files(data_type)
            cache.cache.set('graph', graph_url)
            session['disable_rm_btn'] = False
            if data_type == 'profile':
                target_url = 'profiles'
            else:
                target_url = 'direct_emittance'
            return redirect(url_for(target_url))
    return render_template('upload.html')

@app.route('/direct-reconstruct', methods=['POST'])
def directReconstruct():
    partType = request.form['partType']
    sec = request.form['section']
    current = float(request.form['current'])
    partNum = int(request.form['partNum'])
    gamma = float(request.form['gamma'])
    emitx = float(request.form['emitx'])
    alpx = float(request.form['alpx'])
    betx = float(request.form['betx'])
    emity = float(request.form['emity'])
    alpy = float(request.form['alpy'])
    bety = float(request.form['bety'])
    q1 = float(request.form['q1'])
    q2 = float(request.form['q2'])
    q3 = float(request.form['q3'])
    emitz = 0.2048176
    alpz = -2.9512
    betz = 5.8157454
    
    section = sections[sec]
    mass = section[partType]['mass']
    charge = section[partType]['charge']
    energy = (gamma - 1) * mass
    args = (current, mass, charge, partNum, energy, emitx, alpx, betx,
            emity, alpy, bety, emitz, alpz, betz, q1, q2, q3)

    task = direct_reconstruct.apply_async(args=[args])
    return jsonify({}), 202, {'Location': url_for('directTaskStatus',
                                                  task_id=task.id)}

@celery.task(bind=True)
def direct_reconstruct(self, args):
    """Background task that runs a long function with progress reports."""
    currentDir = Path(basedir)
    os.chdir( currentDir / 'bgworker/direct')
    rst = DirectReconstruction(*args)
    rst.backTrack()
    result = np.loadtxt('results/partran1.out', skiprows=10)
    sigmaX, sigmaY, covX, covY, emitX, emitY = result[-1, [9, 10, 12, 13, 15, 16]]
    mass, energy = args[1], args[4]
    btgm = np.sqrt(np.square(energy/mass+1) - 1)
    alpx = - covX * btgm / emitX
    betx = np.square(sigmaX) * btgm / emitX
    alpy = - covY * btgm / emitY
    bety = np.square(sigmaY) * btgm / emitY
    xContent = r'<p>$\tilde\epsilon_x: {2:.3f}$</p><p>$\alpha_x: {0:.3f}$</p> <p>$\beta_x: {1:.3f}$</p>'.format(
        alpx, betx, emitX)
    yContent = r'<p>$\tilde\epsilon_y: {2:.3f}$</p><p>$\alpha_y: {0:.3f}$</p> <p>$\beta_y: {1:.3f}$</p>'.format(
        alpy, bety, emitY)
    responseContent = xContent + yContent
    return {'status': 'Task completed!', 'result': responseContent}

@app.route('/status/<task_id>')
def directTaskStatus(task_id):
    task = direct_reconstruct.AsyncResult(task_id)
    if task.state == 'PENDING':
        # job did not start yet
        response = {
            'state': task.state,
            'current': 0,
            'total': 1,
            'status': 'Pending...'
        }
    elif task.state != 'FAILURE':
        response = {
            'state': task.state,
            'current': task.info.get('current', 0),
            'total': task.info.get('total', 1),
            'status': task.info.get('status', '')
        }
        if 'result' in task.info:
            response['result'] = task.info['result']
    else:
        # something went wrong in the background job
        response = {
            'state': task.state,
            'current': 1,
            'total': 1,
            'status': str(task.info),  # this is the exception raised
        }
    return jsonify(response)

@app.route('/longtask', methods=['POST'])
def longtask():
    fnames = cache.cache.get('files')
    current = float(request.form['current'])
    particleType = request.form['particleType']
    sec = request.form['section']
    section = sections[sec]
    mass = section[particleType]['mass'] * 1e6
    charge = section[particleType]['charge']
    distribution = section[particleType]['distribution']
    project = section[particleType]['project']
    file_count = len(fnames)
    task = long_task.apply_async(args=[file_count, current, mass, charge, distribution, project])
    return jsonify({}), 202, {'Location': url_for('taskstatus', task_id=task.id)}

@celery.task(bind=True)
def long_task(self, profile_num, current, mass, charge, distribution, project):
    """Background task that runs a long function with progress reports."""
    iter_depth = 50
    nparts = 100000
    I = current
    bg_noise = 0
    final_path = os.path.join(basedir, 'final-results')
    find_centers(iter_depth, profile_num, nparts, I, mass, charge, distribution, project, bg_noise, final_path)

    return {'status': 'Task completed!', 'result': 42}

@app.route('/status/<task_id>')
def taskstatus(task_id):
    task = long_task.AsyncResult(task_id)
    if task.state == 'PENDING':
        # job did not start yet
        response = {
            'state': task.state,
            'current': 0,
            'total': 1,
            'status': 'Pending...'
        }
    elif task.state != 'FAILURE':
        response = {
            'state': task.state,
            'current': task.info.get('current', 0),
            'total': task.info.get('total', 1),
            'status': task.info.get('status', '')
        }
        if 'result' in task.info:
            response['result'] = task.info['result']
    else:
        # something went wrong in the background job
        response = {
            'state': task.state,
            'current': 1,
            'total': 1,
            'status': str(task.info),  # this is the exception raised
        }
    return jsonify(response)

def process_files(data_type):
    filenames = {}
    upload_path = app.config['UPLOAD_FOLDER']
    skip_rows = 2
    graph_url = graph_display(upload_path, filenames, data_type, skip_rows)
    cache.cache.set('files', filenames, 0)
    return 'data:image/png;base64,{}'.format(graph_url)

def plot_2d_density(data, ax):
    x, y, z = data[:, 0], data[:, 1], data[:, 2]  # why bother with split & ravel?
    N_lvls = 256  # high amount to mimic imshow behavior
    ax.plot(x, y, 'ro', ms='0.5')
    ax.tricontourf(x, y, z, levels=N_lvls, cmap="jet")
    #ax.set_aspect("equal")
    xticks = np.linspace(x.min(), x.max(), 10)
    yticks = np.linspace(y.min(), y.max(), 10)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.xaxis.set_tick_params(rotation=45)

def graph_display(path, filenames, data_type, skip_rows):
    global in_hebt
    for fname in os.listdir(path):
        if data_type == 'profile':
            match_object = re.match('[xy](_\d*(\.\d+)?){2,3}', fname)
            if match_object is not None:
                fname_prefix = match_object.group()
                if fname_prefix != fname and in_hebt:
                    in_hebt = False  
                if in_hebt:
                    fname_prefix = '_'.join(fname_prefix.split('_')[:3]) 
                if fname_prefix not in filenames:
                    filenames[fname_prefix] = []
                filenames[fname_prefix].append(fname)
        else:
            filenames[fname] = []
            filenames[fname].append(fname)
    file_num = len(filenames)
    col_num = 8
    row_num = int((file_num + col_num - 1) / col_num)
    img = io.BytesIO()
    plt.switch_backend('SVG')
    subplot_size = 3
    fig = plt.figure(figsize=(col_num*subplot_size, row_num*subplot_size))
    for i, fname_prefix in enumerate(filenames):
        ax = fig.add_subplot(row_num, col_num, i+1)
        if len(filenames[fname_prefix]) == 1:
            data = np.loadtxt(os.path.join(path, filenames[fname_prefix][0]), skiprows=skip_rows)
            if data_type == 'profile':
                ax.plot(data[:, -2], data[:, -1])
            else:
                plot_2d_density(data, ax)
        else:
            for file_one_wire in filenames[fname_prefix]:
                data = np.loadtxt(os.path.join(path, file_one_wire), skiprows=skip_rows)
                ax.plot(data[:, 0], data[:, 1])
        ax.set_title(fname_prefix)
    plt.tight_layout()
    plt.savefig(img, format=('png'))
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    return graph_url

if __name__ == '__main__':
    app.run(debug = True, host='0.0.0.0')
