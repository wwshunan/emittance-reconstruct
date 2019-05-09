from flask import Flask, request, render_template, redirect, url_for, session, jsonify
from bgworker.real_construct_xy import find_centers
from scipy.signal import savgol_filter
from flask_caching import Cache
from werkzeug import secure_filename
from flask_bootstrap import Bootstrap
from flask_moment import Moment
from celery import Celery
import random
import os, re, base64, io, shutil, time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KernelDensity

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
    #data=data0[data0['beam']>maxnoise[0]]
    max_value_index = np.argmax(data_in[:, 2])
    left_noises = np.nonzero(data_in[:max_value_index, 2] < maxnoise[0])[0]
    left_boundary = left_noises.max() if left_noises.size != 0 else 0
    right_noises = np.nonzero(data_in[max_value_index:, 2] < maxnoise[0])[0]
    right_boundary = right_noises.min() + max_value_index if right_noises.size != 0 else -1
    data = data_in[left_boundary:right_boundary]
    data[:, 2] = data[:, 2] - mean
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
    if_exist = False
    graph = cache.cache.get('graph')
    cache.cache.set('graph', None)
    if graph is not None:
        if_exist = True
    return render_template('index.html', graph=graph, if_exist=if_exist)

@app.route('/rm-noise', methods=['POST'])
def rm_noise():
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
        q1, q2, q3 = f.split('_')[1:]
        if f.startswith('x'):
            lattice.write('{0} {1} {2}\n'.format(q1, q2, q3))
    lattice.close()

    profile_final = {}
    graph_url = graph_display('final-results', profile_final)

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
    graph_url = graph_display('average-profiles', profile_averaged)
    return 'data:image/png;base64,{}'.format(graph_url)

@app.route('/upload/',methods = ['GET','POST'])
def upload_file():
    if request.method =='POST':
        files = request.files.getlist('file[]',None)
        if files:
            shutil.rmtree('raw-data')
            os.mkdir('raw-data')
            for file in files:
                filename = secure_filename(file.filename)
                print(filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            graph_url = process_files()
            cache.cache.set('graph', graph_url)
            return redirect(url_for('index'))
    return render_template('upload.html')

@app.route('/longtask', methods=['POST'])
def longtask():
    fnames = cache.cache.get('files')
    file_count = len(fnames)
    task = long_task.apply_async(args=[file_count])
    return jsonify({}), 202, {'Location': url_for('taskstatus',
                                                  task_id=task.id)}

@celery.task(bind=True)
def long_task(self, profile_num):
    """Background task that runs a long function with progress reports."""
    iter_depth = 50
    nparts = 100000
    I = 1
    bg_noise = 0
    final_path = os.path.join(basedir, 'final-results')
    find_centers(iter_depth, profile_num, nparts, I, bg_noise, final_path)

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

def process_files():
    filenames = {}
    upload_path = app.config['UPLOAD_FOLDER']
    graph_url = graph_display(upload_path, filenames)
    cache.cache.set('files', filenames)
    return 'data:image/png;base64,{}'.format(graph_url)

def graph_display(path, filenames):
    for fname in os.listdir(path):
        match_object = re.match('[xy](_\d+){3}', fname)
        if match_object is not None:
            fname_prefix = match_object.group()
            if fname_prefix not in filenames:
                filenames[fname_prefix] = []
            filenames[fname_prefix].append(fname)
            print(fname)
    file_num = len(filenames)
    col_num = 4
    row_num = int((file_num + col_num - 1) / col_num)
    img = io.BytesIO()
    plt.switch_backend('SVG')
    fig = plt.figure()
    for i, fname_prefix in enumerate(filenames):
        ax = fig.add_subplot(row_num, col_num, i+1)
        if len(filenames[fname_prefix]) == 1:
            data = np.loadtxt(os.path.join(path, filenames[fname_prefix][0]), skiprows=2)
            ax.plot(data[:, -2], data[:, -1])
        else:
            for file_one_wire in filenames[fname_prefix]:
                data = np.loadtxt(os.path.join(path, file_one_wire), skiprows=2)
                ax.plot(data[:, 0], data[:, 1])
        ax.set_title(fname_prefix)
    plt.tight_layout()
    plt.savefig(img, format=('png'))
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    return graph_url

if __name__ == '__main__':
    app.run(debug = True, host='0.0.0.0')
