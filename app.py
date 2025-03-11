from flask import Flask, render_template
import threading
import subprocess

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/hip_tutorial')
def hip_tutorial():
    return render_template('hip_tutorial.html')

@app.route('/hand_tutorial')
def hand_tutorial():
    return render_template('hand_tutorial.html')

@app.route('/stretching_tutorial')
def stretching_tutorial():
    return render_template('stretching_tutorial.html')

@app.route('/neck_tutorial')
def neck_tutorial():
    return render_template('neck_tutorial.html')

def run_script(script):
    subprocess.call(["python", script])

@app.route('/start_hip_exercise')
def start_hip_exercise():
    t = threading.Thread(target=run_script, args=("C:/Projects/FinalProject/excesise_proj/hip_exces/hip_exces.py",))
    t.start()
    return render_template('align_hip.html')

@app.route('/start_hand_exercise')
def start_hand_exercise():
    t = threading.Thread(target=run_script, args=("C:/Projects/FinalProject/excesise_proj/hand_ex_model/hand_exce.py",))
    t.start()
    return render_template('align_hand.html')

@app.route('/start_stretching_exercise')
def start_stretching_exercise():
    t = threading.Thread(target=run_script, args=("C:/Projects/FinalProject/excesise_proj/exces_up_down/up_down_excs.py",))
    t.start()
    return render_template('align_stretching.html')

@app.route('/start_neck_exercise')
def start_neck_exercise():
    t = threading.Thread(target=run_script, args=("C:/Projects/FinalProject/excesise_proj/head_exces/head_exces.py",))
    t.start()
    return render_template('align_neck.html')

@app.route('/static/<path:path>')
def serve_static(path):
    root_dir = os.path.dirname(os.getcwd())
    return send_from_directory(os.path.join(root_dir, 'static'), path)

@app.route('/index.html')
def loading_screen():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
