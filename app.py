from flask import Flask, request
import subprocess
import os

app = Flask(__name__)

process = None

@app.route('/start', methods=['GET'])
def start_script():
    global process
    if not process:
        process = subprocess.Popen(['python', 'detect.py'])
        return 'Script started', 200
    return 'Script already running', 400

@app.route('/stop', methods=['GET'])
def stop_script():
    global process
    if process:
        process.terminate()
        process = None
        return 'Script stopped', 200
    return 'No script running', 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
