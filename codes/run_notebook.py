"""Execute this notebook's Python cells and save their real text/figure outputs.

Run from a terminal with: python codes/run_notebook.py
Only the standard library, NumPy, and Matplotlib are required. A Jupyter kernel
is an alternative, and is not required by this script. Start a fresh process for
each full run. Existing figures and experiment records are overwritten.
"""
import base64
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import time


def main():
    code_dir = Path(__file__).resolve().parent
    notebook_path = code_dir / 'HWX2_MLP_MNIST.ipynb'
    project_root = code_dir.parent if code_dir.name.lower() == 'codes' else code_dir
    os.chdir(project_root)
    # Avoid excessive BLAS threads on small mini-batch matrix products.
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    os.environ.setdefault('MPLBACKEND', 'Agg')
    import matplotlib.pyplot as plt
    notebook = json.loads(notebook_path.read_text(encoding='utf-8'))
    namespace = {'__name__': '__main__', 'EXECUTION_METHOD': 'Python cell runner'}
    count = 0
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            cell['execution_count'] = None
            cell['outputs'] = []

    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        count += 1
        buffer = io.StringIO()
        cell['execution_count'] = count
        outputs = cell['outputs']

        def flush_text():
            content = buffer.getvalue()
            if content:
                outputs.append({'output_type': 'stream', 'name': 'stdout', 'text': content})
                sys.__stdout__.write(content)
                sys.__stdout__.flush()
                buffer.seek(0)
                buffer.truncate(0)

        def show(*args, **kwargs):
            flush_text()
            for number in plt.get_fignums():
                fig = plt.figure(number)
                png = io.BytesIO()
                fig.savefig(png, format='png', dpi=120, bbox_inches='tight', facecolor='white')
                outputs.append({'output_type': 'display_data', 'metadata': {}, 'data': {
                    'image/png': base64.b64encode(png.getvalue()).decode('ascii'),
                    'text/plain': f'<Figure with {len(fig.axes)} Axes>',
                }})
                plt.close(fig)

        plt.show = show
        started = time.perf_counter()
        source = cell['source']
        if isinstance(source, list):
            source = ''.join(source)
        try:
            with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
                exec(compile(source, f'{notebook_path.name}:cell{count}', 'exec'), namespace)
        finally:
            flush_text()
            cell.setdefault('metadata', {})['execution_seconds'] = round(time.perf_counter() - started, 4)
            notebook_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
        print(f'Finished code cell {count}.', flush=True)
    print(f'Saved executed notebook: {notebook_path.name}', flush=True)


if __name__ == '__main__':
    main()
