from fileinput import filename
from flask import Flask, render_template, request , jsonify,redirect,url_for
import time
import os

from my_tools.index import index_one
from my_tools.search import Search


app = Flask(__name__)
#general parameters
UPLOAD_FOLDER = 'static/upload_image'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.jinja_env.variable_start_string = '{['
app.jinja_env.variable_end_string = ']}'
detail12 = {}

@app.route('/')
def index():
    return render_template('test.html' )

@app.post('/upload')
def upload():
    detail = []
    # Saving the Uploaded image in the Upload folder
    file = request.files['image']
    new_file_name = str(
        str(time.time()) + '.jpg'
    )
    file.save(os.path.join(
            app.config['UPLOAD_FOLDER'],new_file_name
        )
    )
    # Trích xuất vectơ đối tượng từ các hình ảnh đã tải lên và thêm vectơ này vào cơ sở dữ liệu của chúng tôi
    features = index_one(str(UPLOAD_FOLDER + '/' + new_file_name) )

    print(str(UPLOAD_FOLDER + '/' + new_file_name))
    searcher = Search('my_tools/lfcnn/filecsv/lfcnn_SVD_n.csv')
    results = searcher.search(features)
    for (score, pathImage) in results:
        detail.append(
        {"image": str(pathImage), "score": str(score)}
    )
    detail12['data']=detail
    #return redirect(url_for('index', filename=str(UPLOAD_FOLDER + '/' + new_file_name)))
    return render_template("test.html", filename=str(UPLOAD_FOLDER + '/' + new_file_name))

@app.route('/api/image', methods=['GET', 'POST'])
def image():
    #print(detail12)
    return jsonify(detail12)

    

if __name__ == '__main__':
    app.run(debug=True)