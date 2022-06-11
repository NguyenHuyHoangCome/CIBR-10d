from flask import Flask, render_template, request , jsonify
import time
import os

from my_tools.index import index_one
from my_tools.search import Search


app = Flask(__name__)
#general parameters
UPLOAD_FOLDER = 'static/images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


#Database Offline Indexing
# @app.route('/offlineIndex')
# def test():
#     index(params)
#     return "Done !!"

#Index route
@app.route('/')
def index():
    return render_template('main.html' )

@app.post('/upload')
def upload():
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

    # So sánh và sắp xếp các tính năng của hình ảnh đã tải lên với các tính năng của hình ảnh calulcated ngoại tuyến
    # searcher = Search('my_tools/lfcnn_pca.csv')
    searcher = Search('my_tools/lfcnn/filecsv/lfcnn_SVD_n.csv')
    results = searcher.search(features)
    # results = searcher.gaborSearch(features)
    RESULTS_LIST = list()
    for (score, pathImage) in results:
        RESULTS_LIST.append(
            {"image": str(pathImage), "score": str(score)}
        )
    print(RESULTS_LIST)

    #returning the search results
    return jsonify(RESULTS_LIST)

if __name__ == '__main__':
    app.run(debug=True)