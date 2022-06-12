from flask import Flask, render_template, request , jsonify,redirect,url_for
import time
import os

from my_tools.index import index_one
from my_tools.search import Search


app = Flask(__name__)
#general parameters
UPLOAD_FOLDER = 'static/images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.jinja_env.variable_start_string = '{['
app.jinja_env.variable_end_string = ']}'
detail12 = {}
#Database Offline Indexing
# @app.route('/offlineIndex')
# def test():
#     index(params)
#     return "Done !!"

#Index route
@app.route('/')
def index():
    return render_template('test.html' )


# @app.post('/upload')
# def upload():
#     detail = []
#     # Saving the Uploaded image in the Upload folder

#     file = request.files['image']
#     print(file)

#     tai = file
#     print("[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[")
#     new_file_name = str(
#         str(time.time()) + '.jpg'
#     )
#     file.save(os.path.join(
#             app.config['UPLOAD_FOLDER'],new_file_name
#         )
#     )

#     # Trích xuất vectơ đối tượng từ các hình ảnh đã tải lên và thêm vectơ này vào cơ sở dữ liệu của chúng tôi
#     features = index_one(str(UPLOAD_FOLDER + '/' + new_file_name) )

#     # So sánh và sắp xếp các tính năng của hình ảnh đã tải lên với các tính năng của hình ảnh calulcated ngoại tuyến
#     # searcher = Search('my_tools/lfcnn_pca.csv')
#     searcher = Search('my_tools/lfcnn/filecsv/lfcnn_SVD_n.csv')
#     results = searcher.search(features)
#     # results = searcher.gaborSearch(features)
#     RESULTS_LIST = list()
#     for (score, pathImage) in results:
#         RESULTS_LIST.append(
#             {"image": str(pathImage), "score": str(score)}
#         )
#    # print(RESULTS_LIST)
#     print(results)
#     for (score, pathImage) in results:
#             detail.append(
#             {"image": str(pathImage), "score": str(score)}
#         )
#     print("asdasdasdwqweqweq213213123")
#     print(detail)
#     deto = {}
#     deto['data'] = detail
#     detail12['data']=detail
#     print("asdasdasdwqw#######################################################13123")
#     print(deto)
#     #returning the search results
#     return jsonify(RESULTS_LIST)

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

    # So sánh và sắp xếp các tính năng của hình ảnh đã tải lên với các tính năng của hình ảnh calulcated ngoại tuyến
    # searcher = Search('my_tools/lfcnn_pca.csv')
    searcher = Search('my_tools/lfcnn/filecsv/lfcnn_SVD_n.csv')
    results = searcher.search(features)
    # results = searcher.gaborSearch(features)
    # RESULTS_LIST = list()
    # for (score, pathImage) in results:
    #     RESULTS_LIST.append(
    #         {"image": str(pathImage), "score": str(score)}
    #     )
    # print(RESULTS_LIST)
    for (score, pathImage) in results:
        detail.append(
        {"image": str(pathImage), "score": str(score)}
    )
    # print("asdasdasdwqweqweq213213123")
    # print(detail)
    deto = {}
    deto['data'] = detail
    detail12['data']=detail
    # print("asdasdasdwqw#######################################################13123")
    # print(deto)   
    #returning the search results
    #return render_template("test.html")
    return redirect(url_for('index'))
# @app.route('/api', methods=['GET', 'POST'])
# def api():
#     detail = []
#     if request.method == "POST":
       
#         file = request.files["image"]
        

#     # Saving the Uploaded image in the Upload folder
      
#         new_file_name = str(
#             str(time.time()) + '.jpg'
#         )
#         file.save(os.path.join(
#                 app.config['UPLOAD_FOLDER'],new_file_name
#             )
#         )

#         # Trích xuất vectơ đối tượng từ các hình ảnh đã tải lên và thêm vectơ này vào cơ sở dữ liệu của chúng tôi
#         features = index_one(str(UPLOAD_FOLDER + '/' + new_file_name) )

#         # So sánh và sắp xếp các tính năng của hình ảnh đã tải lên với các tính năng của hình ảnh calulcated ngoại tuyến
#         # searcher = Search('my_tools/lfcnn_pca.csv')
#         searcher = Search('my_tools/lfcnn/filecsv/lfcnn_SVD_n.csv')
#         results = searcher.search(features)
#         # results = searcher.gaborSearch(features)
#         print(results)
#         for (score, pathImage) in results:
#                 detail.append(
#                 {"image": str(pathImage), "score": str(score)}
#             )
#         print("asdasdasdwqweqweq213213123")
#         print(detail)
#         deto = {}
#         deto['data'] = detail
#         detail12['data']=detail
#         print("asdasdasdwqw#######################################################13123")
#         print(deto)

#         #returning the search results
#         #return redirect(url_for('app.kq'))
#         return render_template("kq.html")
#     return render_template("kq.html")

@app.route('/api/image', methods=['GET', 'POST'])
def image():
    print(detail12)
    print("asidhf11111111111111111111111111111111111111111111111111111111111111111111111111111111111111")
    return jsonify(detail12)

    

if __name__ == '__main__':
    app.run(debug=True)