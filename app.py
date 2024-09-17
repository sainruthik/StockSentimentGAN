from flask import Flask, request, render_template
from predict import predict_stock  # Import prediction function from predict.py

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        company = request.form['company']
        tweet = request.form['tweet']
        
        # Call predict.py to get the prediction and sentiment score
        predicted_price, sentiment_score = predict_stock(company, tweet)
        
        return render_template('result.html', company=company, tweet=tweet, sentiment_score=sentiment_score, predicted_price=predicted_price)
    return render_template('index.html')

if __name__ == "__main__":
    from waitress import serve
    serve(app, host='0.0.0.0', port=5000)