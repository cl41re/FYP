from flask import Flask, request, jsonify
import json
import numpy as np

import pandas as pd
from transformers import Trainer

from cyberbully import Dataset, get_cb_model, text_preprocessing_pipeline

app = Flask(__name__)

@app.route('/detect', methods=['GET', 'POST'])
def detect():
    if request.method == 'POST':
        
        data = request.get_json()
        tokenizer, model = get_cb_model()
        input_data = {"text" : [data]}
        print(input_data)
        print(data)
        bully_data = pd.DataFrame(input_data)

        cleaned_input_text = text_preprocessing_pipeline(
                                                df=bully_data,
                                                remove_url=True,
                                                remove_email=True,
                                                remove_user_mention=True,
                                                remove_html=False,
                                                remove_space_single_char=True,
                                                normalize_elongated_char=True,
                                                normalize_accented=True,
                                                lower_case=True,
                                                normalize_slang=True,
                                                normalize_contraction=True,
                                                remove_numeric=True,
                                                remove_stopword=True,
                                                keep_pronoun=True,  
                                                remove_punctuation=True,
                                                lemmatise=True)

        input_text_tokenized = tokenizer(cleaned_input_text, padding=True, truncation=True, max_length=512)

        # Create torch dataset
        input_text_dataset = Dataset(input_text_tokenized)

        # Define test trainer
        pred_trainer = Trainer(model)

        # Make prediction
        raw_pred, _, _ = pred_trainer.predict(input_text_dataset)
    
        # Preprocess raw predictions
        text_pred = np.where(np.argmax(raw_pred, axis=1) == 1, "Cyberbullying Post", "Non-cyberbullying Post")

        detection = text_pred.tolist()[0]

        #return true or false
        response_data = {
        "isBully": (detection == "Cyberbullying Post")
        }

        return jsonify(response_data)
        
    else:
        return "This endpoint only supports POST requests."


if __name__ == '__main__':
    app.run(debug=True)
