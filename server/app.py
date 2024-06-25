# from flask import Flask, request, jsonify
# # import cv2
# # import numpy as np
# # from tensorflow.keras.models import Model, Sequential
# # from tensorflow.keras.layers import Dense, LSTM, TimeDistributed, Embedding, RepeatVector, Concatenate, Activation, Input
# # from tensorflow.keras.applications import ResNet50
# # from tensorflow.keras.applications.resnet50 import preprocess_input
# # from tensorflow.keras.preprocessing.sequence import pad_sequences
# from flask_cors import CORS
# # from tqdm import tqdm
# # # Load model directly
# # # from transformers import AutoProcessor, AutoModelForSeq2SeqLM

# from transformers import BlipForConditionalGeneration, BlipProcessor
# from PIL import Image
# import torch


# # # Load ResNet50 model for feature extraction
# # resnet = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3), pooling='avg')

# # # Load vocabulary
# # vocab = np.load('vocab.npy', allow_pickle=True).item()
# # inv_vocab = {v: k for k, v in vocab.items()}

# # # Define parameters
# # embedding_size = 128
# # max_len = 40
# # vocab_size = len(vocab)

# # # Define image model
# # image_input = Input(shape=(2048,))
# # image_model = Dense(embedding_size, activation='relu')(image_input)
# # image_model = RepeatVector(max_len)(image_model)

# # # Define language model
# # language_input = Input(shape=(max_len,))
# # language_model = Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=max_len)(language_input)
# # language_model = LSTM(256, return_sequences=True)(language_model)
# # language_model = TimeDistributed(Dense(embedding_size))(language_model)

# # # Concatenate image and language models
# # conca = Concatenate()([image_model, language_model])
# # x = LSTM(128, return_sequences=True)(conca)
# # x = LSTM(512, return_sequences=False)(x)
# # x = Dense(vocab_size)(x)
# # out = Activation('softmax')(x)

# # # Define the final model
# # model = Model(inputs=[image_input, language_input], outputs=out)
# # model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])

# # # Load model weights
# # processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
# # model = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/blip-image-captioning-base")
# # # model.load_weights('mine_model_weights.h5')

# # print("=" * 50)
# # print("Model loaded")

# # # Initialize Flask app
# app = Flask(__name__)
# app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1
# cors = CORS(app, resources={r"/*": {"origins": "*"}})

# @app.route('/after', methods=['POST'])
# def after():
#     global model, vocab, inv_vocab
#     file = request.files['file']

#     file.save('static/file.jpg')

#     Image.open("static/file.jpg")
#     image = file
#     inputs = BlipProcessor(images=image, return_tensors="pt").to('cuda' if torch.cuda.is_available() else 'cpu')
#     model.to('cuda' if torch.cuda.is_available() else 'cpu')

# # Generate the caption
#     output_ids = model.generate(**inputs)
#     caption = processor.decode(output_ids[0], skip_special_tokens=True)
#     print(caption)
# #     img = cv2.imread('static/file.jpg')
# #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# #     img = cv2.resize(img, (224, 224))
# #     img = np.reshape(img, (1, 224, 224, 3))

# #     # Preprocess image
# #     img = preprocess_input(img)

# #     # Extract features
# #     features = resnet.predict(img).reshape(1, 2048)

# #     print("=" * 50)
# #     print("Predict Features")

# #     text_in = ['startofseq']
# #     final = ''

# #     print("=" * 50)
# #     print("Generating Captions")

# #     count = 0
# #     while count < 20:
# #         count += 1
# #         encoded = [vocab[word] for word in text_in]
# #         padded = pad_sequences([encoded], maxlen=max_len, padding='post', truncating='post')

# #         sampled_index = np.argmax(model.predict([features, padded]))
# #         sampled_word = inv_vocab[sampled_index]

# #         if sampled_word != 'endofseq':
# #             final += ' ' + sampled_word

# #         text_in.append(sampled_word)

#     return jsonify({'caption': caption})

# if __name__ == "__main__":
#     app.run(debug=True)
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import BlipForConditionalGeneration, BlipProcessor
from PIL import Image
import torch

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1
cors = CORS(app, resources={r"/*": {"origins": "*"}})

# Load the processor and model outside of the route to avoid reloading them for every request
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

@app.route('/after', methods=['POST'])
def after():
    file = request.files['file']
    file_path = 'static/file.jpg'
    file.save(file_path)

    image = Image.open(file_path)
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate the caption
    output_ids = model.generate(**inputs)
    caption = processor.decode(output_ids[0], skip_special_tokens=True)
    print(caption)

    return jsonify({'caption': caption})

if __name__ == "__main__":
    app.run(debug=True)
