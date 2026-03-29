import os
import numpy as np
import pickle
from tqdm import tqdm

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from tensorflow.keras.utils import to_categorical


# ---------------------------
# LOAD CAPTIONS
# ---------------------------
def load_captions(filename):
    captions = {}

    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()

            # skip header
            if line.startswith("image"):
                continue

            if len(line) == 0:
                continue

            parts = line.split(',', 1)

            if len(parts) < 2:
                continue

            img_id, caption = parts[0], parts[1]
            img = img_id.split('#')[0]

            if img not in captions:
                captions[img] = []

            captions[img].append("startseq " + caption + " endseq")

    return captions


# ---------------------------
# FEATURE EXTRACTION
# ---------------------------
def extract_features(image_dir):
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    features = {}

    for img_name in tqdm(os.listdir(image_dir)):
        path = os.path.join(image_dir, img_name)

        img = load_img(path, target_size=(224, 224))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)

        feature = model.predict(img, verbose=0)
        features[img_name] = feature.flatten()

    return features


# ---------------------------
# TOKENIZER
# ---------------------------
def create_tokenizer(captions):
    lines = []
    for key in captions:
        lines.extend(captions[key])

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)

    return tokenizer


# ---------------------------
# MAX LENGTH
# ---------------------------
def max_length(captions):
    return max(len(c.split()) for key in captions for c in captions[key])


# ---------------------------
# DATA GENERATOR
# ---------------------------
def data_generator(captions, features, tokenizer, max_len, vocab_size):
    X1, X2, y = [], [], []

    for img, cap_list in captions.items():
        # skip images not in features (extra safety)
        if img not in features:
            continue

        for cap in cap_list:
            seq = tokenizer.texts_to_sequences([cap])[0]

            for i in range(1, len(seq)):
                in_seq = seq[:i]
                out_seq = seq[i]

                in_seq = pad_sequences([in_seq], maxlen=max_len)[0]
                out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]

                X1.append(features[img])
                X2.append(in_seq)
                y.append(out_seq)

    return np.array(X1), np.array(X2), np.array(y)


# ---------------------------
# MODEL
# ---------------------------
def build_model(vocab_size, max_len):
    inputs1 = Input(shape=(2048,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)

    inputs2 = Input(shape=(max_len,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)

    decoder = add([fe2, se3])
    decoder = Dense(256, activation='relu')(decoder)
    outputs = Dense(vocab_size, activation='softmax')(decoder)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model


# ---------------------------
# CAPTION GENERATION
# ---------------------------
def idx_to_word(index, tokenizer):
    for word, i in tokenizer.word_index.items():
        if i == index:
            return word
    return None


def generate_caption(model, tokenizer, photo, max_len):
    text = "startseq"

    for _ in range(max_len):
        seq = tokenizer.texts_to_sequences([text])[0]
        seq = pad_sequences([seq], maxlen=max_len)

        yhat = model.predict([photo, seq], verbose=0)
        yhat = np.argmax(yhat)

        word = idx_to_word(yhat, tokenizer)

        if word is None:
            break

        text += " " + word

        if word == "endseq":
            break

    return text


# ---------------------------
# MAIN EXECUTION
# ---------------------------
if __name__ == "__main__":

    print("Loading captions...")
    captions = load_captions("captions.txt")
    print("Total captions loaded:", len(captions))

    # reduce dataset (for your system)
    captions = dict(list(captions.items())[:200])

    # load features (avoid reprocessing)
    if os.path.exists("features.pkl"):
        print("Loading saved features...")
        features = pickle.load(open("features.pkl", "rb"))
    else:
        print("Extracting features...")
        features = extract_features("images")
        pickle.dump(features, open("features.pkl", "wb"))

    print("Creating tokenizer...")
    tokenizer = create_tokenizer(captions)
    vocab_size = len(tokenizer.word_index) + 1
    max_len = max_length(captions)

    print("Preparing data...")
    X1, X2, y = data_generator(captions, features, tokenizer, max_len, vocab_size)

    print("Building model...")
    model = build_model(vocab_size, max_len)

    print("Training...")
    model.fit([X1, X2], y, epochs=5, batch_size=32)

    model.save("model.h5")
    pickle.dump(tokenizer, open("tokenizer.pkl", "wb"))

    print("✅ Training Complete!")

    # ---------------------------
    # TEST MODEL
    # ---------------------------
    print("\nTesting model...")

    img_name = list(features.keys())[0]
    photo = features[img_name].reshape((1, 2048))

    caption = generate_caption(model, tokenizer, photo, max_len)

    caption = caption.replace("startseq", "").replace("endseq", "")

    print("🖼 Image:", img_name)
    print("🧠 Caption:", caption)
