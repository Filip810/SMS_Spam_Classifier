# sms_spam_cnn_compress_fixed.py
"""
Pipeline: TF-IDF / Embedding -> Conv1D -> k-fold training
+ class_weight, F1 checkpoint, pruning (opcjonalnie), TFLite conversion
+ evaluation + latency measurement.

Uruchom:
(tfvenv) > python sms_spam_cnn_compress_fixed.py
"""

import os, time, shutil, io, zipfile, urllib.request
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, f1_score
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers

# Optional pruning
try:
    import tensorflow_model_optimization as tfmot
    pruning_available = True
except Exception:
    pruning_available = False

# ------------------ PARAMETERS ------------------
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
RANDOM_SEED = 42
BATCH_SIZE = 32
EPOCHS = 25
KFOLD = 5
USE_TFIDF_SEQUENCE = True  # True: Conv1D over TF-IDF vector; False: token->Embedding->Conv1D
MAX_VOCAB = 8000
TFIDF_MAX_FEATURES = MAX_VOCAB
SEQ_MAXLEN = 160
PRUNE_TARGET_SPARSITY = 0.5
HOLDOUT_SIZE = 0.15
# ------------------------------------------------

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# GPU safety
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass

# -------- F1 checkpoint --------
class F1Checkpoint(tf.keras.callbacks.Callback):
    def __init__(self, X_val, y_val, filepath_keras):
        super().__init__()
        self.X_val, self.y_val = X_val, y_val
        self.best_f1 = -1.0
        self.filepath_keras = filepath_keras
        # usuwamy stary plik jeżeli istnieje
        try:
            if os.path.exists(filepath_keras):
                os.remove(filepath_keras)
        except Exception:
            pass

    def on_epoch_end(self, epoch, logs=None):
        preds = self.model.predict(self.X_val, verbose=0).ravel()
        yhat = (preds >= 0.5).astype(int)
        f1 = f1_score(self.y_val, yhat, zero_division=0)
        print(f" — val_f1: {f1:.4f}")
        if f1 > self.best_f1 + 1e-6:
            self.best_f1 = f1
            # zapisujemy jako plik .keras
            self.model.save(self.filepath_keras, include_optimizer=False)

# ------------------ DATA ------------------
print("Downloading dataset...")
data_b = urllib.request.urlopen(DATA_URL).read()
z = zipfile.ZipFile(io.BytesIO(data_b))
txt = z.read("SMSSpamCollection").decode("utf-8")
rows = [r.split('\t') for r in txt.splitlines() if r.strip()]
df = pd.DataFrame(rows, columns=['label','text'])
df['y'] = (df['label'] == 'spam').astype(int)
print("Loaded", len(df), "messages. Spam:", int(df['y'].sum()))

texts = df['text'].astype(str).tolist()
y = df['y'].values

if USE_TFIDF_SEQUENCE:
    tfv = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, ngram_range=(1,2), strip_accents='unicode')
    X = tfv.fit_transform(texts).toarray().astype(np.float32)
    input_shape = X.shape[1:]
    print("TF-IDF shape:", X.shape)
else:
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    tok = Tokenizer(num_words=MAX_VOCAB, oov_token="<OOV>")
    tok.fit_on_texts(texts)
    seqs = tok.texts_to_sequences(texts)
    X = pad_sequences(seqs, maxlen=SEQ_MAXLEN, padding='post', truncating='post').astype(np.int32)
    input_shape = X.shape[1:]
    print("Token sequence shape:", X.shape)

# ------------------ MODELS ------------------
def build_cnn_tfidf(input_len):
    inp = layers.Input(shape=(input_len,1))
    x = layers.Conv1D(64, 3, activation='relu', padding='same')(inp)
    x = layers.Conv1D(32, 3, activation='relu', padding='same')(x)
    x = layers.GlobalMaxPool1D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inp, out)
    model.compile(optimizer=optimizers.Adam(1e-3),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def build_cnn_emb(seq_len, vocab_size=MAX_VOCAB):
    inp = layers.Input(shape=(seq_len,))
    x = layers.Embedding(vocab_size, 128)(inp)
    x = layers.Conv1D(128, 5, activation='relu', padding='same')(x)
    x = layers.Conv1D(64, 5, activation='relu', padding='same')(x)
    x = layers.GlobalMaxPool1D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inp, out)
    model.compile(optimizer=optimizers.Adam(1e-3),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# ------------------ K-FOLD ------------------
skf = StratifiedKFold(n_splits=KFOLD, shuffle=True, random_state=RANDOM_SEED)
metrics = []
best_saved_paths = []  # pełne ścieżki .keras
last_model = None
last_fold_data = None

for fold,(train_idx,val_idx) in enumerate(skf.split(X,y),1):
    print(f"\n--- Fold {fold}/{KFOLD} ---")
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    cw = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    cw_dict = {0: float(cw[0]), 1: float(cw[1])}
    print("class_weight:", cw_dict)

    tf.keras.backend.clear_session()
    if USE_TFIDF_SEQUENCE:
        Xtr = X_train.reshape((-1, X_train.shape[1], 1))
        Xva = X_val.reshape((-1, X_val.shape[1], 1))
        model = build_cnn_tfidf(Xtr.shape[1])
    else:
        Xtr, Xva = X_train, X_val
        model = build_cnn_emb(Xtr.shape[1])

    saved_path = f"best_fold{fold}.keras"
    f1_cb = F1Checkpoint(Xva, y_val, saved_path)
    es = callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
    rlrop = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)

    model.fit(Xtr, y_train, validation_data=(Xva,y_val),
              epochs=EPOCHS, batch_size=BATCH_SIZE,
              callbacks=[es, rlrop, f1_cb], class_weight=cw_dict, verbose=2)

    y_pred = (model.predict(Xva).ravel()>=0.5).astype(int)
    acc = accuracy_score(y_val,y_pred)
    p,r,f1,_ = precision_recall_fscore_support(y_val,y_pred,average='binary',zero_division=0)
    cm = confusion_matrix(y_val,y_pred)
    metrics.append({'fold':fold,'acc':acc,'precision':p,'recall':r,'f1':f1,'cm':cm})
    print("Fold metrics:", metrics[-1])

    best_saved_paths.append(saved_path)
    last_model = model
    last_fold_data = (Xtr, y_train, Xva, y_val)

print("\nSummary:", metrics)

best_fold_entry = max(metrics, key=lambda x: x['f1'])
best_idx = best_fold_entry['fold'] - 1
best_saved_model_path = best_saved_paths[best_idx]
print(f"Best fold: {best_fold_entry['fold']} (F1={best_fold_entry['f1']:.4f}). Using: {best_saved_model_path}")

# ------------------ Save baseline model (export SavedModel) ------------------
saved_model_dir = "saved_model_baseline"
if os.path.exists(saved_model_dir):
    try: shutil.rmtree(saved_model_dir)
    except Exception: pass

try:
    best_model = tf.keras.models.load_model(best_saved_model_path)
    # Keras 3: do katalogu SavedModel używamy export()
    best_model.export(saved_model_dir)
    print("Exported baseline SavedModel to", saved_model_dir)
except Exception as e:
    print("Failed to load best model, using last_model:", e)
    try:
        last_model.export(saved_model_dir)
        print("Exported last_model as baseline to", saved_model_dir)
    except Exception as e3:
        print("Critical: cannot export baseline model:", e3)
        raise

# ------------------ Pruning (optional) ------------------
if pruning_available and last_fold_data is not None:
    print("\nApplying pruning (one cycle prune->fine-tune)...")
    Xtr, y_train, Xva, y_val = last_fold_data
    steps_per_epoch = int(np.ceil(Xtr.shape[0] / BATCH_SIZE))
    end_step = steps_per_epoch * 6

    def apply_pruning(model, target_sparsity=PRUNE_TARGET_SPARSITY, begin_step=0, end_step=1000):
        pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
                initial_sparsity=0.0,
                final_sparsity=target_sparsity,
                begin_step=begin_step,
                end_step=end_step)
        }
        return tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)

    pruned = apply_pruning(last_model, target_sparsity=PRUNE_TARGET_SPARSITY, begin_step=0, end_step=end_step)
    pruned.compile(optimizer=optimizers.Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    cb = [tfmot.sparsity.keras.UpdatePruningStep(),
          callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)]
    pruned.fit(Xtr, y_train, validation_data=(Xva, y_val), epochs=6, batch_size=BATCH_SIZE, callbacks=cb, verbose=2)
    final_pruned = tfmot.sparsity.keras.strip_pruning(pruned)
    pruned_dir = "saved_model_pruned"
    if os.path.exists(pruned_dir): shutil.rmtree(pruned_dir)
    # export SavedModel
    final_pruned.export(pruned_dir)
    print("Exported pruned model to", pruned_dir)
else:
    print("Pruning not available or no last fold data. Skipping.")

# ------------------ TFLite conversion ------------------
def convert_to_tflite(saved_model_dir, tflite_path, representative_data=None, int8=False):
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    if int8:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if representative_data is None:
            raise ValueError("Representative data required for full integer quantization.")
        def rep_gen():
            for i in representative_data:
                yield [i]
        converter.representative_dataset = rep_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    tflite_model = converter.convert()
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print("Wrote TFLite model:", tflite_path)

# representative data
rep_idx = np.random.choice(len(X), size=min(200, len(X)), replace=False)
rep_data = []
if USE_TFIDF_SEQUENCE:
    for i in rep_idx:
        rep_data.append(X[i].reshape(1, X.shape[1], 1).astype(np.float32))
else:
    for i in rep_idx:
        rep_data.append(X[i].reshape(1, X.shape[1]).astype(np.int32))

# convert baseline -> float TFLite
try:
    convert_to_tflite(saved_model_dir, "model_baseline.tflite", representative_data=rep_data, int8=False)
except Exception as e:
    print("Baseline TFLite conversion failed:", e)

# try INT8
try:
    convert_to_tflite(saved_model_dir, "model_baseline_int8.tflite", representative_data=rep_data, int8=True)
except Exception as e:
    print("INT8 conversion failed (ops not compatible or missing rep data):", e)

# if pruned exists, convert pruned
if pruning_available and os.path.exists("saved_model_pruned"):
    try:
        convert_to_tflite("saved_model_pruned", "model_pruned.tflite", representative_data=rep_data, int8=False)
        convert_to_tflite("saved_model_pruned", "model_pruned_int8.tflite", representative_data=rep_data, int8=True)
    except Exception as e:
        print("TFLite convert pruned failed:", e)

# ------------------ eval_tflite ------------------
def eval_tflite(tflite_path, X_eval, y_eval, batch=1):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    inp_details = interpreter.get_input_details()
    out_details = interpreter.get_output_details()

    # attempt to resize input batch dim
    input_index = inp_details[0]['index']
    expected_shape = inp_details[0]['shape'].tolist()
    batch_size = min(batch, len(X_eval))
    new_shape = expected_shape.copy()
    new_shape[0] = batch_size
    try:
        interpreter.resize_tensor_input(input_index, new_shape)
    except Exception:
        batch_size = 1
        new_shape[0] = 1
        try:
            interpreter.resize_tensor_input(input_index, new_shape)
        except Exception:
            pass

    interpreter.allocate_tensors()
    inp_details = interpreter.get_input_details()
    out_details = interpreter.get_output_details()

    ypred = []
    for i in range(0, len(X_eval), batch_size):
        batch_x = X_eval[i:i+batch_size]
        if USE_TFIDF_SEQUENCE:
            batch_x_proc = batch_x.reshape((batch_x.shape[0], batch_x.shape[1], 1)).astype(np.float32)
        else:
            batch_x_proc = batch_x.astype(np.int32)

        if inp_details[0]['dtype'] == np.int8:
            scale, zero_point = inp_details[0]['quantization']
            scale = scale if scale != 0 else 1.0
            batch_xq = (batch_x_proc / scale + zero_point).round().astype(np.int8)
            interpreter.set_tensor(inp_details[0]['index'], batch_xq)
        else:
            interpreter.set_tensor(inp_details[0]['index'], batch_x_proc)

        interpreter.invoke()
        out = interpreter.get_tensor(out_details[0]['index'])
        if out_details[0]['dtype'] == np.int8:
            scale, zero_point = out_details[0]['quantization']
            scale = scale if scale != 0 else 1.0
            out = (out.astype(np.float32) - zero_point) * scale
        ypred.extend(out.ravel().tolist())

    ypred = np.array(ypred)[:len(y_eval)]
    yhat = (ypred >= 0.5).astype(int)
    acc = accuracy_score(y_eval, yhat)
    p, r, f1, _ = precision_recall_fscore_support(y_eval, yhat, average='binary', zero_division=0)
    cm = confusion_matrix(y_eval, yhat)
    return dict(acc=acc, precision=p, recall=r, f1=f1, cm=cm, probs=ypred)

# ------------------ holdout & latency ------------------
X_train_full, X_hold, y_train_full, y_hold = train_test_split(X, y, test_size=HOLDOUT_SIZE, stratify=y, random_state=RANDOM_SEED)
print("Created holdout split. Holdout size:", len(y_hold))

# prepare sample
if USE_TFIDF_SEQUENCE:
    sample = X_hold.reshape((-1, X_hold.shape[1], 1)).astype(np.float32)
else:
    sample = X_hold.astype(np.int32)

# evaluate baseline tflite (float)
if os.path.exists("model_baseline.tflite"):
    try:
        res = eval_tflite("model_baseline.tflite", X_hold, y_hold, batch=1)
        print("Baseline TFLite eval:", res)
    except Exception as e:
        print("Failed to eval baseline tflite:", e)

# evaluate int8 if exists
if os.path.exists("model_baseline_int8.tflite"):
    try:
        res = eval_tflite("model_baseline_int8.tflite", X_hold, y_hold, batch=1)
        print("Baseline INT8 eval:", res)
    except Exception as e:
        print("Failed to eval baseline int8:", e)

# latency helper
def latency_throughput(tflite_path, X_sample, runs=200):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    # warmup
    for _ in range(10):
        interpreter.set_tensor(inp['index'], X_sample[:1])
        interpreter.invoke()
    times = []
    for i in range(runs):
        s = time.time()
        interpreter.set_tensor(inp['index'], X_sample[i % len(X_sample): (i % len(X_sample))+1])
        interpreter.invoke()
        times.append(time.time() - s)
    times = np.array(times)
    p95 = np.percentile(times, 95)
    mean = times.mean()
    throughput = 1.0 / mean if mean > 0 else float('inf')
    return dict(p95_ms=p95*1000, mean_ms=mean*1000, throughput_per_sec=throughput)

if os.path.exists("model_baseline_int8.tflite"):
    try:
        print("Latency baseline int8:", latency_throughput("model_baseline_int8.tflite", sample, runs=200))
    except Exception as e:
        print("Latency measurement failed:", e)

print("Done. Artifacts (if created): saved_model_baseline/, model_baseline.tflite, model_baseline_int8.tflite, saved_model_pruned/")
