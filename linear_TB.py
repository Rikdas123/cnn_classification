import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ==============================
# LOAD TRAINED MODEL
# ==============================
model = tf.keras.models.load_model(
    r"C:\Users\User\Downloads\codes\horse_human.keras"
)

# ==============================
# DATASET PATH
# ==============================
test_dir = r"C:\Users\User\Downloads\codes\human_horse_test"

# IMPORTANT:
# This MUST match the semantic meaning used during training
# We will ASSUME:
# sigmoid output ≈ 1 → HUMAN
# sigmoid output ≈ 0 → HORSE
class_names = ["horses", "humans"]

# ==============================
# LOAD TEST DATASET
# ==============================
test_ds = image_dataset_from_directory(
    test_dir,
    image_size=(64, 64),
    batch_size=32,
    shuffle=False,
    class_names=class_names
)

print(f"✅ Class order used: {test_ds.class_names}")

# Normalize
test_ds = test_ds.map(lambda x, y: (x / 255.0, y))

# ==============================
# PREDICTIONS
# ==============================
y_true = np.concatenate([y.numpy() for _, y in test_ds])

y_pred_probs = model.predict(test_ds)

# -------- IMPORTANT FIX --------
# If sigmoid = 1 corresponds to HUMAN
# then prediction > 0.5 = HUMAN (class index 1)
# prediction <= 0.5 = HORSE (class index 0)
y_pred = (y_pred_probs >= 0.5).astype("int32").flatten()


# ==============================
# DIAGNOSTICS (VERY IMPORTANT)
# ==============================
print("\n🔎 Prediction statistics:")
print("Min probability :", float(y_pred_probs.min()))
print("Max probability :", float(y_pred_probs.max()))
print("Mean probability:", float(y_pred_probs.mean()))

# ==============================
# METRICS
# ==============================
accuracy = accuracy_score(y_true, y_pred)
print(f"\n✅ Overall Test Accuracy: {accuracy*100:.2f}%\n")

print("📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# ==============================
# CONFUSION MATRIX
# ==============================
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title(f"Confusion Matrix (Acc: {accuracy*100:.2f}%)")
plt.tight_layout()
plt.show()

# ==============================
# MANUAL SANITY CHECK (5 IMAGES)
# ==============================
print("\n🔍 Sample Predictions:")

test_images, test_labels = next(iter(test_ds))
preds = model.predict(test_images)

for i in range(5):
    img = test_images[i].numpy()
    true_label = class_names[test_labels[i].numpy()]
    p = preds[i][0]

    pred_label = "humans" if p > 0.5 else "horses"
    confidence = p if p > 0.5 else 1 - p

    plt.imshow(img)
    plt.title(
        f"True: {true_label} | Pred: {pred_label} | Conf: {confidence:.2f}"
    )
    plt.axis("off")
    plt.show()
