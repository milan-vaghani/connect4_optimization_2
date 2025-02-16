import os
import anvil.server
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, Flatten, Embedding, LayerNormalization, MultiHeadAttention
)
import time

# Constants
ROW_COUNT = 6
COLUMN_COUNT = 7
CNN_MODEL_PATH = "/docker_files/cnn_connect4.h5"
TRANSFORMER_MODEL_PATH = "/docker_files/myconnect4_transformer_model.h5"

# ------------------------------------------------------------------------
# 1) Load CNN Model
# ------------------------------------------------------------------------
cnn_model = None
if os.path.exists(CNN_MODEL_PATH):
    print(f"Loading CNN model from {CNN_MODEL_PATH}...")
    try:
        cnn_model = load_model(CNN_MODEL_PATH, compile=False)
        print("✅ CNN model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading CNN model: {e}")
else:
    print(f"❌ CNN model file not found at {CNN_MODEL_PATH}.")


# ------------------------------------------------------------------------
# 2) Define Transformer Model Components
# ------------------------------------------------------------------------
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, embed_dim, height, width):
        super(PositionalEncoding, self).__init__()
        self.embed_dim = embed_dim
        self.height = height
        self.width = width
        self.position_embeddings = Embedding(input_dim=height * width, output_dim=embed_dim)

    def call(self, inputs):
        position_indices = tf.range(start=0, limit=self.height * self.width, delta=1)
        position_embeddings = self.position_embeddings(position_indices)
        position_embeddings = tf.reshape(position_embeddings, (self.height, self.width, self.embed_dim))
        position_embeddings = tf.expand_dims(position_embeddings, axis=0)  # Add batch dimension
        return inputs + position_embeddings

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation='relu'),
            Dense(embed_dim)
        ])
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.norm1(inputs + attn_output)
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.norm2(out1 + ffn_output)


# ------------------------------------------------------------------------
# 3) Load Transformer Model
# ------------------------------------------------------------------------
transformer_model = None

def load_model_for_inference(model_path):
    """Loads a trained Transformer model for inference."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Error: Model file not found at {model_path}")

    try:
        custom_objects = {
            'PositionalEncoding': PositionalEncoding,
            'TransformerBlock': TransformerBlock
        }
        model = load_model(model_path, custom_objects=custom_objects)
        print(f"✅ Transformer model loaded successfully from {model_path}")
        return model
    except Exception as e:
        raise ValueError(f"❌ Error loading Transformer model: {e}")

if os.path.exists(TRANSFORMER_MODEL_PATH):
    print(f"Loading Transformer model from {TRANSFORMER_MODEL_PATH}...")
    try:
        transformer_model = load_model_for_inference(TRANSFORMER_MODEL_PATH)
    except Exception as e:
        print(e)
else:
    print(f"❌ Transformer model file not found at {TRANSFORMER_MODEL_PATH}.")


# ------------------------------------------------------------------------
# 4) Connect to Anvil
# ------------------------------------------------------------------------
ANVIL_UPLINK_KEY = "server_XCLDOCQ3R6NRUPV2UPUB4USO-3COCDQ6LIK6OUWK7"
anvil.server.connect(ANVIL_UPLINK_KEY)


# ------------------------------------------------------------------------
# 5) Utility functions
# ------------------------------------------------------------------------
def convert_to_6x7x2(board):
    """Converts the board to (6,7,2) format for CNN/Transformer."""
    board_array = np.array(board, dtype=np.float32)
    board_6x7x2 = np.zeros((6, 7, 2), dtype=np.float32)
    board_6x7x2[:, :, 0] = (board_array == 1).astype(np.float32)
    board_6x7x2[:, :, 1] = (board_array == 2).astype(np.float32)
    return board_6x7x2


# ------------------------------------------------------------------------
# 6) AI Move (returns just the column)
# ------------------------------------------------------------------------
@anvil.server.callable
def get_ai_move(board, model_type="cnn"):
    """
    Given the current board, return the best column for the AI to play.
    This function *does not* mutate the board—just picks a move.
    """
    board_input = convert_to_6x7x2(board)
    board_input = np.expand_dims(board_input, axis=0)

    if model_type == "cnn" and cnn_model is not None:
        prediction = cnn_model.predict(board_input)
    elif model_type == "transformer" and transformer_model is not None:
        prediction = transformer_model.predict(board_input)
    else:
        return {"error": f"Model type '{model_type}' not found or not loaded."}

    return int(np.argmax(prediction))


# ------------------------------------------------------------------------
# 7) Check winner (returns True/False)
# ------------------------------------------------------------------------
@anvil.server.callable
def check_winner_server(board, piece):
    """
    Checks if the given piece (1 or 2) has four in a row.
    The board is a 2D list (6 rows x 7 cols).
    """
    board_arr = np.array(board, dtype=int)

    # Horizontal
    for c in range(COLUMN_COUNT - 3):
        for r in range(ROW_COUNT):
            if all(board_arr[r, c + i] == piece for i in range(4)):
                return True
    
    # Vertical
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT - 3):
            if all(board_arr[r + i, c] == piece for i in range(4)):
                return True
    
    # Positively sloped diagonals
    for c in range(COLUMN_COUNT - 3):
        for r in range(ROW_COUNT - 3):
            if all(board_arr[r + i, c + i] == piece for i in range(4)):
                return True
    
    # Negatively sloped diagonals
    for c in range(COLUMN_COUNT - 3):
        for r in range(3, ROW_COUNT):
            if all(board_arr[r - i, c + i] == piece for i in range(4)):
                return True
    
    return False


# ------------------------------------------------------------------------
# 8) Keep the Anvil server link alive
# ------------------------------------------------------------------------
anvil.server.wait_forever()