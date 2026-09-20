"""Example usage of emg_pipeline.

1) Standalone acquisition
2) How to wire it into the trial GUI (emg_gui.py) so each action's
   start/end is written into the CSV as a marker, using the board's own
   clock via insert_marker().
"""

import time

from emg.acquisition import MindRove, EMGStream
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import socket

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, encoder_outputs, hidden):

        seq_len = encoder_outputs.size(1)
        h_tiled = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        energy = torch.tanh(self.attn(torch.cat((encoder_outputs, h_tiled), dim=2)))
        attn_weights = torch.softmax(self.v(energy), dim=1)
        context_vector = torch.sum(attn_weights * encoder_outputs, dim=1)

        return context_vector

class DPARSEncoder(nn.Module):
    def __init__(self, input_size, encode_size, num_classes):
        super(DPARSEncoder, self).__init__()
        second_size = input_size*2
        
        self.bn1 = nn.BatchNorm1d(second_size)
        self.bn2 = nn.BatchNorm1d(encode_size)
        self.dropout = nn.Dropout(p=0.3)
        self.attention = Attention(2*encode_size)

        self.depthwise1 = nn.Conv1d(
            input_size, input_size, kernel_size=64, stride = 16, groups=input_size, bias=False)
        self.pointwise1 = nn.Conv1d(input_size, second_size, kernel_size=1)


        self.depthwise2 = nn.Conv1d(
            second_size, second_size, kernel_size=16, stride = 4, groups=second_size, bias=False)
        self.pointwise2 = nn.Conv1d(second_size, encode_size, kernel_size=1)

        self.fc1 = nn.Linear(encode_size, num_classes)
        self.relu = nn.ReLU()


    def forward(self, x):

        # Input -> (batch_size, 2000, 8)
        x = x.permute(0, 2, 1)
        x = self.depthwise1(x)  # output: (batch, 64, 48)
        x = self.pointwise1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        # print(x.shape)


        x = self.depthwise2(x)  # output: (batch, 64, 48)
        x = self.pointwise2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = x.permute(0, 2, 1)
        x = self.attention(x, x[:, -1, :])
        # print(x.shape)

        return self.fc1(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DPARSEncoder(8, 16, 3).to(device)
state_dict = torch.load("model.pth", weights_only=True)
model.load_state_dict(state_dict)
####


####


emg_device = MindRove()  # uses BoardIds.MINDROVE_WIFI_BOARD by default
try:
    emg_device.connect()
except ConnectionError:
    print("Error")
    raise("Error")

stream = EMGStream(emg_device, csv_path="inference.csv", bandpass=(20, 200), notch_freq=60.0)
stream.start()


# EMA smoothing factor
alpha = 0.2

pred_label = []
pred_label_raw = []
true_label = []

# Previous EMA probabilities
ema_probs = None


model.eval()
PI_IP = "192.168.137.238"
PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


print("Waiting to populate buffer")
time.sleep(5)

while True:

    with torch.no_grad():

        # --------------------------------------------------
        # Get window
        # --------------------------------------------------

        temp_eeg = np.transpose(stream.FilteredBuffer.get_latest(2000).copy())

        # --------------------------------------------------
        # Preprocess EMG
        # --------------------------------------------------

        # temp_eeg -= np.mean(temp_eeg, axis=1, keepdims=True)

        # temp_mean = np.mean(temp_eeg, axis=0, keepdims=True)

        # temp_std = np.std(temp_eeg, axis=0, keepdims=True)

        # temp_eeg = (temp_eeg - temp_mean) / (temp_std + 1e-8)

        # --------------------------------------------------
        # Model inference
        # --------------------------------------------------

        temp_eeg = torch.tensor(
            temp_eeg,
            dtype=torch.float32
        ).unsqueeze(0).to(device)

        outputs = model(temp_eeg)

        # --------------------------------------------------
        # Raw prediction
        # --------------------------------------------------

        raw_pred = outputs.argmax(dim=1).item()
        pred_label_raw.append(raw_pred)

        # --------------------------------------------------
        # Convert logits -> probabilities
        # --------------------------------------------------

        probs = torch.softmax(outputs, dim=1)

        # Shape: (3,)
        probs = probs.cpu().numpy()[0]

        # --------------------------------------------------
        # Causal EMA
        # --------------------------------------------------

        if ema_probs is None:
            # First prediction initializes EMA
            ema_probs = probs.copy()
        else:
            ema_probs = ( alpha * probs + (1.0 - alpha) * ema_probs )

        # --------------------------------------------------
        # Smoothed prediction
        # --------------------------------------------------

        pred = np.argmax(ema_probs)
        pred_label.append(pred)

        packet = ""
        if pred == 0:
            packet = (0, 0, 0)
        elif pred == 1:
            packet = (3, -1, -1)
        else:
            packet = (-2, 2, 2)

        packet = ",".join(str(x) for x in packet)

        print(f"{pred}: {packet}")

        sock.sendto(
                packet.encode(),
                (PI_IP, PORT)
            )
        time.sleep(0.5)

sock.close()

