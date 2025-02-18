import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt

with open("A2_right_vmae_16x4_crop_fold0.pkl", "rb") as f:
    prob_data = pickle.load(f)
video_key = 'Right_side_window_user_id_12670_NoAudio_7'  
prob_list = prob_data[video_key]
prob_array = np.array([np.array(frame_probs, dtype=np.float32) for frame_probs in prob_list])
video_path = "Right_side_window_user_id_12670_NoAudio_7.MP4"
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
output_path = "output_Right_side_window_user_id_12670_NoAudio_7.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height + 350))
# insert image
label_img = cv2.imread("labels.png")
bar_width = int(frame_width * 0.7)
label_width = frame_width - bar_width
label_img = cv2.resize(label_img, (label_width, 350))

def draw_bar_chart(probs):
    fig, ax = plt.subplots(figsize=(10, 4))
    max_idx = np.argmax(probs)
    colors = ['blue'] * 16
    colors[max_idx] = 'red'
    ax.bar(range(16), probs, color=colors)
    ax.set_ylim(0, 1)
    ax.set_xticks(range(16))
    ax.set_xticklabels(range(16))
    ax.set_ylabel("Probability")
    fig.canvas.draw()
    bar_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    bar_img = bar_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    bar_img = cv2.resize(bar_img, (bar_width, 350))
    plt.close(fig)
    return bar_img

frame_idx = 0
second_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    print('Processing frame:', frame_idx)
    print('Second idx:', second_idx)
    if not ret:
        break
    if frame_idx % fps == 0 and second_idx < 180:
        current_probs = prob_array[second_idx]
        bar_chart = draw_bar_chart(current_probs)
        second_idx += 1
    elif second_idx > 0: # show probability for the last frame
        bar_chart = draw_bar_chart(prob_array[second_idx - 1])
    bottom_section = np.hstack((bar_chart, label_img))
    combined_frame = np.vstack((frame, bottom_section))
    out.write(combined_frame)
    frame_idx += 1
cap.release()
out.release()
cv2.destroyAllWindows()
