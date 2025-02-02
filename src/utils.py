import matplotlib.pyplot as plt
import numpy as np

def load_image(data, data_labels, index):
    #make a display function from this 
    with open("../data/mfeat-pix") as datafile:
        raw_data = datafile.read()

    lines = raw_data.split('\n')
    data = [line.split() for line in lines]

    numbers = [
        int(x)
        for line in data
        for x in line
    ]

    arr =(np.array(numbers).reshape((2000, 16,15)))* 255/6
    
    
    

    # Show first data set
    plt.imshow(arr[index], cmap="gray")
    plt.title(f"Label: {data_labels[index]}")
    plt.show()
    
    
def visualization(history, title, epochs = 300):
    #filtered_info = {k: v for k, v in info.items() if v != 0.0}
    


    # Create figure and subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title)
    # Left subplot: Loss vs Epochs
    axes[0].plot(epochs, history['loss_train'], label="Train Loss", color="blue")
    axes[0].plot(epochs, history['loss_test'], label="Test Loss", color="orange")#, linestyle="dashed")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Testing Loss vs Epochs")
    axes[0].legend()
    axes[0].grid(True)

    # Right subplot: Accuracy vs Epochs
    axes[1].plot(epochs, history['accuracy_train'], label="Train Accuracy", color="blue")
    axes[1].plot(epochs, history['accuracy_test'], label="Test Accuracy", color="orange")#, linestyle="dashed")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training & Testing Accuracy vs Epochs")
    axes[1].legend()
    axes[1].grid(True)

    # Adjust layout
    plt.tight_layout()
    plt.show()