import streamlit as st
from streamlit_navigation_bar import st_navbar
from PIL import Image
import numpy as np
import cv2
import pyopencl as cl
import time
import psutil
import os
import tensorflow.keras as K
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
import tracemalloc

# Load the pre-trained ResNet50 model
model = ResNet50(weights='imagenet')

# OpenCL setup
platform = cl.get_platforms()[0]
device = platform.get_devices()[0]
context = cl.Context([device])
queue = cl.CommandQueue(context)

# OpenCL kernel for resizing
kernel_code = """
__kernel void resize(__global const uchar *input_image,
                     __global uchar *output_image,
                     const int input_width,
                     const int input_height,
                     const int output_width,
                     const int output_height) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= output_width || y >= output_height) return;

    int src_x = (int)((float)x / output_width * input_width);
    int src_y = (int)((float)y / output_height * input_height);

    int src_idx = (src_y * input_width + src_x) * 3;
    int dst_idx = (y * output_width + x) * 3;

    output_image[dst_idx] = input_image[src_idx];
    output_image[dst_idx + 1] = input_image[src_idx + 1];
    output_image[dst_idx + 2] = input_image[src_idx + 2];
}
"""

# Compile the OpenCL program
program = cl.Program(context, kernel_code).build()


# Preprocess image using OpenCL
def preprocess_image_opencl(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_height, input_width, channels = image.shape
    output_width, output_height = 224, 224

    image_flattened = image.flatten().astype(np.uint8)
    mf = cl.mem_flags
    input_image_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=image_flattened)
    output_image_buf = cl.Buffer(context, mf.WRITE_ONLY, output_width * output_height * channels)

    kernel = program.resize
    kernel.set_arg(0, input_image_buf)
    kernel.set_arg(1, output_image_buf)
    kernel.set_arg(2, np.int32(input_width))
    kernel.set_arg(3, np.int32(input_height))
    kernel.set_arg(4, np.int32(output_width))
    kernel.set_arg(5, np.int32(output_height))

    cl.enqueue_nd_range_kernel(queue, kernel, (output_width, output_height), None)
    output_image = np.empty(output_width * output_height * channels, dtype=np.uint8)
    cl.enqueue_copy(queue, output_image, output_image_buf).wait()
    output_image = output_image.reshape((output_height, output_width, channels)).astype(np.float32)
    return output_image


# Preprocess image using CPU
def preprocess_image_cpu(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized_image = cv2.resize(image, (224, 224)).astype(np.float32)
    return resized_image


# Function to analyze memory usage
def memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


# Process image and track metrics
def process_image_and_track_metrics(img_array):
    tracemalloc.start()
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    start_time = time.time()
    predictions = model.predict(img_array)
    end_time = time.time()

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    decoded_predictions = decode_predictions(predictions, top=3)[0]

    inference_time = end_time - start_time
    memory_usage_increase = peak / (1024 * 1024)  # Convert bytes to MiB

    return decoded_predictions, inference_time, memory_usage_increase


# Streamlit application layout
# Define the pages for the navbar
pages = ["Home", "Reporting", "User Guide", "Help"]

# Define the styles with the updated navbar background color
styles = {
    "nav": {
        "background-color": "rgb(109, 119, 93)",
    },
    "div": {
        "max-width": "32rem",
    },
    "span": {
        "border-radius": "0.5rem",
        "color": "rgb(49, 51, 63)",
        "margin": "0 0.125rem",
        "padding": "0.4375rem 0.625rem",
    },
    "active": {
        "background-color": "rgba(255, 255, 255, 0.25)",
    },
    "hover": {
        "background-color": "rgba(255, 255, 255, 0.35)",
    },
}

# Create the navbar with the defined styles and pages
page = st_navbar(pages, styles=styles, key="main_navbar")

# Display the current page
st.write(page)
st.title("Joffrey Hammit's Capstone Application: OpenCL Preprocessing for Image Recognition")

# Conditionally display for home page
if page == "Home":
    st.write("Thank you for vising my application!")
    st.write("Upload an image here to test the OpenCL Pattern Recognition Model:")
    st.write("JPEG File Upload")

    # File upload, specifically jpegs
    uploaded_file = st.file_uploader("Choose a JPEG file", type=["jpeg", "jpg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded JPEG Image', use_column_width=True)

        # Convert the uploaded image to a format suitable for processing
        image_np = np.array(image)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCL

        # OpenCL Processing
        st.subheader("Processing with OpenCL...")
        opencl_image = preprocess_image_opencl(image_np)
        opencl_predictions, opencl_inference_time, opencl_memory_increase = process_image_and_track_metrics(opencl_image)

        st.write("**OpenCL Predictions:**")
        for i, (imagenet_id, label, score) in enumerate(opencl_predictions):
            st.write(f"{i + 1}: {label} ({score * 100:.2f}%)")
        st.write(f"OpenCL Inference Time: {opencl_inference_time:.4f} seconds")
        st.write(f"OpenCL Memory Usage Increase: {opencl_memory_increase:.4f} MiB")

        # CPU Processing
        st.subheader("Processing with CPU...")
        cpu_image = preprocess_image_cpu(image_np)
        cpu_predictions, cpu_inference_time, cpu_memory_increase = process_image_and_track_metrics(cpu_image)

        st.write("**CPU Predictions:**")
        for i, (imagenet_id, label, score) in enumerate(cpu_predictions):
            st.write(f"{i + 1}: {label} ({score * 100:.2f}%)")
        st.write(f"CPU Inference Time: {cpu_inference_time:.4f} seconds")
        st.write(f"CPU Memory Usage Increase: {cpu_memory_increase:.4f} MiB")

if page == "Reporting":
    st.write("Recent File Upload...")

if page == "Help":
    st.write("Contact Joffrey Hammit at joffrey.hammit@gmail.com for support if any issues occur during your use of this program.")

if page == "User Guide":

    st.write("""
        **This guide will help you navigate the interface and effectively use the features of this program.**

        ### 1. Getting Started
        To begin using the application:
        - **Open the Streamlit Interface**: Launch the application in your web browser. [Insert link here]
        - **Sign In**: With authentication enabled, sign in with your credentials. If you don’t have an account, use the registration form to create one.

        ### 2. Uploading Data
        Once logged in:
        - **Upload Your Data on the Home Page**: Use the "Upload" button to select and upload your dataset. The application supports JPG format.

        ### 3. Running Pattern Recognition
        After uploading your data:
        - **Start Processing**: Click the "Run Pattern Recognition" button to initiate the capstone project algorithm. The application uses OpenCL for efficient data processing, leveraging your hardware's capabilities.

        ### 4. Viewing Results
        Once the processing is complete:
        - **Access Results**: The results will be displayed on the Results page. This includes:
            - **Recognition Outputs**: Visual and numerical outputs highlighting the identified patterns in your data.
            - **Performance Metrics**: Information on the speed and efficiency of the data processing, leveraging OpenCL’s capabilities.

        ### 5. Exporting Reports
        - **Download Reports**: If you wish to save the results, use the "Download Report" button to export a summary of the findings and performance metrics in CSV format.

        ### 6. Troubleshooting
        - **Error Handling**: If an error occurs, a message will be displayed with details. Ensure your data is in the correct format and try again.
        - **Support**: For further assistance, contact the support team through the "Help" section.

        ### 7. Logging Out
        - **End Your Session**: When you’re finished, click the "Logout" button to securely end your session.

        **Thank you for using the Pattern Recognition Application!**
        """)
